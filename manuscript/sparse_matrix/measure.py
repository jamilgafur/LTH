import os
import time
import uuid
import gc
from typing import Dict, Tuple, Optional

import torch
from codecarbon import OfflineEmissionsTracker
import torch.profiler
import psutil  # pip install psutil

# ----------------------------
# Device and Memory Utilities
# ----------------------------

def clear_memory(device: Optional[torch.device] = None):
    """Free caches and run GC for the given device (if CUDA)."""
    gc.collect()
    if device is not None and getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass  # best-effort

def to_torch_device(device) -> torch.device:
    """Normalize device input (accepts str or torch.device)."""
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    return torch.device("cpu")

def move_to_device_if_needed(obj, device: torch.device):
    """Move module or tensor to device if needed."""
    if isinstance(obj, torch.nn.Module):
        try:
            current_device = next(obj.parameters()).device
        except StopIteration:
            return obj.to(device)
        if current_device != device:
            return obj.to(device)
        return obj
    elif isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    else:
        return obj

# ----------------------------
# Emissions Tracking
# ----------------------------

def start_emissions_tracker(model, device: torch.device, run_id: str, measure_power_secs: float):
    output_dir = os.path.join(".", "codecarbon", model.__class__.__name__, device.type, run_id)
    os.makedirs(output_dir, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        measure_power_secs=measure_power_secs,
        country_iso_code="USA",
        output_dir=output_dir,
        log_level="error",
        save_to_file=True,
        output_file=f"emissions-{run_id}.csv"
    )
    tracker.start()
    return tracker, output_dir

def read_emissions_data(csv_path: str) -> Dict[str, str]:
    """Read last row of codecarbon CSV into dict. Returns empty dict if file missing."""
    try:
        with open(csv_path, "r") as f:
            lines = [ln for ln in f.readlines() if ln.strip()]
            if len(lines) < 2:
                return {}
            header = lines[0].strip().split(",")
            last_row = lines[-1].strip().split(",")
            return dict(zip(header, last_row))
    except FileNotFoundError:
        return {}

# ----------------------------
# Analytical memory calculation
# ----------------------------

def get_tensor_memory_MB(model: torch.nn.Module, x: torch.Tensor) -> float:
    """
    Calculate total memory (MB) of parameters (dense & CSR sparse), buffers, and input tensor `x`.
    """
    total_bytes = 0

    def add_tensor(t: torch.Tensor):
        nonlocal total_bytes
        if t is None:
            return
        if t.is_sparse_csr:
            total_bytes += t.values().numel() * t.values().element_size()
            total_bytes += t.crow_indices().numel() * t.crow_indices().element_size()
            total_bytes += t.col_indices().numel() * t.col_indices().element_size()
        else:
            total_bytes += t.numel() * t.element_size()

    for _, param in model.named_parameters(recurse=True):
        add_tensor(param)
    for _, buf in model.named_buffers(recurse=True):
        add_tensor(buf)
    if isinstance(x, torch.Tensor):
        add_tensor(x)

    return total_bytes / (1024 ** 2)  # MB

# ----------------------------
# Measurement Core Logic
# ----------------------------

@torch.no_grad()
def measure_single_run(
    model: torch.nn.Module,
    x: torch.Tensor,
    device,
    run,
    measure_power_secs: float = 0.001,
    warmup: int = 1,
    use_profiler: bool = True
):
    """
    Run a forward pass with memory & emissions measurement.
    """
    device = to_torch_device(device)
    run_id = str(uuid.uuid4())
    warnings = []

    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)
    model.eval()

    tracker, output_dir = start_emissions_tracker(model, device, run_id, measure_power_secs)

    # Warm-up
    for _ in range(max(0, int(warmup))):
        _ = model(x)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    prof = None
    trace_path, start_time, end_time = None, None, None
    peak_cpu_bytes, peak_cuda_bytes = 0, 0

    proc = psutil.Process(os.getpid())
    baseline_rss = None
    try:
        baseline_rss = proc.memory_info().rss
    except Exception:
        pass

    try:
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

        if use_profiler:
            prof = torch.profiler.profile(
                activities=activities,
                profile_memory=True,
                with_stack=False
            )
            prof.__enter__()

        start_time = time.time()
        _ = model(x)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        end_time = time.time()

        if use_profiler and prof is not None:
            prof.__exit__(None, None, None)
            try:
                for evt in prof.key_averages():
                    for attr in ("self_cpu_memory_usage", "cpu_memory_usage"):
                        val = getattr(evt, attr, None)
                        if isinstance(val, (int, float)) and val > peak_cpu_bytes:
                            peak_cpu_bytes = int(val)
                    for attr in ("self_cuda_memory_usage", "cuda_memory_usage", "device_memory_usage"):
                        val = getattr(evt, attr, None)
                        if isinstance(val, (int, float)) and val > peak_cuda_bytes:
                            peak_cuda_bytes = int(val)
            except Exception as e:
                warnings.append(f"profiler parsing failed: {e}")
            try:
                trace_path = os.path.join(output_dir, f"trace_{device.type}_{run_id}.json")
                prof.export_chrome_trace(trace_path)
            except Exception as e:
                warnings.append(f"trace export failed: {e}")

        measured_bytes = None
        if device.type == "cuda" and torch.cuda.is_available():
            if peak_cuda_bytes > 0:
                measured_bytes = peak_cuda_bytes
            else:
                try:
                    measured_bytes = torch.cuda.max_memory_allocated(device=device)
                except Exception:
                    warnings.append("CUDA memory fallback failed.")
        else:
            if peak_cpu_bytes > 0:
                measured_bytes = peak_cpu_bytes
            else:
                try:
                    total_bytes = sum(
                        obj.numel() * obj.element_size()
                        for obj in gc.get_objects()
                        if torch.is_tensor(obj) and obj.device.type == "cpu"
                    )
                    if total_bytes > 0:
                        measured_bytes = total_bytes
                except Exception as e:
                    warnings.append(f"GC scan failed: {e}")
                if measured_bytes is None and baseline_rss is not None:
                    try:
                        after_rss = proc.memory_info().rss
                        delta = max(0, after_rss - baseline_rss)
                        measured_bytes = delta if delta > 0 else after_rss
                    except Exception as e:
                        warnings.append(f"RSS fallback failed: {e}")

        peak_mem_MB_measured = measured_bytes / (1024 ** 2) if measured_bytes else None

    finally:
        try:
            tracker.stop()
        except Exception:
            pass

    analytical_mem = get_tensor_memory_MB(model, x)

    try:
        del prof
    except Exception:
        pass
    gc.collect()
    clear_memory(device)

    return {
        "run_id": run_id,
        "device_type": device.type,
        "duration": (end_time - start_time) if (start_time and end_time) else None,
        "peak_mem_MB_analytical": analytical_mem,
        "peak_mem_MB_measured": peak_mem_MB_measured,
        "emissions_data": read_emissions_data(os.path.join(output_dir, f"emissions-{run_id}.csv")),
        "trace_path": trace_path,
        "warnings": warnings
    }

# ----------------------------
# Measurement Entry Point
# ----------------------------

def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 2,
    measure_power_secs: float = 0.001,
    warmup: int = 1,
    use_profiler: bool = True
) -> Tuple[Dict[int, Dict], float]:
    clear_memory(device)
    run_data = {}

    for r in range(runs):
        clear_memory(device)
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_data[r] = measure_single_run(model, x, device, r, measure_power_secs, warmup=warmup, use_profiler=use_profiler)
        print(f"run {r} result:", run_data[r])
        clear_memory(device)

    try:
        del model, x
    except Exception:
        pass
    gc.collect()
    clear_memory(device)

    return run_data
