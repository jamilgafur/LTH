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
            # best-effort
            pass

def to_torch_device(device) -> torch.device:
    """Normalize device input (accepts str or torch.device)."""
    if isinstance(device, str):
        return torch.device(device)
    if isinstance(device, torch.device):
        return device
    # fallback: cpu
    return torch.device("cpu")

def move_to_device_if_needed(obj, device: torch.device):
    """Move module or tensor to device if needed."""
    if isinstance(obj, torch.nn.Module):
        try:
            current_device = next(obj.parameters()).device
        except StopIteration:
            # no parameters --> just call .to()
            return obj.to(device)
        if current_device != device:
            return obj.to(device)
        return obj
    elif isinstance(obj, torch.Tensor):
        if obj.device != device:
            return obj.to(device)
        return obj
    else:
        # not a tensor/module: return unchanged
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
        # country_2letter_iso_code parameter in your original code looked odd; keep it if you need a custom region
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
    Calculate total memory (MB) of parameters (dense & CSR sparse), buffers, and input tensor `x`
    but only counting tensors that live on the given device (best-effort).
    """
    total_bytes = 0

    def add_tensor(t: torch.Tensor):
        nonlocal total_bytes
        if t is None:
            return
        
        # sparse CSR
        if t.is_sparse_csr:
            total_bytes += t.values().numel() * t.values().element_size()
            total_bytes += t.crow_indices().numel() * t.crow_indices().element_size()
            total_bytes += t.col_indices().numel() * t.col_indices().element_size()
        else:
            total_bytes += t.numel() * t.element_size()

    # parameters & buffers
    for name, param in model.named_parameters(recurse=True):
        add_tensor(param)
    for name, buf in model.named_buffers(recurse=True):
        add_tensor(buf)

    # input tensor
    if isinstance(x, torch.Tensor):
        add_tensor(x)

    return total_bytes / (1024 ** 2)  # MB

# ----------------------------
# Measurement Core Logic (improved)
# ----------------------------

import time, uuid, gc, os
import torch
import torch.profiler
import psutil  # optional fallback for CPU RSS delta

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
    Uses torch.profiler to measure memory for both CPU and CUDA.

    Returns:
      dict with keys:
        - run_id, device_type, duration,
        - peak_mem_MB_measured (profiler-based or fallback),
        - peak_mem_MB_analytical,
        - emissions_data, trace_path, warnings
    """
    # normalize device
    if isinstance(device, str):
        device = torch.device(device)
    elif not isinstance(device, torch.device):
        device = torch.device("cpu")

    run_id = str(uuid.uuid4())
    warnings = []

    # Move model & input to device
    model = model.to(device)
    x = x.to(device)
    model.eval()

    tracker, output_dir = start_emissions_tracker(model, device, run_id, measure_power_secs)

    # Warm-up forward passes (to stabilize allocator & caches)
    for _ in range(max(0, int(warmup))):
        _ = model(x)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    # Prepare profiler activities
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    prof = None
    trace_path = None
    start_time = None
    end_time = None
    peak_cpu_bytes = 0
    peak_cuda_bytes = 0

    # Optional baseline CPU RSS (fallback)
    proc = psutil.Process(os.getpid())
    baseline_rss = None
    try:
        baseline_rss = proc.memory_info().rss
    except Exception:
        baseline_rss = None

    try:
        # Reset CUDA peak counters (helpful)
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass

        # Start profiler context and time the forward
        if use_profiler:
            prof = torch.profiler.profile(
                activities=activities,
                profile_memory=True,
                with_stack=False  # set True if you need stack traces (slower)
            )

            prof.__enter__()

        start_time = time.time()
        _ = model(x)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        end_time = time.time()

        if use_profiler and prof is not None:
            # close profiler context
            prof.__exit__(None, None, None)

            # Extract peak memory from profiler events (robust attribute checks)
            try:
                key_averages = prof.key_averages()
                for evt in key_averages:
                    # check several plausible attribute names to be robust across versions
                    for attr in ("self_cpu_memory_usage", "cpu_memory_usage"):
                        val = getattr(evt, attr, None)
                        if isinstance(val, (int, float)) and val > peak_cpu_bytes:
                            peak_cpu_bytes = int(val)
                    for attr in ("self_cuda_memory_usage", "cuda_memory_usage", "device_memory_usage"):
                        val = getattr(evt, attr, None)
                        if isinstance(val, (int, float)) and val > peak_cuda_bytes:
                            peak_cuda_bytes = int(val)
            except Exception as e:
                warnings.append(f"profiler key_averages parsing failed: {e}")

            # export trace
            try:
                trace_path = os.path.join(output_dir, f"trace_{device.type}_{run_id}.json")
                prof.export_chrome_trace(trace_path)
            except Exception as e:
                warnings.append(f"profiler.export_chrome_trace failed: {e}")

        # If profiler didn't produce numbers, use safe fallbacks
        measured_bytes = None
        if device.type == "cuda" and torch.cuda.is_available():
            if peak_cuda_bytes and peak_cuda_bytes > 0:
                measured_bytes = peak_cuda_bytes
            else:
                # fallback to PyTorch CUDA allocator peak
                try:
                    measured_bytes = torch.cuda.max_memory_allocated(device=device)
                except Exception:
                    measured_bytes = None
                    warnings.append("Both profiler and torch.cuda.max_memory_allocated() failed to yield CUDA peak.")
        else:
            # CPU case: use profiler number if present, else GC-based sum of live tensors, else RSS delta
            if peak_cpu_bytes and peak_cpu_bytes > 0:
                measured_bytes = peak_cpu_bytes
            else:
                # GC-scan: sum sizes of live torch tensors on CPU
                total_bytes = 0
                try:
                    for obj in gc.get_objects():
                        if torch.is_tensor(obj):
                            try:
                                # only count CPU tensors
                                if obj.device.type == "cpu":
                                    total_bytes += obj.numel() * obj.element_size()
                            except Exception:
                                pass
                    if total_bytes > 0:
                        measured_bytes = total_bytes
                except Exception as e:
                    warnings.append(f"GC tensor-scan failed: {e}")

                # last fallback: RSS delta
                if measured_bytes is None and baseline_rss is not None:
                    try:
                        after_rss = proc.memory_info().rss
                        delta = max(0, after_rss - baseline_rss)
                        measured_bytes = delta
                        # If delta is zero, still set measured_bytes to after_rss (full process footprint)
                        if measured_bytes == 0:
                            measured_bytes = after_rss
                            warnings.append("RSS delta was zero; returning total RSS as fallback.")
                    except Exception as e:
                        warnings.append(f"psutil RSS fallback failed: {e}")
                        measured_bytes = None

        # convert to MB
        peak_mem_MB_measured = None
        if measured_bytes is not None:
            peak_mem_MB_measured = float(measured_bytes) / (1024 ** 2)

    finally:
        # ensure emissions tracker stopped
        try:
            tracker.stop()
        except Exception:
            pass

    # Analytical memory (unchanged)
    analytical_mem = get_tensor_memory_MB(model, x)

    # cleanup
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
# Measurement Entry Point (minor fixes)
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
        # clear CUDA cache if available
        if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        run_data[r] = measure_single_run(model, x, device, r, measure_power_secs, warmup=warmup, use_profiler=use_profiler)
        print(f"run {r} result:", run_data[r])
        clear_memory(device)

    # Final cleanup
    try:
        del model
        del x
    except Exception:
        pass
    gc.collect()
    clear_memory(device)

    return run_data
