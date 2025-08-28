import os
import time
import uuid
import gc
import torch
from typing import Dict, Tuple
from codecarbon import OfflineEmissionsTracker
from memory_profiler import memory_usage

# ----------------------------
# Device and Memory Utilities
# ----------------------------

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def move_to_device_if_needed(obj, device):
    if isinstance(obj, torch.nn.Module):
        current_device = next(obj.parameters()).device
    else:
        current_device = obj.device
    if current_device != device:
        return obj.to(device)
    return obj

# ----------------------------
# Emissions Tracking
# ----------------------------

def start_emissions_tracker(model, device, run_id, measure_power_secs):
    output_dir = f"./codecarbon/{model.__class__.__name__}/{device.type}/{run_id}"
    os.makedirs(output_dir, exist_ok=True)

    tracker = OfflineEmissionsTracker(
        measure_power_secs=measure_power_secs,
        country_iso_code="USA",
        country_2letter_iso_code="US-NW-PSCO",
        output_dir=output_dir,
        log_level="error",
        save_to_file=True,
        output_file=f"emissions-{run_id}.csv"
    )
    tracker.start()
    return tracker, output_dir

def read_emissions_data(csv_path: str) -> Dict[str, str]:
    with open(csv_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        last_row = lines[-1].strip().split(",")
        return dict(zip(header, last_row))

# ----------------------------
# Measurement Core Logic
# ----------------------------

def get_tensor_memory_MB(model: torch.nn.Module, x: torch.Tensor, device: torch.device) -> float:
    """
    Calculates total memory used by:
    - Dense and sparse model parameters (weights & biases)
    - Buffers
    - Input tensor `x`
    Returns total memory in megabytes (MB).
    Takes device into account: sums only tensors located on the specified device.
    """

    total_bytes = 0

    for module in model.modules():
        # --- Dense or standard weight ---
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            weight = module.weight
            if weight.device == device:
                if weight.is_sparse_csr:
                    total_bytes += weight.values().numel() * weight.values().element_size()
                    total_bytes += weight.crow_indices().numel() * weight.crow_indices().element_size()
                    total_bytes += weight.col_indices().numel() * weight.col_indices().element_size()
                else:
                    total_bytes += weight.numel() * weight.element_size()

        # --- Bias ---
        if hasattr(module, 'bias') and module.bias is not None and isinstance(module.bias, torch.Tensor):
            if module.bias.device == device:
                total_bytes += module.bias.numel() * module.bias.element_size()

        # --- SparseLinear or SparseConv2d (custom sparse_weight attribute) ---
        if hasattr(module, 'sparse_weight'):
            sw = module.sparse_weight
            if sw.device == device:
                if sw.is_sparse_csr:
                    total_bytes += sw.values().numel() * sw.values().element_size()
                    total_bytes += sw.crow_indices().numel() * sw.crow_indices().element_size()
                    total_bytes += sw.col_indices().numel() * sw.col_indices().element_size()
                else:
                    total_bytes += sw.numel() * sw.element_size()

        # Bias for sparse layers (if not counted above)
        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
            if module.bias.device == device:
                total_bytes += module.bias.numel() * module.bias.element_size()

    # Buffers (e.g., running mean/var in batchnorm)
    for buffer in model.buffers():
        if buffer.device == device:
            total_bytes += buffer.numel() * buffer.element_size()

    # Input tensor
    if x.device == device:
        total_bytes += x.numel() * x.element_size()

    return total_bytes / (1024 ** 2)  # Convert to MB


@torch.no_grad()
def measure_single_run(model, x, device, run, measure_power_secs):
    run_id = str(uuid.uuid4())
    tracker, output_dir = start_emissions_tracker(model, device, run_id, measure_power_secs)

    model = model.to(device)
    x = x.to(device)
    model.eval()

    # Warm-up run to stabilize memory
    model(x)

    start_time = time.time()

    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        model(x)

        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        print(f"[DEBUG][CUDA] Peak tensor memory (tracked by CUDA): {peak_mem_mb:.2f} MB")
        print(f"[DEBUG][CUDA] Allocated now: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
        print(f"[DEBUG][CUDA] Reserved now: {torch.cuda.memory_reserved(device) / 1024 ** 2:.2f} MB")
    else:
        peak_mem_mb = get_tensor_memory_MB(model, x, device)
        print(f"[DEBUG][CPU] Approximate tensor memory usage: {peak_mem_mb:.2f} MB")

    end_time = time.time()
    tracker.stop()

    csv_path = os.path.join(output_dir, f"emissions-{run_id}.csv")
    emissions_data = read_emissions_data(csv_path)

    # Cleanup
    del tracker
    gc.collect()
    clear_memory()

    return {
        "duration": end_time - start_time,
        "peak_mem_MB": peak_mem_mb,
        "emissions_data": emissions_data,
    }


# ----------------------------
# Measurement Entry Point
# ----------------------------

def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 2,
    measure_power_secs: int = 1
) -> Tuple[Dict[int, Dict[str, float]], float]:
    clear_memory()

    run_data = {}
    for run in range(runs):
        clear_memory()
        torch.cuda.empty_cache()
        run_data[run] = measure_single_run(model, x, device, run, measure_power_secs)
        clear_memory()

    # Final cleanup
    del model
    del x
    gc.collect()
    clear_memory()

    return run_data
