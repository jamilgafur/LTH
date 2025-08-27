import torch
import time
import os
import uuid
import psutil
import gc
from codecarbon import OfflineEmissionsTracker
from typing import Dict, Tuple
import tracemalloc


def clear_memory():
    """Clear GPU and CPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def move_to_device_if_needed(obj, device):
    """Moves tensor or model to the given device if needed."""
    if isinstance(obj, torch.nn.Module):
        current_device = next(obj.parameters()).device
    else:
        current_device = obj.device

    if current_device != device:
        return obj.to(device)
    return obj

def prepare_device_tracking(device: torch.device):
    """Prepare for device memory tracking (GPU/CPU)."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        tracemalloc.start()

def stop_device_tracking(device: torch.device):
    """Stop device memory tracking and return the peak memory."""
    if device.type == 'cuda':
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # in MB
    else:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024 ** 2)  # in MB

# -------------
def get_cpu_peak_memory_MB():
    with open('/proc/self/status', 'r') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                # VmHWM is peak resident set size (kB)
                peak_kb = int(line.split()[1])
                return peak_kb / 1024  # MB
    return 0

def get_model_memory_MB(model):
    total_size_bytes = 0

    for param in model.parameters():
        if param.is_sparse and param.layout == torch.sparse_csr:
            csr = param
            total_size_bytes += csr.crow_indices().element_size() * csr.crow_indices().nelement()
            total_size_bytes += csr.col_indices().element_size() * csr.col_indices().nelement()
            total_size_bytes += csr.values().element_size() * csr.values().nelement()
        else:
            total_size_bytes += param.element_size() * param.nelement()

    total_size_MB = total_size_bytes / (1024 ** 2)
    return total_size_MB

def measure_single_run(model, x, device, run, measure_power_secs):
    import time
    import os
    import uuid
    from codecarbon import OfflineEmissionsTracker

    run_id = str(uuid.uuid4())
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

    model = move_to_device_if_needed(model, device)
    model.eval()

    # Warm-up
    with torch.no_grad():
        model(x)

    tracker.start()
    start_time = time.time()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        # For CPU, you could optionally run inference twice or in a subprocess for peak
        with torch.no_grad():
            model(x)
        peak_mem_mb = get_cpu_peak_memory_MB()

    end_time = time.time()
    tracker.stop()

    csv_path = os.path.join(output_dir, f"emissions-{run_id}.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        last_row = lines[-1].strip().split(",")
        emissions_data = dict(zip(header, last_row))

    return {
        "duration": end_time - start_time,
        "peak_mem_MB": peak_mem_mb,
        "emissions_data": emissions_data,
    }


# -----
def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 2,
    measure_power_secs: int = 1
) -> Tuple[Dict[int, Dict[str, float]], float]:
    """
    Measure inference time, emissions, and memory usage over multiple runs.
    """
    clear_memory()

    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)

    run_data = {}
    for run in range(runs):
        # clear all memory
        clear_memory()
        run_data[run] = measure_single_run(model, x, device, run, measure_power_secs)
        clear_memory()

    return run_data
