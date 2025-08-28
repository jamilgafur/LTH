import os
import time
import uuid
import gc
import torch
import psutil
import tracemalloc
import threading
import memory_profiler
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

def get_model_params_size_MB(model: torch.nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        if param.is_sparse and param.layout == torch.sparse_csr:
            total_bytes += param.crow_indices().element_size() * param.crow_indices().nelement()
            total_bytes += param.col_indices().element_size() * param.col_indices().nelement()
            total_bytes += param.values().element_size() * param.values().nelement()
        else:
            total_bytes += param.element_size() * param.nelement()
    return total_bytes / (1024 ** 2)

def measure_cpu_peak_memory(func, *args, **kwargs):
    mem_usage = memory_usage(
        (func, args, kwargs),
        interval=0.001,
        max_usage=True,
        retval=False
    )
    return mem_usage  # in MB

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

@torch.no_grad()
def measure_single_run(model, x, device, run, measure_power_secs):
    run_id = str(uuid.uuid4())
    tracker, output_dir = start_emissions_tracker(model, device, run_id, measure_power_secs)

    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)
    model.eval()

    # Warm-up
    model(x)

    start_time = time.time()
    model_mem_mb = get_model_params_size_MB(model)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        model(x)
        torch.cuda.synchronize()
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        def run_model():
            model(x)

        peak_mem_mb = measure_cpu_peak_memory(run_model)

    end_time = time.time()
    tracker.stop()

    emissions_data = read_emissions_data(os.path.join(output_dir, f"emissions-{run_id}.csv"))

    # Explicit cleanup
    del tracker
    del emissions_data

    del model
    del x
    gc.collect()
    clear_memory()

    return {
        "duration": end_time - start_time,
        "peak_mem_MB": peak_mem_mb,
        "model_params_MB": model_mem_mb,
        "emissions_data": read_emissions_data(os.path.join(output_dir, f"emissions-{run_id}.csv")),
    }

def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 2,
    measure_power_secs: int = 1
) -> Tuple[Dict[int, Dict[str, float]], float]:
    clear_memory()
    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)

    run_data = {}
    for run in range(runs):
        clear_memory()
        run_data[run] = measure_single_run(model, x, device, run, measure_power_secs)
        clear_memory()

    # Final cleanup
    del model
    del x
    gc.collect()
    clear_memory()

    return run_data
