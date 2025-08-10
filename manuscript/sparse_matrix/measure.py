# measure.py
from codecarbon import OfflineEmissionsTracker
import torch
import time
import gc
from typing import Tuple, Dict
import os
import uuid
import time
import json
from codecarbon import OfflineEmissionsTracker

import os
import time
import uuid
from typing import Dict, Tuple

import torch
from codecarbon import OfflineEmissionsTracker


def clear_memory():
    """Clear GPU and CPU memory."""
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

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def move_to_device_if_needed(obj, device):
    return obj.to(device) if hasattr(obj, 'to') else obj

def prepare_device_tracking(device: torch.device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        import tracemalloc
        tracemalloc.start()

def stop_device_tracking(device: torch.device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    else:
        import tracemalloc
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / (1024**2)

def measure_single_run(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    run: int,
    measure_power_secs: int
) -> Dict[str, float]:
    run_id = str(uuid.uuid4())
    output_dir = f"./codecardbon/{model.__class__.__name__}/{device.type}/{run_id}"
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

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    else:
        import tracemalloc
        tracemalloc.clear_traces()

    tracker.start()
    start = time.time()

    with torch.no_grad():
        model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        import tracemalloc
        current, peak = tracemalloc.get_traced_memory()
        mem_alloc = current / (1024**2)
        peak_mem = peak / (1024**2)

    end = time.time()
    tracker.stop()

    # Read emissions data
    csv_path = os.path.join(output_dir, f"emissions-{run_id}.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        last_row = lines[-1].strip().split(",")
        emissions_data = dict(zip(header, last_row))

    return {
        'duration': end - start,
        'mem_alloc': mem_alloc,
        'peak_mem': peak_mem,
        'emissions_data': emissions_data
    }

def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 5,
    measure_power_secs: int = 1
) -> Tuple[Dict[int, Dict[str, float]], float]:
    """
    Measure inference time, full CO2 emissions metadata, and memory usage for each run.
    """
    clear_memory()

    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)

    prepare_device_tracking(device)
    run_data = {}

    for run in range(runs):
        run_data[run] = measure_single_run(model, x, device, run, measure_power_secs)

    return run_data
