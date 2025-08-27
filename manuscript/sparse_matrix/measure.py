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
import psutil
import os
import time
import torch
import uuid

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
) -> dict:
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

    # Reset memory tracking on GPU or clear tracemalloc for CPU
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    else:
        import tracemalloc
        tracemalloc.clear_traces()

    tracker.start()
    start_time = time.time()

    # For CPU memory measurement with psutil
    start_mem = None
    if device.type == 'cpu':
        process = psutil.Process(os.getpid())
        start_mem = process.memory_info().rss / (1024 ** 2)  # MB

    with torch.no_grad():
        model(x)

    # Record memory after the inference is done
    end_mem = None
    if device.type == 'cpu':
        end_mem = process.memory_info().rss / (1024 ** 2)  # MB

    if device.type == 'cuda':
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        peak_mem = get_model_memory_MB(model)

    end_time = time.time()
    tracker.stop()

    # Read emissions data from file
    csv_path = os.path.join(output_dir, f"emissions-{run_id}.csv")
    with open(csv_path, "r") as f:
        lines = f.readlines()
        header = lines[0].strip().split(",")
        last_row = lines[-1].strip().split(",")
        emissions_data = dict(zip(header, last_row))

    # If we're tracking CPU memory, add it to the results
    cpu_memory_usage = None
    if device.type == 'cpu' and start_mem is not None and end_mem is not None:
        cpu_memory_usage = end_mem - start_mem

    return {
        'duration': end_time - start_time,
        'peak_mem': peak_mem,
        'emissions_data': emissions_data,
        'cpu_memory_usage': cpu_memory_usage  # Add CPU memory usage here
    }

# def measure_single_run(
#     model: torch.nn.Module,
#     x: torch.Tensor,
#     device: torch.device,
#     run: int,
#     measure_power_secs: int
# ) -> dict:
#     run_id = str(uuid.uuid4())
#     output_dir = f"./codecardbon/{model.__class__.__name__}/{device.type}/{run_id}"
#     os.makedirs(output_dir, exist_ok=True)

#     tracker = OfflineEmissionsTracker(
#         measure_power_secs=measure_power_secs,
#         country_iso_code="USA",
#         country_2letter_iso_code="US-NW-PSCO",
#         output_dir=output_dir,
#         log_level="error",
#         save_to_file=True,
#         output_file=f"emissions-{run_id}.csv"
#     )

#     # Reset memory tracking on GPU or clear tracemalloc for CPU
#     if device.type == 'cuda':
#         torch.cuda.reset_peak_memory_stats()
#     else:
#         import tracemalloc
#         tracemalloc.clear_traces()

#     tracker.start()
#     start_time = time.time()

#     # For CPU memory measurement with psutil
#     start_mem = None
#     if device.type == 'cpu':
#         process = psutil.Process(os.getpid())
#         start_mem = process.memory_info().rss / (1024 ** 2)  # MB

#     with torch.no_grad():
#         model(x)

#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#         mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
#         peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
#     else:
#         peak_mem = get_model_memory_MB(model)

#     end_time = time.time()
#     tracker.stop()

#     # Read emissions data from file
#     csv_path = os.path.join(output_dir, f"emissions-{run_id}.csv")
#     with open(csv_path, "r") as f:
#         lines = f.readlines()
#         header = lines[0].strip().split(",")
#         last_row = lines[-1].strip().split(",")
#         emissions_data = dict(zip(header, last_row))

#     return {
#         'duration': end_time - start_time,
#         'peak_mem': peak_mem,
#         'emissions_data': emissions_data
#     }

def measure_inference(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    runs: int = 1,
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

import numpy as np
from scipy.sparse import csr_matrix
import torch

import torch

def get_model_memory_MB(model):
    """
    Function to calculate the total memory used by the model, considering both dense
    and sparse layers (CSR format).

    Args:
        model (torch.nn.Module): The model for which memory needs to be calculated.

    Returns:
        float: The total memory used by the model in megabytes (MB).
    """
    total_size_bytes = 0

    # Iterate over the model's parameters
    for param in model.parameters():
        if param.requires_grad:
            # Check if the parameter is sparse (CSR format)
            if param.is_sparse:
                # For CSR sparse tensors:
                # - param._nnz() is the number of non-zero elements
                # - param.indices().numel() is the number of indices (rows + columns)
                # - param.values().numel() is the number of non-zero values
                nnz = param._nnz()  # number of non-zero elements
                indices_size = param.indices().numel() * param.indices().element_size()  # size of indices
                values_size = nnz * param.values().element_size()  # size of non-zero values
                
                # Total memory for sparse matrix
                total_size_bytes += indices_size + values_size
            else:
                # For dense tensors:
                num_elements = param.numel()
                element_size = param.element_size()  # Size in bytes for the datatype of the tensor
                total_size_bytes += num_elements * element_size

    # Convert bytes to megabytes
    total_size_MB = total_size_bytes / (1024 ** 2)

    return total_size_MB
