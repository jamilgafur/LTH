# measure.py
from codecarbon import OfflineEmissionsTracker
import torch
import time
import gc
from typing import Tuple

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

def measure_inference(
    model: torch.nn.Module, 
    x: torch.Tensor, 
    device: torch.device,
    runs: int = 5,
    measure_power_secs: int = 1
) -> Tuple[float, float]:
    """
    Measure average inference time and CO2 emissions over multiple runs.

    Args:
        model: The PyTorch model.
        x: Input tensor.
        device: Device to run on.
        runs: Number of runs to average.
        measure_power_secs: Seconds for power measurement in CodeCarbon.

    Returns:
        avg_duration: Average inference time (seconds).
        avg_emissions: Average CO2 emissions (kg).
    """
    clear_memory()

    tracker = OfflineEmissionsTracker(measure_power_secs=measure_power_secs)
    model = move_to_device_if_needed(model, device)
    x = move_to_device_if_needed(x, device)

    # Warm-up runs (GPU especially needs this)
    with torch.no_grad():
        for _ in range(3):
            model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    tracker.start()

    times = []
    for _ in range(runs):
        start = time.time()
        with torch.no_grad():
            model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.time()
        times.append(end - start)

    emissions = tracker.stop()

    avg_duration = sum(times) / runs
    avg_emissions = emissions / runs

    return avg_duration, avg_emissions
