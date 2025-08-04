# measure.py
from SparseLinear import SparseLinear
from SparseConv2d import SparseConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

try:
    from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetPowerUsage, nvmlDeviceGetHandleByIndex
    nvmlInit()
    NVML_AVAILABLE = True
    NVML_DEVICE = nvmlDeviceGetHandleByIndex(0)  # GPU 0
except:
    NVML_AVAILABLE = False
    print("NVML not available. GPU energy measurement disabled.")

def dense_tensor_size(tensor):
    return tensor.numel() * tensor.element_size()

def sparse_tensor_size(sparse_tensor):
    if sparse_tensor.layout == torch.sparse_csr:
        values_size = sparse_tensor.values().numel() * sparse_tensor.values().element_size()
        crow_size = sparse_tensor.crow_indices().numel() * sparse_tensor.crow_indices().element_size()
        col_size = sparse_tensor.col_indices().numel() * sparse_tensor.col_indices().element_size()
        return values_size + crow_size + col_size
    elif sparse_tensor.layout == torch.sparse_coo:
        values_size = sparse_tensor.values().numel() * sparse_tensor.values().element_size()
        indices_size = sparse_tensor.indices().numel() * sparse_tensor.indices().element_size()
        return values_size + indices_size
    else:
        raise ValueError(f"Unsupported sparse format: {sparse_tensor.layout}")

def measure_model_memory_by_device(model):
    memory = {'cpu': 0, 'cuda': 0}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            weight = module.weight.data
            device = weight.device.type
            memory[device] += dense_tensor_size(weight)
            if module.bias is not None:
                memory[device] += dense_tensor_size(module.bias.data)
        elif isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
            sparse_weight = module.sparse_weight
            device = sparse_weight.device.type
            memory[device] += sparse_tensor_size(sparse_weight)
            if module.bias is not None:
                memory[device] += dense_tensor_size(module.bias)
    return memory

def print_gpu_memory_usage():
    allocated = torch.cuda.memory_allocated()
    max_allocated = torch.cuda.max_memory_allocated()
    print(f"CUDA Memory Allocated: {allocated / (1024 ** 2):.4f} MB")
    print(f"CUDA Max Memory Allocated: {max_allocated / (1024 ** 2):.4f} MB")

def get_gpu_power_watts():
    """Get current GPU power draw in watts (float)."""
    if NVML_AVAILABLE:
        return nvmlDeviceGetPowerUsage(NVML_DEVICE) / 1000.0  # milliwatts → watts
    return 0.0

def measure_inference_time(model, x, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_power = get_gpu_power_watts()
    start_time = time.time()

    with torch.no_grad():
        output = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = time.time()
    end_power = get_gpu_power_watts()

    duration = end_time - start_time

    # Energy estimation (joules = watts * seconds)
    avg_power = (start_power + end_power) / 2
    energy_joules = avg_power * duration

    return duration, output, energy_joules

def log_memory(mem_dict, device_type, prefix):
    mem_val = mem_dict.get(device_type, 0)
    print(f"{prefix} model memory on {device_type.upper()}: {mem_val / (1024 ** 2):.4f} MB")
