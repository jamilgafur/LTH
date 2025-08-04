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

def dense_tensor_size(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()

def sparse_tensor_size(tensor: torch.Tensor) -> int:
    """
    Estimate memory size in bytes of a sparse CSR tensor.
    Assumes 32-bit indices (4 bytes), and same data dtype as tensor.

    Note: torch.sparse_csr doesn't expose _nnz(), so compute from crow_indices and values.
    """
    if tensor.layout != torch.sparse_csr:
        raise ValueError("Expected sparse_csr tensor")

    nnz = tensor.values().numel()
    row_ptr_bytes = tensor.crow_indices().numel() * tensor.crow_indices().element_size()
    col_ind_bytes = tensor.col_indices().numel() * tensor.col_indices().element_size()
    val_bytes = tensor.values().numel() * tensor.values().element_size()

    total_bytes = row_ptr_bytes + col_ind_bytes + val_bytes
    return total_bytes

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

def log_memory(mem_dict, device_type, prefix):
    mem_val = mem_dict.get(device_type, 0)
    print(f"{prefix} model memory on {device_type.upper()}: {mem_val / (1024 ** 2):.4f} MB")

def measure_model_memory_by_device(model):
    memory = {}  # key: device type string ('cpu', 'cuda'), value: total bytes

    def add_memory(device, bytes_):
        if device not in memory:
            memory[device] = 0
        memory[device] += bytes_

    for name, module in model.named_modules():
        # Dense layers
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            weight = module.weight.detach()
            device = weight.device.type
            add_memory(device, dense_tensor_size(weight))

            if module.bias is not None:
                bias = module.bias.detach()
                add_memory(device, dense_tensor_size(bias))

        # Sparse layers
        elif isinstance(module, (SparseLinear, SparseConv2d)):
            sparse_weight = module.sparse_weight
            device = sparse_weight.device.type
            add_memory(device, sparse_tensor_size(sparse_weight))

            if module.bias is not None:
                bias = module.bias
                # Check if bias is sparse or dense
                if hasattr(bias, 'layout') and bias.layout == torch.sparse_csr:
                    add_memory(device, sparse_tensor_size(bias))
                else:
                    add_memory(device, dense_tensor_size(bias))

    # If no memory found for a device, set to 0 explicitly
    if 'cpu' not in memory:
        memory['cpu'] = 0
    if 'cuda' not in memory:
        memory['cuda'] = 0

    return memory

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

