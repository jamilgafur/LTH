from SparseLinear import SparseLinear
from SparseConv2d import SparseConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 
import os
import re

from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16ImageNet import VGG16_ImageNet
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from pyPrune.models.RegNetX import RegNetX_400MF

def convert_all_to_sparse(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            sparse_weight = child.weight.data.to_sparse_csr()
            bias = child.bias.data if child.bias is not None else None
            setattr(module, name, SparseLinear(sparse_weight, bias))
        elif isinstance(child, nn.Conv2d):
            sparse_weight = child.weight.data.to_sparse_csr()
            bias = child.bias.data if child.bias is not None else None
            sparse_conv = SparseConv2d(
                sparse_weight=sparse_weight,
                bias=bias,
                in_channels=child.in_channels,
                stride=child.stride[0],
                padding=child.padding[0],
                dilation=child.dilation[0]
            )
            setattr(module, name, sparse_conv)
        else:
            convert_all_to_sparse(child)

def validate_original_weights_not_sparse(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            weight = child.weight.data
            print(f"Layer {name}:")
            print(f"  Type: {type(weight)}")
            print(f"  Is sparse? {weight.is_sparse}")
            assert not weight.is_sparse, f"Weight in layer {name} should NOT be sparse!"
        else:
            validate_original_weights_not_sparse(child)

def validate_sparse_csr_format(module):
    for name, child in module.named_children():
        if isinstance(child, SparseLinear) or isinstance(child, SparseConv2d):
            weight = child.sparse_weight
            print(f"Layer {name}:")
            print(f"  Is sparse? {weight.is_sparse}")
            print(f"  Layout: {weight.layout}")
            assert weight.is_sparse, f"Weight in layer {name} is not sparse!"
            assert weight.layout == torch.sparse_csr, f"Weight in layer {name} is not CSR format!"
        else:
            validate_sparse_csr_format(child)

def extract_sparsity_from_filename(filename, prefix="checkpoint_Pruned_"):
    """Extract sparsity value from checkpoint filename."""
    pattern = rf"{prefix}(\d+\.\d+).pth"
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None

def get_checkpoints(path, prefix="checkpoint_Pruned_"):
    """Return sorted list of (sparsity, full_path) tuples for checkpoints matching prefix."""
    files = glob.glob(os.path.join(path, f"{prefix}*.pth"))
    checkpoints = [(extract_sparsity_from_filename(f, prefix), f) for f in files]  # don't re-join the path
    checkpoints = [ckpt for ckpt in checkpoints if ckpt[0] is not None]
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def getModelType(model_name):
    if model_name == "ResNet20":
        return ResNet20
    elif model_name == "Vgg16_ImageNet":
        return VGG16_ImageNet
    elif model_name == "Vgg16_":
        return VGG16_CIFAR10
    elif model_name == "LeNet":
        return LeNet
    elif model_name == "RegNetX":
        return RegNetX_400MF
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def load_model_from_checkpoint(model_cls, checkpoint_path: str, device: torch.device, use_compile: bool = True):
    model = model_cls()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict if wrapped in 'model'
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if use_compile:
        model = torch.compile(model)  # PyTorch 2.0+ speedup

    return model

def initialize_results_dict(devices, batch_sizes):
    mem_results = {
        d.type: {
            bs: {
                'cpu_dense': [], 'gpu_dense': [],
                'cpu_sparse': [], 'gpu_sparse': []
            } for bs in batch_sizes
        } for d in devices
    }

    time_results = {
        d.type: {
            bs: {'dense': [], 'sparse': []} for bs in batch_sizes
        } for d in devices
    }

    energy_results = {
        d.type: {
            bs: {'dense': [], 'sparse': []} for bs in batch_sizes
        } for d in devices
    }

    return mem_results, time_results, energy_results

def sparse_tensor_size_bytes(sparse_tensor: torch.Tensor) -> int:
    """
    Calculate memory footprint of a sparse CSR tensor in bytes,
    summing data, crow_indices, and col_indices buffers.
    """
    assert sparse_tensor.layout == torch.sparse_csr, "Expected CSR sparse tensor"

    data_bytes = sparse_tensor.values().element_size() * sparse_tensor.values().numel()
    crow_bytes = sparse_tensor.crow_indices().element_size() * sparse_tensor.crow_indices().numel()
    col_bytes = sparse_tensor.col_indices().element_size() * sparse_tensor.col_indices().numel()
    total = data_bytes + crow_bytes + col_bytes
    return total

def model_size_bytes(model: torch.nn.Module) -> int:
    total_bytes = 0
    for module in model.modules():
        if isinstance(module, SparseLinear) or isinstance(module, SparseConv2d):
            sparse_weight = module.sparse_weight
            total_bytes += sparse_tensor_size_bytes(sparse_weight)
            if module.bias is not None:
                total_bytes += module.bias.element_size() * module.bias.numel()
        else:
            # For normal modules, count dense weights and biases
            for param in module.parameters(recurse=False):
                total_bytes += param.element_size() * param.numel()
    return total_bytes
