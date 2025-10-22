import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import json
from fvcore.nn import FlopCountAnalysis
import time
from torchinfo import summary
import numpy as np
from pyPrune.utils import load_cifar10, load_cifar100, load_tiny_imagenet, load_imagenet
from copy import deepcopy
def load_dataset(dataset_name, model_name="VGG16"):
    if model_name == "VGG16":
        if dataset_name == "TinyImageNet":
            print("Loading Tiny ImageNet data...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet":
            print("Loading ImageNet data...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    elif model_name == "RegNetX_400MF":
        if dataset_name == "TinyImageNet":
            print("Loading Tiny ImageNet data for RegNetX_400MF...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for RegNetX_400MF...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for RegNetX_400MF...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet":
            print("Loading ImageNet data for RegNetX_400MF...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset for {model_name}: {dataset_name}")
    
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return train_loader, test_loader, input_size, input_channels, num_classes

# -------------------------
# Benchmark Inference
# -------------------------
import torch
import time
from copy import deepcopy
from fvcore.nn import FlopCountAnalysis

def benchmark_model(model, loader, device, num_batches=20, warmup_batches=5):
    """
    Returns: (avg_time_seconds, flops_total, total_feature_map_size_mb)

    total_feature_map_size_mb = cumulative size of all intermediate feature maps (excluding input and output),
    measured during one forward pass (batch 0).
    """
    tempmodel = deepcopy(model)
    tempmodel.eval()
    tempmodel.to(device)

    times = []
    flops = 0
    feature_map_sizes = []

    feature_maps = []

    # ---- Register hooks to capture feature maps ----
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            feature_maps.append(output)
        elif isinstance(output, (list, tuple)):
            feature_maps.extend(o for o in output if isinstance(o, torch.Tensor))

    hooks = []
    for name, module in tempmodel.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.BatchNorm2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        it = iter(loader)
        for _ in range(warmup_batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device)
            _ = tempmodel(xb)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)

        it = iter(loader)
        for i in range(num_batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device)

            feature_maps.clear()

            if torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                _ = tempmodel(xb)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender) / 1000.0)  # ms -> s
            else:
                start = time.time()
                _ = tempmodel(xb)
                times.append(time.time() - start)

            # Only collect feature map size for the first batch
            if i == 0:
                total_bytes = 0
                for fmap in feature_maps:
                    # Total number of elements in the tensor * size of each element (assume float32 = 4 bytes)
                    total_bytes += fmap.numel() * 4
                feature_map_sizes.append(total_bytes / (1024 ** 2))  # Convert to MB

                # FLOPs
                try:
                    flops = FlopCountAnalysis(tempmodel, xb).total()
                except Exception:
                    try:
                        flops = FlopCountAnalysis(tempmodel.cpu(), xb.cpu()).total()
                    except Exception:
                        flops = 0

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    # Remove hooks
    for h in hooks:
        h.remove()

    del tempmodel

    avg_time = sum(times) / len(times) if times else 0.0
    total_feature_map_size_mb = feature_map_sizes[0] if feature_map_sizes else 0.0

    return avg_time, flops, total_feature_map_size_mb

def describe_model(model, loader, device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    summary(model, input_size=next(iter(loader))[0].shape, device=device)
    layer_stats(model)
    print("=" * 60)

# ===============================
# Basic Counting Utilities
# ===============================

def count_zeros(tensor): 
    return torch.sum(tensor == 0).item()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# Model Statistics
# ===============================

def layer_stats(model):
    print("\nLayer-wise zero parameter stats:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")



# ===============================
# Cloning Utility
# ===============================

def clone_model(model, model_class):
    """Utility to clone a model and load weights to keep experiments isolated."""
    new_model = model_class()
    new_model.load_state_dict(model.state_dict())
    return new_model
