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
def benchmark_model(model, loader, device, num_batches=20, warmup_batches=5):
    """
    Returns: (avg_time_seconds, flops_total, total_size_mb)
    """
    from copy import deepcopy
    tempmodel = deepcopy(model)
    tempmodel.eval()
    tempmodel.to(device)

    times = []
    flops = 0
    total_size_mb = 0

    # warmup
    with torch.no_grad():
        it = iter(loader)
        for _ in range(warmup_batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device)
            _ = tempmodel(xb)

        # timed runs
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        for i, (xb, _) in enumerate(loader):
            if i >= num_batches:
                break
            xb = xb.to(device)

            if torch.cuda.is_available():
                # GPU timing using events
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

            if i == 0:
                try:
                    # FlopCountAnalysis may require CPU tensors or the model on CPU depending on implementation.
                    # If it works on device, use device tensor; else move tempmodel/x to cpu for FlopCountAnalysis.
                    flops = FlopCountAnalysis(tempmodel, xb).total()
                except Exception:
                    try:
                        flops = FlopCountAnalysis(tempmodel.cpu(), xb.cpu()).total()
                    except Exception:
                        flops = 0

        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    del tempmodel

    # Calculate the estimated total size of the model (in MB)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_size_mb = (param_count * 4) / (1024 ** 2)  # 4 bytes per float32 parameter

    avg_time = sum(times) / len(times) if times else 0.0
    return avg_time, flops, total_size_mb

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
