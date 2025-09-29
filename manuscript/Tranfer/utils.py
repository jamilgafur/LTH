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

def load_dataset(dataset_name):
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
        num_classes = 10

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_loader, test_loader, input_size, input_channels, num_classes

# -------------------------
# Benchmark Inference
# -------------------------
def benchmark_model(model, loader, device, num_batches=10):
    model.eval()
    model.to(device)
    times = []
    flops = 0
    with torch.no_grad():
        for i, (xb, _) in enumerate(loader):
            if i >= num_batches:
                break
            xb = xb.to(device)

            # Measure inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            output = model(xb)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start_time)

            # Measure FLOPs (only on the first batch for simplicity)
            if i == 0:
                flops = FlopCountAnalysis(model, xb).total()

    avg_time = sum(times) / len(times) if times else 0
    return avg_time, flops

def describe_model(model, loader, device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    layer_stats(model)
    summary(model, input_size=next(iter(loader))[0].shape, device=device)
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
