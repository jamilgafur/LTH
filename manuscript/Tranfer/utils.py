# utils.pt
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



# -------------------------
# Helper utilities
# -------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def is_dict_like(x):
    return isinstance(x, dict)

def normalize_metrics(metrics):
    """
    Normalize incoming metrics into a dict[str -> dict] mapping for plotting functions.
    Accepts:
      - dict mapping experiment_name -> metrics (ideal)
      - list of dicts (will pick 'name'/'experiment' if present, else index-based)
      - single dict that might contain nested dicts
    Returns dict.
    """
    if is_dict_like(metrics):
        # If it looks like {exp_name: { ... }}, keep only dict values
        # If metrics itself is single experiment (contains final_accuracy etc), wrap it
        contains_nested = any(isinstance(v, dict) for v in metrics.values())
        if contains_nested:
            result = {k: v for k, v in metrics.items() if isinstance(v, dict)}
            # If result empty but metrics seems like one experiment record, wrap it
            if not result and metrics and all(k in metrics for k in ("accuracies", "losses", "param_count")):
                return {"metric_record": metrics}
            return result
        # fallback: treat as single experiment
        if all(k in metrics for k in ("accuracies", "losses", "param_count")):
            return {"metric_record": metrics}
        return {}
    elif isinstance(metrics, list):
        out = {}
        for i, entry in enumerate(metrics):
            if not is_dict_like(entry):
                continue
            name = entry.get("name") or entry.get("experiment") or f"exp_{i}"
            out[name] = entry
        return out
    else:
        return {}

def safe_get(d, key, default=None):
    if not is_dict_like(d):
        return default
    return d.get(key, default)

def timestamped_filename(base):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base)
    return f"{name}_{t}{ext}" if ext else f"{base}_{t}"



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
from torch.utils.data import DataLoader

def benchmark_model(model, loader, device, num_batches=20, warmup_batches=5):
    """
    Returns: (avg_time_seconds, flops_total, total_feature_map_size_mb)

    Notes:
    - Uses a local DataLoader with num_workers=0 to ensure forward runs in the main process
      (avoids worker deaths hiding OOMs).
    - Hooks only accumulate the number of bytes of feature maps (do NOT keep tensors).
    """
    # clone model to avoid modifying original
    tempmodel = deepcopy(model)
    tempmodel.eval()
    tempmodel.to(device)

    times = []
    flops = 0
    total_feature_map_size_mb = 0.0

    # Build a single-process DataLoader from the provided loader's dataset & batch_size
    # Fallback to small defaults if attributes are missing.
    dataset = getattr(loader, "dataset", None)
    batch_size = getattr(loader, "batch_size", 1)
    if dataset is None:
        # If loader doesn't expose dataset (rare), fall back to iterating the loader directly
        data_iterable = loader
        def make_iterable():
            return iter(data_iterable)
        use_loader_obj = False
    else:
        safe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
        def make_iterable():
            return iter(safe_loader)
        use_loader_obj = True

    # Helper to register lightweight hooks that accumulate bytes instead of storing tensors.
    def register_size_hooks(mod):
        acc = {"bytes": 0}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                # safe: only inspect size/numel, do NOT store the tensor
                try:
                    if isinstance(output, torch.Tensor):
                        acc["bytes"] += output.numel() * output.element_size()
                    elif isinstance(output, (list, tuple)):
                        for o in output:
                            if isinstance(o, torch.Tensor):
                                acc["bytes"] += o.numel() * o.element_size()
                except Exception:
                    # be resilient: if anything goes wrong in hook, skip adding
                    pass
            return hook

        for _, m in mod.named_modules():
            # Limit to typical feature-producing modules (keeps number of hooks manageable)
            if isinstance(m, (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d,
                              torch.nn.MaxPool2d, torch.nn.BatchNorm2d,
                              torch.nn.ReLU, torch.nn.Linear)):
                hooks.append(m.register_forward_hook(make_hook(None)))
        return hooks, acc

    # Warmup passes (use the safe single-process iterable)
    it = make_iterable()
    for _ in range(warmup_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        xb = xb.to(device)
        with torch.no_grad():
            _ = tempmodel(xb)

    # Reset peak stats if using CUDA
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    # Measurement passes
    it = make_iterable()
    for i in range(num_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        xb = xb.to(device)

        # For the *first* measured batch, attach size hooks so we compute total feature-map bytes
        size_hooks = []
        size_acc = None
        if i == 0:
            size_hooks, size_acc = register_size_hooks(tempmodel)

        # Time the forward (CUDA events if available)
        with torch.no_grad():
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

        # After forward, capture total bytes for first batch (if measured)
        if i == 0 and size_acc is not None:
            total_bytes = size_acc.get("bytes", 0)
            total_feature_map_size_mb = total_bytes / (1024 ** 2)
            # Compute FLOPs for this batch (best-effort with fallbacks)
            try:
                flops = FlopCountAnalysis(tempmodel, xb).total()
            except Exception:
                try:
                    flops = FlopCountAnalysis(tempmodel.cpu(), xb.cpu()).total()
                except Exception:
                    flops = 0

        # Remove size hooks for safety after first batch
        if size_hooks:
            for h in size_hooks:
                h.remove()

    # peak memory (if desired)
    if torch.cuda.is_available():
        try:
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            peak_mem = None

    # cleanup
    try:
        del tempmodel
    except Exception:
        pass

    avg_time = sum(times) / len(times) if times else 0.0

    return avg_time, flops, total_feature_map_size_mb

def describe_model(model, loader, device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    summary(model, input_size=next(iter(loader))[0].shape, device=device)
    # layer_stats(model)
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
