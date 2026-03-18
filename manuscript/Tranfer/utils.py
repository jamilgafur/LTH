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
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transfer import EXPERIMENTS # Imports your dictionaries

def plot_experiment_heuristics(model_name, dataset_name, stats_csv_path):
    # Load the raw layer stats
    df_layers = pd.read_csv(stats_csv_path)
    layer_names = df_layers['Layer'].tolist()
    variances = dict(zip(df_layers['Layer'], df_layers['Variance']))
    activations = dict(zip(df_layers['Layer'], df_layers['Mean Activation']))

    exp_dict = EXPERIMENTS[model_name][dataset_name]
    
    exp_names, total_vars, avg_acts = [], [], []

    # Calculate Total Variance and Average Activation per experiment
    for exp_name, layer_range in exp_dict.items():
        if layer_range is None or exp_name == "Original Model":
            continue
            
        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        b_vars, b_acts = [], []
        
        for start_layer, end_layer in ranges:
            in_range = False
            for name in layer_names:
                if start_layer in name: in_range = True
                if in_range:
                    if name in variances: b_vars.append(variances[name])
                    if name in activations: b_acts.append(activations[name])
                if end_layer in name: break
                
        if b_vars and b_acts:
            exp_names.append(exp_name)
            total_vars.append(np.sum(b_vars)) # SUM of variance (Total Information)
            avg_acts.append(np.mean(b_acts))  # MEAN of activation (Average Volume)

    # Generate Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    df_plot = pd.DataFrame({"Experiment": exp_names, "Total Variance": total_vars, "Mean Activation": avg_acts})

    # Top Plot: Mean Activation
    sns.barplot(data=df_plot, x="Experiment", y="Mean Activation", color="#4C72B0", edgecolor="black", ax=ax1)
    ax1.set_title(f"Heuristic Profiling by Target Region: {model_name}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Avg Mean Activation", fontweight='bold')
    ax1.axhline(0, color='black', linewidth=1.5)

    # Bottom Plot: Total Variance
    sns.barplot(data=df_plot, x="Experiment", y="Total Variance", color="#C44E52", edgecolor="black", ax=ax2)
    ax2.set_ylabel("Total Sum of Variance", fontweight='bold')
    ax2.set_xlabel("Targeted Collapse Region", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{model_name}_heuristic_target_summary.png", dpi=300)
    print(f"Saved {model_name}_heuristic_target_summary.png")


def draw_collapse_visual(model_name="MobileNet", target_collapse="Block 11"):
    fig, ax = plt.subplots(figsize=(15, 4))
    
    # 1. Define the abstract blocks of the network
    blocks = ["Stem", "Block 0", "Block 1", "...", "Block 10", "Block 11", "Block 12", "Classifier"]
    x_pos = 0
    box_width = 1.2
    box_height = 0.8
    
    # 2. Draw the "Features" Container background
    feat_rect = patches.Rectangle((1.5, -0.6), 8.5, 2.0, linewidth=2, edgecolor='gray', facecolor='#f0f0f0', linestyle='--')
    ax.add_patch(feat_rect)
    ax.text(5.75, 1.6, "PyTorch 'features' Sequential Container\n(Contains Stages/Blocks)", ha='center', fontsize=11, fontweight='bold', color='gray')

    # 3. Draw the Blocks
    for i, name in enumerate(blocks):
        # Determine color based on whether it is the collapsed target
        is_collapsed = (name == target_collapse)
        facecolor = '#fce8e6' if is_collapsed else '#eaf2f8'
        edgecolor = '#c44e52' if is_collapsed else '#4c72b0'
        
        rect = patches.Rectangle((x_pos, 0), box_width, box_height, linewidth=2, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        
        # Block Text
        ax.text(x_pos + box_width/2, box_height/2, name, ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Draw Arrows between blocks
        if i < len(blocks) - 1:
            ax.annotate("", xy=(x_pos + box_width + 0.4, box_height/2), xytext=(x_pos + box_width, box_height/2),
                        arrowprops=dict(arrowstyle="->", lw=2))
        
        # Draw the "Collapse" Bypass Arrow
        if is_collapsed:
            # Draw a big red X over the block
            ax.plot([x_pos, x_pos + box_width], [0, box_height], color='red', lw=3)
            ax.plot([x_pos, x_pos + box_width], [box_height, 0], color='red', lw=3)
            
            # Draw the bypass route (Identity)
            ax.annotate("nn.Identity()\n(Bypass)", xy=(x_pos + box_width + 0.2, -0.2), xytext=(x_pos - 0.2, -0.2),
                        arrowprops=dict(arrowstyle="->", lw=2, color='green', connectionstyle="arc3,rad=-0.5"),
                        ha='center', va='top', color='green', fontweight='bold')
            
        x_pos += box_width + 0.5

    ax.set_xlim(-0.5, x_pos)
    ax.set_ylim(-1.5, 2.5)
    ax.axis('off')
    ax.set_title(f"Visualizing Structural Collapse: {model_name} (Target: {target_collapse})", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"architecture_collapse_diagram.png", dpi=300)
    print("Saved architecture_collapse_diagram.png")
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
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
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

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    elif model_name == "RegNetX_400MF":
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
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

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for RegNetX_400MF...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset for {model_name}: {dataset_name}")
    
    elif model_name == "InceptionNet":
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for InceptionNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for InceptionNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for InceptionNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for InceptionNet...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset for {model_name}: {dataset_name}")
    
    elif model_name == "XceptionNet":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for XceptionNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for XceptionNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for XceptionNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

    elif model_name == "MobileNet":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for MobileNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for MobileNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for MobileNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200
    elif model_name == "ConvNeXt":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for ConvNeXt...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for ConvNeXt...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for ConvNeXt...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200
        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for ConvNeXt...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes
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

def benchmark_model(model, loader, device, num_batches=20, warmup_batches=5, quant=False):
    """
    Returns: (avg_time_seconds, flops_total, total_feature_map_size_mb)

    Notes:
    - Uses a local DataLoader with num_workers=0 to ensure forward runs in the main process
      (avoids worker deaths hiding OOMs).
    - Hooks only accumulate the number of bytes of feature maps (do NOT keep tensors).
    - If quant=True and CUDA is available, uses mixed precision (fp16) for forward pass.
    """
    from copy import deepcopy
    import torch
    import time
    from torch.utils.data import DataLoader
    from fvcore.nn import FlopCountAnalysis

    # clone model to avoid modifying original
    tempmodel = deepcopy(model)
    tempmodel.eval()
    tempmodel.to(device)

    times = []
    flops = 0
    total_feature_map_size_mb = 0.0

    # Build a single-process DataLoader
    dataset = getattr(loader, "dataset", None)
    batch_size = getattr(loader, "batch_size", 1)
    if dataset is None:
        data_iterable = loader
        def make_iterable():
            return iter(data_iterable)
    else:
        safe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
        def make_iterable():
            return iter(safe_loader)

    # Helper to register lightweight hooks that accumulate bytes
    def register_size_hooks(mod):
        acc = {"bytes": 0}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                try:
                    if isinstance(output, torch.Tensor):
                        acc["bytes"] += output.numel() * output.element_size()
                    elif isinstance(output, (list, tuple)):
                        for o in output:
                            if isinstance(o, torch.Tensor):
                                acc["bytes"] += o.numel() * o.element_size()
                except Exception:
                    pass
            return hook

        for _, m in mod.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d,
                              torch.nn.MaxPool2d, torch.nn.BatchNorm2d,
                              torch.nn.ReLU, torch.nn.Linear)):
                hooks.append(m.register_forward_hook(make_hook(None)))
        return hooks, acc

    # Warmup passes
    it = make_iterable()
    use_autocast = quant and device.type == 'cuda'
    for _ in range(warmup_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        xb = xb.to(device)
        with torch.no_grad():
            if use_autocast:
                with torch.cuda.amp.autocast():
                    _ = tempmodel(xb)
            else:
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

        # Attach hooks on first batch
        size_hooks = []
        size_acc = None
        if i == 0:
            size_hooks, size_acc = register_size_hooks(tempmodel)

        # Forward timing
        with torch.no_grad():
            if torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        _ = tempmodel(xb)
                else:
                    _ = tempmodel(xb)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender) / 1000.0)  # ms -> s
            else:
                start = time.time()
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        _ = tempmodel(xb)
                else:
                    _ = tempmodel(xb)
                times.append(time.time() - start)

        # Capture total bytes for first batch
        if i == 0 and size_acc is not None:
            total_bytes = size_acc.get("bytes", 0)
            total_feature_map_size_mb = total_bytes / (1024 ** 2)
            try:
                flops = FlopCountAnalysis(tempmodel, xb).total()
            except Exception:
                try:
                    flops = FlopCountAnalysis(tempmodel.cpu(), xb.cpu()).total()
                except Exception:
                    flops = 0

        # Remove hooks
        if size_hooks:
            for h in size_hooks:
                h.remove()

    avg_time = sum(times) / len(times) if times else 0.0
    return avg_time, flops, total_feature_map_size_mb

def describe_model(model, loader, device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    summary(model, input_size=next(iter(loader))[0].shape, device=device)
    # layer_stats(model)
    print("=" * 60)


def calibrate_hyperparameters(df):
    """
    Analyzes the heuristic DataFrame to find optimal scaling factors.
    Returns a dict of tuned parameters: {'lambda_v', 'd_0'}
    """
    # 1. Calibrate Lambda (Variance Sensitivity)
    # We want the bottom 20% of layers (by variance) to have a Silence Score > 0.5
    # Formula: exp(-lambda * var_20th) = 0.5
    # Solve for lambda: lambda = -ln(0.5) / var_20th
    
    # Filter for valid variances (conv/linear layers only)
    variances = df[df['act_var'] > 0]['act_var']
    
    if variances.empty:
        return {'lambda_v': 10.0, 'd_0': 0.15} # Fallback defaults
        
    var_20th_percentile = np.percentile(variances, 20)
    
    # Avoid division by zero if variance is extremely small
    var_threshold = max(var_20th_percentile, 1e-6)
    
    lambda_v = -np.log(0.5) / var_threshold
    
    # 2. Calibrate Depth Gate (d_0)
    # We assume the "Stem" is roughly the first 10% of layers, 
    # but at least the first 5 layers.
    total_layers = len(df)
    stem_layers = max(5, int(total_layers * 0.10))
    d_0 = stem_layers / total_layers
    
    print(f"[Auto-Calibrate] Tuned lambda_v: {lambda_v:.4f} (based on p20 var: {var_threshold:.4e})")
    print(f"[Auto-Calibrate] Tuned d_0: {d_0:.4f} (Protecting first {stem_layers} layers)")
    
    return {'lambda_v': lambda_v, 'd_0': d_0}

def calculate_adaptive_score(row, total_layers, tuned_params):
    """
    Calculates CS using the auto-calibrated parameters.
    """
    # Unpack tuned params
    lambda_v = tuned_params['lambda_v']
    d_0 = tuned_params['d_0']
    
    # Fixed params (these are generally robust)
    k = 20.0       # Steepness of depth gate (20 makes it a sharp wall)
    gamma = 0.2    # Residual bonus
    
    # --- Metrics from Dataframe ---
    variance = row['act_var']
    identity = row['identity_score']
    # We approximate 'has_residual' by checking layer name or using a passed flag
    # For now, we'll assume False or you can map it from your model graph
    has_residual = False 
    
    # Calculate Relative Depth (0.0 to 1.0)
    # Assuming the dataframe index corresponds to depth
    relative_depth = (row.name + 1) / total_layers
    
    # 1. Depth Gating
    depth_gate = 1 / (1 + np.exp(-k * (relative_depth - d_0)))
    
    # 2. Functional Score
    silence_score = np.exp(-lambda_v * variance)
    redundancy_score = identity
    
    # Weighted average (favoring silence slightly as it's a stronger signal)
    functional_score = 0.6 * silence_score + 0.4 * redundancy_score
    
    # 3. Residual Bonus
    residual_multiplier = 1.0 + (gamma if has_residual else 0.0)
    
    final_score = depth_gate * functional_score * residual_multiplier
    
    return final_score
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
