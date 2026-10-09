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
import torch.nn.utils.prune as prune
import torch
import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import torch.nn as nn

from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet

from collapse import collapse_only
import argparse
import glob
import re
from main import PowerTracker
# =========================================================
# Utility Functions
# =========================================================

def parse_directory_context(filepath: str, expected_model: str, expected_dataset: str):
    """Safely extracts context without relying on naive underscore splitting."""
    base_dir = filepath.split("/checkpoints/")[0].split("/")[-1]
    parts = base_dir.split("_")
    
    # Safely find epochs and pretrain tokens
    epochs_str = next((p for p in parts if p.startswith("epochs")), "epochs100")
    pretrain_str = next((p for p in parts if p.startswith("pretrain")), "pretrain300")
    
    match = re.search(r"epoch(\d+)\.pt", filepath)
    ckpt_epoch = int(match.group(1)) if match else 0
    
    # Return the explicitly provided model and dataset names
    return expected_model, expected_dataset, epochs_str, pretrain_str, ckpt_epoch, base_dir

def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    """Matches the corresponding checkpoint within the same base experiment directory."""
    base_dir = before_path.split("/checkpoints/")[0]

    print(f"[INFO] Searching for checkpoint matching: {base_dir}")

    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            print(f"[INFO] Found matching checkpoint: {candidate}")
            return candidate

    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")

def load_weights(model: nn.Module, filepath: str, device: torch.device):
    """Loads a PyTorch model checkpoint state_dict into the instantiated model."""
    print(f"[INFO] Loading weights: {filepath}")
    start_time = time.time()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    # --- KEY MAPPING INTERCEPT ---
    state_dict = {}
    for key, value in raw_state_dict.items():
        # Fix ConvNeXt collapse mismatch
        new_key = key.replace("conv_dw", "conv_g1x1")
        # Fix potential XceptionNet DataParallel artifacts
        new_key = new_key.replace("module.", "")
        
        state_dict[new_key] = value
    # -----------------------------

    print(f"[INFO] Checkpoint loaded. State dict keys: {len(state_dict)}")

    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        print(f"\n[CRITICAL WARNING] {len(incompatible_keys.missing_keys)} Missing keys!")
        print(f"Sample missing: {incompatible_keys.missing_keys[:5]}\n")
        
    if incompatible_keys.unexpected_keys:
        print(f"[CRITICAL WARNING] {len(incompatible_keys.unexpected_keys)} Unexpected keys in checkpoint!")
        print(f"Sample unexpected: {incompatible_keys.unexpected_keys[:5]}\n")

    model.to(device)
    model.eval()

    print(f"[INFO] Weights loaded in {time.time() - start_time:.2f}s")

    return model

def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    """Applies L1 Unstructured Pruning globally to all convolutional and linear layers."""
    print(f"[INFO] Applying {amount * 100:.1f}% global unstructured pruning...")
    start_time = time.time()

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    print(f"[INFO] Found {len(parameters_to_prune)} Conv/Linear layers to prune.")

    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    print(f"[INFO] Pruning completed in {time.time() - start_time:.2f}s")

    return model


# =========================================================
# Model Evaluation with Energy Tracking
# =========================================================

def evaluate_accuracy(model: nn.Module, dataloader, device: torch.device, power_interval: int = 1):
    """Evaluates top-1 accuracy while profiling power draw (W) and energy (J)."""
    print("[INFO] Starting accuracy and energy evaluation...")
    model.eval()
    correct = 0
    total = 0

    with PowerTracker(device=device, interval_s=power_interval) as tracker:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 50 == 0:
                    print(f"[INFO] Accuracy evaluation: batch={batch_idx + 1}, samples={total}")

    avg_watts, total_joules, elapsed_sec = tracker.get_results()
    accuracy = correct / total if total > 0 else 0.0

    print(
        f"[INFO] Evaluation complete: acc={accuracy:.4f}, "
        f"power={avg_watts:.2f}W, energy={total_joules:.2f}J, time={elapsed_sec:.2f}s"
    )
    return accuracy, avg_watts, total_joules

# =========================================================
# Iterative Magnitude Pruning (IMP) Training Loop
# =========================================================

def train_imp(model, train_loader, device, epochs, target_sparsity, save_path, power_interval: int = 1):
    """Trains on-the-fly using Iterative Magnitude Pruning to match target sparsity."""
    print(f"\n[INFO] Starting Iterative Magnitude Pruning (IMP) for {epochs} epochs...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    pruning_steps = max(1, int(epochs * 0.75))
    incremental_amount = 1.0 - (1.0 - target_sparsity) ** (1.0 / pruning_steps)
    print(f"[INFO] Compounding prune rate: {incremental_amount * 100:.2f}% per step over {pruning_steps} steps.")

    model.train()
    with PowerTracker(device=device, interval_s=power_interval) as tracker:
        for epoch in range(epochs):
            if epoch < pruning_steps:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=incremental_amount
                )
                
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            print(f"[INFO] Epoch {epoch + 1}/{epochs} - Train Loss: {running_loss / len(train_loader):.4f}")

    train_watts, train_joules, train_time = tracker.get_results()
    print(f"[INFO] IMP training energy: {train_watts:.2f}W avg, {train_joules:.2f}J total ({train_time:.2f}s)")

    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[INFO] IMP model checkpoint saved to {save_path}\n")
    
    model.eval()
    return model

# =========================================================
# Architecture and Region Utilities
# =========================================================

def initialize_architecture(model_name: str, dataset_name: str):
    """Initializes architecture and returns data loaders and sample tensor."""
    print(f"[INFO] Initializing architecture: model={model_name}, dataset={dataset_name}")
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    
    model_kwargs = {"num_classes": num_classes}
    dummy_input = next(iter(train_loader))[0][0:2]
    
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    model = model_class(**model_kwargs)
    return model, train_loader, test_loader, dummy_input

def find_experiment_checkpoints(model_name, dataset_name, pre_epochs, post_epochs):
    """
    Finds model checkpoints across varied directory conventions (handles optional _None_ tags
    and case-insensitive dataset matching).
    """
    dataset_patterns = [dataset_name, dataset_name.lower(), dataset_name.capitalize()]
    before_ckpts = []
    collapsed_ckpts = []
    after_ckpts = []

    for d_name in set(dataset_patterns):
        # FIX: Added an underscore after {d_name} to prevent 'Cifar10' from matching 'Cifar100'
        base_pattern = f"../Tranfer/{model_name}_{d_name}_*epochs{post_epochs}*pre*{pre_epochs}"
        before_ckpts.extend(glob.glob(f"{base_pattern}/checkpoints/final_JF_Control.pt"))
        collapsed_ckpts.extend(glob.glob(f"{base_pattern}/checkpoints/final_JF_Dynamic_Region_All_Combined.pt"))
        after_ckpts.extend(glob.glob(f"{base_pattern}/checkpoints/final_JF_Control_Continuted.pt"))

    return {
        "before": sorted(list(set(before_ckpts))),
        "collapsed": sorted(list(set(collapsed_ckpts))),
        "after": sorted(list(set(after_ckpts)))
    }

def load_collapse_regions(model_name, dataset_name, pre_epochs, post_epochs):
    """Searches for discovered regions JSON across matching naming permutations."""
    dataset_patterns = [dataset_name, dataset_name.lower(), dataset_name.capitalize()]
    json_filename = None

    for d_name in set(dataset_patterns):
        candidate = f"../Tranfer/{model_name}_{d_name}_epochs{post_epochs}_pretrain{pre_epochs}_JF_discovered_regions.json"
        if os.path.isfile(candidate):
            json_filename = candidate
            break

    if json_filename is None:
        raise FileNotFoundError(
            f"Discovered regions file not found for {model_name}_{dataset_name} (pre={pre_epochs}, post={post_epochs})"
        )

    print(f"[INFO] Loading discovered regions: {json_filename}")
    with open(json_filename, "r") as f:
        discovered_regions = json.load(f)

    json_to_collapse = discovered_regions.get("Dynamic_Region_All_Combined")
    
    # Fallback for simpler JSON structures (like MobileNet)
    if not json_to_collapse:
        json_to_collapse = discovered_regions.get("Set_0")
        
    if not json_to_collapse:
        raise ValueError(f"No valid collapse regions (Dynamic_Region_All_Combined or Set_0) found in {json_filename}.")
        
    return {f"Region_{i}": pair for i, pair in enumerate(json_to_collapse)}


def extract_features(
    model: nn.Module, dataloader, device: torch.device, max_batches: int = 10
):
    """Extracts flattened activations from the penultimate layer for CKA."""
    print(f"[INFO] Extracting features from {max_batches} batches...")
    start_time = time.time()

    features = []

    def hook(module, input, output):
        features.append(output.flatten(start_dim=1).detach())

    target_layer = list(model.modules())[-2]
    print(f"[INFO] Feature extraction target layer: {target_layer.__class__.__name__}")

    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            inputs = inputs.to(device)
            model(inputs)

            print(f"[INFO] Feature extraction batch {i + 1}/{max_batches}")

    handle.remove()

    features = torch.cat(features, dim=0)

    print(
        f"[INFO] Feature extraction complete: "
        f"shape={tuple(features.shape)}, "
        f"time={time.time() - start_time:.2f}s"
    )

    return features


def compare_CKA(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between two feature matrices."""
    print(
        f"[INFO] Computing CKA: "
        f"features_x={tuple(features_x.shape)}, "
        f"features_y={tuple(features_y.shape)}"
    )

    start_time = time.time()

    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    dot_prod_xx = torch.norm(features_x.T @ features_x, p="fro")
    dot_prod_yy = torch.norm(features_y.T @ features_y, p="fro")
    dot_prod_xy = torch.norm(features_y.T @ features_x, p="fro")

    cka_score = (dot_prod_xy**2) / (dot_prod_xx * dot_prod_yy)
    cka_score = cka_score.item()

    print(
        f"[INFO] CKA complete: score={cka_score:.6f}, "
        f"time={time.time() - start_time:.2f}s"
    )

    return cka_score


# =========================================================
# Main Post-Processing Routine
# =========================================================

# -======================
def plot_experiment_heuristics(model_name, dataset_name, stats_csv_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # [✓] MOVED HERE: This breaks the circular import!
    from transfer import EXPERIMENTS 

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
    plt.savefig(f"runs/plots/{model_name}_heuristic_target_summary.png", dpi=300)
    print(f"Saved runs/plots/{model_name}_heuristic_target_summary.png")



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
    # summary(model, input_size=next(iter(loader))[0].shape, device=device)
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
            # print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")



# ===============================
# Cloning Utility
# ===============================

def clone_model(model, model_class):
    """Utility to clone a model and load weights to keep experiments isolated."""
    new_model = model_class()
    new_model.load_state_dict(model.state_dict())
    return new_model

