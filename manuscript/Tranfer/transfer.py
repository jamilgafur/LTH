# transfer.py
import os
import glob
import json
import random
import argparse
from manuscript.Tranfer.experiments import merge_all_metrics, safe_update_metrics_json
from manuscript.Tranfer.experiments import merge_all_metrics
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import gc
import json
from pathlib import Path
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies import MagnitudePruningStrategy
from experiments import *
from utils import *
from plots import *
from pyPrune.utils import *
from trainer import train_one_epoch
import gc  # Needed for cleaning up memory after evaluating models
from pathlib import Path # The missing piece causing your current error
from datetime import datetime
# set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
       
def safe_glob(path_pattern):
    matches = glob.glob(path_pattern)
    return matches[0] + "/" if matches else "None"

CHECKPOINT_BASES = {
    "VGG16": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasettinyimagenet_*"),
    },
    "RegNetX_400MF": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasettinyimagenet_*"),
    },
    "InceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasettinyimagenet_*"),
    },
    "MobileNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasettinyimagenet_*"),
    },
    "XceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasettinyimagenet_*"),
    },
    "ConvNeXt": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasettinyimagenet_*"),
    },
}

CHECKPOINT_FILES = {
    "VGG16": {
        "Cifar10": ("checkpoint_Finetuned_0.914101.pth", "checkpoint_Original_0.000000.pth"),
        "Cifar100": ("checkpoint_Finetuned_0.981986.pth", "checkpoint_Original_0.000000.pth"),
        "imagenet": ("checkpoint_Finetuned_0.790285.pth", "checkpoint_Original_0.000000.pth"),
        "tinyimagenet": ("None", "None"),
    },
    "RegNetX_400MF": {
        "Cifar10": ("checkpoint_Finetuned_0.945024.pth", "checkpoint_Original_0.000000.pth"),
        "Cifar100": ("checkpoint_Finetuned_0.488000.pth", "checkpoint_Original_0.000000.pth"),
        "imagenet": ("checkpoint_Finetuned_0.914101.pth", "checkpoint_Original_0.000000.pth"),
        "tinyimagenet": ("None", "None"),
    },
    "InceptionNet": {
        "Cifar10": ("None", "None"), "Cifar100": ("None", "None"),
        "imagenet": ("None", "None"), "tinyimagenet": ("None", "None"),
    },
    "MobileNet": {
        "Cifar10": ("None", "None"), "Cifar100": ("None", "None"),
        "imagenet": ("None", "None"), "tinyimagenet": ("None", "None"),
    },
    "XceptionNet": {
        "Cifar10": ("None", "None"), "Cifar100": ("None", "None"),
        "imagenet": ("None", "None"), "tinyimagenet": ("None", "None"),
    },
    "ConvNeXt": {
        "Cifar10": ("None", "None"), "Cifar100": ("None", "None"),
        "imagenet": ("None", "None"), "tinyimagenet": ("None", "None"),
    },
}

# ==============================================================================
# Dynamic Collapse Logic
# ==============================================================================
def get_layer_variances(model, dummy_input):
    """Minimal hook to capture the variance of each layer's activations."""
    variances = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if isinstance(out, torch.Tensor):
                variances[name] = out.var(dim=[0, 2, 3]).mean().item() if out.ndim == 4 else out.var().item()
        return hook
    
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))
            
    model.eval()
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
    return variances

def calculate_bav_states(variances, veto_fraction=0.25):
    """
    Computes the Bounded Activation Variance (BAV) state for each layer.
    Uses a dynamic sliding window that borrows from previous layers when 
    approaching the end of the network to maintain a stable local context.
    """
    states = []
    num_layers = len(variances)
    veto_idx = int(num_layers * veto_fraction)
    window_size = 3
    
    for i, sigma_i in enumerate(variances):
        # Try to look ahead by window_size
        look_ahead = variances[i+1 : i+1+window_size]
        
        # If we hit the end of the network, we won't have enough layers in 'look_ahead'.
        # Calculate the shortfall and borrow from immediately preceding layers.
        shortfall = window_size - len(look_ahead)
        if shortfall > 0 and i > 0:
            look_back_start = max(0, i - shortfall)
            look_back = variances[look_back_start : i]
            context_vars = look_back + look_ahead
        else:
            context_vars = look_ahead
            
        # Fallback to global mean ONLY if the network itself is tiny (e.g., 1 layer total)
        sigma_bar = np.mean(context_vars) if len(context_vars) > 0 else np.mean(variances)
        sigma_bar = max(sigma_bar, 1e-12)
        
        diff = sigma_i - sigma_bar
        h = max(diff / sigma_bar, -1.0) if diff < 0 else min(diff / sigma_bar, 1.0)
        
        if i < veto_idx:
            states.append("VETO")
        elif h < 0:
            states.append("SAFE")
        else:
            states.append("DANGER")
            
    return states

def find_efficient_subregions(model, layers_list, input_shape):
    """
    Recursively divides a contiguous block of layers to find the largest
    sub-blocks that reduce parameters without crossing structural boundaries.
    """
    import copy
    from collapse import collapse_only
    from utils import count_trainable_params

    # Base case: We need at least 2 layers to perform a collapse
    if len(layers_list) < 2:
        return []

    base_params = count_trainable_params(model)

    try:
        test_model = copy.deepcopy(model)
        collapsed_model = collapse_only(
            model=test_model,
            compression_set={"test": layers_list},
            input_shape=input_shape,
            dry_run=True
        )
        new_params = count_trainable_params(collapsed_model)

        # If it reduces or maintains memory, this entire block is valid!
        if new_params <= base_params:
            return [layers_list]

    except Exception:
        # If the collapse physically fails (e.g., shape mismatch across pooling boundaries)
        # we catch it and force a split below.
        pass

    # If we reach here, the block crossed a boundary (LCA explosion) and increased memory.
    # Split the block in half and recursively check both sides!
    mid = len(layers_list) // 2
    left_valid = find_efficient_subregions(model, layers_list[:mid], input_shape)
    right_valid = find_efficient_subregions(model, layers_list[mid:], input_shape)

    return left_valid + right_valid

def get_dynamic_experiment_config(model, cnn_layers, variances, input_shape=(1, 3, 224, 224), window_size=5):
    import numpy as np

    # 1. Calculate Rolling Mean and H-values (Your existing logic)
    rolling_means = []
    for i in range(len(variances)):
        start = max(0, i - window_size)
        end = min(len(variances), i + window_size + 1)
        rolling_means.append(np.mean(variances[start:end]))
        
    h_values = []
    for var, mean_var in zip(variances, rolling_means):
        if var - mean_var < 0:
            h_values.append(max((var - mean_var) / mean_var, -1))
        else:
            h_values.append(min((var - mean_var) / mean_var, 1))

    # 2. Extract Raw Contiguous Sets (h < 0)
    raw_sets = []
    current_set = []
    for i, h in enumerate(h_values):
        if h < 0:
            current_set.append(cnn_layers[i])
        else:
            if len(current_set) >= 2:
                raw_sets.append(current_set)
            current_set = []
    if len(current_set) >= 2:
        raw_sets.append(current_set)

    # 3. Process via Recursive Boundary Splitting
    experiment_regions = {}
    set_counter = 0
    all_combined_sets = {}

    for raw_set in raw_sets:
        # This will fracture [39...52] into safe sub-chunks like [39..45] and [46..52]
        valid_subregions = find_efficient_subregions(model, raw_set, input_shape)

        for valid_set in valid_subregions:
            set_name = f"Set_{set_counter}"
            experiment_regions[set_name] = valid_set
            all_combined_sets[set_name] = valid_set
            set_counter += 1

    # 4. Generate the "Set of All Sets"
    if all_combined_sets:
        experiment_regions["Dynamic_Region_All_Combined"] = all_combined_sets

    return experiment_regions
def is_feasible_experiment_config(experiment_regions, cnn_layers, model=None, input_shape=None, device='cpu'):
    import copy
    from collapse import collapse_only
    from utils import count_trainable_params
    from fvcore.nn import FlopCountAnalysis
    import gc
    import torch
    
    def get_module_prefix(layer_name):
        # FIX FOR ISSUE 3: Group by the top-level macro-stage ONLY.
        # e.g., 's1.b1.conv' and 's1.b2.conv' both return 's1'.
        parts = str(layer_name).split('.')
        return parts[0] if len(parts) > 0 else str(layer_name)

    validated_regions = []
    
    # --- Guardrail: Prevent cross-architectural boundary regions ---
    if experiment_regions and len(experiment_regions) > 0:
        for start_layer, end_layer in experiment_regions:
            start_prefix = get_module_prefix(start_layer)
            end_prefix = get_module_prefix(end_layer)
            
            if start_prefix == end_prefix:
                validated_regions.append((start_layer, end_layer))
            else:
                print(f"[WARN] Region {start_layer} -> {end_layer} crosses macro boundaries. Splitting...")
                s_idx = cnn_layers.index(start_layer)
                e_idx = cnn_layers.index(end_layer)
                
                curr_start = s_idx
                curr_prefix = get_module_prefix(cnn_layers[curr_start])
                
                for i in range(s_idx, e_idx + 1):
                    prefix_i = get_module_prefix(cnn_layers[i])
                    if prefix_i != curr_prefix:
                        # Enforce |s| >= 2 even during boundary splits
                        if (i - 1) - curr_start >= 1: # Represents 2 actual layers inclusive
                            validated_regions.append((cnn_layers[curr_start], cnn_layers[i-1]))
                        curr_start = i
                        curr_prefix = prefix_i
                        
                if e_idx - curr_start >= 1:
                    validated_regions.append((cnn_layers[curr_start], cnn_layers[e_idx]))

    def to_dict(region_list):
        return {f"Dynamic_Region_{i} (Full)": reg for i, reg in enumerate(region_list)}

    # --- Step 2: Surrogate Memory & FLOP Filter ---
    if model is None or input_shape is None:
        return to_dict(validated_regions)

    print(f"\n[INFO] Running Surrogate Hardware Filter on {len(validated_regions)} proposed regions...")
    feasible_regions = []
    
    # Generate dummy input to accurately measure FLOPs
    dummy_input = torch.randn(1, *input_shape[1:]).to(device)
    
    model.eval()
    original_params = count_trainable_params(model)
    # Silence fvcore warnings
    original_flops = FlopCountAnalysis(model, dummy_input).unsupported_ops_warnings(False).total()

    for region in validated_regions:
        start_layer, end_layer = region
        print(f"[STEP] Validating collapse hardware delta: {start_layer} -> {end_layer}")
        
        try:
            test_model = copy.deepcopy(model).to(device)
            collapsed_model = collapse_only(
                model=test_model,
                compression_set=[(start_layer, end_layer)],
                input_shape=input_shape,
                device=device,
                safe_param_reduction=True,
                handle_skips=True,
                debug=False,
                dry_run=False
            )
            
            collapsed_model.eval()
            collapsed_params = count_trainable_params(collapsed_model)
            collapsed_flops = FlopCountAnalysis(collapsed_model, dummy_input).unsupported_ops_warnings(False).total()
            
            delta_params = original_params - collapsed_params
            delta_flops = original_flops - collapsed_flops
            
            # FIX FOR ISSUES 1 & 4: STRICT EVALUATION OF BOTH PARAMS AND FLOPS
            if collapsed_params < original_params and collapsed_flops < original_flops:
                print(f"    [✓] Kept: Params reduced by {delta_params:,} | FLOPs reduced by {delta_flops:,}")
                feasible_regions.append(region)
            else:
                print(f"    [X] Dropped: Surrogate inflated Params ({delta_params:,}) or FLOPs ({delta_flops:,}).")
                
        except (Exception, SystemExit) as e:
            print(f"    [!] Dropped: Surrogate validation failed - {type(e).__name__}: {e}")
            
        finally:
            if 'test_model' in locals(): del test_model
            if 'collapsed_model' in locals(): del collapsed_model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    final_experiments = to_dict(feasible_regions)
    if len(feasible_regions) > 1:
        final_experiments["Dynamic_Region_All_Combined"] = feasible_regions
        
    return final_experiments

# ==============================================================================
# Helper functions
# ==============================================================================
def create_optimizer_scheduler(model, learning_rate=1e-3):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

def initialize_model_and_data(args):
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    if args.model == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

def setup_directories(save_root_dir):
    dirs = {
        "var": os.path.join(save_root_dir, "Heuristic_Variance"),
        "sim": os.path.join(save_root_dir, "Heuristic_Redundancy"),
        "kl": os.path.join(save_root_dir, "Heuristic_Bypass_KL"),
        "cscore": os.path.join(save_root_dir, "Heuristic_Collapse_Score"),
        "layer_stats": os.path.join(save_root_dir, "Layer_Statistics")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def run_baseline_pass(model, input_tensor):
    saved_tensors = {}
    layer_variances = {}
    layer_activations = {}

    def unbroken_hook(name):
        def fn(module, inp, out):
            if not isinstance(out, torch.Tensor) or not isinstance(inp[0], torch.Tensor):
                return
            x = inp[0].detach().cpu()
            y = out.detach().cpu()
            saved_tensors[name] = {"in": x, "out": y}
            if y.ndim == 4:
                act_var = y.var(dim=[2, 3]).mean().item()
                act_mean = y.mean(dim=[2, 3]).mean().item()
            else:
                act_var = y.var().item()
                act_mean = y.mean().item()
                
            layer_variances[name] = act_var
            layer_activations[name] = act_mean
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(unbroken_hook(name)))

    with torch.no_grad():
        baseline_logits = model(input_tensor)
        baseline_probs = F.softmax(baseline_logits, dim=1)

    for h in hooks:
        h.remove()

    global_median_var = float(np.median(list(layer_variances.values()))) if layer_variances else 1.0
    return saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs

def plot_individual_layers(layer_activations, layer_variances, directory, model_name, dataset_name, exp_config=None):
    if not layer_activations:
        return
    layers = list(layer_activations.keys())
    activations = list(layer_activations.values())
    variances = list(layer_variances.values())

    df = pd.DataFrame({"Layer": layers, "Mean Activation": activations, "Variance": variances})
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats.csv"), index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    regions = {}
    if exp_config:
        for key, val in exp_config.items():
            if "(Full)" in key and isinstance(val, tuple):
                start_layer, end_layer = val
                start_idx = next((i for i, n in enumerate(layers) if start_layer in n), None)
                end_idx = next((i for i, n in reversed(list(enumerate(layers))) if end_layer in n), None)
                if start_idx is not None and end_idx is not None:
                    clean_name = key.replace(" (Full)", "")
                    regions[clean_name] = (start_idx, end_idx)

    bg_colors = ['#eaf2f8', '#fdf2e9', '#e8f8f5', '#f5eef8', '#f4f6f7']
    for i, (region_name, (start, end)) in enumerate(regions.items()):
        color = bg_colors[i % len(bg_colors)]
        ax1.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        ax2.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        y_max = max(variances) if variances else 1
        ax2.text((start + end) / 2, y_max * 0.95, region_name, ha='center', va='top', fontsize=11, fontweight='bold', color='#555555', alpha=0.8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    sns.lineplot(data=df, x="Layer", y="Mean Activation", marker="o", color="steelblue", linewidth=2, ax=ax1, zorder=3)
    ax1.set_ylabel("Mean Activation", fontweight='bold', labelpad=10)
    ax1.set_title(f"Layer-wise Activation & Structural Stages\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=12)

    sns.lineplot(data=df, x="Layer", y="Variance", marker="s", color="crimson", linewidth=2, linestyle="--", ax=ax2, zorder=3)
    ax2.set_ylabel("Variance", fontweight='bold', labelpad=10)
    ax2.set_xlabel("Network Layer", fontweight='bold', labelpad=10)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=90, fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats_annotated.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_metrics(layer_activations, layer_variances, directory, model_name, dataset_name):
    if not layer_activations: return
    
    layers = list(layer_activations.keys())
    means = np.array(list(layer_activations.values()))
    vars_arr = np.array(list(layer_variances.values()))
    
    avg_var = np.mean(vars_arr)
    norm_vars = vars_arr / (avg_var + 1e-12)
    
    cvs = vars_arr / (np.abs(means) + 1e-12) 
    avg_cv = np.mean(cvs)
    norm_cvs = cvs / (avg_cv + 1e-12)
    
    df = pd.DataFrame({"Layer": layers, "Normalized Variance": norm_vars, "Normalized CV": norm_cvs})
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_normalized_layer_stats.csv"), index=False)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized Variance", color="coral", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-Variance (1.0)")
    ax.set_title(f"Normalized Layer Variance\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Variance / Avg Variance", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_variance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized CV", color="mediumpurple", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-CV (1.0)")
    ax.set_title(f"Normalized Coefficient of Variation (CV)\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Layer CV / Avg CV", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_cv.png"), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_experiments(model, input_tensor, exp_config, layer_names, module_dict, saved_tensors, layer_variances, global_median_var, baseline_probs):
    plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore = [], [], [], []

    for exp_name, layer_range in exp_config.items():
        exp_display = exp_name.replace("_", " ")

        if layer_range is None:
            plot_data_var.append({"Experiment": exp_display, "Relative Variance": 1.0})
            sim_scores = []
            for t_data in saved_tensors.values():
                t_in, t_out = t_data["in"], t_data["out"]
                if t_in.shape == t_out.shape:
                    in_flat = t_in.flatten(start_dim=1)
                    out_flat = t_out.flatten(start_dim=1)
                    try: sim_scores.append(F.cosine_similarity(in_flat, out_flat, dim=1).mean().item())
                    except: pass
            
            global_sim = float(np.median(sim_scores)) if sim_scores else 0.0
            plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": global_sim})
            plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": 0.0})
            baseline_cscore = global_sim / (1.0 * (1.0 + 0.0))
            plot_data_cscore.append({"Experiment": exp_display, "Collapse Score": baseline_cscore})
            continue

        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        block_vars, block_sims = [], []
        bypass_handles, bypass_cache = [], {}
        valid_bypass = True

        def get_start_hook(idx):
            def hook(module, inp, out): bypass_cache[idx] = inp[0]
            return hook

        def get_end_hook(idx):
            def hook(module, inp, out):
                if idx in bypass_cache:
                    cached_inp = bypass_cache[idx]
                    if cached_inp.shape == out.shape: return cached_inp
                    else: return torch.zeros_like(out)
                return out
            return hook

        for idx, (start_layer, end_layer) in enumerate(ranges):
            start_name = next((n for n in layer_names if start_layer in n), None)
            end_name = next((n for n in reversed(layer_names) if end_layer in n), None)

            if start_name and end_name:
                in_range = False
                for name in layer_names:
                    if name == start_name: in_range = True
                    if in_range and name in layer_variances:
                        block_vars.append(layer_variances[name])
                    if name == end_name: break
                
                if start_name in saved_tensors and end_name in saved_tensors:
                    block_in = saved_tensors[start_name]["in"]
                    block_out = saved_tensors[end_name]["out"]
                    if block_in.shape == block_out.shape:
                        in_flat = block_in.flatten(start_dim=1)
                        out_flat = block_out.flatten(start_dim=1)
                        try: sim = F.cosine_similarity(in_flat, out_flat, dim=1).mean().item()
                        except: sim = 0.0
                    else: sim = 0.0 
                    block_sims.append(sim)

                start_mod = module_dict[start_name]
                end_mod = module_dict[end_name]
                bypass_handles.append(start_mod.register_forward_hook(get_start_hook(idx)))
                bypass_handles.append(end_mod.register_forward_hook(get_end_hook(idx)))
            else:
                valid_bypass = False

        exp_rel_var = (float(np.median(block_vars)) / global_median_var) if block_vars and global_median_var > 0 else 1.0
        plot_data_var.append({"Experiment": exp_display, "Relative Variance": exp_rel_var})
        exp_sim = float(np.median(block_sims)) if block_sims else 0.0
        plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": exp_sim})

        if not valid_bypass: kl_div = 50.0 
        else:
            try:
                with torch.no_grad():
                    bypass_logits = model(input_tensor)
                    bypass_log_probs = F.log_softmax(bypass_logits, dim=1)
                kl_div = F.kl_div(bypass_log_probs, baseline_probs, reduction='batchmean').item()
            except: kl_div = 50.0 
        
        for h in bypass_handles: h.remove()
        bypass_cache.clear()

        display_kl = kl_div if kl_div < 50.0 else 50.0
        plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": display_kl})
        safe_rel_var = max(exp_rel_var, 1e-8)
        c_score = exp_sim / (safe_rel_var * (1.0 + display_kl))
        plot_data_cscore.append({"Experiment": exp_display, "Collapse Score": c_score})

    return plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore

def save_and_plot_metric(data, y_col, directory, title_prefix, ylabel, hline_val, hline_label, color_base, color_alt, model_name, dataset_name, invert_safe_zone=False):
    if not data: return
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_{y_col.replace(' ', '_')}.csv"), index=False)
    df.to_latex(os.path.join(directory, f"{model_name}_{dataset_name}.tex"), index=False, float_format="%.4f")

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 7))
    df['Color_Group'] = ['Baseline' if exp == 'Original Model' else 'Experiment' for exp in df['Experiment']]
    palette = {'Baseline': color_base, 'Experiment': color_alt}

    sns.barplot(data=df, x="Experiment", y=y_col, hue="Color_Group", palette=palette, dodge=False, edgecolor="black", linewidth=0.8, zorder=3, ax=ax)
    ax.legend_.remove()

    ymin, ymax = ax.get_ylim()
    if df[y_col].min() >= 0: ymin = 0.0  
    else: ymin = min(df[y_col].min() * 1.05, ymin)
    if hline_val > ymax: ymax = hline_val * 1.15
    ax.set_ylim(ymin, ymax)

    ax.axhline(0, color='black', linewidth=1.5, zorder=4) 
    ax.axhline(hline_val, color='crimson', linestyle='--', linewidth=2.5, zorder=4, label=hline_label)

    if invert_safe_zone:
        ax.axhspan(hline_val, ymax, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe (High Redundancy)')
        ax.axhspan(ymin, hline_val, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')
    else:
        ax.axhspan(ymin, hline_val, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe')
        ax.axhspan(hline_val, ymax, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')

    ax.set_title(f"{title_prefix}\n{model_name} | {dataset_name}", fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel("Structural Modification", fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)
    sns.despine(bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_experiment_{y_col.split(' ')[0]}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name, exp_config=None):
    print(f"[•] Running Comprehensive Heuristic Analysis for {model_name} on {dataset_name}...")
    
    model.eval()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    module_dict = dict(model.named_modules())
    layer_names = list(module_dict.keys())

    dirs = setup_directories(save_root_dir)
    saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs = run_baseline_pass(model, input_tensor)

    # Use the provided config, or generate dynamically if None is passed
    if exp_config is None:
        device = next(model.parameters()).device
        exp_config = get_dynamic_experiment_config(
            list(layer_variances.keys()), 
            list(layer_variances.values()),
            model=model,
            input_shape=input_tensor.shape,
            device=device
        )
        
    if not exp_config: print("[WARN] No experiment config dynamically generated.")

    plot_individual_layers(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name, exp_config)
    plot_normalized_metrics(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name)

    try:
        plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore = evaluate_experiments(
            model, input_tensor, exp_config, layer_names, module_dict, 
            saved_tensors, layer_variances, global_median_var, baseline_probs
        )
    except Exception as e:
        print(f"[!] Failed to process experiments: {e}")
        return pd.DataFrame()

    save_and_plot_metric(plot_data_var, "Relative Variance", dirs["var"], "Relative Activation Variance", "Relative Variance (Multiplier)", 1.0, "1.0x Baseline", 'crimson', 'steelblue', model_name, dataset_name)
    global_sim_val = plot_data_sim[0]["Block Redundancy"] if plot_data_sim else 0.0
    save_and_plot_metric(plot_data_sim, "Block Redundancy", dirs["sim"], "Feature Redundancy (Cosine Similarity)", "Cosine Similarity (1.0 = Identity)", global_sim_val, "Global Median", 'crimson', 'mediumseagreen', model_name, dataset_name, invert_safe_zone=True)
    save_and_plot_metric(plot_data_kl, "Prediction Shift (KL)", dirs["kl"], "Virtual Bypass Prediction Damage", "KL Divergence (0.0 = Safe | 50.0 = Failed)", 1.0, "Critical Threshold (Approx)", 'crimson', 'teal', model_name, dataset_name)
    global_cscore_val = plot_data_cscore[0]["Collapse Score"] if plot_data_cscore else 1.0
    save_and_plot_metric(plot_data_cscore, "Collapse Score", dirs["cscore"], "Composite Activational Collapse Score", "C_Score (Higher = Safer to Collapse)", global_cscore_val, "Baseline Architecture Score", 'crimson', 'purple', model_name, dataset_name, invert_safe_zone=True)

    return pd.DataFrame(plot_data_cscore)

def run_jf_or_kevin_experiment(experiment_name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path, post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args):
    model_class = eval(model_class) if isinstance(model_class, str) else model_class
    if args.JF:
        return run_jf_experiment({experiment_name: layers}, model_path_097, train_loader, test_loader, device, epochs, pretrain, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path, post_compress_epochs=post_compress_epochs, quant=quant)
    elif args.Kevin:
        if experiment_name == "Original Model": epochs = pretrain + epochs
        return run_kevin_experiment({experiment_name: layers}, model_path_000, train_loader, test_loader, device, epochs, model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path, post_compress_epochs=post_compress_epochs, quant=quant)
    else: raise ValueError("Specify either --JF or --Kevin.")

def run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, post_compress_epochs, experiment_func, quant=False, args=None):
    if model_class in [InceptionNet, XceptionNet, MobileNet]:
        steps = [0]
        epochs = pretrain
        pretrain = 0
    else:
        steps = exponential_decay_list(steps=21)

    save_path = f"{model_class}_{dataset}_{CHECKPOINT_FILES[args.model][dataset][0]}_epochs{epochs}_pretrain{pretrain}_postcompress{post_compress_epochs}"

    if train_loader is None or test_loader is None:
        train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, args.model)
    else:
        input_size = model_kwargs['one_batch'].shape

    for name, layers in experiments.items():
        print(f"\n--- Running experiment: {name} ---")
        run_jf_or_kevin_experiment(name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path, post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args)

# ==============================================================================
# Main Entry Point
# ==============================================================================
import os
import json
import argparse
import torch

def auto_recover_metrics(checkpoint_path, experiment_name, base_folder):
    """
    Checks if the experiment's metrics exist in the merged_metrics.json.
    If not, it dynamically extracts them from the .pt checkpoint dictionary and saves them.
    """
    metrics_dir = os.path.join(base_folder, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    merged_json_path = os.path.join(metrics_dir, "merged_metrics.json")

    # Load existing JSON or create a fresh dictionary
    if os.path.exists(merged_json_path):
        with open(merged_json_path, 'r') as f:
            try:
                metrics_data = json.load(f)
            except json.JSONDecodeError:
                metrics_data = {}
    else:
        metrics_data = {}

    # If the metric is missing, heal it
    if experiment_name not in metrics_data:
        print(f"[Auto-Heal] Missing JSON entry for '{experiment_name}'. Extracting from checkpoint...")
        
        try:
            # Map to CPU to prevent VRAM spikes during fast recovery
            ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract the raw data directly from the saved dictionary
            metrics_data[experiment_name] = {
                "final_accuracy": ckpt.get("final_accuracy", 0.0),
                "param_count": ckpt.get("param_count", 0),
                "total_size_mb": ckpt.get("total_size_mb", 0.0),
                "flops": ckpt.get("flops", 0),
                "timestamp_recovered": "auto_recovered"
            }
            
            # Save the healed JSON
            with open(merged_json_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            print(f"[✓] Successfully healed merged_metrics.json for {experiment_name}.")
            
        except Exception as e:
            print(f"[!] Auto-Heal failed for {experiment_name}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RegNetX_400MF", choices=["VGG16", "RegNetX_400MF", "InceptionNet", "XceptionNet", "MobileNet", "ConvNeXt"], help="Model architecture to use")
    parser.add_argument("--dataset", type=str, default="Cifar10", help="Dataset to use")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--pretrain", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--experiment", type=str, default="discover", help="Experiment to run, or 'discover' to generate regions")
    parser.add_argument("--post_compress_epochs", type=int, default=0, help="Number of post-pruning compression epochs")
    parser.add_argument("--imp", action="store_false", help="Apply Iterative Magnitude Pruning")
    parser.add_argument("--JF", action="store_true", help="Run JF experiments")
    parser.add_argument("--Kevin", action="store_true", help="Run Kevin experiments")
    parser.add_argument("--quant", action="store_true", help="Apply Quantization Aware Training")
    # --regenerate flag removed! Pipeline is now fully automated and self-healing.
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"[INIT] PyPrune Transfer Learning Framework")
    print(f"[INIT] Arguments: {args}")
    print(f"{'='*60}\n")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[INFO] Hardware detected: {device.type.upper()} (CUDA Available: {torch.cuda.is_available()})")

    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    base_path = CHECKPOINT_BASES[args.model][args.dataset]
    print(f"[DEBUG] Base checkpoint path resolved: {base_path}")
    
    model_path_097 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][0])
    model_path_000 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][1])

    json_file = f"{args.model}_{args.dataset}_{'JF' if args.JF else 'Kevin'}_discovered_regions.json"

    # =========================================================================
    # PRE-FLIGHT PROBE: Train (if needed), capture variances, output JSON
    # =========================================================================
    if args.experiment == "discover":
        print(f"\n{'='*60}")
        print(f"[MODE] STAGE 1: DISCOVERY & PRE-FLIGHT PROBE")
        print(f"       Model: {args.model} | Dataset: {args.dataset}")
        print(f"{'='*60}\n")
        
        # 1. Run/Ensure the Original Model is trained using YOUR existing logic. 
        print(f"[INFO] Phase 1: Validating/Training 'Original Model' Baseline...")
        run_experiments_for_dataset(
            {"Original Model": None}, args.dataset, model_path_097, model_path_000, 
            train_loader, test_loader, device, args.epochs, args.pretrain, model_class, 
            model_kwargs, args.post_compress_epochs, None, args.quant, args
        )

        # 2. Reconstruct save path to grab the finalized checkpoint
        print(f"\n[INFO] Phase 2: Loading Finalized Baseline Weights...")
        
        # ---> CRITICAL FIX: Build the path using the raw args BEFORE they are modified! <---
        save_path = f"{args.model}_{args.dataset}_{CHECKPOINT_FILES[args.model][args.dataset][0]}_epochs{args.epochs}_pretrain{args.pretrain}_postcompress{args.post_compress_epochs}"
        
        ckpt_dir = os.path.join(save_path, "checkpoints")
        flag_str = "JF" if args.JF else "Kevin"
        quant_str = "_quant" if args.quant else ""
        trained_baseline_path = os.path.join(ckpt_dir, f"final_{flag_str}_Original Model{quant_str}.pt")
        print(f"[DEBUG] Expected trained baseline path: {trained_baseline_path}")
        
        # 3. Load the weights and calculate variances
        model_class_obj = eval(model_class) if isinstance(model_class, str) else model_class
        eval_model = model_class_obj(**model_kwargs).to(device)
        
        if os.path.exists(trained_baseline_path):
            print(f"[INFO] Checkpoint located.")
            
            # ---> INJECT AUTO-HEAL HERE <---
            # Before loading the weights, ensure the JSON file is intact
            experiment_name = f"Original Model{quant_str}"
            auto_recover_metrics(
                checkpoint_path=trained_baseline_path, 
                experiment_name=experiment_name, 
                base_folder=save_path
            )

            print(f"       Restoring model state dict...")
            ckpt = torch.load(trained_baseline_path, map_location=device, weights_only=False)
            eval_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['model'], strict=False)
            print(f"[✓] Weights successfully applied.")
        else:
            print(f"[ERROR] Failed to find finalized model at {trained_baseline_path}")
            print(f"[ERROR] Discovery cannot proceed without a valid baseline.")
            return

        print(f"\n[INFO] Phase 3: Executing Network Probe...")
        input_tensor = model_kwargs["one_batch"].to(device)
        if len(input_tensor.shape) == 3: input_tensor = input_tensor.unsqueeze(0)
        
        layer_variances = get_layer_variances(eval_model, input_tensor)
        
        # Type error fix implemented here
        dynamic_experiments = get_dynamic_experiment_config(args.model, layer_variances, variances=list(layer_variances.values()), input_shape=input_tensor.shape, window_size=3)

        print(f"\n[INFO] Phase 4: Generating Heuristic Plots & Analytics...")
        plots_root = os.path.join("runs", "plots")
        analyze_collapse_heuristics(
            model=eval_model, 
            input_tensor=input_tensor, 
            save_root_dir=plots_root, 
            model_name=args.model, 
            dataset_name=args.dataset,
            exp_config=dynamic_experiments # <--- Phase 3 config explicitly passed
        )
        csv_path = os.path.join(plots_root, "Layer_Statistics", f"{args.model}_{args.dataset}_layer_stats.csv")
        # if os.path.exists(csv_path):
        #     plot_experiment_heuristics(args.model, args.dataset, csv_path)
        print(f"\n[INFO] Phase 5: Exporting Configuration Map...")
        with open(json_file, 'w') as f:
            json.dump(dynamic_experiments, f, indent=4)
        print(f"[✓] Stage 1 Complete. Exported {len(dynamic_experiments)} targets to '{json_file}'.")
        return

    # =========================================================================
    # HPC EXECUTION: Read JSON and run the requested experiment
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"[MODE] STAGE 2: HPC EXPERIMENT EXECUTION")
    print(f"       Targeting: '{args.experiment}'")
    print(f"{'='*60}\n")
    
    print(f"[INFO] Validating JSON configuration map at: {json_file}")
    if not os.path.exists(json_file):
        print(f"[ERROR] Map not found. You must run Stage 1 (discover) prior to HPC array execution.")
        raise FileNotFoundError(f"Missing {json_file}")
        
    with open(json_file, 'r') as f:
        dynamic_experiments = json.load(f)
        
    print(f"[DEBUG] Loaded {len(dynamic_experiments)} configurations from map.")
        
    if args.experiment not in dynamic_experiments:
        print(f"[ERROR] The requested experiment '{args.experiment}' does not exist in the generated JSON.")
        raise ValueError(f"Experiment '{args.experiment}' not found in {json_file}.")

    print(f"[INFO] Handoff to PyPrune Experiment Framework...")
    run_experiments_for_dataset(
        {args.experiment: dynamic_experiments[args.experiment]}, args.dataset, model_path_097, model_path_000, 
        train_loader, test_loader, device, args.epochs, args.pretrain, model_class, 
        model_kwargs, args.post_compress_epochs, None, args.quant, args
    )
if __name__ == "__main__":
    main()
