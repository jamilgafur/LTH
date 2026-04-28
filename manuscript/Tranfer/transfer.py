# transfer.py
import os
import torch
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
from torch.backends import cudnn
import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
import argparse

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
# DYNAMIC EXPERIMENT GENERATOR
# ==============================================================================
def get_dynamic_experiment_config(layer_variances):
    """
    Dynamically generates collapse regions based on the piecewise variance function.
    Identifies contiguous sets of layers where variance is below the network mean,
    and returns full collapse sets alongside their individual layers.
    """
    exp_config = {"Original Model": None}
    if not layer_variances:
        return exp_config

    layer_names = list(layer_variances.keys())
    variances = list(layer_variances.values())
    sigma_bar = np.mean(variances)

    def calculate_h(sigma_i, s_bar):
        diff = sigma_i - s_bar
        if diff < 0:
            return max(diff / s_bar, -1.0)
        else:
            return min(diff / s_bar, 1.0)

    collapse_sets = []
    current_set = []

    for i, name in enumerate(layer_names):
        sigma_i = variances[i]
        h_val = calculate_h(sigma_i, sigma_bar)

        if h_val < 0:
            current_set.append(name)
        else:
            if current_set:
                collapse_sets.append(current_set)
                current_set = []
                
    if current_set:
        collapse_sets.append(current_set)

    for k, s in enumerate(collapse_sets):
        if len(s) > 1:
            exp_config[f"Set {k+1} (Full)"] = (s[0], s[-1])
        for layer in s:
            exp_config[f"Single Layer: {layer}"] = (layer, layer)

    return exp_config

# Helper functions
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
    model_kwargs["one_batch"] = next(iter(train_loader))[0]
    
    if args.model == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

# -------------------------------------------------------------
# HPC Safe Heuristic Profiling
# -------------------------------------------------------------
def run_heuristic_profiling_safely(model_class, model_kwargs, train_loader, epochs, device, dataset, model_name):
    import os
    import glob
    import torch
    import time
    
    lock_dir_path = f"{model_name}_{dataset}_heuristics.lock_dir"
    done_marker_path = f"{model_name}_{dataset}_heuristics_done.marker"
    plots_root_dir = os.path.join("runs", "plots")
    ckpt_root_dir = os.path.join("runs", "checkpoints", "heuristics")
    
    ensure_dir(plots_root_dir)
    ensure_dir(ckpt_root_dir)

    is_already_done = os.path.exists(done_marker_path)
    if is_already_done:
        print(f"[INFO] Done marker found for {model_name}. Will load checkpoint and skip to analysis.")

    print(f"[INFO] Attempting to acquire lock: {lock_dir_path}")
    try:
        os.mkdir(lock_dir_path)
        print("[INFO] Lock acquired! Starting/Resuming process...")
    except FileExistsError:
        print("[INFO] Lock busy. Another job is running this. Skipping.")
        return
    except OSError as e:
        print(f"[WARN] OS Error acquiring lock: {e}")
        return

    try:
        if isinstance(model_class, str):
            model_class_obj = eval(model_class)
        else:
            model_class_obj = model_class

        model = model_class_obj(**model_kwargs).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        ckpt_prefix = f"{model_name}_{dataset}_heuristic"
        ckpt_pattern = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch*.pt")
        existing_ckpts = sorted(
            glob.glob(ckpt_pattern),
            key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0])
        )

        start_epoch = 0
        if existing_ckpts:
            last_ckpt = existing_ckpts[-1]
            print(f"[INFO] Found checkpoint: {last_ckpt}. Loading state...")
            checkpoint = torch.load(last_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"[✓] Model loaded. State is at Epoch {start_epoch}")

        if not is_already_done and start_epoch < epochs:
            print(f"[INFO] Training model for {epochs} epochs...")
            for epoch in range(start_epoch + 1, epochs + 1):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
                print(f"    [Epoch {epoch}/{epochs}] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
                ckpt_path = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                }, ckpt_path)
                prev_ckpt = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch{epoch-1}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
        else:
            print(f"[INFO] Skipping training loop. Proceeding to analysis.")

        print("[INFO] Running analysis...")
        input_sample = model_kwargs["one_batch"].to(device)
        analyze_collapse_heuristics(
            model=model, 
            input_tensor=input_sample, 
            save_root_dir=plots_root_dir, 
            model_name=model_name,
            dataset_name=dataset
        )
        
        with open(done_marker_path, 'w') as f:
            f.write(f"Completed/Verified at {time.ctime()}")
            
        print("[INFO] Heuristic profiling complete. Plots saved.")

    except Exception as e:
        print(f"[ERROR] An error occurred during heuristic profiling: {e}")

    finally:
        if os.path.exists(lock_dir_path):
            try:
                os.rmdir(lock_dir_path)
                print("[INFO] Lock released.")
            except OSError:
                pass

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
    """Updated to use Bar Charts for discrete layer representation."""
    if not layer_activations: return
    
    layers = list(layer_activations.keys())
    activations = list(layer_activations.values())
    variances = list(layer_variances.values())

    df = pd.DataFrame({
        "Layer": layers,
        "Mean Activation": activations,
        "Variance": variances
    })
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

    # ---- UPDATED to Barplot ----
    sns.barplot(data=df, x="Layer", y="Mean Activation", color="steelblue", ax=ax1, zorder=3)
    ax1.set_ylabel("Mean Activation", fontweight='bold', labelpad=10)
    ax1.set_title(f"Layer-wise Activation & Structural Stages\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=12)

    sns.barplot(data=df, x="Layer", y="Variance", color="crimson", ax=ax2, zorder=3)
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

def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
    print(f"[•] Running Comprehensive Heuristic Analysis for {model_name} on {dataset_name}...")
    model.eval()
    if len(input_tensor.shape) == 3: input_tensor = input_tensor.unsqueeze(0)

    module_dict = dict(model.named_modules())
    layer_names = list(module_dict.keys())

    dirs = setup_directories(save_root_dir)
    saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs = run_baseline_pass(model, input_tensor)

    # 3. Dynamic Experiment Fetch
    exp_config = get_dynamic_experiment_config(layer_variances)
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
    save_path = f"{model_class}_{dataset}_{CHECKPOINT_FILES[args.model][dataset][0]}_epochs{epochs}_pretrain{pretrain}_postcompress{post_compress_epochs}"

    if args.model in ["InceptionNet", "XceptionNet", "MobileNet"]:
        epochs = pretrain
        pretrain = 0

    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, args.model)

    run_heuristic_profiling_safely(model_class=model_class, model_kwargs=model_kwargs, train_loader=train_loader, epochs=epochs, device=device, dataset=dataset, model_name=args.model)
    
    print("\n[STEP] Generating Heuristic and Architectural Visuals...")
    csv_path = os.path.join("runs/plots", "Layer_Statistics", f"{args.model}_{dataset}_layer_stats.csv")
    if os.path.exists(csv_path):
        try: plot_experiment_heuristics(args.model, dataset, csv_path)
        except Exception as e: print(f"[WARN] Failed to generate visual plots: {e}")
    else: print(f"[WARN] Could not find {csv_path} to generate heuristic plots.")

    for name, layers in experiments.items():
        print(f"\n--- Running experiment: {name} ---")
        run_jf_or_kevin_experiment(name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path, post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RegNetX_400MF", choices=["VGG16", "RegNetX_400MF", "InceptionNet", "XceptionNet", "MobileNet", "ConvNeXt"], help="Model architecture to use")
    parser.add_argument("--dataset", type=str, default="Cifar10", help="Dataset to use (Cifar10, Cifar100, ImageNet, TinyImageNet)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--pretrain", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--experiment", type=str, default="all", help="Experiment to run, or 'all' to run dynamically discovered sets")
    parser.add_argument("--post_compress_epochs", type=int, default=0, help="Number of post-pruning compression epochs")
    parser.add_argument("--imp", action="store_false", help="Apply Iterative Magnitude Pruning")
    parser.add_argument("--JF", action="store_true", help="Run JF experiments")
    parser.add_argument("--Kevin", action="store_true", help="Run Kevin experiments")
    parser.add_argument("--quant", action="store_true", help="Apply Quantization Aware Training")
    args = parser.parse_args()
    
    print(args)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    base_path = CHECKPOINT_BASES[args.model][args.dataset]
    model_path_097 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][0])
    model_path_000 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][1])

    print("\n[INFO] Running baseline pass to discover collapse targets...")
    model_class_obj = eval(model_class) if isinstance(model_class, str) else model_class
    eval_model = model_class_obj(**model_kwargs).to(device)
    input_tensor = model_kwargs["one_batch"].to(device)
    
    _, layer_variances, _, _, _ = run_baseline_pass(eval_model, input_tensor)
    dynamic_experiments = get_dynamic_experiment_config(layer_variances)
    
    if args.experiment == "all":
        experiment_dict = dynamic_experiments
    else:
        if args.experiment not in dynamic_experiments:
            raise ValueError(f"Experiment '{args.experiment}' not dynamically found. Available experiments: {list(dynamic_experiments.keys())}")
        experiment_dict = {args.experiment: dynamic_experiments[args.experiment]}

    run_experiments_for_dataset(
        experiment_dict,
        args.dataset,
        model_path_097,
        model_path_000,
        None, None, device, args.epochs, args.pretrain, model_class,
        model_kwargs, args.post_compress_epochs, None, args.quant, args
    )

if __name__ == "__main__":
    main()