# experiment.py

import os
import glob
import json
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from fvcore.nn import FlopCountAnalysis
from copy import deepcopy

from pyPrune.models.Vgg16 import VGG16
from pyPrune.utils import *
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
from plots import plot_accuracy_loss_curve, plot_results

# =====================================================
# === Safe JSON Merging (HPC Compatible)
# =====================================================
def safe_update_metrics_json(model_root, exp_name, new_data, base_dir="./runs/metrics"):
    """
    Safely merges new experiment data into model-level JSON (HPC-safe).
    Saved as: <base_dir>/<model_root>_metrics.json
    """
    os.makedirs(base_dir, exist_ok=True)
    json_path = os.path.join(base_dir, f"{model_root}_metrics.json")

    try:
        # Load if exists
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing = json.load(f)
        else:
            existing = {}

        # Ensure the structure is a dict
        if not isinstance(existing, dict):
            print(f"[!] Warning: Existing JSON is not a dict. Converting to empty dict.")
            existing = {}

        # Merge experiment
        existing[exp_name] = new_data

        # Atomic write
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(existing, f, indent=4)
        os.replace(tmp_path, json_path)

        print(f"[✓] Saved metrics for '{exp_name}' → {json_path}")
        return json_path

    except Exception as e:
        print(f"[!] Failed to update metrics JSON: {e}")
        return None

# =====================================================
# === Core Experiment Function ===
# =====================================================
def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

    print(f"[•] Starting experiment '{exp_name}' in workflow '{workflow}'")

    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    ckpt_path = os.path.join(
        ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs)
    )
    model.to(device)
    describe_model(model, loader=train_loader, device=device)

    # Train / Load metrics
    data = None
    model_root = f"{model.__class__.__name__}_{train_loader.dataset.__class__.__name__}"
    json_path = os.path.join(metrics_dir, f"{model_root}_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            if not isinstance(all_metrics, dict):
                print(f"[!] Warning: Loaded metrics JSON is not a dict. Skipping preloaded metrics.")
                all_metrics = {}
            exp_group = all_metrics.get(model_root, {})
            if isinstance(exp_group, dict) and exp_name in exp_group:
                exp_data = exp_group[exp_name]
                if isinstance(exp_data, dict):
                    print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                    data = exp_data
                    plot_accuracy_loss_curve(data.get('accuracies', []), data.get('losses', []),
                                             workflow, exp_name, save_dir=plots_dir)
                else:
                    print(f"[!] Warning: Experiment '{exp_name}' data is not a dict. Skipping preloaded metrics.")

    if data is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[•] Using device: {device}")
        data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)

    # Save checkpoint
    torch.save({'model': model.state_dict()}, ckpt_path)

    # Benchmark core metrics
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)
    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": data.get("accuracies", [0])[-1] if data.get("accuracies") else 0,
    })

    # Run diagnostics
    diagnostics = run_full_diagnostics(
        model, data_shape, {exp_name: data}, plots_dir, exp_name,
        collapse_range=collapse_range, device=device
    )
    data["diagnostics"] = diagnostics

    # Save metrics safely
    safe_update_metrics_json(model_root, exp_name, data, base_dir=metrics_dir)

    # === Cross-experiment unified plots ===
    plot_memory_per_layer_across_experiments(metrics_dir, plots_dir, title=f"Per-Layer Diagnostics Across {workflow} Experiments")
    plot_unified_metrics(metrics_dir, plots_dir, workflow)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
    return data

# =====================================================
# === Diagnostics and Plot Functions ===
# =====================================================
def run_full_diagnostics(model, input_shape, metrics_dict, save_dir, exp_name, collapse_range=None, device="cuda"):
    print(f"[•] Running diagnostics for {exp_name}...")
    print(f"[•] Save directory: {save_dir}")
    print(f"[•] Input shape: {input_shape}")
    print(f"[•] metrics_dict keys: {list(metrics_dict.keys())}")
    print(f"[•] Device: {device}")
    print(f"[•] Collapse range: {collapse_range}")
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()

    # Ensure input tensor is 4D
    if len(input_shape) == 2:
        input_tensor = torch.randn((1, 3, *input_shape), device=device)
    elif len(input_shape) == 3:
        input_tensor = torch.randn((1, *input_shape), device=device)
    else:
        input_tensor = torch.randn(input_shape, device=device)

    diagnostics = {}

    # Per-layer parameters and FLOPs
    try:
        df_params = analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
        diagnostics["per_layer_params_flops"] = df_params.to_dict(orient="records") if hasattr(df_params, "to_dict") else df_params
    except Exception as e:
        print(f"[!] Params/FLOPs analysis error: {e}")
        diagnostics["per_layer_params_flops"] = []

    # Activation sizes
    try:
        df_act = analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
        diagnostics["activation_sizes"] = df_act.to_dict(orient="records") if hasattr(df_act, "to_dict") else df_act
    except Exception as e:
        print(f"[!] Activation analysis error: {e}")
        diagnostics["activation_sizes"] = []

    # Memory decomposition
    try:
        mem = memory_decomposition(model, input_tensor, save_dir, exp_name)
        diagnostics["memory_decomposition"] = mem if isinstance(mem, dict) else {"memory": mem}
    except Exception as e:
        print(f"[!] Memory decomposition error: {e}")
        diagnostics["memory_decomposition"] = {}

    # Plots
    for func in [plot_flops_vs_latency, analyze_collapse_effects, plot_delta_accuracy_vs_params,
                 plot_flops_vs_memory, plot_accuracy_vs_memory, plot_heatmap, plot_stage_collapse_cost_curve]:
        try:
            func(metrics_dict, save_dir, exp_name)
        except Exception as e:
            print(f"[!] {func.__name__} error: {e}")

    print(f"[✓] Diagnostics complete for {exp_name}")
    return diagnostics

# =====================================================
# === Cross-Experiment Comparison (Extended)
# =====================================================
def plot_memory_per_layer_across_experiments(metrics_dir, save_dir, title="Per-Layer Diagnostics Across Experiments"):
    print("[DEBUG] Generating extended cross-experiment per-layer diagnostics plot...")
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))
    all_params, all_activations, all_memory = [], [], []

    for path in json_paths:
        with open(path, "r") as f:
            experiments = json.load(f)
            if not isinstance(experiments, dict):
                print(f"[!] Warning: JSON at {path} is not a dict. Skipping.")
                continue
            for exp_group in experiments.values():
                if not isinstance(exp_group, dict):
                    continue
                for exp_name, exp_data in exp_group.items():
                    if not isinstance(exp_data, dict):
                        print(f"[!] Warning: Experiment '{exp_name}' data is not a dict. Skipping diagnostics.")
                        continue
                    diag = exp_data.get("diagnostics", {})

                    # Params
                    for entry in diag.get("per_layer_params_flops", []):
                        all_params.append({"experiment": exp_name, "layer": entry.get("layer", ""),
                                           "params": entry.get("params", 0)})

                    # Activations
                    for entry in diag.get("activation_sizes", []):
                        all_activations.append({"experiment": exp_name, "layer": entry.get("layer", ""),
                                                "activation_elements": entry.get("activation_elements", 0)})

                    # Memory
                    if "memory_decomposition" in diag:
                        parts = diag["memory_decomposition"]
                        for cat in ["Params_MB", "Activations+Temps_MB", "Peak_MB"]:
                            all_memory.append({"experiment": exp_name, "category": cat, "value": parts.get(cat, 0)})

    if not (all_params or all_activations or all_memory):
        print("[!] No diagnostics found in JSON files.")
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle(title, fontsize=16)

    # Parameters per layer
    if all_params:
        df_p = pd.DataFrame(all_params)
        sns.barplot(data=df_p, x="layer", y="params", hue="experiment", ax=axs[0])
        axs[0].set_yscale("log")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45, ha="right")
        axs[0].set_ylabel("Params (log scale)")
        axs[0].grid(True, axis="y", linestyle="--", alpha=0.7)

    # Activation sizes per layer
    if all_activations:
        df_a = pd.DataFrame(all_activations)
        sns.barplot(data=df_a, x="layer", y="activation_elements", hue="experiment", ax=axs[1])
        axs[1].set_yscale("log")
        axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45, ha="right")
        axs[1].set_ylabel("Activation Elements (log scale)")
        axs[1].grid(True, axis="y", linestyle="--", alpha=0.7)

    # Memory decomposition
    if all_memory:
        df_m = pd.DataFrame(all_memory)
        sns.barplot(data=df_m, x="category", y="value", hue="experiment", ax=axs[2])
        axs[2].set_ylabel("Memory (MB)")
        axs[2].grid(True, axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout(rect=[0,0,0.85,0.95])
    save_path = os.path.join(save_dir, "cross_experiment_per_layer_diagnostics.svg")
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved extended per-layer diagnostics plot: {save_path}")

# =====================================================
# === Unified Metrics Plotting (Safe) ===
# =====================================================
def plot_unified_metrics(metrics_dir, save_dir, workflow):
    os.makedirs(save_dir, exist_ok=True)
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))
    if not json_paths:
        return

    all_data = []
    for path in json_paths:
        with open(path, "r") as f:
            content = json.load(f)
            if not isinstance(content, dict):
                continue
            for exp_group in content.values():
                if not isinstance(exp_group, dict):
                    continue
                for name, m in exp_group.items():
                    if not isinstance(m, dict):
                        print(f"[!] Warning: Experiment '{name}' data is not a dict. Skipping.")
                        continue
                    all_data.append({
                        "Experiment": name,
                        "Params": m.get("param_count", 0),
                        "Accuracy": m.get("final_accuracy", 0),
                        "FLOPs": m.get("flops", 0),
                        "Inference Time": m.get("inference_time", 0),
                        "Memory": m.get("total_size_mb", 0)
                    })
    df = pd.DataFrame(all_data)
    if df.empty:
        return

    # === Accuracy vs Params ===
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="Params", y="Accuracy", hue="Experiment", style="Experiment", s=100)
    plt.xscale("log")
    plt.xlabel("Parameters (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Parameters — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_accuracy_vs_params.svg"))
    plt.close()

    # === FLOPs vs Memory ===
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="FLOPs", y="Memory", hue="Experiment", style="Experiment", s=100)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FLOPs (log)")
    plt.ylabel("Memory (MB, log)")
    plt.title(f"FLOPs vs Memory — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_flops_vs_memory.svg"))
    plt.close()

    # === Accuracy vs Memory ===
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="Memory", y="Accuracy", hue="Experiment", style="Experiment", s=100)
    plt.xlabel("Memory (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Memory — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_accuracy_vs_memory.svg"))
    plt.close()

    print(f"[✓] Saved unified metrics plots for workflow '{workflow}'")
