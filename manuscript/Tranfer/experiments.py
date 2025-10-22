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
            if exp_name in all_metrics.get(model_root, {}):
                print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                data = all_metrics[model_root][exp_name]
                plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

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
        "final_accuracy": data["accuracies"][-1] if data.get("accuracies") else 0,
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
def plot_unified_metrics(metrics_dir, save_dir, workflow):
    """Generates unified comparison plots for all experiments in a workflow."""
    os.makedirs(save_dir, exist_ok=True)
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))
    if not json_paths: return

    all_data = []
    for path in json_paths:
        with open(path, "r") as f:
            for exp_group in json.load(f).values():
                for name, m in exp_group.items():
                    all_data.append({
                        "Experiment": name,
                        "Params": m.get("param_count", 0),
                        "Accuracy": m.get("final_accuracy", 0),
                        "FLOPs": m.get("flops", 0),
                        "Inference Time": m.get("inference_time", 0),
                        "Memory": m.get("total_size_mb", 0)
                    })
    df = pd.DataFrame(all_data)
    if df.empty: return

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

def run_full_diagnostics(model, input_shape, metrics_dict, save_dir, exp_name, collapse_range=None, device="cuda"):
    print(f"[•] Running diagnostics for {exp_name}...")
    print(f"[•] Save directory: {save_dir}")
    print(f"[•] Input shape: {input_shape}")
    print(f"[•] metrics_dict keys: {list(metrics_dict.keys())}")
    print(f"[•] Device: {device}")
    print(f"[•] Collapse range: {collapse_range}")
    print(f"[•] Model summary: {describe_model(model)}")
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
    try:
        plot_flops_vs_latency(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] FLOPs vs latency plot error: {e}")

    try:
        analyze_collapse_effects(model, collapse_range, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Collapse effects error: {e}")

    # Additional plots
    for func in [plot_delta_accuracy_vs_params, plot_flops_vs_memory, plot_accuracy_vs_memory,
                 plot_heatmap, plot_stage_collapse_cost_curve]:
        try:
            func(metrics_dict, save_dir, exp_name)
        except Exception as e:
            print(f"[!] {func.__name__} error: {e}")

    print(f"[✓] Diagnostics complete for {exp_name}")
    return diagnostics


def analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name):
    model.eval()

    # Debug the input tensor shape
    debug_tensor_shape(input_tensor, "Input Tensor")

    # Ensure input tensor is 4D (batch_size, channels, height, width)
    if len(input_tensor.shape) == 3:
        print("Input tensor is missing batch dimension, adding batch size of 1.")
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    debug_tensor_shape(input_tensor, "Input Tensor (After Batch Dimension Check)")

    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()

    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Skip non-leaf nodes (i.e., not directly trained modules)
            params = sum(p.numel() for p in module.parameters())
            flops_for_layer = per_module_flops.get(name, 0)
            layer_data.append({"layer": name, "params": params, "flops": flops_for_layer})
            print(f"Layer: {name}, Params: {params}, FLOPs: {flops_for_layer}")

    # Create DataFrame and save it
    df = pd.DataFrame(layer_data)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_params_flops.csv"), index=False)

    # Plot params and FLOPs per layer
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    df.plot(x="layer", y="params", kind="bar", ax=axes[0], color="skyblue", legend=False)
    axes[0].set_title("Parameters per Layer")
    axes[0].tick_params(axis='x', rotation=90)
    df.plot(x="layer", y="flops", kind="bar", ax=axes[1], color="salmon", legend=False)
    axes[1].set_title("FLOPs per Layer")
    axes[1].tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_params_flops_layers.svg"))
    plt.close(fig)

    return df

def analyze_activation_sizes(model, input_tensor, save_dir, exp_name):
    activations = {}

    def hook(name):
        def fn(_, __, output):
            if isinstance(output, torch.Tensor):
                activations[name] = output.numel()
        return fn

    # Debug the input tensor shape
    debug_tensor_shape(input_tensor, "Input Tensor (For Activation Analysis)")

    # Ensure input tensor is 4D
    if len(input_tensor.shape) == 3:
        print("Input tensor is missing batch dimension, adding batch size of 1.")
        input_tensor = input_tensor.unsqueeze(0)

    debug_tensor_shape(input_tensor, "Input Tensor (After Batch Dimension Check)")

    # Register hooks to track activations
    hooks = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remove hooks after analysis
    for h in hooks:
        h.remove()

    # Create DataFrame of activations and save it
    df = pd.DataFrame(list(activations.items()), columns=["layer", "activation_elements"])
    df.to_csv(os.path.join(save_dir, f"{exp_name}_activation_sizes.csv"), index=False)

    # Plot activation sizes
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="layer", y="activation_elements", color="lightgreen")
    plt.xticks(rotation=90)
    plt.title("Activation Size per Layer (# elements)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_activation_heatmap.svg"))
    plt.close()

    return df

def memory_decomposition(model, input_tensor, save_dir, exp_name):
    # Debug input tensor shape
    debug_tensor_shape(input_tensor, "Input Tensor (For Memory Decomposition)")

    # Ensure input tensor is 4D
    if len(input_tensor.shape) == 3:
        print("Input tensor is missing batch dimension, adding batch size of 1.")
        input_tensor = input_tensor.unsqueeze(0)

    debug_tensor_shape(input_tensor, "Input Tensor (After Batch Dimension Check)")

    param_mem = sum(p.numel() for p in model.parameters()) * 4 / 1e6  # MB (assuming FP32)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        _ = model(input_tensor)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None

    activation_mem = max(peak_mem - param_mem, 0) if peak_mem is not None else None
    parts = {"Params_MB": param_mem, "Activations+Temps_MB": activation_mem, "Peak_MB": peak_mem}
    
    # Debug memory decomposition details
    print(f"Memory Decomposition: Params: {param_mem}MB, Activations: {activation_mem}MB, Peak: {peak_mem}MB")

    # Plot memory usage
    plt.figure(figsize=(6, 6))
    plt.bar(parts.keys(), parts.values(), color=["steelblue", "salmon", "gold"])
    plt.title(f"Memory Breakdown — {exp_name}")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_memory_breakdown.svg"))
    plt.close()

    return parts

def plot_flops_vs_latency(metrics_dict, save_dir, exp_name):
    # Extract FLOPs and latency data for the plot
    flops = [v["flops"] for v in metrics_dict.values()]
    times = [v["inference_time"] for v in metrics_dict.values()]
    names = list(metrics_dict.keys())

    # Debug the collected metrics
    print(f"FLOPs: {flops}")
    print(f"Latency Times: {times}")

    plt.figure(figsize=(8, 6))
    plt.scatter(flops, times, color="orange")
    for i, txt in enumerate(names):
        plt.annotate(txt, (flops[i], times[i]))
    plt.xlabel("FLOPs")
    plt.ylabel("Inference Time (s)")
    plt.title(f"FLOPs vs Inference Time — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_flops_vs_latency.svg"))
    plt.close()

def debug_tensor_shape(tensor, description="Tensor"):
    """ Helper function to debug tensor shapes. """
    if tensor is not None:
        print(f"{description} Shape: {tensor.shape}")
    else:
        print(f"{description} is None!")

def plot_delta_accuracy_vs_params(metrics_dict, save_dir, exp_name):
    base_name = list(metrics_dict.keys())[0]
    base_acc = metrics_dict[base_name]["final_accuracy"]
    base_params = metrics_dict[base_name]["param_count"]

    deltas = []
    for name, data in metrics_dict.items():
        d_acc = data["final_accuracy"] - base_acc
        d_params = (data["param_count"] - base_params) / base_params * 100
        deltas.append({"name": name, "ΔAcc": d_acc, "ΔParams(%)": d_params})

    df = pd.DataFrame(deltas)
    plt.figure(figsize=(8,6))
    plt.scatter(df["ΔParams(%)"], df["ΔAcc"], c="blue")
    for _, r in df.iterrows():
        plt.annotate(r["name"], (r["ΔParams(%)"], r["ΔAcc"]))
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Δ Parameters (%)")
    plt.ylabel("Δ Accuracy")
    plt.title(f"Compression Efficiency — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.svg"))
    plt.close()

def plot_flops_vs_memory(metrics_dict, save_dir, exp_name):
    flops = [v["flops"] for v in metrics_dict.values()]
    mems = [v.get("total_size_mb", 0) for v in metrics_dict.values()]
    names = list(metrics_dict.keys())
    plt.figure(figsize=(8,6))
    plt.scatter(flops, mems, color="purple")
    for i, n in enumerate(names):
        plt.annotate(n, (flops[i], mems[i]))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FLOPs (log)")
    plt.ylabel("Total Memory (MB, log)")
    plt.title(f"FLOPs vs Memory — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_flops_vs_memory.svg"))
    plt.close()

def plot_accuracy_vs_memory(metrics_dict, save_dir, exp_name):
    accs = [v["final_accuracy"] for v in metrics_dict.values()]
    mems = [v["total_size_mb"] for v in metrics_dict.values()]
    names = list(metrics_dict.keys())
    plt.figure(figsize=(8,6))
    plt.scatter(mems, accs, c="green")
    for i, n in enumerate(names):
        plt.annotate(n, (mems[i], accs[i]))
    plt.xlabel("Model Size (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Memory — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_acc_vs_memory.svg"))
    plt.close()

def plot_heatmap(metrics_dict, save_dir, exp_name):
    df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": v["final_accuracy"],
            "Params": v["param_count"],
            "FLOPs": v["flops"],
            "Inference Time": v["inference_time"],
            "Memory (MB)": v["total_size_mb"]
        }
        for name, v in metrics_dict.items()
    ])
    df_norm = df.set_index("Model").apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    plt.figure(figsize=(10,6))
    sns.heatmap(df_norm, annot=True, cmap="coolwarm")
    plt.title(f"Normalized Metrics Heatmap — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_metrics_heatmap.svg"))
    plt.close()

def plot_stage_collapse_cost_curve(metrics_dict, save_dir, exp_name):
    df = pd.DataFrame([
        {"Model": name, "Params": v["param_count"], "Time": v["inference_time"], "Accuracy": v["final_accuracy"]}
        for name, v in metrics_dict.items()
    ])
    df = df.sort_values("Model")
    plt.figure(figsize=(9,6))
    plt.plot(df["Model"], df["Params"], label="Parameters", marker="o")
    plt.plot(df["Model"], df["Time"], label="Inference Time", marker="s")
    plt.plot(df["Model"], df["Accuracy"], label="Accuracy", marker="^")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f"Stage Collapse Cost Curve — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_collapse_cost_curve.svg"))
    plt.close()

def predict_collapse_parameters(in_channels, out_channels, kernel_size, num_layers_collapsed):
    """Theoretical expected parameters after collapsing n layers."""
    original_params = num_layers_collapsed * (in_channels * out_channels * kernel_size * kernel_size + out_channels)
    collapsed_params = in_channels * out_channels * kernel_size * kernel_size + out_channels
    delta = collapsed_params - original_params
    return {
        "original": original_params,
        "collapsed": collapsed_params,
        "delta": delta
    }

def analyze_collapse_effects(model, collapse_range, save_dir, exp_name):
    if not collapse_range:
        return
    start_stage, end_stage = collapse_range
    stage_channels = [64, 128, 256, 512, 512, 4096]
    in_ch = stage_channels[start_stage - 1]
    out_ch = stage_channels[end_stage - 1]
    num_layers = (end_stage - start_stage + 1) * 3  # approximate

    pred = predict_collapse_parameters(in_ch, out_ch, 3, num_layers)
    observed_params = count_trainable_params(model)
    df = pd.DataFrame([{
        "stage_range": f"{start_stage}-{end_stage}",
        "predicted_params": pred["collapsed"],
        "original_est": pred["original"],
        "delta_predicted": pred["delta"],
        "observed_total": observed_params
    }])
    df.to_csv(os.path.join(save_dir, f"{exp_name}_collapse_prediction.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(["Original","Predicted Collapsed","Observed Total"],
            [pred["original"], pred["collapsed"], observed_params],
            color=["gray","orange","blue"])
    plt.ylabel("Parameter Count")
    plt.title(f"Collapse {start_stage}-{end_stage} Parameter Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_collapse_prediction.svg"))
    plt.close()

# =====================================================
# === Cross-Experiment Comparison (Extended)
# =====================================================
def plot_memory_per_layer_across_experiments(metrics_dir, save_dir, title="Per-Layer Diagnostics Across Experiments"):
    print("[DEBUG] Generating extended cross-experiment per-layer diagnostics plot...")
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))  # <-- Fixed glob
    all_params, all_activations, all_memory = [], [], []

    for path in json_paths:
        with open(path, "r") as f:
            experiments = json.load(f)
            for exp_group in experiments.values():
                for exp_name, exp_data in exp_group.items():
                    diag = exp_data.get("diagnostics", {})

                    # Params
                    for entry in diag.get("per_layer_params_flops", []):
                        all_params.append({"experiment": exp_name, "layer": entry["layer"], "params": entry.get("params", 0)})

                    # Activations
                    for entry in diag.get("activation_sizes", []):
                        all_activations.append({"experiment": exp_name, "layer": entry["layer"], "activation_elements": entry.get("activation_elements", 0)})

                    # Memory
                    if "memory_decomposition" in diag:
                        parts = diag["memory_decomposition"]
                        for cat in ["Params_MB","Activations+Temps_MB","Peak_MB"]:
                            all_memory.append({"experiment": exp_name, "category": cat, "value": parts.get(cat,0)})

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
# === Experiment Entry Points (JF & Kevin) ===
# =====================================================
def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                      model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                      post_compress_epochs=False):

    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_097, map_location='cpu')['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    if collapse_range:
        base_model = collapse_only(
            model_weights_1=model_path_097,
            compression_set=[collapse_range],
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=model_kwargs['one_batch'].shape,
            device=device
        )

    data = run_experiment(base_model, model_kwargs, train_loader, test_loader, device, epochs,
                          workflow="JF", exp_name=exp_name, data_shape=data_shape,
                          save_path=save_path, post_compress_epochs=post_compress_epochs)
    return base_model

def run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs,
                         model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                         post_compress_epochs=False):

    model_kwargs = model_kwargs or {}
    print("\n=== Running Kevin experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_000, map_location='cpu')['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    if collapse_range:
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tmp_path = os.path.join(save_path, f"temp_model_kevin_{formatted_time}.pth")
        os.makedirs(save_path, exist_ok=True)
        torch.save({'model': base_model.state_dict()}, tmp_path)
        base_model = collapse_only(
            model_weights_1=tmp_path,
            compression_set=[collapse_range],K)
            model_class=model_class,
            model_kwargs=model_kwargs,
            input_shape=model_kwargs['one_batch'].shape,
            device=device
        )
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    data = run_experiment(base_model, model_kwargs, train_loader, test_loader, device, epochs,
                          workflow="Kevin", exp_name=exp_name, data_shape=data_shape,
                          save_path=save_path, post_compress_epochs=post_compress_epochs)
    return base_model
