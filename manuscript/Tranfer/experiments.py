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
def safe_update_metrics_json(workflow_path, exp_name, new_data, base_dir="./runs/metrics"):
    """
    Safely merges new metrics into existing JSON.
    Prevents overwriting when running in parallel (HPC).
    """
    json_path = os.path.join(base_dir, f"{workflow_path}_metrics.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing = json.load(f)
        else:
            existing = {}

        # auto-detect model root
        model_root = list(existing.keys())[0] if existing else workflow_path.split("/")[-1]
        if model_root not in existing:
            existing[model_root] = {}

        existing[model_root][exp_name] = new_data
        with open(json_path, "w") as f:
            json.dump(existing, f, indent=4)
        print(f"[✓] Safely merged metrics for '{exp_name}' → {json_path}")

    except Exception as e:
        print(f"[!] Failed to safely merge metrics JSON: {e}")

# =====================================================
# === Core Experiment Function ===
# =====================================================
def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

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

    # === Load existing JSONs if any ===
    glob_path = os.path.join(metrics_dir, f"{workflow}/*metrics.json")
    json_paths = glob.glob(glob_path)
    data = None
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            first_key = list(all_metrics.keys())[0]
            if exp_name in all_metrics.get(first_key, {}):
                print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                data = all_metrics[first_key][exp_name]
                plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # === Train if needed ===
    if data is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[•] Using device: {device}")
        print(f"[•] Training model: {exp_name}")
        data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)
    else:
        print(f"[✓] Skipping training for '{exp_name}' as results already exist.")

    torch.save({'model': model.state_dict()}, ckpt_path)
    plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # === Benchmark core metrics ===
    print(f"[DEBUG] Running benchmark on {exp_name}")
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)

    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
    })

    # === Diagnostics ===
    print(f"[DEBUG] Running full diagnostics for {exp_name}")
    diagnostics = run_full_diagnostics(
        model, data_shape, {exp_name: data}, plots_dir, exp_name,
        collapse_range=collapse_range, device=device
    )
    data["diagnostics"] = diagnostics

    # === Save metrics safely ===
    safe_update_metrics_json(
        f"{workflow}/{model.__class__.__name__}_postcomp_{post_compress_epochs}",
        exp_name,
        data,
        base_dir=metrics_dir,
    )

    # === Comparison Plot ===
    glob_path = os.path.join(metrics_dir, f"{workflow}/*metrics.json")
    json_paths = glob.glob(glob_path)
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            all_metrics = all_metrics[list(all_metrics.keys())[0]]

            params, accs, names, infer_times, mem_usages, flops_list = [], [], [], [], [], []
            for name, metrics in all_metrics.items():
                names.append(name)
                params.append(metrics.get("param_count", 0))
                accs.append(metrics.get("final_accuracy", 0))
                infer_times.append(metrics.get("inference_time", 0))
                mem_usages.append(metrics.get("total_size_mb", 0))
                flops_list.append(metrics.get("flops", 0))

            save_path_plot = json_path.replace("metrics", "plots").replace("json", "svg")
            plot_results(
                params, accs, names, f"{workflow} Experiments", save_path_plot,
                dataset=workflow, infer_times=infer_times, mem_usages=mem_usages, flops=flops_list
            )
            print(f"[✓] Saved comparison plot to {save_path_plot}")

    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)
    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
    return data


# =====================================================
# === Diagnostics and Plots (Updated)
# =====================================================
def run_full_diagnostics(model, input_shape, metrics_dict, save_dir, exp_name, collapse_range=None, device="cuda"):
    print(f"[•] Running diagnostics for {exp_name}...")
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    model.eval()
    input_tensor = torch.randn(input_shape, device=device)

    diagnostics = {}

    try:
        print("[DEBUG] Running per-layer params/FLOPs analysis...")
        df_params = analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
        diagnostics["per_layer_params_flops"] = df_params.to_dict(orient="records")
    except Exception as e:
        print(f"[!] Error in per-layer params/FLOPs analysis: {e}")

    try:
        print("[DEBUG] Running activation size analysis...")
        df_act = analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
        diagnostics["activation_sizes"] = df_act.to_dict(orient="records")
    except Exception as e:
        print(f"[!] Error in activation sizes analysis: {e}")

    try:
        if torch.cuda.is_available():
            print("[DEBUG] Running memory decomposition...")
            parts = memory_decomposition(model, input_tensor, save_dir, exp_name)
            diagnostics["memory_decomposition"] = parts
    except Exception as e:
        print(f"[!] Error in memory decomposition: {e}")

    try:
        print("[DEBUG] Plotting FLOPs vs Latency...")
        plot_flops_vs_latency(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in FLOPs vs Latency plot: {e}")

    try:
        print("[DEBUG] Analyzing collapse effects...")
        analyze_collapse_effects(model, collapse_range, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in collapse effects analysis: {e}")

    print(f"[✓] Diagnostics complete for {exp_name}.")
    return diagnostics



# =====================================================
# === Cross-Experiment Comparison Plot (Extended)
# =====================================================
def plot_memory_per_layer_across_experiments(metrics_dir, save_dir, title="Per-Layer Diagnostics Across Experiments"):
    """
    Compare per-layer Params, Activation Sizes, and Memory Decomposition
    across multiple experiments (JF, Kevin, etc.).
    """
    print("[DEBUG] Generating extended cross-experiment diagnostics plot...")

    json_paths = glob.glob(os.path.join(metrics_dir, "*/*metrics.json"))
    all_params, all_activations, all_memory = [], [], []

    for path in json_paths:
        with open(path, "r") as f:
            experiments = json.load(f)
            for exp_group in experiments.values():
                for exp_name, exp_data in exp_group.items():
                    diag = exp_data.get("diagnostics", {})

                    # --- Params/FLOPs ---
                    if "per_layer_params_flops" in diag:
                        for entry in diag["per_layer_params_flops"]:
                            all_params.append({
                                "experiment": exp_name,
                                "layer": entry["layer"],
                                "params": entry.get("params", 0),
                                "flops": entry.get("flops", 0),
                            })

                    # --- Activation sizes ---
                    if "activation_sizes" in diag:
                        for entry in diag["activation_sizes"]:
                            all_activations.append({
                                "experiment": exp_name,
                                "layer": entry["layer"],
                                "activation_MB": entry.get("activation_MB", 0),
                            })

                    # --- Memory decomposition ---
                    if "memory_decomposition" in diag:
                        mem_dict = diag["memory_decomposition"]
                        for layer_name, values in mem_dict.items():
                            all_memory.append({
                                "experiment": exp_name,
                                "layer": layer_name,
                                "forward_MB": values.get("forward_MB", 0),
                                "backward_MB": values.get("backward_MB", 0),
                                "total_MB": values.get("total_MB", 0),
                            })

    if not all_params and not all_activations and not all_memory:
        print("[!] No diagnostics found in JSON files.")
        return

    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    fig.suptitle(title, fontsize=16)

    # --- Params per layer ---
    if all_params:
        df_params = pd.DataFrame(all_params)
        sns.barplot(data=df_params, x="layer", y="params", hue="experiment", ax=axs[0])
        axs[0].set_title("Parameters per Layer")
        axs[0].set_ylabel("Params")
        axs[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[0].grid(True)

    # --- Activation sizes ---
    if all_activations:
        df_act = pd.DataFrame(all_activations)
        sns.barplot(data=df_act, x="layer", y="activation_MB", hue="experiment", ax=axs[1])
        axs[1].set_title("Activation Size per Layer (MB)")
        axs[1].set_ylabel("Activation (MB)")
        axs[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[1].grid(True)

    # --- Memory decomposition ---
    if all_memory:
        df_mem = pd.DataFrame(all_memory)
        df_mem_melted = df_mem.melt(
            id_vars=["experiment", "layer"],
            value_vars=["forward_MB", "backward_MB", "total_MB"],
            var_name="memory_type",
            value_name="memory_MB",
        )
        sns.barplot(
            data=df_mem_melted, x="layer", y="memory_MB",
            hue="experiment", ax=axs[2]
        )
        axs[2].set_title("Memory Usage per Layer (Forward + Backward + Total)")
        axs[2].set_ylabel("Memory (MB)")
        axs[2].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        axs[2].grid(True)

    for ax in axs:
        ax.set_xlabel("Layer")
        ax.tick_params(axis="x", rotation=90)

    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    save_path = os.path.join(save_dir, "cross_experiment_per_layer_diagnostics.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Saved extended per-layer diagnostics plot: {save_path}")


# =====================================================
# === Experiment Entry Points (JF & Kevin)
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
    base_model.load_state_dict(torch.load(model_path_000, map_location="cpu")['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    if collapse_range:
        formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tmp_path = os.path.join(save_path, f"temp_model_kevin_{formatted_time}.pth")
        os.makedirs(save_path, exist_ok=True)
        torch.save({'model': base_model.state_dict()}, tmp_path)
        base_model = collapse_only(
            model_weights_1=tmp_path,
            compression_set=[collapse_range],
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
