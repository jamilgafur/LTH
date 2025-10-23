# experiment.py

# Standard libraries
import os
import glob
import json
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

# Third-party libraries
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis

# Local modules
from pyPrune.models.Vgg16 import VGG16
from pyPrune.utils import *
from plots import *
from diagnostic import *
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
from plots import plot_accuracy_loss_curve, plot_results


# -------------------------
# Safe JSON Merging
# -------------------------
def safe_update_metrics_json(model_root, exp_name, new_data, base_dir="./runs/metrics"):
    ensure_dir(base_dir)
    json_path = os.path.join(base_dir, f"{model_root}_metrics.json")
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                existing = json.load(f)
        else:
            existing = {}

        if not isinstance(existing, dict):
            print(f"[!] Warning: Existing JSON at {json_path} is not a dict. Replacing it.")
            existing = {}

        existing[exp_name] = new_data

        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(existing, f, indent=4)
        os.replace(tmp_path, json_path)
        print(f"[✓] Saved metrics for '{exp_name}' → {json_path}")
        return json_path
    except Exception as e:
        print(f"[!] Failed to update metrics JSON: {e}")
        return None

# -------------------------
# Core Experiment
# -------------------------
def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

    print(f"[•] Starting experiment '{exp_name}' in workflow '{workflow}'")
    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    ensure_dir(ckpt_dir)
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    ckpt_path = os.path.join(
        ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs)
    )
    model.to(device)
    describe_model(model, loader=train_loader, device=device)

    # Load existing metrics (if valid)
    data = None
    model_root = f"{model.__class__.__name__}_{train_loader.dataset.__class__.__name__}"
    json_path = os.path.join(metrics_dir, f"{model_root}_metrics.json")
    

    all_metrics = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            try:
                all_metrics = json.load(f)
            except Exception:
                print(f"[!] Warning: could not parse {json_path}, starting fresh.")
                all_metrics = {}

        if not is_dict_like(all_metrics):
            print(f"[!] Warning: metrics JSON {json_path} malformed (not dict). Ignoring preloaded metrics.")
            all_metrics = {}

        exp_group = all_metrics.get(model_root, all_metrics) if is_dict_like(all_metrics) else {}
        # exp_group may be dict mapping exp_name->data
        if is_dict_like(exp_group) and exp_name in exp_group and is_dict_like(exp_group[exp_name]):
            data = exp_group[exp_name]
            print(f"[✓] Found existing results for '{exp_name}' in {json_path}.")
            plot_accuracy_loss_curve(
                data.get('accuracies', []),
                data.get('losses', []),
                workflow,
                exp_name,
                save_dir=plots_dir
            )

            # ✅ Check if diagnostics exist, if not, compute and update JSON
            if "diagnostics" not in data or not data["diagnostics"]:
                print(f"[•] Diagnostics missing for '{exp_name}', running now...")
                diagnostics = run_full_diagnostics(
                    model, data_shape, {exp_name: data}, plots_dir, exp_name,
                    collapse_range=collapse_range, device=device
                )
                data["diagnostics"] = diagnostics
                # Update the JSON with new diagnostics
                safe_update_metrics_json(model_root, f"{exp_name}_{workflow}", data, base_dir=metrics_dir)
                print(f"[✓] Diagnostics added for '{exp_name}'.")
            else:
                print(f"[✓] Diagnostics already exist for '{exp_name}' — skipping.")

    # If experiment data not found, run new training
    if data is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[•] Using device: {device}")
        if not torch.cuda.is_available():
            print("[!] Warning: CUDA not available.")
            quit()

            
        data = train_and_evaluate(
            model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs
        )

        torch.save({'model': model.state_dict()}, ckpt_path)

        # Benchmark & attach core metrics
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

        # Save metrics JSON
        safe_update_metrics_json(model_root, f"{exp_name}_{workflow}", data, base_dir=metrics_dir)

    with open(json_path, "r") as f:
        all_metrics = json.load(f)

        params = []
        accs = []
        names = []
        infer_times = []
        mem_usages = []
        flops = []
        total_sizes = []  # List to store total size for plotting

        # Iterate through each model's metrics to prepare data for plotting
        for name, metrics in all_metrics.items():
            names.append(name)
            params.append(metrics.get("param_count", 0))
            accs.append(metrics.get("final_accuracy", 0))
            infer_times.append(metrics.get("inference_time", 0))
            mem_usages.append(metrics.get("total_size_mb", 0))
            flops.append(metrics.get("flops", 0))  # Collect FLOPs

        # Save comparison plot
        save_path = json_path.replace("metrics", "plots").replace("json", "svg")
        
        plot_results(params, accs, names, f"{workflow} Experiments", save_path,
                    dataset=workflow, infer_times=infer_times, mem_usages=mem_usages, flops=flops, total_sizes=total_sizes)
        print(f"Saved comparison plot to {save_path}")

    norm_metrics = normalize_metrics(all_metrics)

    # Plots (each function is robust to input)
    for func in [plot_flops_vs_latency, analyze_collapse_effects, plot_delta_accuracy_vs_params,
                 plot_flops_vs_memory, plot_accuracy_vs_memory, plot_heatmap, plot_stage_collapse_cost_curve]:
        try:
            if func.__name__ == "analyze_collapse_effects":
                try:
                    func(model, collapse_range, plots_dir, exp_name)
                except TypeError:
                    func(norm_metrics, plots_dir, exp_name)
            else:
                func(norm_metrics, plots_dir, exp_name)
        except Exception as e:
            print(f"[!] {func.__name__} error: {e}")

    # Cross-experiment plots
    plot_memory_per_layer_across_experiments(glob.glob(os.path.join(metrics_dir, "*.json")), plots_dir, workflow)
    plot_unified_metrics(metrics_dir, plots_dir, workflow)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
    return data

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

# -------------------------

