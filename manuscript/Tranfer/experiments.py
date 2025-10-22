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
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            if not is_dict_like(all_metrics):
                print(f"[!] Warning: metrics JSON {json_path} malformed (not dict). Ignoring preloaded metrics.")
                all_metrics = {}
            exp_group = all_metrics.get(model_root, all_metrics) if is_dict_like(all_metrics) else {}
            # exp_group may be dict mapping exp_name->data
            if is_dict_like(exp_group) and exp_name in exp_group and is_dict_like(exp_group[exp_name]):
                print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                data = exp_group[exp_name]
                plot_accuracy_loss_curve(data.get('accuracies', []), data.get('losses', []), workflow, exp_name, save_dir=plots_dir)
            else:
                # sometimes older files stored experiments directly under root; try fallback
                if is_dict_like(all_metrics) and exp_name in all_metrics and is_dict_like(all_metrics[exp_name]):
                    data = all_metrics[exp_name]
                    plot_accuracy_loss_curve(data.get('accuracies', []), data.get('losses', []), workflow, exp_name, save_dir=plots_dir)

    if data is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"[•] Using device: {device}")
        data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)

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
    diagnostics = run_full_diagnostics(model, data_shape, {exp_name: data}, plots_dir, exp_name,
                                       collapse_range=collapse_range, device=device)
    data["diagnostics"] = diagnostics
    # Save metrics
    safe_update_metrics_json(model_root, exp_name, data, base_dir=metrics_dir)

    # Cross-experiment plots
    plot_memory_per_layer_across_experiments(glob.glob(os.path.join(metrics_dir, "*.json")), plots_dir, workflow)
    plot_unified_metrics(metrics_dir, plots_dir, workflow)

    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    # def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    plot_results(data, workflow, exp_name, plots_dir,filename=f"{workflow}_{exp_name}_results.svg",dataset=test_loader.dataset.__class__.__name__, infer_times=[data['inference_time']], mem_usages=[data['diagnostics']['total_memory_usage_mb']], flops=[data['flops']], total_sizes=[data['total_size_mb']])
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

