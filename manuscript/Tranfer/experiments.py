# experiment.py
import os
import torch
import json
import glob
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fvcore.nn import FlopCountAnalysis

from pyPrune.models.Vgg16 import VGG16
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
from plots import plot_accuracy_loss_curve, plot_results
from pyPrune.utils import *
from copy import deepcopy


# =====================================================
# === Core Experiment Logic ===
# =====================================================

def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

    # --- Directory Setup ---
    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # --- Save checkpoint path ---
    ckpt_path = os.path.join(ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs))
    model.to(device)

    describe_model(model, loader=train_loader, device=device)

    # --- Load existing metrics (if exist) ---
    glob_path = os.path.join(metrics_dir, f"{workflow}/*metrics.json")
    json_paths = glob.glob(glob_path)
    data = None
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            if exp_name in all_metrics.get(list(all_metrics.keys())[0], {}):
                print(f"[✓] Found existing results for '{exp_name}' in {json_path}, skipping training.")
                data = all_metrics[list(all_metrics.keys())[0]][exp_name]
                plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # --- Train if needed ---
    if data is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if torch.cuda.is_available():
            print(f"[•] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("[•] No GPU found")
        
        print(f"[•] Training model: {exp_name}")
        data = train_and_evaluate(model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs)
    else:
        print(f"[✓] Skipping training for '{exp_name}' as results already exist.")

    # --- Save model checkpoint ---
    torch.save({'model': model.state_dict()}, ckpt_path)

    # --- Accuracy/Loss curve ---
    plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # --- Benchmarking ---
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)

    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
    })

    # --- Save metrics ---
    save_metrics_json(f"{workflow}/{model.__class__.__name__}_postcomp_{post_compress_epochs}", exp_name, data, base_dir=metrics_dir)

    # --- Comparison Plot ---
    glob_path = os.path.join(metrics_dir, f"{workflow}/*metrics.json")
    json_paths = glob.glob(glob_path)
    if json_paths:
        json_path = json_paths[0]
        with open(json_path, "r") as f:
            all_metrics = json.load(f)
            all_metrics = all_metrics[list(all_metrics.keys())[0]]

            params, accs, names, infer_times, mem_usages, flops_list, total_sizes = [], [], [], [], [], [], []

            for name, metrics in all_metrics.items():
                names.append(name)
                params.append(metrics.get("param_count", 0))
                accs.append(metrics.get("final_accuracy", 0))
                infer_times.append(metrics.get("inference_time", 0))
                mem_usages.append(metrics.get("memory_usage_mb", 0))
                flops_list.append(metrics.get("flops", 0))
                total_sizes.append(metrics.get("total_size_mb", 0))

            save_path = json_path.replace("metrics", "plots").replace("json", "svg")
            plot_results(params, accs, names, f"{workflow} Experiments", save_path,
                         dataset=workflow, infer_times=infer_times, mem_usages=mem_usages, flops=flops_list, total_sizes=total_sizes)
            print(f"Saved comparison plot to {save_path}")

    # --- Run Full Diagnostics (new section) ---
    dummy_input = torch.randn(data_shape).to(device)
    run_full_diagnostics(model, dummy_input, {exp_name: data}, plots_dir, exp_name)

    # --- Save Final Model ---
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    del model
    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved. full data: {data}")
    return data


def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                      model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                      post_compress_epochs=False):
    
    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    print(f"\nRunning JF experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_097, map_location=torch.device('cpu'))['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    if collapse_range is not None:
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
    print(f"\nRunning Kevin experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_000, map_location="cpu")['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")
    
    if collapse_range is not None:
        from datetime import datetime
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


# =====================================================
# === Diagnostics & Visualization Utilities ===
# =====================================================

def analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name):
    model.eval()
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()
    
    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            layer_data.append({"layer": name, "params": params, "flops": per_module_flops.get(name, 0)})
    
    df = pd.DataFrame(layer_data)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_params_flops.csv"), index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    df.plot(x="layer", y="params", kind="bar", ax=axes[0], color="skyblue", legend=False)
    axes[0].set_title("Parameters per Layer")
    df.plot(x="layer", y="flops", kind="bar", ax=axes[1], color="salmon", legend=False)
    axes[1].set_title("FLOPs per Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_params_flops_layers.png"))
    plt.close(fig)
    return df


def analyze_activation_sizes(model, input_tensor, save_dir, exp_name):
    activations = {}
    def hook(name):
        def fn(_, __, output):
            if isinstance(output, torch.Tensor):
                activations[name] = np.prod(output.shape)
        return fn

    hooks = [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    for h in hooks: h.remove()

    df = pd.DataFrame(list(activations.items()), columns=["layer", "activation_size"])
    df.to_csv(os.path.join(save_dir, f"{exp_name}_activation_sizes.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="layer", y="activation_size", color="lightgreen")
    plt.xticks(rotation=90)
    plt.title("Activation Size per Layer (elements)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_activation_heatmap.png"))
    plt.close()
    return df


def plot_flops_vs_latency(metrics_dict, save_dir, exp_name):
    flops = [v["flops"] for v in metrics_dict.values()]
    times = [v["inference_time"] for v in metrics_dict.values()]
    names = list(metrics_dict.keys())
    plt.figure(figsize=(8,6))
    plt.scatter(flops, times, color="orange")
    for i, txt in enumerate(names):
        plt.annotate(txt, (flops[i], times[i]))
    plt.xlabel("FLOPs")
    plt.ylabel("Inference Time (s)")
    plt.title(f"FLOPs vs Inference Time — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_flops_vs_latency.png"))
    plt.close()


def memory_decomposition(model, input_tensor, save_dir, exp_name):
    param_mem = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()
    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    activation_mem = max(peak_mem - param_mem, 0)
    parts = {"Params": param_mem, "Activations+Temps": activation_mem}
    plt.figure(figsize=(6,6))
    plt.bar(parts.keys(), parts.values(), color=["steelblue", "salmon"])
    plt.title(f"Memory Breakdown — {exp_name}")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_memory_breakdown.png"))
    plt.close()
    return parts


def generate_layer_summary_table(model, input_tensor, save_dir, exp_name):
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()

    rows = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            shape = getattr(module, 'weight', None)
            shape = list(shape.shape) if shape is not None else None
            rows.append({"layer": name, "param_count": params, "flops": per_module_flops.get(name, 0), "weight_shape": shape})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_summary.csv"), index=False)
    return df


def run_full_diagnostics(model, input_tensor, metrics_dict, save_dir, exp_name):
    print(f"[•] Running diagnostics for {exp_name}...")
    os.makedirs(save_dir, exist_ok=True)
    df_params_flops = analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
    df_activations = analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
    memory_decomposition(model, input_tensor, save_dir, exp_name)
    generate_layer_summary_table(model, input_tensor, save_dir, exp_name)
    plot_flops_vs_latency(metrics_dict, save_dir, exp_name)
    print(f"[✓] Diagnostics complete for {exp_name}. Results saved in {save_dir}.")
