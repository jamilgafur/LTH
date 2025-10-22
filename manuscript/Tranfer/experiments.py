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
# === Core Experiment Function ===
# =====================================================
def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

    # Paths
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

    # Load existing metrics if available
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

    # Train if needed
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

    torch.save({'model': model.state_dict()}, ckpt_path)
    plot_accuracy_loss_curve(data['accuracies'], data['losses'], workflow, exp_name, save_dir=plots_dir)

    # Benchmarking
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)

    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": data["accuracies"][-1] if data["accuracies"] else 0,
    })

    # Save metrics
    save_metrics_json(
        f"{workflow}/{model.__class__.__name__}_postcomp_{post_compress_epochs}",
        exp_name,
        data,
        base_dir=metrics_dir,
    )

    # Comparison Plot
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
                mem_usages.append(metrics.get("total_size_mb", 0))
                flops_list.append(metrics.get("flops", 0))
                total_sizes.append(metrics.get("total_size_mb", 0))

            save_path_plot = json_path.replace("metrics", "plots").replace("json", "svg")
            plot_results(
                params, accs, names, f"{workflow} Experiments", save_path_plot,
                dataset=workflow, infer_times=infer_times, mem_usages=mem_usages, flops=flops_list, total_sizes=total_sizes
            )
            print(f"Saved comparison plot to {save_path_plot}")

    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    del model
    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
    return data


# =====================================================
# === Diagnostic Visualizations (Research Section) ===
# =====================================================

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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.png"))
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_flops_vs_memory.png"))
    plt.close()


def plot_stage_parameter_density(model, save_dir, exp_name):
    stage_params, stage_names = [], []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            params = sum(p.numel() for p in module.parameters())
            stage = name.split(".")[0]
            stage_params.append(params)
            stage_names.append(stage)
    df = pd.DataFrame({"stage": stage_names, "params": stage_params})
    df = df.groupby("stage").sum().reset_index()
    df["norm_params"] = df["params"] / df["params"].sum() * 100
    plt.figure(figsize=(10,5))
    sns.barplot(data=df, x="stage", y="norm_params", color="teal")
    plt.title(f"Parameter Density per Stage — {exp_name}")
    plt.ylabel("Share of Total Params (%)")
    plt.xlabel("Stage")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_stage_param_density.png"))
    plt.close()
    return df


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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_acc_vs_memory.png"))
    plt.close()


# === Advanced Visualizations ===

def plot_heatmap(metrics_dict, save_dir, exp_name):
    """Heatmap of normalized metrics."""
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_metrics_heatmap.png"))
    plt.close()


def plot_stage_collapse_cost_curve(metrics_dict, save_dir, exp_name):
    """Show cost of collapsing later stages."""
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_collapse_cost_curve.png"))
    plt.close()


# === Run All Diagnostics ===

def run_full_diagnostics(model, input_tensor, metrics_dict, save_dir, exp_name, collapse_range=None):
    print(f"[•] Running diagnostics for {exp_name}...")
    os.makedirs(save_dir, exist_ok=True)

    analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
    analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
    memory_decomposition(model, input_tensor, save_dir, exp_name)
    plot_flops_vs_latency(metrics_dict, save_dir, exp_name)
    analyze_collapse_effects(model, collapse_range, save_dir, exp_name)

    # Extended diagnostics
    plot_delta_accuracy_vs_params(metrics_dict, save_dir, exp_name)
    plot_flops_vs_memory(metrics_dict, save_dir, exp_name)
    plot_stage_parameter_density(model, save_dir, exp_name)
    plot_accuracy_vs_memory(metrics_dict, save_dir, exp_name)
    plot_heatmap(metrics_dict, save_dir, exp_name)
    plot_stage_collapse_cost_curve(metrics_dict, save_dir, exp_name)

    print(f"[✓] Diagnostics complete for {exp_name}. Results saved in {save_dir}.")


# =====================================================
# === JF & Kevin Experiment Functions ===
# =====================================================
def run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                      model_class=VGG16, model_kwargs=None, data_shape=None, save_path="./runs",
                      post_compress_epochs=False):

    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]
    print(f"\nRunning JF experiment: {exp_name}")

    base_model = model_class(**model_kwargs)
    base_model.load_state_dict(torch.load(model_path_097, map_location='cpu')['model'])
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
