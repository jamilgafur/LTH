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

    run_full_diagnostics(model, data_shape, {exp_name: data}, plots_dir, exp_name, collapse_range=collapse_range)

    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)

    del model
    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
    return data


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


# (1) Per-layer parameters & FLOPs
def analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name):
    model.eval()
    with torch.no_grad():
        if isinstance(input_tensor, tuple):
            input_tensor = input_tensor[0]
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()
    
    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf layer
            params = sum(p.numel() for p in module.parameters())
            layer_data.append({
                "layer": name,
                "params": params,
                "flops": per_module_flops.get(name, 0)
            })
    
    df = pd.DataFrame(layer_data)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_params_flops.csv"), index=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    df.plot(x="layer", y="params", kind="bar", ax=axes[0], color="skyblue", legend=False)
    axes[0].set_title("Parameters per Layer")
    axes[0].tick_params(axis='x', rotation=90)
    df.plot(x="layer", y="flops", kind="bar", ax=axes[1], color="salmon", legend=False)
    axes[1].set_title("FLOPs per Layer")
    axes[1].tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_params_flops_layers.png"))
    plt.close(fig)
    return df


# (2) Activation footprint heatmap
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


# (3) Predicted vs Observed Params/FLOPs scatter
def plot_predicted_vs_observed(pred_df, observed_df, save_dir, exp_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(pred_df["pred_params"], observed_df["params"], color="blue")
    ax[0].plot([pred_df["pred_params"].min(), pred_df["pred_params"].max()],
               [pred_df["pred_params"].min(), pred_df["pred_params"].max()], 'k--')
    ax[0].set_title("Predicted vs Observed Params")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Observed")

    ax[1].scatter(pred_df["pred_flops"], observed_df["flops"], color="red")
    ax[1].plot([pred_df["pred_flops"].min(), pred_df["pred_flops"].max()],
               [pred_df["pred_flops"].min(), pred_df["pred_flops"].max()], 'k--')
    ax[1].set_title("Predicted vs Observed FLOPs")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Observed")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_pred_vs_obs.png"))
    plt.close(fig)


# (4) FLOPs vs Inference Time scatter
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


# (5) Memory decomposition
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


# (9) Summary table combining all metrics
def generate_layer_summary_table(model, input_tensor, save_dir, exp_name):
    with torch.no_grad():
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()

    rows = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            shape = None
            if hasattr(module, 'weight'):
                shape = list(module.weight.shape)
            rows.append({
                "layer": name,
                "param_count": params,
                "flops": per_module_flops.get(name, 0),
                "weight_shape": shape
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_summary.csv"), index=False)
    return df

def analyze_collapse_effects(model, collapse_range, save_dir, exp_name):
    if not collapse_range:
        return
    start_stage, end_stage = collapse_range
    # Approximation: assume each stage doubles filters
    stage_channels = [64, 128, 256, 512, 512, 4096]
    in_ch = stage_channels[start_stage - 1]
    out_ch = stage_channels[end_stage - 1]
    num_layers = (end_stage - start_stage + 1) * 3  # assume 3 convs per stage

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
    plt.bar(["Original", "Predicted Collapsed", "Observed Total"],
            [pred["original"], pred["collapsed"], observed_params],
            color=["gray", "orange", "blue"])
    plt.ylabel("Parameter Count")
    plt.title(f"Collapse {start_stage}-{end_stage} Parameter Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_collapse_prediction.png"))
    plt.close()

def predict_collapse_parameters(in_channels, out_channels, kernel_size, num_layers_collapsed):
    """Theoretical expected parameters after collapsing n layers."""
    # Original parameters before collapse
    original_params = num_layers_collapsed * (in_channels * out_channels * kernel_size * kernel_size + out_channels)
    # After collapse — a single equivalent convolution layer
    collapsed_params = in_channels * out_channels * kernel_size * kernel_size + out_channels
    delta = collapsed_params - original_params
    return {
        "original": original_params,
        "collapsed": collapsed_params,
        "delta": delta
    }

# === Run All Diagnostics ===
def run_full_diagnostics(model, input_shape, metrics_dict, save_dir, exp_name, collapse_range=None, device="cuda"):
    print(f"[•] Running diagnostics for {exp_name}...")
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    model.eval()
    input_tensor = torch.randn(input_shape, device=device)

    print(f"[•] Running diagnostics for {exp_name}...")
    os.makedirs(save_dir, exist_ok=True)
    try:
        analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in per-layer params/FLOPs analysis: {e}")
    try:
        analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in activation sizes analysis: {e}")
    try:
        memory_decomposition(model, input_tensor, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in memory decomposition: {e}")
    try:
        plot_flops_vs_latency(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in FLOPs vs Latency plot: {e}")
    try:
        analyze_collapse_effects(model, collapse_range, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in collapse effects analysis: {e}")

    # Extended diagnostics
    try:
        plot_delta_accuracy_vs_params(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in delta accuracy vs params plot: {e}")
    try:
        plot_flops_vs_memory(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in FLOPs vs Memory plot: {e}")
    try:
        plot_stage_parameter_density(model, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in stage parameter density plot: {e}")
    try:
        plot_accuracy_vs_memory(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in accuracy vs memory plot: {e}")
    try:
        plot_heatmap(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in heatmap plot: {e}")
    try:
        plot_stage_collapse_cost_curve(metrics_dict, save_dir, exp_name)
    except Exception as e:
        print(f"[!] Error in stage collapse cost curve plot: {e}")

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
