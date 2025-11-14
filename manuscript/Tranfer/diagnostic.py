#diagnostic.py
# =====================================
# Imports (Cleaned and Organized)
# =====================================
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import psutil

# =====================================
# Utility Imports (Project-specific)
# =====================================
from utils import   ensure_dir, is_dict_like, normalize_metrics,    count_trainable_params
import torch, psutil, os, gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -------------------------
# Diagnostics (robust)
# -------------------------
def run_full_diagnostics(model, input_shape, metrics_dict, save_dir, exp_name, collapse_range=None, device="cuda"):
    print(f"[•] Running diagnostics for {exp_name}...")
    ensure_dir(save_dir)
    model.to(device)
    model.eval()

    # Prepare input tensor (4D)
    if len(input_shape) == 2:
        input_tensor = torch.randn((1, 3, *input_shape), device=device)
    elif len(input_shape) == 3:
        input_tensor = torch.randn((1, *input_shape), device=device)
    else:
        input_tensor = torch.randn(input_shape, device=device)
    
    diagnostics = {}

    # Per-layer params/FLOPs (returns DataFrame or [] on error)
    try:
        df_params = analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
        diagnostics["per_layer_params_flops"] = df_params.to_dict(orient="records") if hasattr(df_params, "to_dict") else []
    except Exception as e:
        print(f"[!] Params/FLOPs analysis error: {e}")
        diagnostics["per_layer_params_flops"] = []

    # Activation sizes
    try:
        df_act = analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
        diagnostics["activation_sizes"] = df_act.to_dict(orient="records") if hasattr(df_act, "to_dict") else []
    except Exception as e:
        print(f"[!] Activation analysis error: {e}")
        diagnostics["activation_sizes"] = []

    # Memory decomposition
    try:
        mem = memory_decomposition(model, input_tensor, save_dir, exp_name)
        diagnostics["memory_decomposition"] = mem if isinstance(mem, dict) else {}
    except Exception as e:
        print(f"[!] Memory decomposition error: {e}")
        diagnostics["memory_decomposition"] = {}

    print(f"[✓] Diagnostics complete for {exp_name}")
    return diagnostics
# -------------------------
# Per-layer analysis & activation analysis (robust + save)
# -------------------------
def analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name):
    model.eval()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    from fvcore.nn import FlopCountAnalysis
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        per_module_flops = flops.by_module()
    except Exception:
        per_module_flops = {}

    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            flops_layer = per_module_flops.get(name, 0)
            layer_data.append({
                "layer": name,
                "params": params,
                "flops": flops_layer / 1e9  # GFLOPs
            })

    df = pd.DataFrame(layer_data)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_layer_params_flops.csv"), index=False)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(df)*0.5), 10))
    sns.barplot(x="layer", y="params", data=df, ax=axes[0], color="skyblue")
    axes[0].set_title("Parameters per Layer")
    axes[0].set_ylabel("Params (#)")
    axes[0].tick_params(axis='x', rotation=90)
    for i, v in enumerate(df["params"]):
        axes[0].text(i, v, f"{int(v):,}", ha='center', va='bottom', fontsize=8)

    sns.barplot(x="layer", y="flops", data=df, ax=axes[1], color="salmon")
    axes[1].set_title("FLOPs per Layer (GFLOPs)")
    axes[1].set_ylabel("GFLOPs")
    axes[1].tick_params(axis='x', rotation=90)
    for i, v in enumerate(df["flops"]):
        axes[1].text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_params_flops_layers.svg")
    df.to_csv(os.path.join(save_dir, f"{exp_name}_params_flops_layers.csv"), index=False)
    plt.savefig(svg_path)
    plt.close(fig)
    return df

def analyze_activation_sizes(model, input_tensor, save_dir, exp_name):
    activations = {}

    def hook(name):
        def fn(_, __, output):
            try:
                activations[name] = int(output.numel()) if isinstance(output, torch.Tensor) else 0
            except Exception:
                activations[name] = 0
        return fn

    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                hooks.append(m.register_forward_hook(hook(n)))
            except Exception:
                continue

    model.eval()
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception:
            pass

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    df = pd.DataFrame(list(activations.items()), columns=["layer", "activation_elements"])
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_activation_sizes.csv"), index=False)

    fig = plt.figure(figsize=(max(12, len(df)*0.4), 6))
    if len(df) > 30:
        pivot = df.set_index("layer").T
        sns.heatmap(pivot, cmap="viridis", annot=False, cbar_kws={"label": "Activation Elements"})
        plt.title(f"Activation Elements per Layer (Heatmap) for {exp_name}")
    else:
        sns.barplot(x="layer", y="activation_elements",
                    data=df.sort_values("activation_elements", ascending=False),
                    color="lightgreen")
        plt.xticks(rotation=90)
        plt.title(f"Activation Elements per Layer for {exp_name}")
        for i, v in enumerate(df["activation_elements"]):
            plt.text(i, v, f"{int(v):,}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_activation_sizes.svg")
    plt.savefig(svg_path)
    plt.close()
    return df

# --------------------------
# Memory Measurement Functions
# --------------------------
def get_process_cpu_memory_MB():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e6  # MB

def get_model_params_memory_MB(model):
    return sum(p.numel() for p in model.parameters()) * 4 / 1e6  # float32

def track_activation_memory(model, input_tensor, device_label="cuda"):
    device = torch.device(device_label)
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    if device_label.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
    else:
        before = 0.0

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)

    if device_label.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()
    else:
        after = 0.0

    return (after - before) / 1e6

# --------------------------
# Core Measurement Routine
# --------------------------
def run_and_measure(model, input_tensor, device_label="cpu"):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    device = torch.device(device_label)
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # CPU memory before
    cpu_before = get_process_cpu_memory_MB()

    # Params memory
    params_MB = get_model_params_memory_MB(model)
    params_CPU = params_MB if device_label == "cpu" else 0.0
    params_GPU = params_MB if device_label.startswith("cuda") else 0.0

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    if torch.cuda.is_available() and device_label.startswith("cuda"):
        torch.cuda.synchronize()

    # CPU memory after
    cpu_after = get_process_cpu_memory_MB()
    cpu_used = cpu_after - cpu_before
    cpu_total = cpu_after + params_CPU

    # GPU memory
    peak_gpu = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() and device_label.startswith("cuda") else 0.0
    activation_MB = track_activation_memory(model, input_tensor, device_label=device_label)
    other_GPU = peak_gpu - params_GPU - activation_MB if peak_gpu > 0 else 0.0

    return {
        "device": device_label,
        "Params_MB_CPU": params_CPU,
        "Params_MB_GPU": params_GPU,
        "Activation_MB_GPU": activation_MB,
        "Other_GPU_MB": other_GPU,
        "Peak_GPU_MB": peak_gpu,
        "CPU_Process_MB": cpu_after,
        "CPU_Used_MB": cpu_used,
        "CPU_Total_MB": cpu_total
    }

# --------------------------
# Memory Decomposition with Compilation
# --------------------------
def memory_decomposition(model, input_tensor, save_dir=".", exp_name="experiment"):
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    results = []

    # --- Original CPU run ---
    results.append(run_and_measure(model, input_tensor, "cpu"))

    # --- Original GPU run ---
    if torch.cuda.is_available():
        results.append(run_and_measure(model, input_tensor, "cuda"))

    # --- Compiled CPU run ---
    compiled_model_cpu = torch.compile(model)
    results.append(run_and_measure(compiled_model_cpu, input_tensor, "cpu_compiled"))

    # --- Compiled GPU run ---
    if torch.cuda.is_available():
        compiled_model_gpu = torch.compile(model)
        results.append(run_and_measure(compiled_model_gpu, input_tensor, "cuda_compiled"))

    # ----------------------
    # Prepare plotting data
    # ----------------------
    devices = [r["device"] for r in results]

    # GPU memory
    params_gpu = [r["Params_MB_GPU"] for r in results]
    activations_gpu = [r["Activation_MB_GPU"] for r in results]
    other_gpu = [r["Other_GPU_MB"] for r in results]
    peak_gpu = [r["Peak_GPU_MB"] for r in results]

    # CPU memory
    cpu_process = [r["CPU_Process_MB"] for r in results]
    cpu_params = [r["Params_MB_CPU"] for r in results]

    # ----------------------
    # Plotting
    # ----------------------
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # GPU stacked bar
    axes[0].bar(devices, params_gpu, color="#1f77b4", label="Params (GPU)")
    axes[0].bar(devices, activations_gpu, bottom=params_gpu, color="#ff7f0e", label="Activations (GPU)")
    bottom_gpu = [params_gpu[i] + activations_gpu[i] for i in range(len(devices))]
    axes[0].bar(devices, other_gpu, bottom=bottom_gpu, color="#2ca02c", label="Other GPU Memory")

    for i, peak in enumerate(peak_gpu):
        if peak > 0:
            axes[0].plot([i - 0.3, i + 0.3], [peak, peak], linestyle="--", color="gray")
            axes[0].text(i, peak + 5, f"Peak: {peak:.1f} MB", ha="center")

    axes[0].set_ylabel("GPU Memory (MB)")
    axes[0].set_title(f"GPU Memory Usage — {exp_name}")
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)

    # CPU stacked bar
    width = 0.35
    x = np.arange(len(devices))
    axes[1].bar(x - width/2, cpu_process, width, color="#9467bd", label="Process RAM")
    axes[1].bar(x + width/2, cpu_params, width, color="#8c564b", label="Params (CPU)")

    for i, v in enumerate(cpu_process):
        axes[1].text(x[i] - width/2, v + 5, f"{v:.1f}", ha="center")
    for i, v in enumerate(cpu_params):
        axes[1].text(x[i] + width/2, v + 5, f"{v:.1f}", ha="center")

    axes[1].set_ylabel("CPU Memory (MB)")
    axes[1].set_xlabel("Device / Compilation")
    axes[1].set_title(f"CPU Memory Usage — {exp_name}")
    axes[1].legend()
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    plt.xticks(x, devices)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    svg_path = os.path.join(save_dir, f"{exp_name}_memory_comparison.svg")
    plt.savefig(svg_path, dpi=300)
    plt.close()
    print(f"Saved memory breakdown to: {svg_path}")

    return results

# -------------------------
# Collapse analysis (unchanged but robust)
# -------------------------
def predict_collapse_parameters(in_channels, out_channels, kernel_size, num_layers_collapsed):
    original_params = num_layers_collapsed * (in_channels * out_channels * kernel_size * kernel_size + out_channels)
    collapsed_params = in_channels * out_channels * kernel_size * kernel_size + out_channels
    delta = collapsed_params - original_params
    return {"original": original_params, "collapsed": collapsed_params, "delta": delta}

def analyze_collapse_effects(model, collapse_range, save_dir, exp_name):
    if not collapse_range:
        return
    try:
        start_stage, end_stage = collapse_range
        stage_channels = [64, 128, 256, 512, 512, 4096]
        in_ch = stage_channels[start_stage - 1]
        out_ch = stage_channels[end_stage - 1]
        num_layers = (end_stage - start_stage + 1) * 3
        pred = predict_collapse_parameters(in_ch, out_ch, 3, num_layers)
        observed_params = count_trainable_params(model)
        df = pd.DataFrame([{
            "stage_range": f"{start_stage}-{end_stage}",
            "predicted_params": pred["collapsed"],
            "original_est": pred["original"],
            "delta_predicted": pred["delta"],
            "observed_total": observed_params
        }])
        ensure_dir(save_dir)
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
    except Exception as e:
        print(f"[!] analyze_collapse_effects error: {e}")

# -------------------------
# Cross-experiment per-layer aggregation (robust + readable)
# -------------------------
def debug_tensor_shape(tensor, description="Tensor"):
    """ Helper function to debug tensor shapes. """
    if tensor is not None:
        print(f"{description} Shape: {tensor.shape}")
    else:
        print(f"{description} is None!")

# -------------------------
# Robust plotting helpers
# -------------------------
def plot_flops_vs_latency(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return

    names = list(metrics.keys())
    flops = []
    times = []
    for n in names:
        m = metrics[n] if is_dict_like(metrics[n]) else {}
        flops.append(float(m.get("flops", 0)))
        times.append(float(m.get("inference_time", 0)))

    print(f"[DEBUG] FLOPs: {flops}")
    print(f"[DEBUG] Times: {times}")

    if not any(flops) and not any(times):
        return

    # Save the data to CSV
    df_flops_latency = pd.DataFrame({"Model": names, "FLOPs": flops, "Inference Time (s)": times})
    df_flops_latency.to_csv(os.path.join(save_dir, f"{exp_name}_flops_vs_latency.csv"), index=False)

    ensure_dir(save_dir)
    plt.figure(figsize=(8, 6))
    plt.scatter(flops, times, marker='o')
    for i, txt in enumerate(names):
        plt.annotate(txt, (flops[i], times[i]), xytext=(5, 2), textcoords='offset points', fontsize=8)
    plt.xscale("log")
    plt.xlabel("FLOPs (log)")
    plt.ylabel("Inference Time (s)")
    plt.title(f"FLOPs vs Inference Time — {exp_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    file_svg = os.path.join(save_dir, f"flops_vs_latency.svg")
    plt.tight_layout()
    plt.savefig(file_svg)
    df_flops_latency.to_csv(os.path.join(save_dir, f"{exp_name}_flops_vs_latency.csv"), index=False)
    plt.close()

def plot_delta_accuracy_vs_params(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return

    # --- Find the original model ---
    base_key = next((k for k in metrics if k.startswith("Original_Model_")), None)
    if base_key is None:
        print("[WARN] No base model found matching 'Original_model_*'. Using first entry as fallback.")
        base_key = next(iter(metrics))

    base = metrics[base_key]
    if not is_dict_like(base):
        return

    base_acc = base.get("final_accuracy", 0)
    base_params = base.get("param_count", 1)

    # --- Compute deltas ---
    deltas = []
    for name, data in metrics.items():
        if not is_dict_like(data):
            continue
        d_acc = float(data.get("final_accuracy", 0) - base_acc)
        try:
            if float(base_params) != 0:
                d_params = (float(data.get("param_count", 0)) - float(base_params)) / float(base_params) * 100
            else:
                d_params = 0.0
        except Exception:
            d_params = 0.0
        deltas.append({"name": name, "ΔAcc": d_acc, "ΔParams(%)": d_params})

    print(f"[DEBUG] Base model: {base_key}")
    print(f"[DEBUG] Delta Accuracy vs Params Data: {deltas}")

    if not deltas:
        return

    df = pd.DataFrame(deltas)
    ensure_dir(save_dir)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.csv"), index=False)

    # --- Plot ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="ΔParams(%)", y="ΔAcc")
    for _, r in df.iterrows():
        plt.annotate(r["name"], (r["ΔParams(%)"], r["ΔAcc"]), fontsize=8)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("Δ Parameters (%)")
    plt.ylabel("Δ Accuracy")
    plt.title(f"Compression Efficiency — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"delta_acc_vs_params.svg"))
    plt.close()

def plot_flops_vs_memory(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return
    names = list(metrics.keys())
    flops = [float(metrics[n].get("flops", 0)) if is_dict_like(metrics[n]) else 0 for n in names]
    mems = [float(metrics[n].get("total_size_mb", 0) or metrics[n].get("memory") or 0) if is_dict_like(metrics[n]) else 0 for n in names]
    
    print(f"[DEBUG] FLOPs: {flops}")
    print(f"[DEBUG] Memory: {mems}")

    if not any(flops) and not any(mems):
        return
    
    # Save the data to CSV
    df_flops_memory = pd.DataFrame({"Model": names, "FLOPs": flops, "Memory (MB)": mems})
    df_flops_memory.to_csv(os.path.join(save_dir, f"{exp_name}_flops_vs_memory.csv"), index=False)

    ensure_dir(save_dir)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=flops, y=mems)
    for i, n in enumerate(names):
        plt.annotate(n, (flops[i], mems[i]), fontsize=8, xytext=(4, 2), textcoords='offset points')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FLOPs (log)")
    plt.ylabel("Total Memory (MB, log)")
    plt.title(f"FLOPs vs Memory — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"flops_vs_memory.svg"))
    df_flops_memory.to_csv(os.path.join(save_dir, f"{exp_name}_flops_vs_memory.csv"), index=False)
    plt.close()

def plot_accuracy_vs_memory(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return
    names = list(metrics.keys())
    accs = [float(metrics[n].get("final_accuracy", 0)) if is_dict_like(metrics[n]) else 0 for n in names]
    mems = [float(metrics[n].get("total_size_mb", 0) or metrics[n].get("memory") or 0) if is_dict_like(metrics[n]) else 0 for n in names]
    
    print(f"[DEBUG] Accuracy: {accs}")
    print(f"[DEBUG] Memory: {mems}")

    if not any(accs) and not any(mems):
        return
    
    # Save the data to CSV
    df_acc_memory = pd.DataFrame({"Model": names, "Accuracy": accs, "Memory (MB)": mems})
    df_acc_memory.to_csv(os.path.join(save_dir, f"{exp_name}_acc_vs_memory.csv"), index=False)

    ensure_dir(save_dir)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=mems, y=accs)
    for i, n in enumerate(names):
        plt.annotate(n, (mems[i], accs[i]), fontsize=8, xytext=(4, 2), textcoords='offset points')
    plt.xlabel("Memory (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Memory — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"acc_vs_memory.svg"))
    df_acc_memory.to_csv(os.path.join(save_dir, f"{exp_name}_acc_vs_memory.csv"), index=False)
    plt.close()

def plot_heatmap(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return
    rows = []
    for name, v in metrics.items():
        if not is_dict_like(v):
            continue
        rows.append({
            "Model": name,
            "Accuracy": v.get("final_accuracy", 0),
            "Params": v.get("param_count", 0),
            "FLOPs": v.get("flops", 0),
            "Inference Time": v.get("inference_time", 0),
            "Memory (MB)": v.get("total_size_mb", 0)
        })
    
    print(f"[DEBUG] Heatmap Rows: {rows}")

    if not rows:
        return
    df = pd.DataFrame(rows).set_index("Model")
    # Normalize columns for heatmap stability
    df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else (x * 0.0))

    # Save the data to CSV
    df.to_csv(os.path.join(save_dir, f"{exp_name}_metrics.csv"))

    ensure_dir(save_dir)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_norm, annot=True, cmap="coolwarm")
    plt.title(f"Normalized Metrics Heatmap — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"metrics_heatmap.svg"))
    df_norm.to_csv(os.path.join(save_dir, f"{exp_name}_metrics_heatmap.csv"))
    plt.close()

def plot_memory_per_layer_across_experiments(metrics_sources, save_dir, exp_name, dtype_bytes=4):
    import json
    from collections import defaultdict

    ensure_dir(save_dir)
    per_layer_mem = defaultdict(dict)
    experiment_names = []

    def _load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    if isinstance(metrics_sources, str):
        loaded = _load_json(metrics_sources)
        metrics = {os.path.splitext(os.path.basename(metrics_sources))[0]: loaded}
    elif isinstance(metrics_sources, dict):
        metrics = metrics_sources.copy()
    elif isinstance(metrics_sources, (list, tuple)):
        metrics = {}
        for path in metrics_sources:
            if isinstance(path, str) and os.path.isfile(path):
                loaded = _load_json(path)
                name_hint = os.path.splitext(os.path.basename(path))[0]
                metrics[name_hint] = loaded

    for exp_name_key, metric_obj in metrics.items():
        experiment_names.append(exp_name_key)
        diagnostics = metric_obj.get("diagnostics") or {}
        activation_sizes = diagnostics.get("activation_sizes", [])
        for item in activation_sizes:
            layer = item.get("layer") or item.get("name")
            elems = item.get("activation_elements") or 0
            mem_mb = (float(elems) * dtype_bytes) / 1e6
            per_layer_mem[layer][exp_name_key] = mem_mb

    if not per_layer_mem:
        return

    df = pd.DataFrame(per_layer_mem).T.fillna(0.0).T
    df["total_mb"] = df.sum(axis=1)
    df = df.sort_values("total_mb", ascending=False).drop(columns=["total_mb"])

    plt.figure(figsize=(max(8, len(df)*0.4), max(6, len(df.columns)*0.5)))
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Activation Memory (MB)"})
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Layer")
    plt.title(f"Per-layer Activation Memory (MB) — {exp_name}")
    plt.tight_layout()
    outpath = os.path.join(save_dir, f"{exp_name}_per_layer_activation_memory_heatmap.svg")
    df.to_csv(outpath.replace(".svg", ".csv"))
    plt.savefig(outpath)
    plt.close()
    return outpath

def plot_accuracy_loss_curve(acc_list, loss_list, workflow, experiment, save_dir="plots"):
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    os.makedirs(save_dir, exist_ok=True)
    
    # Create subplots: 2 rows, 1 column
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot accuracy
    ax1.plot(acc_list, label='Accuracy', marker='o', linewidth=2, markersize=6, color='tab:blue')
    ax1.set_title(f'{workflow} - {experiment} Accuracy', fontsize=16)
    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=12)

    # Plot loss
    ax2.plot(loss_list, label='Loss', marker='x', linewidth=2, markersize=6, color='tab:red')
    ax2.set_title(f'{workflow} - {experiment} Loss', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Loss', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Set common x-axis labels
    ax2.set_xticks(range(len(acc_list)))
    
    # Adjust layout and save the plot
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{workflow}_{experiment.replace(' ', '_')}_metrics.svg")
    plt.savefig(filename, format='svg')
    plt.close()

    print(f"[✓] Saved plot: {filename}")
    
def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    # Extract experiment from filename
    experiment = ' '.join(filename.split('/')[-1].replace('.svg','').split('_')[:1])
    sns.set(style="whitegrid", palette="Set2", font_scale=1.1)
    
    fig, axs = plt.subplots(3, 1, figsize=(18, 18))
    
    # --- Sort by parameter size, then alphabetically ---
    sorted_data = sorted(
        zip(params, accs, names, infer_times or [], mem_usages or [], flops or []),
        key=lambda x: (x[0], x[2].lower())
    )
    params, accs, names, infer_times, mem_usages, flops = zip(*sorted_data)
    
    # --- Accuracy vs Parameters ---
    sns.barplot(x=list(names), y=list(accs), ax=axs[0], palette="Blues_d")
    axs[0].set_title(f"{dataset or ''} - {experiment} - {title} - Final Accuracy (%)", fontsize=16)
    axs[0].set_ylabel("Accuracy (%)", fontsize=14)
    axs[0].grid(alpha=0.3)
    
    # Annotate bars
    for i, v in enumerate(accs):
        axs[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Secondary axis for parameters
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(range(len(params)), params, 'ro--', linewidth=2, markersize=6, label='Parameters')
    ax0_twin.set_ylabel('Trainable Parameters', color='red', fontsize=14)
    ax0_twin.set_yscale('linear')
    
    # Set zero at the smallest trainable parameter
    min_param = min(params)
    ax0_twin.set_ylim(bottom=min_param * 0.9, top=max(params) * 1.1)
    ax0_twin.tick_params(axis='y', colors='red')
    
    # --- Inference Time ---
    if infer_times:
        sns.barplot(x=list(names), y=list(infer_times), ax=axs[1], palette="Oranges_d")
        axs[1].set_title("Average Inference Time per Batch (s)", fontsize=16)
        axs[1].set_ylabel("Time (s)", fontsize=14)
        axs[1].grid(alpha=0.3)
    
    # --- Memory or FLOPs ---
    if mem_usages:
        mem_mb = [m / 1e6 for m in mem_usages]
        sns.barplot(x=list(names), y=mem_mb, ax=axs[2], palette="Greens_d")
        axs[2].set_title("Peak GPU Memory (MB)", fontsize=16)
        axs[2].set_ylabel("Memory (MB)", fontsize=14)
    elif flops:
        flops_g = [f / 1e9 for f in flops]
        sns.barplot(x=list(names), y=flops_g, ax=axs[2], palette="Greens_d")
        axs[2].set_title("FLOPs (GFLOPs)", fontsize=16)
        axs[2].set_ylabel("GFLOPs", fontsize=14)
    else:
        axs[2].axis('off')
    
    # Rotate x-ticks
    for ax in axs:
        ax.set_xticklabels(names, rotation=30, ha='right')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, format='svg')
    plt.show()
    print(f"[✓] Saved plot: {filename}")
