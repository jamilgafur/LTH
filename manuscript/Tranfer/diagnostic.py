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

# =====================================
# Utility Imports (Project-specific)
# =====================================
# These should exist elsewhere in your repo
# If not, replace with local equivalents or implement them
from utils import   ensure_dir, is_dict_like, normalize_metrics,    count_trainable_params


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
        plt.title("Activation Elements per Layer (Heatmap)")
    else:
        sns.barplot(x="layer", y="activation_elements",
                    data=df.sort_values("activation_elements", ascending=False),
                    color="lightgreen")
        plt.xticks(rotation=90)
        plt.title("Activation Size per Layer (# elements)")
        for i, v in enumerate(df["activation_elements"]):
            plt.text(i, v, f"{int(v):,}", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_activation_sizes.svg")
    plt.savefig(svg_path)
    plt.close()
    return df

def memory_decomposition(model, input_tensor, save_dir, exp_name):
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    param_mem = sum(p.numel() for p in model.parameters()) * 4 / 1e6  # MB

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception:
            pass

    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else None
    activation_mem = max(peak_mem - param_mem, 0) if peak_mem is not None else 0.0
    parts = {"Params_MB": float(param_mem), "Activations_MB": float(activation_mem),
             "Peak_GPU_MB": float(peak_mem) if peak_mem else 0.0}

    plt.figure(figsize=(6, 6))
    sns.barplot(x=list(parts.keys()), y=list(parts.values()), palette=["steelblue", "salmon", "gold"])
    for i, v in enumerate(parts.values()):
        plt.text(i, v, f"{v:.1f}", ha='center', va='bottom', fontsize=10)
    plt.ylabel("Memory (MB)")
    plt.title(f"GPU Memory Breakdown — {exp_name}")
    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_memory_breakdown.svg")
    plt.savefig(svg_path)
    plt.close()
    return parts
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


import numpy as np

# -------------------------
# Unified metrics plots (wrap non-dict metrics, average lists)
# -------------------------
def plot_unified_metrics(metrics_dir, save_dir, workflow):
    import glob, json, numpy as np
    from utils import ensure_dir, is_dict_like, normalize_metrics
    ensure_dir(save_dir)
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))
    all_data = []

    for path in json_paths:
        with open(path, "r") as f:
            content = json.load(f)
        for exp_group_name, exp_group in content.items():
            for name, m in exp_group.items():
                if not is_dict_like(m):
                    m = {name: m}
                def safe_float(x):
                    try:
                        return float(np.mean(x)) if isinstance(x, list) else float(x)
                    except Exception:
                        return 0.0
                all_data.append({
                    "Experiment": name,
                    "Params": safe_float(m.get("param_count", 0)),
                    "Accuracy": safe_float(m.get("final_accuracy", m.get("accuracies", 0))),
                    "FLOPs": safe_float(m.get("flops", 0)),
                    "Inference Time": safe_float(m.get("inference_time", 0)),
                    "Memory": safe_float(m.get("total_size_mb", 0))
                })

    df = pd.DataFrame(all_data)
    if df.empty:
        return

    # Accuracy vs Parameters
    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(data=df, x="Params", y="Accuracy", hue="Experiment", s=120)
    for i, row in df.iterrows():
        ax.text(row["Params"], row["Accuracy"], row["Experiment"], fontsize=8, ha='right')
    ax.set_xscale("log")
    plt.grid(alpha=0.3)
    plt.title(f"Accuracy vs Parameters — {workflow}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_accuracy_vs_params.svg"))
    plt.close()

    # FLOPs vs Memory
    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(data=df, x="FLOPs", y="Memory", hue="Experiment", s=120)
    ax.set_xscale("log")
    ax.set_yscale("log")
    plt.grid(alpha=0.3)
    plt.title(f"FLOPs vs Memory — {workflow}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_flops_vs_memory.svg"))
    plt.close()
    df.to_csv(os.path.join(save_dir, f"{workflow}_unified_metrics.csv"), index=False)

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
    file_svg = os.path.join(save_dir, f"{exp_name}_flops_vs_latency.svg")
    plt.tight_layout()
    plt.savefig(file_svg)
    df_flops_latency.to_csv(os.path.join(save_dir, f"{exp_name}_flops_vs_latency.csv"), index=False)
    plt.close()

def plot_delta_accuracy_vs_params(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return
    try:
        base = list(metrics.values())[0]
        if not is_dict_like(base):
            return
        base_acc = base.get("final_accuracy", 0)
        base_params = base.get("param_count", 1)
    except Exception:
        return

    deltas = []
    for name, data in metrics.items():
        if not is_dict_like(data):
            continue
        d_acc = float(data.get("final_accuracy", 0) - base_acc)
        try:
            d_params = (float(data.get("param_count", 0)) - float(base_params)) / float(base_params) * 100 if float(base_params) != 0 else 0.0
        except Exception:
            d_params = 0.0
        deltas.append({"name": name, "ΔAcc": d_acc, "ΔParams(%)": d_params})

    print(f"[DEBUG] Delta Accuracy vs Params Data: {deltas}")

    if not deltas:
        return
    df = pd.DataFrame(deltas)

    # Save the data to CSV
    df.to_csv(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.csv"), index=False)

    ensure_dir(save_dir)
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.svg"))
    df.to_csv(os.path.join(save_dir, f"{exp_name}_delta_acc_vs_params.csv"), index=False)
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_flops_vs_memory.svg"))
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_acc_vs_memory.svg"))
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
    plt.savefig(os.path.join(save_dir, f"{exp_name}_metrics_heatmap.svg"))
    df_norm.to_csv(os.path.join(save_dir, f"{exp_name}_metrics_heatmap.csv"))
    plt.close()

def plot_stage_collapse_cost_curve(metrics_dict, save_dir, exp_name):
    metrics = normalize_metrics(metrics_dict)
    if not metrics:
        return
    rows = []
    for name, v in metrics.items():
        if not is_dict_like(v):
            continue
        rows.append({"Model": name, "Params": v.get("param_count", 0),
                     "Time": v.get("inference_time", 0), "Accuracy": v.get("final_accuracy", 0)})
    
    print(f"[DEBUG] Collapse Curve Rows: {rows}")

    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("Model")

    # Save the data to CSV
    df.to_csv(os.path.join(save_dir, f"{exp_name}_collapse_cost_curve.csv"), index=False)

    ensure_dir(save_dir)
    plt.figure(figsize=(9, 6))
    plt.plot(df["Model"], df["Params"], label="Parameters", marker="o")
    plt.plot(df["Model"], df["Time"], label="Inference Time", marker="s")
    plt.plot(df["Model"], df["Accuracy"], label="Accuracy", marker="^")
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f"Stage Collapse Cost Curve — {exp_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{exp_name}_collapse_cost_curve.svg"))
    # save data
    df.to_csv(os.path.join(save_dir, f"{exp_name}_collapse_cost_curve.csv"), index=False)
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