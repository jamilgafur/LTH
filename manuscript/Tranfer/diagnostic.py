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
    debug_tensor_shape(input_tensor, "Input Tensor")

    # Ensure batch dimension
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    debug_tensor_shape(input_tensor, "Input Tensor (After Batch Dimension Check)")

    with torch.no_grad():
        try:
            flops = FlopCountAnalysis(model, input_tensor)
            per_module_flops = flops.by_module()
            # optionally total flops:
            try:
                total_flops_val = flops.total()
            except Exception:
                total_flops_val = None
        except Exception as e:
            print(f"[!] FlopCountAnalysis failed: {e} — continuing with empty FLOPs map")
            per_module_flops = {}
            total_flops_val = None


    layer_data = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            params = sum(p.numel() for p in module.parameters())
            flops_for_layer = per_module_flops.get(name, 0)
            layer_data.append({"layer": name, "params": params, "flops": flops_for_layer})

    if not layer_data:
        print(f"[!] No per-layer data collected for {exp_name}")
        return pd.DataFrame(columns=["layer", "params", "flops"])

    df = pd.DataFrame(layer_data)
    ensure_dir(save_dir)
    csv_path = os.path.join(save_dir, f"{exp_name}_layer_params_flops.csv")
    df.to_csv(csv_path, index=False)

    # Plot (if many layers, switch to heatmap/pivot for readability)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    unique_layers = df["layer"].nunique()
    if unique_layers > 30:
        # pivot (experiments would usually be aggregated later) - show log10 values for compactness
        df_plot = df.set_index("layer")[["params", "flops"]].astype(float).where(lambda x: x > 0, other=0.0)
        # transpose for heatmap
        sns.heatmap(df_plot.T.fillna(0.0), ax=axes[0])

        axes[0].set_title("Parameters & FLOPs per Layer (Heatmap)")
    else:
        df.plot(x="layer", y="params", kind="bar", ax=axes[0], color="skyblue", legend=False)
        axes[0].set_title("Parameters per Layer")
        axes[0].tick_params(axis='x', rotation=90)

    df.plot(x="layer", y="flops", kind="bar", ax=axes[1], color="salmon", legend=False)
    axes[1].set_title("FLOPs per Layer")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_params_flops_layers.svg")
    plt.savefig(svg_path)
    plt.close(fig)
    print(f"[✓] Saved per-layer params/flops CSV: {csv_path} and plot: {svg_path}")

    return df

def analyze_activation_sizes(model, input_tensor, save_dir, exp_name):
    activations = {}

    def hook(name):
        def fn(_, __, output):
            try:
                if isinstance(output, torch.Tensor):
                    activations[name] = int(output.numel())
                else:
                    activations[name] = 0
            except Exception:
                activations[name] = 0
        return fn

    # Ensure batch dimension
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    hooks = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            try:
                hooks.append(m.register_forward_hook(hook(n)))
            except Exception:
                continue

    model.eval()
    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception as e:
            print(f"[!] Forward pass for activation sizes failed: {e}")

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if not activations:
        print(f"[!] No activations collected for {exp_name}")
        return pd.DataFrame(columns=["layer", "activation_elements"])

    df = pd.DataFrame(list(activations.items()), columns=["layer", "activation_elements"])
    ensure_dir(save_dir)
    csv_path = os.path.join(save_dir, f"{exp_name}_activation_sizes.csv")
    df.to_csv(csv_path, index=False)

    # If many layers, show heatmap pivot
    fig = plt.figure(figsize=(12, 6))
    unique_layers = df["layer"].nunique()
    if unique_layers > 30:
        pivot = df.set_index("layer").T
        sns.heatmap(pivot, annot=False)
        plt.title("Activation Elements per Layer (heatmap)")
    else:
        sns.barplot(data=df.sort_values("activation_elements", ascending=False), x="layer", y="activation_elements", color="lightgreen")
        plt.xticks(rotation=90)
        plt.title("Activation Size per Layer (# elements)")
    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_activation_sizes.svg")
    plt.savefig(svg_path)
    plt.close()
    print(f"[✓] Saved activation sizes CSV: {csv_path} and plot: {svg_path}")

    return df

def memory_decomposition(model, input_tensor, save_dir, exp_name):
    # Ensure batch dimension
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    param_mem = sum(p.numel() for p in model.parameters()) * 4 / 1e6  # MB (fp32)
    peak_mem = None
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    with torch.no_grad():
        try:
            _ = model(input_tensor)
        except Exception as e:
            print(f"[!] Forward pass for memory decomposition failed: {e}")

    if torch.cuda.is_available():
        try:
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
        except Exception:
            peak_mem = None

    activation_mem = max(peak_mem - param_mem, 0) if peak_mem is not None else None
    parts = {"Params_MB": float(param_mem),
             "Activations+Temps_MB": float(activation_mem) if activation_mem is not None else 0.0,
             "Peak_MB": float(peak_mem) if peak_mem is not None else 0.0}

    ensure_dir(save_dir)
    print(f"Memory Decomposition: Params: {parts['Params_MB']}MB, Activations: {parts['Activations+Temps_MB']}MB, Peak: {parts['Peak_MB']}MB")

    # Save and plot
    svg_path = os.path.join(save_dir, f"{exp_name}_memory_breakdown.svg")
    plt.figure(figsize=(6, 6))
    cats = list(parts.keys())
    vals = [parts[k] for k in cats]
    plt.bar(cats, vals, color=["steelblue", "salmon", "gold"])

    plt.title(f"Memory Breakdown — {exp_name}")
    plt.ylabel("Memory (MB)")
    plt.tight_layout()
    plt.savefig(svg_path)
    plt.close()
    print(f"[✓] Saved memory decomposition plot: {svg_path}")

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
    ensure_dir(save_dir)
    json_paths = glob.glob(os.path.join(metrics_dir, "*metrics.json"))
    if not json_paths:
        print(f"[!] No JSON metrics files found in {metrics_dir}")
        return

    all_data = []
    for path in json_paths:
        print(f"[DEBUG] Processing file: {path}")
        try:
            with open(path, "r") as f:
                content = json.load(f)
        except Exception as e:
            print(f"[!] Failed to read JSON '{path}': {e}")
            continue
        if not is_dict_like(content):
            print(f"[!] JSON root is not a dict in '{path}'")
            continue

        for exp_group_name, exp_group in content.items():
            if not is_dict_like(exp_group):
                print(f"[!] Experiment group '{exp_group_name}' is not a dict, skipping")
                continue
            for name, m in exp_group.items():
                # Wrap non-dict metrics into a dict with the metric name as key
                if not is_dict_like(m):
                    print(f"[DEBUG] Wrapping non-dict experiment '{name}' data: {m}")
                    m = {name: m}

                # Convert values to float safely
                def safe_float(x):
                    try:
                        if isinstance(x, list):
                            if not x:  # empty list
                                return 0.0
                            return float(np.mean(x))
                        return float(x)
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
        print("[!] No valid metrics data found for unified plots.")
        return

    print("[DEBUG] Data for plotting:")
    print(df)

    ensure_dir(save_dir)

    # Accuracy vs Params
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="Params", y="Accuracy", hue="Experiment", legend="brief", s=120)
    plt.xscale("log")
    plt.xlabel("Parameters (log scale)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Parameters — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_accuracy_vs_params.svg"))
    plt.close()

    # FLOPs vs Memory
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="FLOPs", y="Memory", hue="Experiment", legend=False, s=120)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("FLOPs (log)")
    plt.ylabel("Memory (MB, log)")
    plt.title(f"FLOPs vs Memory — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_flops_vs_memory.svg"))
    plt.close()

    # Accuracy vs Memory
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="Memory", y="Accuracy", hue="Experiment", s=120)
    plt.xlabel("Memory (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Accuracy vs Memory — {workflow}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{workflow}_accuracy_vs_memory.svg"))
    plt.close()

    print(f"[✓] Saved unified metrics plots for workflow '{workflow}'")

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
    plt.close()


def plot_memory_per_layer_across_experiments(metrics_sources, save_dir, exp_name, dtype_bytes=4):
    """
    Plot per-layer activation memory (MB) across experiments.

    metrics_sources can be:
      - dict: mapping experiment_name -> metrics-dict
      - str: path to a single JSON file. The file may either:
          * represent one experiment (contains "diagnostics"), or
          * contain many experiments as top-level keys (each value a metrics dict)
      - list of str: list of JSON file paths (each file may be a single experiment or a mapping)

    dtype_bytes: bytes per activation element (default 4 for float32).
    """
    import os
    import json
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from collections import defaultdict

    # helper to ensure save dir
    try:
        ensure_dir(save_dir)
    except NameError:
        os.makedirs(save_dir, exist_ok=True)

    # load a JSON file
    def _load_json(path):
        with open(path, 'r') as f:
            return json.load(f)

    # normalize a single metrics-like object into a dict mapping experiment_name -> metrics_dict
    def _metrics_from_obj(obj, default_name):
        """
        If obj is a mapping whose values are per-experiment dicts, return it directly.
        Else if obj itself appears to be a single experiment dict (has 'diagnostics' or 'final_accuracy' etc.),
        return {default_name: obj}.
        """
        if isinstance(obj, dict):
            # detect if top-level appears to be multiple experiments:
            # heuristic: many values are dicts and at least one has diagnostics/accuracies keys
            multi_like = False
            candidate_count = 0
            for v in obj.values():
                if isinstance(v, dict):
                    candidate_count += 1
                    if any(k in v for k in ("diagnostics", "accuracies", "final_accuracy", "param_count")):
                        multi_like = True
                        break
            if multi_like and candidate_count >= 1:
                return dict(obj)  # treat top-level keys as experiments
            # else treat as single experiment
            return {default_name: dict(obj)}
        # not a dict -> can't handle
        return {default_name: {"__raw__": obj}}

    # Build unified metrics mapping
    metrics = {}

    # If caller passed already a dict of metrics
    if isinstance(metrics_sources, dict):
        metrics = metrics_sources.copy()
    elif isinstance(metrics_sources, str):
        # single file path
        try:
            loaded = _load_json(metrics_sources)
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON from {metrics_sources}: {e}")
        name_hint = os.path.splitext(os.path.basename(metrics_sources))[0]
        metrics = _metrics_from_obj(loaded, name_hint)
    elif isinstance(metrics_sources, (list, tuple)):
        # list of paths (or possibly a mixture)
        for path in metrics_sources:
            if isinstance(path, str) and os.path.isfile(path):
                try:
                    loaded = _load_json(path)
                except Exception:
                    # skip unreadable files
                    continue
                name_hint = os.path.splitext(os.path.basename(path))[0]
                obj_map = _metrics_from_obj(loaded, name_hint)
                # if there are duplicates, prefix with filename when necessary
                for k, v in obj_map.items():
                    if k in metrics:
                        new_k = f"{name_hint}__{k}"
                        metrics[new_k] = v
                    else:
                        metrics[k] = v
            elif isinstance(path, dict):
                # if the list contained pre-loaded dicts
                for k, v in path.items():
                    if k in metrics:
                        metrics[f"{k}_dup"] = v
                    else:
                        metrics[k] = v
            else:
                # ignore invalid entries
                continue
    else:
        raise ValueError("metrics_sources must be a dict, a path (str), or a list of paths/dicts.")

    if not metrics:
        # nothing to plot
        return

    # Extract per-layer activation sizes and convert to MB
    # We'll build: layer_name -> {experiment_name: memory_mb}
    per_layer_mem = defaultdict(dict)
    experiment_names = []

    for exp_name_key, metric_obj in metrics.items():
        experiment_names.append(exp_name_key)
        # metric_obj may not be a dict (coerced earlier), guard
        if not isinstance(metric_obj, dict):
            continue

        diagnostics = metric_obj.get("diagnostics") or metric_obj.get("diagnostic") or {}

        # If diagnostics missing, try legacy keys
        if not diagnostics:
            # sometimes the JSON has activation_sizes at top level
            if "activation_sizes" in metric_obj:
                diagnostics = {"activation_sizes": metric_obj.get("activation_sizes")}
            else:
                diagnostics = {}

        activation_sizes = diagnostics.get("activation_sizes", [])
        # activation_sizes expected as list of {"layer": name, "activation_elements": N}
        if activation_sizes and isinstance(activation_sizes, (list, tuple)):
            for item in activation_sizes:
                if not isinstance(item, dict):
                    continue
                layer = item.get("layer") or item.get("name") or item.get("layer_name")
                elems = item.get("activation_elements") or item.get("elements") or item.get("activation_size")
                try:
                    elems = float(elems)
                except Exception:
                    elems = 0.0
                mem_mb = (elems * float(dtype_bytes)) / 1e6
                per_layer_mem[layer][exp_name_key] = mem_mb
        else:
            # fallback: some diagnostics include activation element counts under different keys
            # try to extract from 'per_layer_params_flops' sizes (not ideal)
            alt = diagnostics.get("per_layer_params_flops", [])
            if alt and isinstance(alt, (list, tuple)):
                # not activation sizes, but include as a fallback to show something
                for item in alt:
                    if not isinstance(item, dict):
                        continue
                    layer = item.get("layer")
                    # use params as proxy (very rough) -> convert params to bytes
                    params = item.get("params", 0)
                    try:
                        params = float(params)
                    except Exception:
                        params = 0.0
                    mem_mb = (params * float(dtype_bytes)) / 1e6
                    per_layer_mem[layer][exp_name_key] = mem_mb
            # if nothing, continue (will be NaN)
            continue

    if not per_layer_mem:
        # nothing extracted
        return

    # Create DataFrame: index = layers, columns = experiments
    df = pd.DataFrame(per_layer_mem).T.fillna(0.0)  # currently dict keyed by layer -> {exp: mem}
    # After T, rows are experiments -> we want index = layers, so transpose back:
    df = df.T  # now index=layers, columns=experiments
    # Ensure all experiments appear as columns (even if missing)
    for e in experiment_names:
        if e not in df.columns:
            df[e] = 0.0

    # Sort layers by total memory descending for clearer heatmap
    df["total_mb"] = df.sum(axis=1)
    df = df.sort_values("total_mb", ascending=False).drop(columns=["total_mb"])

    if df.empty:
        return

    # Plot heatmap
    plt.figure(figsize=(max(8, min(0.4 * len(df.index), 24)), max(6, min(0.5 * len(df.columns), 16))))
    sns.set_theme()  # let seaborn choose nice defaults
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "Activation Memory (MB)"})
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Layer")
    plt.title(f"Per-layer Activation Memory (MB) — {exp_name}")
    plt.tight_layout()

    outpath = os.path.join(save_dir, f"{exp_name}_per_layer_activation_memory_heatmap.svg")
    plt.savefig(outpath)
    plt.close()

    print(f"[✓] Saved per-layer activation memory heatmap: {outpath}")
    return outpath
