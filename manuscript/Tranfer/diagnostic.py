#diagnostic.py
# =====================================
# Imports (Cleaned and Organized)
# =====================================
import glob
import json
import os
from collections import defaultdict
import os
import gc
import psutil
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
def run_full_diagnostics(
    model,
    input_shape,
    metrics_dict,
    save_dir,
    exp_name,
    collapse_range=None,
    device="cuda",
    quant=False,
    collapse_results=None,     # list[dict]
    accuracy_tolerance=None,   # float τ
):
    """
    Run a complete diagnostic suite on a PyTorch model.
    
    Supports:
        - GPU-safe quantization (FP16/FP8/INT8)
        - Per-layer FLOPs and params
        - Activation size analysis
        - Memory decomposition (CPU/GPU)
        - Optional collapse analysis

    Args:
        model: PyTorch model
        input_shape: tuple or list, model input shape (C,H,W) or (N,C,H,W)
        metrics_dict: dict, to store evaluation metrics
        save_dir: str, path to save CSVs/plots
        exp_name: str, experiment name
        collapse_range: tuple, optional layer collapse stages
        device: str, "cuda" or "cpu"
        quant: bool, whether to use quantization (FP16/INT8/QAT)
    Returns:
        diagnostics dict
    """
    print(f"[•] Running diagnostics for {exp_name}...")
    ensure_dir(save_dir)
    
    # Device setup
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # -----------------------------
    # Prepare input tensor safely
    # -----------------------------
    if len(input_shape) == 2:
        input_tensor = torch.randn((1, 3, *input_shape), device=device)
    elif len(input_shape) == 3:
        input_tensor = torch.randn((1, *input_shape), device=device)
    else:
        input_tensor = torch.randn(input_shape, device=device)

    # -----------------------------
    # Quantization handling
    # -----------------------------
    if quant:
        try:
            # GPU-safe mixed precision
            if device.type == "cuda":
                if torch.cuda.is_bf16_supported():
                    model = model.to(dtype=torch.bfloat16)
                    input_tensor = input_tensor.to(dtype=torch.bfloat16)
                else:
                    model = model.half()
                    input_tensor = input_tensor.half()
            else:
                model = model.float()  # CPU does not support FP16 well
                input_tensor = input_tensor.float()
            print(f"[•] Quantization enabled: model dtype {next(model.parameters()).dtype}")
        except Exception as e:
            print(f"[!] Quantization conversion failed: {e}. Falling back to FP32.")
            model = model.float()
            input_tensor = input_tensor.float()

    diagnostics = {}

    # -----------------------------
    # Per-layer params/FLOPs
    # -----------------------------
    try:
        df_params = analyze_per_layer_params_flops(model, input_tensor, save_dir, exp_name)
        diagnostics["per_layer_params_flops"] = df_params.to_dict(orient="records")
    except Exception as e:
        print(f"[!] Per-layer params/FLOPs analysis failed: {e}")
        diagnostics["per_layer_params_flops"] = []

    # -----------------------------
    # Activation sizes
    # -----------------------------
    try:
        df_act = analyze_activation_sizes(model, input_tensor, save_dir, exp_name)
        diagnostics["activation_sizes"] = df_act.to_dict(orient="records")
    except Exception as e:
        print(f"[!] Activation size analysis failed: {e}")
        diagnostics["activation_sizes"] = []

    # -----------------------------
    # Memory decomposition
    # -----------------------------
    try:
        mem = memory_decomposition(model, input_tensor, save_dir, exp_name)
        diagnostics["memory_decomposition"] = mem
    except Exception as e:
        print(f"[!] Memory decomposition failed: {e}")
        diagnostics["memory_decomposition"] = {}

    # -----------------------------
    # Optional collapse analysis
    # -----------------------------
    if collapse_range:
        try:
            analyze_collapse_effects(model, collapse_range, save_dir, exp_name)
        except Exception as e:
            print(f"[!] Collapse analysis failed: {e}")

    print(f"[✓] Diagnostics complete for {exp_name}")
    try:
        run_results_analysis(
            collapse_results=collapse_results,
            tau=accuracy_tolerance,
            save_dir=save_dir,
        )
    except Exception as e:
        print(f"[!] Results analysis failed: {e}")
    return diagnostics

def run_results_analysis(
    collapse_results,
    tau,
    save_dir,
):
    """
    Generates all Results section figures and tables (Fig.1–6, Table.1–6)
    """

    if not collapse_results or tau is None:
        print("[•] No collapse results or τ provided — skipping Results figures.")
        return

    print("[•] Generating Results section figures and tables...")

    # ---- Fig 1 + Table 1 ----
    plot_accuracy_vs_collapsed_depth(collapse_results, tau, save_dir)
    table1 = table_max_collapsible_depth(collapse_results, tau, save_dir)

    # ---- Fig 2 + Table 2 ----
    plot_block_acceptance_by_depth(collapse_results, save_dir)
    table_block_statistics(collapse_results, save_dir)

    # ---- Fig 3 + Table 3 ----
    plot_surrogate_error_vs_accuracy(collapse_results, save_dir)
    table_surrogate_summary(collapse_results, save_dir)

    # ---- Fig 4 + Table 4 ----
    plot_collapsible_depth_across_models(table1, save_dir)
    table_collapsible_depth_stats(table1, save_dir)

    # ---- Fig 5 + Table 5 ----
    plot_efficiency_vs_collapse(collapse_results, save_dir)
    table_efficiency_comparison(collapse_results, save_dir)

    # ---- Fig 6 + Table 6 ----
    plot_failure_case(collapse_results, save_dir)
    table_failure_modes(collapse_results, save_dir)

    print("[✓] Results section artifacts generated.")

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
    return process.memory_info().rss / 1e6   # MB


def get_model_params_memory_MB(model):
    """Accurate parameter memory using true element sizes."""
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.nelement() * p.element_size()
    return total_bytes / 1e6  # MB


def warmup_gpu(model, input_tensor, steps=3):
    """Warm up GPU to ensure kernels and cuDNN workspace are initialized."""
    if not torch.cuda.is_available():
        return
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            _ = model(input_tensor)
        torch.cuda.synchronize()


def measure_gpu_memory(model, input_tensor):
    """
    Correct way to measure GPU memory:
        - reset peak stats
        - forward pass
        - read allocated, reserved, peak
    """
    if not torch.cuda.is_available():
        return dict(
            allocated_MB=0.0,
            reserved_MB=0.0,
            peak_MB=0.0,
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    peak = torch.cuda.max_memory_allocated() / 1e6

    return dict(
        allocated_MB=allocated,
        reserved_MB=reserved,
        peak_MB=peak,
    )


# --------------------------------------------------
# Main profiling function
# --------------------------------------------------
def profile_model_memory(model, input_tensor, device_label="cpu"):

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Move to device
    device = torch.device("cuda" if "cuda" in device_label else "cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warm-up if using GPU
    if "cuda" in device_label and torch.cuda.is_available():
        warmup_gpu(model, input_tensor)

    # CPU before
    cpu_before = get_process_cpu_memory_MB()

    # Parameter memory (MB)
    params_MB = get_model_params_memory_MB(model)
    params_CPU = params_MB if "cpu" in device_label else 0.0
    params_GPU = params_MB if "cuda" in device_label else 0.0

    # Perform forward and measure GPU memory
    gpu_stats = measure_gpu_memory(model, input_tensor)

    # CPU after
    cpu_after = get_process_cpu_memory_MB()
    cpu_used = cpu_after - cpu_before
    cpu_total = cpu_after + params_CPU

    # Activation memory: peak - parameters
    activation_MB_gpu = (
        max(gpu_stats["peak_MB"] - params_GPU, 0.0)
        if "cuda" in device_label
        else 0.0
    )

    # GPU workspace/other: allocated - (params + activations)
    other_gpu_MB = (
        max(gpu_stats["allocated_MB"] - params_GPU - activation_MB_gpu, 0.0)
        if "cuda" in device_label
        else 0.0
    )

    return {
        "device": device_label,
        "Params_MB_CPU": params_CPU,
        "Params_MB_GPU": params_GPU,
        "Activation_MB_GPU": activation_MB_gpu,
        "Other_GPU_MB": other_gpu_MB,
        "Peak_GPU_MB": gpu_stats["peak_MB"],
        "Allocated_GPU_MB": gpu_stats["allocated_MB"],
        "Reserved_GPU_MB": gpu_stats["reserved_MB"],
        "CPU_Process_MB": cpu_after,
        "CPU_Used_MB": cpu_used,
        "CPU_Total_MB": cpu_total,
    }


def memory_decomposition(model, input_tensor, save_dir=".", exp_name="experiment"):
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    results = []

    # CPU
    results.append(profile_model_memory(model, input_tensor, "cpu"))

    # TODO uncomment when torch.compile is stable
    # # CPU compiled
    # compiled_cpu = torch.compile(model)
    # results.append(profile_model_memory(compiled_cpu, input_tensor, "cpu_compiled"))

    # GPU (if available)
    if torch.cuda.is_available():
        results.append(profile_model_memory(model, input_tensor, "cuda"))

        # compiled_gpu = torch.compile(model)
        # results.append(profile_model_memory(compiled_gpu, input_tensor, "cuda_compiled"))

    # Save results CSV
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(save_dir, f"{exp_name}_memory.csv")
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------
    # PLOTTING
    # --------------------------------------------------
    sns.set_theme(style="whitegrid")

    devices = df["device"]
    params_gpu = df["Params_MB_GPU"]
    activ_gpu = df["Activation_MB_GPU"]
    other_gpu = df["Other_GPU_MB"]
    peak_gpu = df["Peak_GPU_MB"]

    cpu_process = df["CPU_Process_MB"]
    cpu_params = df["Params_MB_CPU"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    # ---------- GPU Memory Plot ----------

    # Base bars: parameters
    axes[0].bar(devices, params_gpu, label="Parameters (GPU)", 
                color="#1f77b4", edgecolor="black", linewidth=0.4)

    # Activations on top
    axes[0].bar(devices, activ_gpu, bottom=params_gpu,
                label="Activations (GPU)", color="#ff7f0e", alpha=0.9)

    # Plot peak lines and annotations
    max_peak = 0
    for i, peak in enumerate(peak_gpu):
        if peak > 0:
            axes[0].plot([i - 0.3, i + 0.3], [peak, peak], 
                        linestyle="--", color="grey")
            axes[0].text(i, peak + 5, f"{peak:.1f} MB", ha="center")
            max_peak = max(max_peak, peak)

    axes[0].set_ylim(0, max_peak * 1.18)
    axes[0].set_ylabel("GPU Memory (MB)")
    axes[0].set_title(f"GPU Memory Breakdown — {exp_name}")
    axes[0].legend()
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.6)



    # ---------- CPU Memory Plot ----------
    cpu_overhead = np.array(cpu_process) - np.array(cpu_params)

    axes[1].bar(devices, cpu_params, label="Parameters (CPU)", 
                color="#8c564b", edgecolor="black", linewidth=0.4)

    axes[1].bar(devices, cpu_overhead, bottom=cpu_params,
                label="Overhead (Process RAM)", color="#9467bd", alpha=0.9)

    # Annotate
    for i, v in enumerate(cpu_process):
        axes[1].text(i, v + 10, f"{v:.1f}", ha="center")

    axes[1].set_ylabel("CPU Memory (MB)")
    axes[1].set_xlabel("Device / Compilation")
    axes[1].set_title(f"CPU Memory Breakdown — {exp_name}")
    axes[1].legend()
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)


    plt.tight_layout()
    svg_path = os.path.join(save_dir, f"{exp_name}_memory.svg")
    plt.savefig(svg_path)
    plt.close()

    print(f"Saved memory CSV to {csv_path}")
    print(f"Saved memory plot to {svg_path}")
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
    base_key = next((k for k in metrics if k.startswith("Original Model_")), None)
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
