# plots.py
import os
import matplotlib.pyplot as plt
import matplotlib
import logging
from utils import *
from analysis import *
from experiments import *
matplotlib.set_loglevel('ERROR')


def plot_accuracy_loss_curve(acc_list, loss_list, workflow, experiment, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(acc_list, label='Accuracy', marker='o')
    plt.plot(loss_list, label='Loss', marker='x')
    plt.title(f'{workflow} - {experiment} Accuracy & Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    filename = os.path.join(save_dir, f"{workflow}_{experiment.replace(' ', '_')}_metrics.svg")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"[✓] Saved plot: {filename}")

import os
import matplotlib.pyplot as plt

def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    # Create dataset directory if specified and doesn't exist
    if dataset:
        os.makedirs(dataset, exist_ok=True)
        filename = os.path.join(dataset, filename)

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))  # 3 subplots (accuracy + params, inference time, memory usage)

    # Plot Accuracy vs Model (Bar Plot)
    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{dataset if dataset else ''} - {title} - Final Accuracy (%)", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)

    # Add a twin axis to plot params on a log scale
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(names, params, 'ro--', label='Trainable Parameters (log)', linewidth=2)
    ax0_twin.set_ylabel('Trainable Parameters', color='red')
    ax0_twin.set_yscale('log')
    ax0_twin.tick_params(axis='y', colors='red')

    # Annotate params
    for i, param in enumerate(params):
        ax0_twin.annotate(f'{param:,}', xy=(i, param), xytext=(0, -15),
                          textcoords='offset points', ha='center', fontsize=9, color='red')

    # Plot Inference Times (Bar Plot)
    if infer_times:
        axs[1].bar(names, infer_times, color='orange')
        axs[1].set_title("Inference Time (avg per batch in seconds)", fontsize=14)
        axs[1].set_ylabel("Time (s)")
        axs[1].grid(True)
    else:
        axs[1].axis('off')  # Hide the axis if infer_times is not provided

    # Plot Memory Usage or FLOPs (Bar Plot)
    if mem_usages:
        mem_mb = [m / 1e6 for m in mem_usages]  # Convert memory usage to MB
        axs[2].bar(names, mem_mb, color='green')
        axs[2].set_title("feature map size (MB)", fontsize=14)
        axs[2].set_ylabel("feature map size (MB)")
        axs[2].grid(True)
    elif flops:
        axs[2].bar(names, flops, color='lightgreen')
        axs[2].set_title("FLOPs (Millions)", fontsize=14)
        axs[2].set_ylabel("FLOPs (Millions)")
        axs[2].grid(True)
    else:
        axs[2].axis('off')  # Hide the axis if neither mem_usages nor flops are provided

    # Set x-ticks and labels
    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')

    # Tight layout and save the plot
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()

    print(f"[✓] Saved plot: {filename}")



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

    if not any(flops) and not any(times):
        return

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

    if not deltas:
        return
    df = pd.DataFrame(deltas)
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
    if not any(flops) and not any(mems):
        return
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
    if not any(accs) and not any(mems):
        return
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
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("Model")
    # Normalize columns for heatmap stability
    df_norm = df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else (x * 0.0))

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
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("Model")
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
