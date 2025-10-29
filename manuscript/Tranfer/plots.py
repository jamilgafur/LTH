# plots.py
import os
import matplotlib.pyplot as plt
import matplotlib
import logging
from utils import ensure_dir
import pandas as pd
import seaborn as sns
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
    """
    Unified plot: Accuracy vs Params, Inference Time, and Memory/FLOPs.
    Corrected labeling and axis scaling.
    """
    import os
    import matplotlib.pyplot as plt
    import pandas as pd

    # Create figure with 3 vertically stacked plots
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))

    # Sort all data by parameter count
    sorted_data = sorted(
        zip(params, accs, names, infer_times or [], mem_usages or [], flops or []),
        key=lambda x: x[0]
    )
    params, accs, names, infer_times, mem_usages, flops = zip(*sorted_data)

    # --- Accuracy vs Parameters ---
    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{dataset or ''} - {title} - Final Accuracy (%)", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)

    ax0_twin = axs[0].twinx()
    ax0_twin.plot(names, params, 'ro--', label='Trainable Parameters', linewidth=2)
    ax0_twin.set_ylabel('Trainable Parameters (log scale)', color='red')
    ax0_twin.set_yscale('log')
    ax0_twin.tick_params(axis='y', colors='red')

    # --- Inference Times ---
    if infer_times:
        axs[1].bar(names, infer_times, color='orange')
        axs[1].set_title("Average Inference Time per Batch (s)", fontsize=14)
        axs[1].set_ylabel("Time (seconds)")
        axs[1].grid(True)
    else:
        axs[1].axis('off')

    # --- Memory or FLOPs ---
    if mem_usages:
        # mem_usages should be in bytes — convert to MB
        mem_mb = [m / 1e6 for m in mem_usages]
        axs[2].bar(names, mem_mb, color='green')
        axs[2].set_title("Peak GPU Memory (MB)", fontsize=14)
        axs[2].set_ylabel("Memory (MB)")
    elif flops:
        flops_g = [f / 1e9 for f in flops]
        axs[2].bar(names, flops_g, color='lightgreen')
        axs[2].set_title("FLOPs (GFLOPs)", fontsize=14)
        axs[2].set_ylabel("GFLOPs")
    else:
        axs[2].axis('off')

    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()

    # Export CSV for reproducibility
    data = {
        'Model': names,
        'Parameters': params,
        'Accuracy (%)': accs,
        'Inference Time (s)': infer_times,
        'Peak GPU Memory (MB)': [m / 1e6 for m in mem_usages] if mem_usages else None,
        'FLOPs (GFLOPs)': [f / 1e9 for f in flops] if flops else None
    }
    df = pd.DataFrame(data)
    csv_filename = filename.replace('.svg', '.csv')
    df.to_csv(csv_filename, index=False)
    print(f"[✓] Saved plot and CSV: {filename}")



