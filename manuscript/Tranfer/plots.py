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
    import seaborn as sns
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    plt.plot(acc_list, label='Accuracy', marker='o', linewidth=2, markersize=6)
    plt.plot(loss_list, label='Loss', marker='x', linewidth=2, markersize=6)
    
    plt.title(f'{workflow} - {experiment} Accuracy & Loss', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.xticks(range(len(acc_list)))
    plt.grid(alpha=0.3)
    plt.legend(fontsize=12)
    
    filename = os.path.join(save_dir, f"{workflow}_{experiment.replace(' ', '_')}_metrics.svg")
    plt.tight_layout()
    plt.savefig(filename, format='svg')
    plt.close()
    print(f"[✓] Saved plot: {filename}")

import os
import matplotlib.pyplot as plt
def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    import seaborn as sns
    sns.set(style="whitegrid", palette="Set2", font_scale=1.1)
    
    fig, axs = plt.subplots(3, 1, figsize=(18, 18))
    
    # Sort by parameter size
    sorted_data = sorted(zip(params, accs, names, infer_times or [], mem_usages or [], flops or []), key=lambda x: x[0])
    params, accs, names, infer_times, mem_usages, flops = zip(*sorted_data)
    
    # --- Accuracy vs Parameters ---
    sns.barplot(x=list(names), y=list(accs), ax=axs[0], palette="Blues_d")
    axs[0].set_title(f"{dataset or ''} - {title} - Final Accuracy (%)", fontsize=16)
    axs[0].set_ylabel("Accuracy (%)", fontsize=14)
    axs[0].grid(alpha=0.3)
    
    # Annotate bars
    for i, v in enumerate(accs):
        axs[0].text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Secondary axis for parameters
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(range(len(params)), params, 'ro--', linewidth=2, markersize=6, label='Parameters')
    ax0_twin.set_ylabel('Trainable Parameters (log scale)', color='red', fontsize=14)
    ax0_twin.set_yscale('log')
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