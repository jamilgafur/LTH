import os
import matplotlib.pyplot as plt
import matplotlib
import logging
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
    plt.savefig(filename)
    plt.show()

    print(f"[✓] Saved plot: {filename}")
