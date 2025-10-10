import os
import matplotlib.pyplot as plt
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

def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None):
  
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))

    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{dataset if dataset else ''} - {title} - Final Accuracy (%)", fontsize=14)
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].grid(True)

    ax0_twin = axs[0].twinx()
    ax0_twin.plot(names, params, 'ro--', label='Trainable Parameters (log)', linewidth=2)
    ax0_twin.set_ylabel('Trainable Parameters', color='red')
    ax0_twin.set_yscale('log')
    ax0_twin.tick_params(axis='y', colors='red')
    for i, param in enumerate(params):
        ax0_twin.annotate(f'{param:,}', xy=(i, param), xytext=(0, -15),
                          textcoords='offset points', ha='center', fontsize=9, color='red')

    if infer_times:
        axs[1].bar(names, infer_times, color='orange')
        axs[1].set_title("Inference Time (avg per batch in seconds)", fontsize=14)
        axs[1].set_ylabel("Time (s)")
        axs[1].grid(True)
    else:
        axs[1].axis('off')

    if mem_usages:
        mem_mb = [m / 1e6 for m in mem_usages]
        axs[2].bar(names, mem_mb, color='green')
        axs[2].set_title("FLOPs (Millions)", fontsize=14)
        axs[2].set_ylabel("FLOPs (Millions)")
        axs[2].grid(True)
    else:
        axs[2].axis('off')

    for ax in axs:
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha='right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.show()
    print(f"[✓] Saved plot: {filename}")
