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

def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    fig, axs = plt.subplots(4, 1, figsize=(16, 20))  # We now have 4 subplots

    # Sort by params
    sorted_data = sorted(zip(params, accs, names, infer_times if infer_times else [None]*len(names),
                             mem_usages if mem_usages else [None]*len(names),
                             flops if flops else [None]*len(names),
                             total_sizes if total_sizes else [None]*len(names)),
                         key=lambda x: x[0])

    params, accs, names, infer_times, mem_usages, flops, total_sizes = zip(*sorted_data)

    axs[0].bar(names, accs, color='skyblue')
    axs[0].set_title(f"{dataset if dataset else ''} Accuracy vs Number of Parameters")
    axs[0].set_xlabel("Model")
    axs[0].set_ylabel("Accuracy")
    
    axs[1].bar(names, params, color='lightgreen')
    axs[1].set_title(f"{dataset if dataset else ''} Number of Parameters")
    axs[1].set_xlabel("Model")
    axs[1].set_ylabel("Params (Millions)")

    if infer_times:
        axs[2].bar(names, infer_times, color='lightcoral')
        axs[2].set_title("Inference Time (seconds)")
        axs[2].set_xlabel("Model")
        axs[2].set_ylabel("Time (seconds)")

    if total_sizes:
        axs[3].bar(names, total_sizes, color='lightgoldenrodyellow')
        axs[3].set_title("Estimated Total Size (MB)")
        axs[3].set_xlabel("Model")
        axs[3].set_ylabel("Size (MB)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close()
