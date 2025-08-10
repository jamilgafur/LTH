

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob

from SparseConv2d import SparseConv2d
from SparseLinear import SparseLinear
from util import *
from measure import *
from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16ImageNet import VGG16_ImageNet
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from pyPrune.models.RegNetX import RegNetX_400MF

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def evaluate_model_performance(model: torch.nn.Module, device: torch.device, batch_size: int, modelName: str):
    """
    Evaluate a model's performance with detailed per-run metrics.
    """
    x = generateData(modelName, batch_size, device)
            
    with torch.no_grad():
        run_data = measure_inference(model, x, device)

    return run_data


def evaluate_models_for_sparsities(modelName, sparsity_paths, batch_size=64, threshold=0.0):
    devices = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    row_labels = ["cpu", "cuda"]
    metric_names = ["time", "cpu_energy", "gpu_energy", "memory"]

    metrics = {device: {m: [] for m in metric_names} for device in row_labels}
    labels = [str(round(float(sp), 4)) for sp, _ in sparsity_paths]

    for sparsity, path in sparsity_paths:
        for device_key in row_labels:
            device = devices[device_key]
            model = load_model_from_checkpoint(getModelType(modelName), path, device, use_compile=False)

            # Always apply sparsification based on threshold
            convert_all_to_sparse(model, threshold=threshold)

            run_data = evaluate_model_performance(model, device, batch_size, modelName)
            # normalize everything based on batch_size
            for run in run_data.values():
                run["duration"] /= batch_size
                run["mem_alloc"] /= batch_size
                run["emissions_data"]["cpu_energy"] = float(run["emissions_data"]["cpu_energy"]) / batch_size
                run["emissions_data"]["gpu_energy"] = float(run["emissions_data"]["gpu_energy"]) / batch_size

            

            times = [run["duration"] for run in run_data.values()]
            cpu_energy = [float(run["emissions_data"]["cpu_energy"]) for run in run_data.values()]
            gpu_energy = [float(run["emissions_data"]["gpu_energy"]) for run in run_data.values()]
            mems = [run["mem_alloc"] for run in run_data.values()]

            metrics[device_key]["time"].append(times)
            metrics[device_key]["cpu_energy"].append(cpu_energy)
            metrics[device_key]["gpu_energy"].append(gpu_energy)
            metrics[device_key]["memory"].append(mems)

    return metrics, labels, row_labels, metric_names


def save_metrics_to_csv(all_data, modelName):
    df = pd.DataFrame(all_data)
    os.makedirs("plots", exist_ok=True)
    csv_path = f"./plots/{modelName}_performance_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved combined metrics CSV to {csv_path}")


def plot_metrics(metrics, labels, row_labels, metric_names, threshold_label, axs, colors):
    import numpy as np

    metric_titles = {
        "memory": "Peak Memory (MB)",
        "time": "Time (s)",
        "cpu_energy": "CPU Energy (kWh)",
        "gpu_energy": "GPU Energy (kWh)",
    }

    desired_order = ["memory", "time", "cpu_energy", "gpu_energy"]
    metric_names = [m for m in desired_order if m in metric_names]

    n_rows = len(row_labels)
    n_cols = len(metric_names)

    for i, device_key in enumerate(row_labels):
        for j, metric in enumerate(metric_names):
            ax = axs[i, j] if n_rows > 1 else axs[j]
            all_runs = metrics[device_key][metric]

            medians, q25s, q75s = [], [], []

            for k in range(len(labels)):
                runs = np.array(all_runs[k])
                if len(runs) == 0:
                    medians.append(0)
                    q25s.append(0)
                    q75s.append(0)
                    continue

                medians.append(np.median(runs))
                q25s.append(np.percentile(runs, 25))
                q75s.append(np.percentile(runs, 75))

            color = colors[threshold_label]

            ax.plot(labels, medians, marker='o', linestyle='-', label=threshold_label, color=color)
            ax.fill_between(labels, q25s, q75s, alpha=0.2, color=color)

            if i == 0:
                ax.set_title(metric_titles.get(metric, metric), fontsize=14)
            if j == 0:
                ax.set_ylabel(device_key.replace('_', ' ').upper(), fontsize=12)

            ax.grid(True)

import matplotlib.pyplot as plt

def main():
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    models = [ "Vgg16ImageNet", "Vgg16_", "ResNet20"]
    batch_sizes = [32]
    thresholds = [0.0, 0.5, 1.0]
    for batch_size in batch_sizes:
        for modelName in models:
            path_to_checkpoints = f"../../../../plots/LTH_output/*{modelName}*"
            all_paths = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")
            # Use first, left middle, right middle, middle and last paths for thresholds (0,25% 50 and 75% 100%)
            all_paths = [all_paths[0], all_paths[len(all_paths) // 4], all_paths[len(all_paths) // 2], all_paths[3 * len(all_paths) // 4], all_paths[-1]]
            fig, axs = plt.subplots(2, 4, figsize=(20, 8), sharex=True)

            axs = np.array(axs)

            color_map = cm.get_cmap('tab10')
            colors = {f"Threshold={th}": color_map(i) for i, th in enumerate(thresholds)}

            all_rows = []  

            for threshold in thresholds:
                threshold_label = f"Threshold={threshold}"

                metrics, labels, row_labels, metric_names = evaluate_models_for_sparsities(
                    modelName, all_paths, batch_size, threshold=threshold
                )

                # 👇 Add rows for this threshold to the combined list
                for i, label in enumerate(labels):
                    for device_key in row_labels:
                        all_rows.append({
                            "Sparsity": label,
                            "Device": device_key,
                            "Threshold": threshold,
                            "Time_s": metrics[device_key]["time"][i],
                            "CPU_Energy_kWh": metrics[device_key]["cpu_energy"][i],
                            "GPU_Energy_kWh": metrics[device_key]["gpu_energy"][i],
                            "Peak_Memory_MB": metrics[device_key]["memory"][i],
                        })

                plot_metrics(metrics, labels, row_labels, metric_names, threshold_label, axs, colors)

            # Save combined CSV once per model
            save_metrics_to_csv(all_rows, modelName)

            for ax in axs[-1, :]:
                ax.set_xlabel("Sparsity")


            for j in range(axs.shape[1]):
                axs[0, j].legend(loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.suptitle(f"{modelName} Performance Metrics across Thresholds", fontsize=18, y=1.02)
            plt.savefig(f"./plots/{modelName}_all_thresholds_comparison.png")
            plt.savefig(f"./plots/{modelName}_all_thresholds_comparison.svg")
            plt.show()
   
if __name__ == "__main__":
    main()
