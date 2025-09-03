import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.cm as cm
import gc
import re

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

torch.backends.cudnn.benchmark = False
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def evaluate_single_experiment(args):
    modelName, sparsity, checkpoint_path, batch_size, threshold, device_key = args

    devices = {
        "cpu": torch.device("cpu"),
        "cuda": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    device = devices[device_key]

    clear_memory()

    # Load model and move to device
    model = load_model_from_checkpoint(getModelType(modelName), checkpoint_path, device, use_compile=True)
    model = model.to(device)
    model.eval()

    # Apply sparsification threshold
    print(f"[{device_key.upper()}] Sparsity {sparsity} → Applying threshold {threshold}")
    convert_all_to_sparse(model, threshold=threshold)

    # Generate data and move to device
    x = generateData(modelName, batch_size, device).to(device)

    # Measure inference
    with torch.no_grad():
        run_data = measure_inference(model, x, device, runs=5)  # increase runs for stability

    # Collect metrics (keep per-batch values, no division by batch_size)
    times = [run["duration"] for run in run_data.values()]
    cpu_energy = [float(run["emissions_data"]["cpu_energy"]) for run in run_data.values()]
    gpu_energy = [float(run["emissions_data"]["gpu_energy"]) for run in run_data.values()]
    mems = [run["peak_mem_MB_analytical"] for run in run_data.values()]
    measured_mems = [run["peak_mem_MB_measured"] for run in run_data.values()]

    # Cleanup
    del model, x, run_data
    torch.cuda.empty_cache()
    gc.collect()
    clear_memory()

    return {
        "Sparsity": str(round(float(sparsity), 4)),
        "Device": device_key,
        "Threshold": threshold,
        "Time_s": times,
        "CPU_Energy_kWh": cpu_energy,
        "GPU_Energy_kWh": gpu_energy,
        "Analytical_Memory_MB": mems,
        "Measured_Memory_MB": measured_mems,
    }

def evaluate_models_for_sparsities_serial(modelName, sparsity_paths, batch_size=64, threshold=0.0):
    """
    Evaluate models across sparsity paths and devices serially.
    Returns combined metrics, labels, row_labels, metric_names.
    """
    row_labels = ["cpu", "cuda"]
    metric_names = ["time", "cpu_energy", "gpu_energy", "analytical_memory", "measured_memory"]
    labels = [str(round(float(sp), 4)) for sp, _ in sparsity_paths]

    # Prepare tasks for all sparsity/device combinations
    tasks = []
    for sparsity, path in sparsity_paths:
        for device_key in row_labels:
            tasks.append((modelName, sparsity, path, batch_size, threshold, device_key))

    all_results = []
    for task in tasks:
        all_results.append(evaluate_single_experiment(task))

    # Initialize metrics dictionary
    metrics = {device: {m: [] for m in metric_names} for device in row_labels}

    # Group results by device and sparsity order
    for device_key in row_labels:
        device_results = [res for res in all_results if res["Device"] == device_key]
        # Sort by sparsity label to keep order same as input labels
        device_results.sort(key=lambda r: labels.index(r["Sparsity"]))
        for res in device_results:
            metrics[device_key]["time"].append(res["Time_s"])
            metrics[device_key]["cpu_energy"].append(res["CPU_Energy_kWh"])
            metrics[device_key]["gpu_energy"].append(res["GPU_Energy_kWh"])
            metrics[device_key]["analytical_memory"].append(res["Analytical_Memory_MB"])
            metrics[device_key]["measured_memory"].append(res["Measured_Memory_MB"])

    return metrics, labels, row_labels, metric_names


def sanitize_filename(text):
    """Make a string safe for use in filenames."""
    text = str(text).strip()
    # Replace "." with "p" for floats (e.g., 0.5 → 0p5)
    text = text.replace('.', 'p')
    # Remove or replace any other unsafe characters
    return re.sub(r'[^\w\-]', '_', text)


def save_metrics_to_csv(all_data, modelName, batch_size, thresholds):
    # Clean up model name (remove accidental trailing underscores)
    modelName = modelName.strip('_')

    # Create a DataFrame from the data
    df = pd.DataFrame(all_data)

    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)

    # Sanitize threshold list for filename
    thresholds_str = "_".join(sanitize_filename(th) for th in thresholds)

    # Build output file path
    csv_filename = f"{modelName}_{batch_size}batchsize_thresholds_{thresholds_str}_performance_metrics.csv"
    csv_path = os.path.join("plots", csv_filename)

    # Save CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")
    del df
    gc.collect()


def plot_metrics(metrics, labels, row_labels, metric_names, threshold_label, axs, colors):
    # Define the metric titles for plotting
    metric_titles = {
        "analytical_memory": "Analytical Memory (MB)",
        "measured_memory": "Measured Memory (MB)",
        "time": "Inference Time (s)",
        "cpu_energy": "CPU Energy (kWh)",
        "gpu_energy": "GPU Energy (kWh)",
    }

    # Define the order in which we want to plot the metrics
    desired_order = ["analytical_memory", "measured_memory", "time", "cpu_energy", "gpu_energy"]
    metric_names = [m for m in desired_order if m in metric_names]

    n_rows = len(row_labels)
    n_cols = len(metric_names)

    # Convert labels (sparsity) to floats for numeric x-axis if possible
    try:
        x_vals = list(map(float, labels))
    except Exception:
        x_vals = labels

    # Iterate over devices (CPU and GPU) and metrics
    for i, device_key in enumerate(row_labels):
        for j, metric in enumerate(metric_names):
            ax = axs[i, j] if n_rows > 1 else axs[j]
            all_runs = metrics[device_key][metric]
            # Calculate medians and interquartile ranges for each metric
            medians, q25s, q75s = [], [], []
            for runs in all_runs:
                runs_arr = np.array(runs)
                medians.append(np.median(runs_arr))
                q25s.append(np.percentile(runs_arr, 25))
                q75s.append(np.percentile(runs_arr, 75))

            # Plot with the appropriate color for the threshold
            color = colors[threshold_label]
            ax.plot(x_vals, medians, marker='o', linestyle='-', label=threshold_label, color=color)
            ax.fill_between(x_vals, q25s, q75s, alpha=0.2, color=color)

            if i == 0:
                ax.set_title(metric_titles.get(metric, metric), fontsize=14)
            if j == 0:
                ax.set_ylabel(device_key.upper(), fontsize=12)

            ax.grid(True)
            ax.set_xlabel("Sparsity" if i == n_rows - 1 else "")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate model performance across different sparsities.")
    parser.add_argument('--models', nargs='+', default=["Vgg16_"], help="List of models to evaluate.")
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[32], help="List of batch sizes.")
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0, .5,  1], help="Percent CSR.")
    parser.add_argument('--path_to_checkpoints', type=str, default="../structured_study/pruning_checkpoints/*", help="Path to checkpoint files.")
    
    args = parser.parse_args()

    for batch_size in args.batch_sizes:
        for modelName in args.models:
            path_to_checkpoints = f"{args.path_to_checkpoints}*{modelName}*"
            all_paths = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")
            all_paths = [all_paths[0], all_paths[int(len(all_paths) * 0.25)], all_paths[int(len(all_paths) * 0.5)], all_paths[int(len(all_paths) * 0.75)], all_paths[-1]] 

            fig, axs = plt.subplots(2, 5, figsize=(20, 8), sharex=True)  # Adjusted for 5 columns of metrics
            axs = np.array(axs)

            color_map = cm.get_cmap('tab10')
            colors = {f"Threshold={th}": color_map(i) for i, th in enumerate(args.thresholds)}

            all_rows = []

            for threshold in args.thresholds:
                threshold_label = f"Threshold={threshold}"

                metrics, labels, row_labels, metric_names = evaluate_models_for_sparsities_serial(
                    modelName, all_paths, batch_size, threshold=threshold
                )

                for i, label in enumerate(labels):
                    for device_key in row_labels:
                        all_rows.append({
                            "Sparsity": label,
                            "Device": device_key,
                            "Threshold": threshold,
                            "Time_s": metrics[device_key]["time"][i],
                            "CPU_Energy_kWh": metrics[device_key]["cpu_energy"][i],
                            "GPU_Energy_kWh": metrics[device_key]["gpu_energy"][i],
                            "Analytical_Memory_MB": metrics[device_key]["analytical_memory"][i],
                            "Measured_Memory_MB": metrics[device_key]["measured_memory"][i],
                        })

                plot_metrics(metrics, labels, row_labels, metric_names, threshold_label, axs, colors)

            save_metrics_to_csv(all_rows, modelName, batch_size, args.thresholds)

            # Add legends on the top row only once per metric column
            for j in range(axs.shape[1]):
                axs[0, j].legend(loc='upper right', fontsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.suptitle(f"{modelName} Performance Metrics across Thresholds", fontsize=18)
            plot_filename = f"./plots/{modelName}_{batch_size}_{'_'.join(map(str, args.thresholds))}_comparison"
            plt.savefig(f"{plot_filename}.png")
            plt.savefig(f"{plot_filename}.svg")
            plt.close(fig)


if __name__ == "__main__":
    main()
