import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
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


def evaluate_model_performance(model: torch.nn.Module, device: torch.device, batch_size: int, modelName: str) -> float:
    x = generateData(modelName, batch_size, device)
    with torch.no_grad():
        time_s = measure_inference_time(model, x, device)
    return time_s


def evaluate_models_for_all_configs(modelName, checkpoints, batch_sizes, devices, sparsity_levels):
    modelType = getModelType(modelName)
    all_timings = {"CPU": {}, "GPU": {}}

    for batch_size in batch_sizes:
        for device in devices:
            device_type = "GPU" if device.type == "cuda" else "CPU"

            print(f"\n[Model: {modelName}] Batch size: {batch_size} | Device: {device_type}")

            all_timings[device_type][batch_size] = {
                "unpruned_dense": None,
                "pruned_dense": [],
                "pruned_sparse": []
            }

            # 1) Unpruned Dense model (sparsity = 0)
            dense_model = modelType()
            dense_model.to(device)
            dense_model.eval()
            if torch.__version__ >= "2.0":
                dense_model = torch.compile(dense_model)

            unpruned_time = evaluate_model_performance(dense_model, device, batch_size, modelName)
            all_timings[device_type][batch_size]["unpruned_dense"] = unpruned_time

            # 2) Pruned models: Dense and Sparse timings
            pruned_dense_times = []
            pruned_sparse_times = []

            for sparsity, ckpt_path in checkpoints:
                # Load pruned dense model
                pruned_dense_model = load_model_from_checkpoint(modelType, ckpt_path, device, use_compile=True)

                dense_time = evaluate_model_performance(pruned_dense_model, device, batch_size, modelName)
                pruned_dense_times.append(dense_time)

                # Convert to sparse representation
                sparse_model = load_model_from_checkpoint(modelType, ckpt_path, device, use_compile=False)  # no compile before convert
                convert_all_to_sparse(sparse_model)
                sparse_model.to(device)
                sparse_model.eval()

                sparse_time = evaluate_model_performance(sparse_model, device, batch_size, modelName)
                pruned_sparse_times.append(sparse_time)

            all_timings[device_type][batch_size]["pruned_dense"] = pruned_dense_times
            all_timings[device_type][batch_size]["pruned_sparse"] = pruned_sparse_times

    return all_timings


def plot_timing_grid(modelName, sparsity_levels, batch_sizes, all_timings):
    n_rows = 2
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharey=True)

    device_order = ["CPU", "GPU"]
    device_row_idx = {"CPU": 0, "GPU": 1}

    for col_idx, batch_size in enumerate(batch_sizes):
        for device_type in device_order:
            row_idx = device_row_idx[device_type]
            ax = axes[row_idx][col_idx] if n_cols > 1 else axes[row_idx]

            timing_data = all_timings[device_type][batch_size]

            # x axis is sparsity in %
            sparsity_percent = [s * 100 for s in sparsity_levels]

            # Plot unpruned dense time (constant line)
            ax.axhline(
                y=timing_data["unpruned_dense"],
                linestyle='--',
                color='black',
                label=f"Unpruned Dense (Time={timing_data['unpruned_dense']:.4f}s)"
            )

            # Plot pruned dense model times
            ax.plot(
                sparsity_percent,
                timing_data["pruned_dense"],
                marker='o',
                color='blue',
                label="Pruned Dense Model"
            )

            # Plot pruned sparse model times
            ax.plot(
                sparsity_percent,
                timing_data["pruned_sparse"],
                marker='x',
                color='green',
                label="Pruned Sparse Model (CSR)"
            )

            if row_idx == 0:
                ax.set_title(f"Batch Size = {batch_size}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"{device_type} Inference Time (s)", fontsize=11)

            ax.set_xlabel("Sparsity Level (%)", fontsize=11)
            ax.grid(True)
            ax.legend(fontsize=9)

    fig.suptitle(f"{modelName} – Inference Time Comparison (CPU & GPU)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{modelName}_inference_time_comparison.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"[Saved Plot] {plot_filename}")


def main():
    torch.manual_seed(0)
    models = ["ResNet20", "Vgg16_ImageNet", "Vgg16_", "RegNetX"]
    batch_sizes = [1, 32,64]

    results = {}

    for modelName in models:
        path_to_checkpoints = f"../../../../plots/LTH_output/*{modelName}*"
        checkpoints = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")

        if not checkpoints:
            print(f"[Warning] No checkpoints found for {modelName} at {path_to_checkpoints}")
            continue

        sparsity_levels = [s for s, _ in checkpoints]
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        # Measure all timings
        all_timings = evaluate_models_for_all_configs(
            modelName=modelName,
            checkpoints=checkpoints,
            batch_sizes=batch_sizes,
            devices=devices,
            sparsity_levels=sparsity_levels
        )

        # Store results
        results[modelName] = {
            "sparsity_levels": sparsity_levels,
            "batch_sizes": batch_sizes,
            "timings": all_timings
        }

        # Plot
        plot_timing_grid(modelName, sparsity_levels, batch_sizes, all_timings)

        # Save results as CSV with flattening timing dict to DataFrame for clarity
        records = []
        for device_type in all_timings:
            for batch_size in all_timings[device_type]:
                timing_data = all_timings[device_type][batch_size]
                for i, sparsity in enumerate(sparsity_levels):
                    records.append({
                        "Device": device_type,
                        "Batch Size": batch_size,
                        "Sparsity": sparsity,
                        "Unpruned Dense": timing_data["unpruned_dense"],
                        "Pruned Dense": timing_data["pruned_dense"][i],
                        "Pruned Sparse": timing_data["pruned_sparse"][i],
                    })

        df = pd.DataFrame(records)
        csv_path = f"{modelName}_inference_timings.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved CSV] {csv_path}")

        print(f"[Completed] {modelName} evaluation and plotting.")
        break  # remove if want all models processed

    return results


if __name__ == "__main__":
    main()
