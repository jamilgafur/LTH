import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
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

def evaluate_model_performance(model: torch.nn.Module, device: torch.device, batch_size: int, modelName: str) -> float:
    x = generateData(modelName, batch_size, device)
    with torch.no_grad():
        time_s, emissions = measure_inference(model, x, device)
    return time_s, emissions

def evaluate_models_for_all_configs(modelName, checkpoints, batch_sizes, devices, sparsity_levels, thresholds):
    modelType = getModelType(modelName)
    threshold_results = {}

    for threshold in thresholds:
        print(f"\n[Evaluating Threshold = {threshold}]")
        all_timings = {"CPU": {}, "GPU": {}}
        all_emissions = {"CPU": {}, "GPU": {}}

        for batch_size in batch_sizes:
            for device in devices:
                device_type = "GPU" if device.type == "cuda" else "CPU"
                print(f"[{modelName}] Batch size: {batch_size} | Device: {device_type} | Threshold: {threshold}")

                all_timings[device_type][batch_size] = {
                    "unpruned_dense": None,
                    "pruned_dense": [],
                    "pruned_sparse": []
                }
                all_emissions[device_type][batch_size] = {
                    "unpruned_dense": None,
                    "pruned_dense": [],
                    "pruned_sparse": []
                }

                dense_model = modelType()
                dense_model.to(device)
                dense_model.eval()
                if torch.__version__ >= "2.0":
                    dense_model = torch.compile(dense_model)
                unpruned_time, unpruned_emissions = evaluate_model_performance(dense_model, device, batch_size, modelName)

                all_timings[device_type][batch_size]["unpruned_dense"] = unpruned_time
                all_emissions[device_type][batch_size]["unpruned_dense"] = unpruned_emissions

                pruned_dense_times, pruned_dense_emissions = [], []
                pruned_sparse_times, pruned_sparse_emissions = [], []

                for sparsity, ckpt_path in checkpoints:
                    pruned_dense_model = load_model_from_checkpoint(modelType, ckpt_path, device, use_compile=True)
                    dense_time, dense_emissions = evaluate_model_performance(pruned_dense_model, device, batch_size, modelName)
                    pruned_dense_times.append(dense_time)
                    pruned_dense_emissions.append(dense_emissions)

                    sparse_model = load_model_from_checkpoint(modelType, ckpt_path, device, use_compile=False)
                    convert_all_to_sparse(sparse_model, threshold=threshold)
                    sparse_model.to(device)
                    sparse_model.eval()
                    if torch.__version__ >= "2.0":
                        sparse_model = torch.compile(sparse_model)

                    sparse_time, sparse_emissions = evaluate_model_performance(sparse_model, device, batch_size, modelName)
                    pruned_sparse_times.append(sparse_time)
                    pruned_sparse_emissions.append(sparse_emissions)

                all_timings[device_type][batch_size]["pruned_dense"] = pruned_dense_times
                all_timings[device_type][batch_size]["pruned_sparse"] = pruned_sparse_times
                all_emissions[device_type][batch_size]["pruned_dense"] = pruned_dense_emissions
                all_emissions[device_type][batch_size]["pruned_sparse"] = pruned_sparse_emissions

        threshold_results[threshold] = {
            "timings": all_timings,
            "emissions": all_emissions,
        }

    return threshold_results

def plot_timing_and_emissions_separate_rows(modelName, sparsity_levels, batch_sizes, threshold_results, filename, thresholds):
    n_rows = 4
    n_cols = len(batch_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True)

    devices = ["CPU", "GPU"]
    sparsity_percent = [s * 100 for s in sparsity_levels]
    colors = ["green", "red", "purple", "orange", "brown", "gray"]
    threshold_labels = [f"{int(t*100)}% Threshold" for t in thresholds]

    for col_idx, batch_size in enumerate(batch_sizes):
        for device_idx, device_type in enumerate(devices):
            timing_row = device_idx * 2
            emissions_row = timing_row + 1

            ax_time = axes[timing_row][col_idx] if n_cols > 1 else axes[timing_row]
            ax_emission = axes[emissions_row][col_idx] if n_cols > 1 else axes[emissions_row]

            for t_idx, threshold in enumerate(thresholds):
                all_timings = threshold_results[threshold]["timings"]
                all_emissions = threshold_results[threshold]["emissions"]

                timing_data = all_timings[device_type][batch_size]
                emissions_data = all_emissions[device_type][batch_size]

                unpruned_time = timing_data["unpruned_dense"] / batch_size
                pruned_dense_times = [t / batch_size for t in timing_data["pruned_dense"]]
                pruned_sparse_times = [t / batch_size for t in timing_data["pruned_sparse"]]

                unpruned_emission = emissions_data["unpruned_dense"] / batch_size
                pruned_dense_emissions = [e / batch_size for e in emissions_data["pruned_dense"]]
                pruned_sparse_emissions = [e / batch_size for e in emissions_data["pruned_sparse"]]

                if t_idx == 0:
                    ax_time.axhline(y=unpruned_time, linestyle='--', color='black',
                                    label=f"Unpruned Dense Time/item={unpruned_time:.4f}s")
                    ax_emission.axhline(y=unpruned_emission, linestyle='--', color='black',
                                        label=f"Unpruned Dense Emission/item={unpruned_emission:.6f}kg")

                # Sparse lines for this threshold
                ax_time.plot(sparsity_percent, pruned_sparse_times, marker='x',
                             color=colors[t_idx % len(colors)],
                             label=f"Sparse ({threshold_labels[t_idx]})")

                ax_emission.plot(sparsity_percent, pruned_sparse_emissions, marker='x',
                                 linestyle='-', color=colors[t_idx % len(colors)],
                                 label=f"Sparse ({threshold_labels[t_idx]})")

            if col_idx == 0:
                ax_time.set_ylabel(f"{device_type} Inference Time\nper Data (s)", fontsize=11)
                ax_emission.set_ylabel(f"{device_type} CO₂ Emission\nper Data (kg)", fontsize=11)
            if timing_row == 0:
                ax_time.set_title(f"Batch Size = {batch_size}", fontsize=12)
            ax_emission.set_xlabel("Sparsity Level (%)", fontsize=11)

            ax_time.set_yscale("log")
            ax_time.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax_emission.grid(True, linestyle="--", linewidth=0.5)

            ax_time.legend(fontsize=9)
            ax_emission.legend(fontsize=9)

    fig.suptitle(f"{modelName} – Per-Data Inference Time & CO₂ Emissions (CPU & GPU)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir = "./plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = f"{filename}_timing_emissions_thresholds.png"
    plt.savefig(os.path.join(output_dir, plot_filename))
    plt.close()
    print(f"[Saved Plot] {plot_filename}")

def main():
    import torch
    import os
    import pandas as pd
    import glob

    torch.manual_seed(0)
    
    models = ["ResNet20", "Vgg16_ImageNet", "Vgg16_"]
    batch_sizes = [1, 32, 64]
    thresholds = [0.3, 0.5, 0.7, 0.9]  # Sparsity conversion thresholds

    results = {}

    for modelName in models:
        path_to_checkpoints = f"../../../../plots/LTH_output/*{modelName}*"
        allRuns = glob.glob(path_to_checkpoints)

        for oneRun in allRuns:
            print(f"[Processing] {oneRun}")
            checkpoints = get_checkpoints(oneRun, prefix="checkpoint_Pruned_")

            if not checkpoints:
                print(f"[Warning] No checkpoints found for {modelName} at {path_to_checkpoints}")
                continue

            sparsity_levels = [s for s, _ in checkpoints]
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))

            # Run evaluation across thresholds
            threshold_results = evaluate_models_for_all_configs(
                modelName=modelName,
                checkpoints=checkpoints,
                batch_sizes=batch_sizes,
                devices=devices,
                sparsity_levels=sparsity_levels,
                thresholds=thresholds
            )

            results[modelName] = {
                "sparsity_levels": sparsity_levels,
                "batch_sizes": batch_sizes,
                "threshold_results": threshold_results
            }

            filename = checkpoints[0][1].split("/")[-2]

            # Plot timing and emissions across thresholds
            plot_timing_and_emissions_separate_rows(
                modelName, sparsity_levels, batch_sizes, threshold_results, filename, thresholds
            )

            # Save combined CSV
            all_records = []
            for threshold in thresholds:
                all_timings = threshold_results[threshold]["timings"]
                all_emissions = threshold_results[threshold]["emissions"]

                for device_type in all_timings:
                    for batch_size in all_timings[device_type]:
                        timing_data = all_timings[device_type][batch_size]
                        emission_data = all_emissions[device_type][batch_size]

                        for i, sparsity in enumerate(sparsity_levels):
                            all_records.append({
                                "Threshold": threshold,
                                "Device": device_type,
                                "Batch Size": batch_size,
                                "Sparsity": sparsity,
                                "Unpruned Dense Time": timing_data["unpruned_dense"],
                                "Pruned Dense Time": timing_data["pruned_dense"][i],
                                "Pruned Sparse Time": timing_data["pruned_sparse"][i],
                                "Unpruned Dense Emissions": emission_data["unpruned_dense"],
                                "Pruned Dense Emissions": emission_data["pruned_dense"][i],
                                "Pruned Sparse Emissions": emission_data["pruned_sparse"][i],
                            })

            df = pd.DataFrame(all_records)
            # make a csv dir and save all csv to it
            os.makedirs("csv", exist_ok=True)
            # Save to CSV
            csv_path = f"{filename}_timings_emissions_thresholds.csv"
            csv_path = os.path.join("csv", csv_path)
            df.to_csv(csv_path, index=False)
            print(f"[Saved CSV] {csv_path}")

            print(f"[Completed] {modelName} evaluation and plotting.")

    return results


if __name__ == "__main__":
    torch.manual_seed(6991)
    main()
