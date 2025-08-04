import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from SparseConv2d import SparseConv2d
from SparseLinear import SparseLinear
from exampleModel import ExampleModel
from util import *
import glob
from measure import *
from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16ImageNet import VGG16_ImageNet
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from pyPrune.models.RegNetX import RegNetX_400MF


def evaluate_model_performance(model: torch.nn.Module, device: torch.device, batch_size: int = 32, modelName: str = "ResNet20"):
    if modelName == "ResNet20":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "Vgg16_ImageNet":
        x = torch.randn(batch_size, 3, 224, 224, device=device)
    elif modelName == "Vgg16_":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "LeNet":
        x = torch.randn(batch_size, 1, 32, 32, device=device)
    elif modelName == "RegNetX":
        x = torch.randn(batch_size, 3, 224, 224, device=device)
    else:
        raise ValueError(f"Unknown model type: {modelName}")

    mem = measure_model_memory_by_device(model)

    with torch.no_grad():
        time_s, out, energy = measure_inference_time(model, x, device)

    return mem, time_s, energy, out


def plot_results(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes, filename):
    n_metrics = 6
    n_batches = len(batch_sizes)

    fig, axs = plt.subplots(n_metrics, n_batches, figsize=(5 * n_batches, 30), squeeze=False)

    metric_titles = [
        'Memory Usage (% of Sparse vs Dense)',
        'GPU Memory Usage (MB)',
        'Inference Time',
        'Total Inference Energy (Joules)',
        'Energy per Sample (Joules/sample)',
        'Average Power During Inference (W)'
    ]
    ylabels = [
        'Memory Usage (%)',
        'Memory (MB)',
        'Time (s)',
        'Energy (J)',
        'Energy/sample (J)',
        'Watts (J/s)'
    ]

    colors = {'cpu': 'tab:blue', 'cuda': 'tab:orange'}
    line_styles = {'dense': '-', 'sparse': '--'}
    markers = {'dense': 'o', 'sparse': 'x'}

    for col, bs in enumerate(batch_sizes):
        for device in devices:
            device_type = device.type
            color = colors.get(device_type, 'gray')

            cpu_dense = mem_results[device_type][bs]['cpu_dense']
            cpu_sparse = mem_results[device_type][bs]['cpu_sparse']
            gpu_dense = mem_results[device_type][bs]['gpu_dense']
            gpu_sparse = mem_results[device_type][bs]['gpu_sparse']

            epsilon = 1e-6
            if device_type == 'cpu':
                cpu_mem_percent = [100 * s / (d + epsilon) if d > 0 else 0 for s, d in zip(cpu_sparse, cpu_dense)]
                axs[0, col].plot(
                    sparsity_levels, cpu_mem_percent,
                    label='CPU Sparse % vs Dense', marker=markers['sparse'],
                    linestyle=line_styles['sparse'], color=colors['cpu']
                )
            if device_type == 'cuda':
                gpu_mem_percent = [100 * s / (d + epsilon) if d > 0 else 0 for s, d in zip(gpu_sparse, gpu_dense)]
                axs[0, col].plot(
                    sparsity_levels, gpu_mem_percent,
                    label='GPU Sparse % vs Dense', marker=markers['sparse'],
                    linestyle=line_styles['sparse'], color=colors['cuda']
                )

            if device_type == 'cuda':
                axs[1, col].plot(
                    sparsity_levels, gpu_dense,
                    label='Dense GPU', marker=markers['dense'],
                    linestyle=line_styles['dense'], color=colors['cuda']
                )
                axs[1, col].plot(
                    sparsity_levels, gpu_sparse,
                    label='Sparse GPU', marker=markers['sparse'],
                    linestyle=line_styles['sparse'], color=colors['cuda']
                )

            axs[2, col].plot(
                sparsity_levels, time_results[device_type][bs]['dense'],
                label=f'{device_type} Dense', marker=markers['dense'],
                linestyle=line_styles['dense'], color=color
            )
            axs[2, col].plot(
                sparsity_levels, time_results[device_type][bs]['sparse'],
                label=f'{device_type} Sparse', marker=markers['sparse'],
                linestyle=line_styles['sparse'], color=color
            )

            axs[3, col].plot(
                sparsity_levels, energy_results[device_type][bs]['dense'],
                label=f'{device_type} Dense', marker=markers['dense'],
                linestyle=line_styles['dense'], color=color
            )
            axs[3, col].plot(
                sparsity_levels, energy_results[device_type][bs]['sparse'],
                label=f'{device_type} Sparse', marker=markers['sparse'],
                linestyle=line_styles['sparse'], color=color
            )

            energy_per_sample_dense = [e / bs if bs > 0 else 0 for e in energy_results[device_type][bs]['dense']]
            energy_per_sample_sparse = [e / bs if bs > 0 else 0 for e in energy_results[device_type][bs]['sparse']]
            axs[4, col].plot(
                sparsity_levels, energy_per_sample_dense,
                label=f'{device_type} Dense', marker=markers['dense'],
                linestyle=line_styles['dense'], color=color
            )
            axs[4, col].plot(
                sparsity_levels, energy_per_sample_sparse,
                label=f'{device_type} Sparse', marker=markers['sparse'],
                linestyle=line_styles['sparse'], color=color
            )

            power_dense = [
                e / t if t > 0 else 0
                for e, t in zip(energy_results[device_type][bs]['dense'], time_results[device_type][bs]['dense'])
            ]
            power_sparse = [
                e / t if t > 0 else 0
                for e, t in zip(energy_results[device_type][bs]['sparse'], time_results[device_type][bs]['sparse'])
            ]
            axs[5, col].plot(
                sparsity_levels, power_dense,
                label=f'{device_type} Dense', marker=markers['dense'],
                linestyle=line_styles['dense'], color=color
            )
            axs[5, col].plot(
                sparsity_levels, power_sparse,
                label=f'{device_type} Sparse', marker=markers['sparse'],
                linestyle=line_styles['sparse'], color=color
            )

        axs[0, col].set_title(f'Batch size: {bs}', fontsize=12, fontweight='bold')
        for row in range(n_metrics):
            axs[row, col].set_xlabel('Sparsity', fontsize=10)
            axs[row, col].set_ylabel(ylabels[row], fontsize=10)
            axs[row, col].grid(True, linestyle='--', alpha=0.6)
            if col == 0:
                axs[row, col].legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")


def main():
    torch.manual_seed(0)
    models = ["ResNet20", "Vgg16_ImageNet", "Vgg16_", "RegNetX"]
    for modelName in models:
        path_to_checkpoints = f"../../../../plots/LTH_output/*{modelName}*"
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))

        checkpoints = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")

        if not checkpoints:
            print(f"No checkpoints found for {modelName} at {path_to_checkpoints}")
            continue

        filename = checkpoints[0][1].split('/')[-2] + "_sparsity_vs_batch_sizes.png"

        sparsity_levels = [s for s, _ in checkpoints]

        batch_sizes = [1]  # Batch sizes to evaluate

        mem_results, time_results, energy_results = initialize_results_dict(devices, batch_sizes)

        for device in devices:
            print(f"\nRunning on {device}\n" + "=" * 60)
            for batch_size in batch_sizes:
                print(f"\nBatch size: {batch_size}\n" + "-" * 40)
                for sparsity, ckpt_path in checkpoints:
                    print(f"  Sparsity: {sparsity:.2f} | Loading: {os.path.basename(ckpt_path)}")
                    modelType = getModelType(modelName)
                    model = load_model_from_checkpoint(modelType, ckpt_path, device, use_compile=True)

                    mem, inf_time, energy, _ = evaluate_model_performance(model, device, batch_size=batch_size, modelName=modelName)

                    log_memory(mem, device.type, "Sparse")

                    # Store sparse results
                    mem_results[device.type][batch_size]['cpu_sparse'].append(mem.get('cpu', 0) / (1024 ** 2))
                    mem_results[device.type][batch_size]['gpu_sparse'].append(mem.get('cuda', 0) / (1024 ** 2))
                    time_results[device.type][batch_size]['sparse'].append(inf_time)
                    energy_results[device.type][batch_size]['sparse'].append(energy)

                    # Append zeros for dense (or implement dense evaluation if desired)
                    mem_results[device.type][batch_size]['cpu_dense'].append(0)
                    mem_results[device.type][batch_size]['gpu_dense'].append(0)
                    time_results[device.type][batch_size]['dense'].append(0)
                    energy_results[device.type][batch_size]['dense'].append(0)
            
        printInfo(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes)
        plot_results(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes, filename)
        quit()

def printInfo(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes):
    print("\nMemory Results:")
    for device in devices:
        print(f"\nDevice: {device.type}")
        for bs in batch_sizes:
            print(f"  Batch Size: {bs}")
            print(f"    CPU Dense: {mem_results[device.type][bs]['cpu_dense']}")
            print(f"    CPU Sparse: {mem_results[device.type][bs]['cpu_sparse']}")
            print(f"    GPU Dense: {mem_results[device.type][bs]['gpu_dense']}")
            print(f"    GPU Sparse: {mem_results[device.type][bs]['gpu_sparse']}")

    print("\nTime Results:")
    for device in devices:
        print(f"\nDevice: {device.type}")
        for bs in batch_sizes:
            print(f"  Batch Size: {bs}")
            print(f"    Dense: {time_results[device.type][bs]['dense']}")
            print(f"    Sparse: {time_results[device.type][bs]['sparse']}")

    print("\nEnergy Results:")
    for device in devices:
        print(f"\nDevice: {device.type}")
        for bs in batch_sizes:
            print(f"  Batch Size: {bs}")
            print(f"    Dense: {energy_results[device.type][bs]['dense']}")
            print(f"    Sparse: {energy_results[device.type][bs]['sparse']}")

if __name__ == "__main__":
    main()
