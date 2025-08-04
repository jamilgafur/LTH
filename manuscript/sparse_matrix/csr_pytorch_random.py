import torch
import matplotlib.pyplot as plt
import os
import re
from SparseConv2d import SparseConv2d
from SparseLinear import SparseLinear
from exampleModel import ExampleModel
from util import *
from measure import *
from pyPrune.models.LeNet import LeNet


def extract_sparsity_from_filename(filename, prefix="checkpoint_Pruned_"):
    """Extract sparsity value from checkpoint filename."""
    pattern = rf"{prefix}(\d+\.\d+).pth"
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None


def get_checkpoints(path, prefix="checkpoint_Pruned_"):
    """Return sorted list of (sparsity, full_path) tuples for checkpoints matching prefix."""
    files = [f for f in os.listdir(path) if f.startswith(prefix) and f.endswith(".pth")]
    checkpoints = [(extract_sparsity_from_filename(f, prefix), os.path.join(path, f)) for f in files]
    checkpoints = [ckpt for ckpt in checkpoints if ckpt[0] is not None]
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def load_model_from_checkpoint(model_cls, checkpoint_path: str, device: torch.device, use_compile: bool = True):
    model = model_cls()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict if wrapped in 'model'
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if use_compile:
        model = torch.compile(model)  # PyTorch 2.0+ speedup

    return model


def evaluate_model_performance(model: torch.nn.Module, device: torch.device, batch_size: int = 32):
    x = torch.randn(batch_size, 1, 32, 32, device=device)

    mem = measure_model_memory_by_device(model)

    with torch.no_grad():
        # No mixed precision, run directly
        time_s, out, energy = measure_inference_time(model, x, device)

    return mem, time_s, energy, out


def initialize_results_dict(devices, batch_sizes):
    """Initialize nested dicts to store memory, time, energy results per batch size."""
    mem_results = {
        d.type: {bs: {'cpu_dense': [], 'gpu_dense': [], 'cpu_sparse': [], 'gpu_sparse': []} for bs in batch_sizes} for d in devices
    }
    time_results = {
        d.type: {bs: {'dense': [], 'sparse': []} for bs in batch_sizes} for d in devices
    }
    energy_results = {
        d.type: {bs: {'dense': [], 'sparse': []} for bs in batch_sizes} for d in devices
    }
    return mem_results, time_results, energy_results


def plot_results(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes):
    n_metrics = 6
    n_batches = len(batch_sizes)

    fig, axs = plt.subplots(n_metrics, n_batches, figsize=(5 * n_batches, 30), squeeze=False)
    # axs shape: (6, n_batches)

    metric_titles = [
        'Memory Usage',
        'Memory Usage',
        'Inference Time',
        'Total Inference Energy (Joules)',
        'Energy per Sample (Joules/sample)',
        'Average Power During Inference (W)'
    ]
    ylabels = [
        'Memory (MB)',
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
            
            # CPU memory usage (top row)
            if device_type == 'cpu':
                axs[0, col].plot(
                    sparsity_levels, mem_results[device_type][bs]['cpu_dense'],
                    label=f'Dense CPU', marker=markers['dense'], linestyle=line_styles['dense'], color=color
                )
                axs[0, col].plot(
                    sparsity_levels, mem_results[device_type][bs]['cpu_sparse'],
                    label=f'Sparse CPU', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
                )
            
            # GPU memory usage (second row)
            if device_type == 'cuda':
                axs[1, col].plot(
                    sparsity_levels, mem_results[device_type][bs]['gpu_dense'],
                    label=f'Dense GPU', marker=markers['dense'], linestyle=line_styles['dense'], color=color
                )
                axs[1, col].plot(
                    sparsity_levels, mem_results[device_type][bs]['gpu_sparse'],
                    label=f'Sparse GPU', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
                )
            
            # Inference time (third row)
            axs[2, col].plot(
                sparsity_levels, time_results[device_type][bs]['dense'],
                label=f'{device_type} Dense', marker=markers['dense'], linestyle=line_styles['dense'], color=color
            )
            axs[2, col].plot(
                sparsity_levels, time_results[device_type][bs]['sparse'],
                label=f'{device_type} Sparse', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
            )
            
            # Total energy (fourth row)
            axs[3, col].plot(
                sparsity_levels, energy_results[device_type][bs]['dense'],
                label=f'{device_type} Dense', marker=markers['dense'], linestyle=line_styles['dense'], color=color
            )
            axs[3, col].plot(
                sparsity_levels, energy_results[device_type][bs]['sparse'],
                label=f'{device_type} Sparse', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
            )
            
            # Energy per sample (fifth row)
            energy_per_sample_dense = [e / bs if bs > 0 else 0 for e in energy_results[device_type][bs]['dense']]
            energy_per_sample_sparse = [e / bs if bs > 0 else 0 for e in energy_results[device_type][bs]['sparse']]
            axs[4, col].plot(
                sparsity_levels, energy_per_sample_dense,
                label=f'{device_type} Dense', marker=markers['dense'], linestyle=line_styles['dense'], color=color
            )
            axs[4, col].plot(
                sparsity_levels, energy_per_sample_sparse,
                label=f'{device_type} Sparse', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
            )
            
            # Average power (energy/time) (sixth row)
            eff_dense = [
                e / t if t > 0 else 0 for e, t in zip(energy_results[device_type][bs]['dense'], time_results[device_type][bs]['dense'])
            ]
            eff_sparse = [
                e / t if t > 0 else 0 for e, t in zip(energy_results[device_type][bs]['sparse'], time_results[device_type][bs]['sparse'])
            ]
            axs[5, col].plot(
                sparsity_levels, eff_dense,
                label=f'{device_type} Dense', marker=markers['dense'], linestyle=line_styles['dense'], color=color
            )
            axs[5, col].plot(
                sparsity_levels, eff_sparse,
                label=f'{device_type} Sparse', marker=markers['sparse'], linestyle=line_styles['sparse'], color=color
            )
        
        # Set titles, labels, and grid
        axs[0, col].set_title(f'Batch size: {bs}', fontsize=12, fontweight='bold')
        for row in range(n_metrics):
            axs[row, col].set_xlabel('Sparsity', fontsize=10)
            axs[row, col].set_ylabel(ylabels[row], fontsize=10)
            axs[row, col].grid(True, linestyle='--', alpha=0.6)
            
            # Only add legend on the first column to avoid clutter
            if col == 0:
                axs[row, col].legend(fontsize=8, loc='best')
            
            # Optional: tighten y-axis limits or set reasonable limits if needed
            # axs[row, col].set_ylim(bottom=0)
  
    plt.tight_layout()
    plt.savefig('model_performance_vs_sparsity_batch_sizes_grid.png')
    print("Saved: model_performance_vs_sparsity_batch_sizes_grid.png")


def main():
    torch.manual_seed(0)

    path_to_checkpoints = "../structured_study/pruning_checkpoints/LeNet_pretrain1_finetune1_steps21_batch2048_devicecuda_strategy_magnitude"
    filename = path_to_checkpoints.split('/')[-1] + "_sparsity_vs_batch_sizes.png"
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    checkpoints = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")
    sparsity_levels = [s for s, _ in checkpoints]

    batch_sizes = [1, 128, 1024]  # Batch sizes to evaluate

    mem_results, time_results, energy_results = initialize_results_dict(devices, batch_sizes)

    for device in devices:
        print(f"\nRunning on {device}\n" + "=" * 60)
        for batch_size in batch_sizes:
            print(f"\nBatch size: {batch_size}\n" + "-" * 40)
            for sparsity, ckpt_path in checkpoints:
                print(f"  Sparsity: {sparsity:.2f} | Loading: {os.path.basename(ckpt_path)}")

                model = load_model_from_checkpoint(LeNet, ckpt_path, device, use_compile=True)

                mem, inf_time, energy, _ = evaluate_model_performance(model, device, batch_size=batch_size)

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

    plot_results(sparsity_levels, mem_results, time_results, energy_results, devices, batch_sizes, filename)


if __name__ == "__main__":
    main()
