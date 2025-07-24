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
    checkpoints = [(extract_sparsity_from_filename(f, prefix), os.path.join(path, f))
                   for f in files]
    # Remove None sparsity entries and sort by sparsity
    checkpoints = [ckpt for ckpt in checkpoints if ckpt[0] is not None]
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def load_model_from_checkpoint(model_cls, checkpoint_path, device):
    model = model_cls()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract the inner model weights if wrapped in a 'model' key
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    return model

def evaluate_model_performance(model, device):
    x = torch.randn(32, 1, 32, 32).to(device)  # pass the device instance

    mem = measure_model_memory_by_device(model)
    time_s, out, energy = measure_inference_time(model, x, device)

    return mem, time_s, energy, out

def initialize_results_dict(devices):
    """Initialize nested dicts to store memory, time, energy results."""
    mem_results = {d.type: {'cpu_dense': [], 'gpu_dense': [], 'cpu_sparse': [], 'gpu_sparse': []} for d in devices}
    time_results = {d.type: {'dense': [], 'sparse': []} for d in devices}
    energy_results = {d.type: {'dense': [], 'sparse': []} for d in devices}
    return mem_results, time_results, energy_results

def plot_results(sparsity_levels, mem_results, time_results, energy_results, devices):
    """Plot the results in six subplots."""
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))  # 6 subplots

    # --- Memory Plots ---
    for device in devices:
        if device.type == 'cpu':
            axs[0].plot(sparsity_levels, mem_results[device.type]['cpu_dense'], label='Dense CPU', marker='o')
            axs[0].plot(sparsity_levels, mem_results[device.type]['cpu_sparse'], label='Sparse CPU', marker='x')
    axs[0].set(title='CPU Memory Usage', xlabel='Sparsity', ylabel='Memory (MB)')
    axs[0].legend()
    axs[0].grid(True)

    for device in devices:
        if device.type == 'cuda':
            axs[1].plot(sparsity_levels, mem_results[device.type]['gpu_dense'], label='Dense GPU', marker='o')
            axs[1].plot(sparsity_levels, mem_results[device.type]['gpu_sparse'], label='Sparse GPU', marker='x')
    axs[1].set(title='GPU Memory Usage', xlabel='Sparsity', ylabel='Memory (MB)')
    axs[1].legend()
    axs[1].grid(True)

    # --- Inference Time Plots ---
    for device in devices:
        axs[2].plot(sparsity_levels, time_results[device.type]['dense'], label=f'{device} Dense', marker='o')
        axs[2].plot(sparsity_levels, time_results[device.type]['sparse'], label=f'{device} Sparse', marker='x')
    axs[2].set(title='Inference Time', xlabel='Sparsity', ylabel='Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    # --- Total Energy (Joules) ---
    for device in devices:
        axs[3].plot(sparsity_levels, energy_results[device.type]['dense'], label=f'{device} Dense', marker='o')
        axs[3].plot(sparsity_levels, energy_results[device.type]['sparse'], label=f'{device} Sparse', marker='x')
    axs[3].set(title='Total Inference Energy (Joules)', xlabel='Sparsity', ylabel='Energy (J)')
    axs[3].legend()
    axs[3].grid(True)

    # --- Energy per Sample ---
    for device in devices:
        energy_per_sample_dense = [e / 32 for e in energy_results[device.type]['dense']]
        energy_per_sample_sparse = [e / 32 for e in energy_results[device.type]['sparse']]
        axs[4].plot(sparsity_levels, energy_per_sample_dense, label=f'{device} Dense', marker='o')
        axs[4].plot(sparsity_levels, energy_per_sample_sparse, label=f'{device} Sparse', marker='x')
    axs[4].set(title='Energy per Sample (Joules/sample)', xlabel='Sparsity', ylabel='Energy/sample')
    axs[4].legend()
    axs[4].grid(True)

    # --- Energy Efficiency (Joules / second) = Power ---
    for device in devices:
        eff_dense = [e / t if t > 0 else 0 for e, t in zip(energy_results[device.type]['dense'], time_results[device.type]['dense'])]
        eff_sparse = [e / t if t > 0 else 0 for e, t in zip(energy_results[device.type]['sparse'], time_results[device.type]['sparse'])]
        axs[5].plot(sparsity_levels, eff_dense, label=f'{device} Dense', marker='o')
        axs[5].plot(sparsity_levels, eff_sparse, label=f'{device} Sparse', marker='x')
    axs[5].set(title='Average Power During Inference (W)', xlabel='Sparsity', ylabel='Watts (J/s)')
    axs[5].legend()
    axs[5].grid(True)

    plt.tight_layout()
    plt.savefig('model_performance_vs_sparsity_with_energy.png')
    print("Saved: model_performance_vs_sparsity_with_energy.png")

def main():
    torch.manual_seed(0)

    # Set your checkpoint directory here
    path_to_checkpoints = "../structured_study/pruning_checkpoints/LeNet_pretrain1_finetune1_steps21_batch2048_devicecuda_strategy_magnitude"

    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    # Load checkpoints and extract sparsity levels
    checkpoints = get_checkpoints(path_to_checkpoints, prefix="checkpoint_Pruned_")
    sparsity_levels = [s for s, _ in checkpoints]

    mem_results, time_results, energy_results = initialize_results_dict(devices)

    for device in devices:
        print(f"\nRunning on {device}\n" + "=" * 60)
        for sparsity, ckpt_path in checkpoints:
            print(f"\n--- Sparsity: {sparsity:.2f} | Loading: {os.path.basename(ckpt_path)} ---")

            model = load_model_from_checkpoint(LeNet, ckpt_path, device)

            mem, inf_time, energy, _ = evaluate_model_performance(model, device)

            log_memory(mem, device.type, "Sparse")

            # Store sparse results
            mem_results[device.type]['cpu_sparse'].append(mem.get('cpu', 0) / (1024 ** 2))
            mem_results[device.type]['gpu_sparse'].append(mem.get('cuda', 0) / (1024 ** 2))
            time_results[device.type]['sparse'].append(inf_time)
            energy_results[device.type]['sparse'].append(energy)

            # Append zeros for dense (or implement dense evaluation if desired)
            mem_results[device.type]['cpu_dense'].append(0)
            mem_results[device.type]['gpu_dense'].append(0)
            time_results[device.type]['dense'].append(0)
            energy_results[device.type]['dense'].append(0)

    plot_results(sparsity_levels, mem_results, time_results, energy_results, devices)


if __name__ == "__main__":
    main()
