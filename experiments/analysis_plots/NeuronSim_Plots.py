import os
import gc
import glob
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cosine

from pyPrune.utils import get_pruneable_named_modules, clean_memory, plot_loss_accuracy_sparsity

# === Global Plot Settings ===
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# === Utility Functions ===
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_json(file_path):
    print(f"Processing: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def poly_lr_with_warmup(args, epoch):
    warmup_epochs = args.pretrain_epochs // 10
    total_epochs = args.pretrain_epochs + args.finetune_epochs
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    decay_progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return (1 - decay_progress) ** 2

def lr_lambda(args, epoch):
    pct = epoch / (args.pretrain_epochs + args.finetune_epochs)
    return 1.0 if pct < 0.5 else 0.1 if pct < 0.75 else 0.01

# === Grouping Functions ===
def group_data_by_layername(data_dict):
    grouped = defaultdict(list)
    for name, vals in data_dict.items():
        key = name.split('.')[1] + "-" + name.split('.')[3] if "stage" in name else name.split('.')[0]
        grouped[key].extend(vals)
    return grouped

def aggregate_similarity(sim_dict):
    step_dict = defaultdict(list)
    for values in sim_dict.values():
        for step, sim in values:
            step_dict[step].append(sim)
    steps = sorted(step_dict)
    return steps, [np.mean(step_dict[s]) for s in steps], [np.percentile(step_dict[s], 25) for s in steps], [np.percentile(step_dict[s], 75) for s in steps]

# === Plotting Functions ===
def plot_similarity(data_dict, title, ylabel, save_path):
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("tab10", n_colors=len(data_dict))
    linestyles = ['-', '--', '-.', ':']
    for i, (layer, sims) in enumerate(sorted(data_dict.items())):
        steps, vals = zip(*sims)
        plt.plot(steps, vals, marker='o', linestyle=linestyles[i % len(linestyles)],
                 color=palette[i], label=layer)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Layer")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_aggregated_similarity(steps, avg, q25, q75, title, ylabel, save_path):
    plt.figure(figsize=(12, 8))
    plt.plot(steps, avg, marker='o', label="Average", color='tab:blue')
    plt.fill_between(steps, q25, q75, color='tab:blue', alpha=0.3, label="25-75% Percentile")
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300)
    plt.close()

# === Processing Functions ===
def compute_cosine_similarity_pairs(data):
    result = []
    for i in range(1, len(data)):
        step = data[i][0]
        vec1 = data[i - 1][1].cpu().numpy().flatten()
        vec2 = data[i][1].cpu().numpy().flatten()
        result.append((step, 1 - cosine(vec1, vec2 + 1e-10)))
    return result

def compute_similarity_to_baseline(data):
    baseline = data[0][1].cpu().numpy().flatten()
    return [(step, 1 - cosine(baseline, act.cpu().numpy().flatten() + 1e-10)) for step, act in data]

def process_neuron_similarity(neuron_dir, model_name):
    files = glob.glob(os.path.join(neuron_dir, "*.pkl"))
    if not files:
        print(f"No neuron similarity files found in {neuron_dir}")
        return

    checkpoint = os.path.basename(os.path.normpath(os.path.dirname(neuron_dir)))
    merged = defaultdict(list)
    for f in files:
        print(f"Loading {f}")
        pruner = load_pickle(f)
        for layer, steps in pruner.activations_step.items():
            merged[layer].extend(steps)

    for k in merged:
        merged[k] = sorted(merged[k], key=lambda x: x[0])

    cosine_sim = {layer: compute_cosine_similarity_pairs(data) for layer, data in merged.items()}
    cosine_sim = group_data_by_layername(cosine_sim)

    save_dir_prev = os.path.join("plots", checkpoint, "activation_similarity")
    os.makedirs(save_dir_prev, exist_ok=True)
    plot_similarity(cosine_sim, "Cosine Similarity of Activations Across Steps", "Cosine Similarity",
                    os.path.join(save_dir_prev, "cosine_similarity_across_steps.svg"))

    baseline_sim = {layer: compute_similarity_to_baseline(data) for layer, data in merged.items()}
    baseline_sim = group_data_by_layername(baseline_sim)

    save_dir_base = os.path.join("plots", model_name, checkpoint, "baseline_similarity")
    os.makedirs(save_dir_base, exist_ok=True)
    plot_similarity(baseline_sim, "Cosine Similarity w.r.t. Prepruning", "Cosine Similarity",
                    os.path.join(save_dir_base, "baseline_cosine_similarity.svg"))

    steps, avg, q25, q75 = aggregate_similarity(cosine_sim)
    plot_aggregated_similarity(steps, avg, q25, q75,
        "Aggregated Cosine Similarity with Previous Step", "Cosine Similarity",
        os.path.join(save_dir_prev, "aggregated_previous_similarity.svg"))

    steps, avg, q25, q75 = aggregate_similarity(baseline_sim)
    plot_aggregated_similarity(steps, avg, q25, q75,
        "Aggregated Cosine Similarity with Prepruning", "Cosine Similarity",
        os.path.join(save_dir_base, "aggregated_prepruning_similarity.svg"))

    print(f"Neuron similarity processing complete for model: {model_name}, checkpoint: {checkpoint}")

def process_maxneuron_similarity(neuron_dir, model_name):
    files = glob.glob(os.path.join(neuron_dir, "*.pkl"))
    if not files:
        print(f"No max neuron similarity files found in {neuron_dir}")
        return

    pruner = load_pickle(files[0])
    checkpoint = os.path.basename(os.path.normpath(os.path.dirname(neuron_dir)))
    save_dir = os.path.join("plots", model_name, checkpoint, "activation_similarity")
    os.makedirs(save_dir, exist_ok=True)

    max_sims = defaultdict(list)
    for layer, data in pruner.activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        for step, act in data_sorted:
            act = act.cpu().double().view(act.size(1), -1) + 1e-10
            act_normed = act / act.norm(dim=1, keepdim=True)
            sim_matrix = torch.abs(act_normed @ act_normed.T)
            sim_matrix.fill_diagonal_(0)
            max_sims[layer].append((step, torch.max(sim_matrix, dim=1)[0]))

    # Plot histograms for selected steps
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    steps = [0, 3, 14]
    labels = ['0%', '50%', '95%']

    for i, step_idx in enumerate(steps):
        combined = []
        for sims in max_sims.values():
            if step_idx < len(sims):
                combined.append(sims[step_idx][1])
        if combined:
            data = torch.cat(combined)
            axs[i].hist(data.numpy(), bins=20, color='skyblue', edgecolor='black')
            axs[i].set_title(f"Sparsity {labels[i]} (Step {step_idx})")
            axs[i].set_xlabel("Max Cosine Similarity")
            axs[i].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "maximum_cosine_similarity.png"))
    plt.close()

    print(f"Max neuron similarity plot complete for model: {model_name}, checkpoint: {checkpoint}")

# === Entry Point ===
def main():
    checkpoint_glob = "/scratch/jgafur/LTH_output/*LeNet_pretrain20_finetune5_steps21_batch128_devicecuda_strategy_magnitud*"
    for output_dir in glob.glob(checkpoint_glob):
        clear_memory()
        print(f"Processing directory: {output_dir}")
        sim_dir = os.path.join(output_dir, "neuron_similarity")
        process_neuron_similarity(sim_dir, model_name='')
        process_maxneuron_similarity(sim_dir, model_name='')

if __name__ == "__main__":
    main()
