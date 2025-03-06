import os
import torch
import pickle
import glob
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pyPrune.utils import get_pruneable_named_modules
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine
import seaborn as sns

#############################################
# Utility Functions for Loading Metrics
#############################################

def load_metric(pkl_file):
    """Load metrics from a pickle file."""
    try:
        with open(pkl_file, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Failed to load metrics from {pkl_file}: {e}")
        return {}

def load_json(json_file):
    """Load metrics from a JSON checkpoint file."""
    print(f"Processing: {json_file}")
    with open(json_file, 'r') as file:
        return json.load(file)

#############################################
# Plotting Functions for Accuracy & Loss
#############################################

def plot_accuracy_and_loss(json_data, model_name):
    """Plot accuracy and loss over sparsity steps, including separate plots for each metric."""
    metrics = json_data['overall_metrics']
    accuracy = metrics.get('accuracy', [])
    loss = metrics.get('loss', [])
    sparsity = metrics.get('step', [])

    # Combined dual-axis plot (Accuracy & Loss)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax1.set_xlabel('Sparsity', fontsize=14)
    ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=14)
    ax1.plot(sparsity, accuracy, color='tab:blue', label='Accuracy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red', fontsize=14)
    ax2.plot(sparsity, loss, color='tab:red', label='Loss', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    plt.title('Accuracy and Loss over Sparsity Steps', fontsize=16, pad=20)
    fig.tight_layout()
    os.makedirs(f"./plots/{model_name}/layer_sparsity/", exist_ok=True)
    combined_path = os.path.join(f"./plots/{model_name}/layer_sparsity/", "accuracy_and_loss_plot.png")
    plt.savefig(combined_path, dpi=300)
    plt.close()

    # Separate plot: Accuracy vs Step
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity, accuracy, marker='o', linestyle='-', color='tab:blue', label='Accuracy')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Accuracy vs Step', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    accuracy_path = os.path.join(f"./plots/{model_name}/layer_sparsity/", "accuracy_vs_step.png")
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    # Separate plot: Loss vs Step
    plt.figure(figsize=(10, 6))
    plt.plot(sparsity, loss, marker='o', linestyle='-', color='tab:red', label='Loss')
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss vs Step', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    loss_path = os.path.join(f"./plots/{model_name}/layer_sparsity/", "loss_vs_step.png")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return metrics

#############################################
# Plotting Functions for Layer Sparsity
#############################################

def plot_layer_sparsity(layer_sparsity_data, model_name):
    """Generate and save layer sparsity plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot: Sparsity vs Zero Weights in Layer
    for layer_name, data in layer_sparsity_data.items():
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        sorted_sparsity = [data['sparsity'][i] for i in sorted_indices]
        sorted_zero_weights_in_layer = [data['zero_weights_in_layer'][i] for i in sorted_indices]
        sorted_total_weights_in_layer = [data['total_weights_in_layer'][i] for i in sorted_indices]
        zero_weights_in_layer_ratio = [zero / total for zero, total in zip(sorted_zero_weights_in_layer, sorted_total_weights_in_layer)]
        axes[0].plot(sorted_sparsity, zero_weights_in_layer_ratio, marker='o', linestyle='-', label=layer_name)

    axes[0].set_xlabel('Sparsity (%)', fontsize=12)
    axes[0].set_ylabel('Zero Weights / Total Weights in Layer', fontsize=12)
    axes[0].set_title('Zero Weights / Total Weights in Layer', fontsize=14)
    axes[0].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot: Sparsity vs Zero Weights in Model
    for layer_name, data in layer_sparsity_data.items():
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        sorted_sparsity = [data['sparsity'][i] for i in sorted_indices]
        sorted_zero_weights_in_model = [data['zero_weights_in_model'][i] for i in sorted_indices]
        zero_weights_in_model_ratio = [zero / sum(data['zero_weights_in_layer']) for zero in sorted_zero_weights_in_model]
        axes[1].plot(sorted_sparsity, zero_weights_in_model_ratio, marker='o', linestyle='-', label=layer_name)

    axes[1].set_xlabel('Sparsity (%)', fontsize=12)
    axes[1].set_ylabel('Zero Weights / Total Weights in Model', fontsize=12)
    axes[1].set_title('Zero Weights / Total Weights in Model', fontsize=14)
    axes[1].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(f"./plots/{model_name}/layer_sparsity/", exist_ok=True)
    sparsity_path = os.path.join(f"./plots/{model_name}/layer_sparsity/", "weights_and_sparsity_plots_sorted.png")
    plt.savefig(sparsity_path)
    plt.close()

#############################################
# Neuron Similarity Analysis Functions
#############################################

def process_neuron_similarity(output_dir, model_name):
    """
    Process neuron similarity data and generate plots:
      1. Cosine similarity between consecutive activation steps.
      2. Cosine similarity of layer activations compared to the smallest (baseline) step.
      3. Heatmaps of similarity matrices for each layer at all recorded steps.
      4. Histograms of cosine similarity distributions.
    """
    files_found = glob.glob(os.path.join(output_dir, "*.pkl"))
    print(f"Files found for neuron similarity: {files_found}")
    if not files_found:
        return

    with open(files_found[0], 'rb') as f:
        pruner = pickle.load(f)

    if not hasattr(pruner, 'activations_step'):
        print("No 'activations_step' attribute found in pruner for neuron similarity.")
        return

    activations_step = pruner.activations_step

    # 1. Compute cosine similarity between consecutive activation steps for each layer.
    cosine_similarities = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        steps = [item[0] for item in data_sorted]
        acts = [item[1] for item in data_sorted]
        for i in range(1, len(acts)):
            act_prev = np.array(acts[i-1].to('cpu').detach().numpy()).flatten()
            act_curr = np.array(acts[i].to('cpu').detach().numpy()).flatten()
            cos_sim = 1 - cosine(act_prev, act_curr)
            cosine_similarities[layer].append((steps[i], cos_sim))

    plt.figure(figsize=(10, 6))
    for layer, sim_data in cosine_similarities.items():
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        plt.plot(steps, sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cosine Similarity with Previous Step', fontsize=12)
    plt.title('Cosine Similarity of Layer Activations Across Steps', fontsize=14)
    plt.legend(title='Layer', fontsize=10)
    plt.grid(True)
    os.makedirs(f"./plots/{model_name}/activation_similarity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/activation_similarity/cosine_similarity_across_steps.png")
    plt.close()

    # 2. Plot cosine similarity of layer activations compared to the smallest (baseline) step.
    baseline_cosine_similarity_by_layer = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        if not data_sorted:
            continue
        baseline_activation = np.array(data_sorted[0][1].to('cpu').detach().numpy()).flatten()
        for step, act in data_sorted:
            current_activation = np.array(act.to('cpu').detach().numpy()).flatten()
            cos_sim = 1 - cosine(baseline_activation, current_activation)
            baseline_cosine_similarity_by_layer[layer].append((step, cos_sim))
    
    plt.figure(figsize=(10, 6))
    for layer, data in baseline_cosine_similarity_by_layer.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        steps = [x[0] for x in data_sorted]
        sims = [x[1] for x in data_sorted]
        plt.plot(steps, sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cosine Similarity with Baseline', fontsize=12)
    plt.title('Cosine Similarity of Layer Activations Compared to the Smallest Step', fontsize=14)
    plt.legend(title='Layer', fontsize=10)
    plt.grid(True)
    os.makedirs(f"./plots/{model_name}/baseline_similarity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/baseline_similarity/baseline_cosine_similarity.png")
    plt.close()

    # 3. Plot heatmaps of similarity matrices for each layer at all recorded steps.
    if hasattr(pruner, 'metrics'):
        metrics = pruner.metrics
        for step in sorted(metrics.keys()):
            step_metrics = metrics[step]
            for sim_dict in step_metrics['similarity_matrices']:
                layer = sim_dict['layer_name']
                sim_matrix = np.array(sim_dict['similarity_matrix'])
                plt.figure(figsize=(8, 6))
                sns.heatmap(sim_matrix, cmap='viridis')
                plt.title(f"Similarity Matrix Heatmap for {layer} at Step {step}", fontsize=14)
                plt.xlabel("Neuron Index", fontsize=12)
                plt.ylabel("Neuron Index", fontsize=12)
                os.makedirs(f"./plots/{model_name}/heatmaps/", exist_ok=True)
                plt.savefig(f"./plots/{model_name}/heatmaps/heatmap_{layer}_step_{step}.png")
                plt.close()
    else:
        print("Skipping heatmap plotting as 'metrics' attribute is missing.")

    # 4. Plot histograms of cosine similarity distributions for each layer.
    for layer, sim_data in cosine_similarities.items():
        sims = [x[1] for x in sim_data]
        plt.figure(figsize=(8, 6))
        plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Cosine Similarity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f"Distribution of Cosine Similarities for {layer}", fontsize=14)
        os.makedirs(f"./plots/{model_name}/histograms/", exist_ok=True)
        plt.savefig(f"./plots/{model_name}/histograms/histogram_{layer}.png")
        plt.close()

    print(f"Neuron similarity post-processing complete for model: {model_name}")

#############################################
# Sparsity and Accuracy/Loss Analysis Functions
#############################################

def process_sparsity_and_accuracy(output_dir, model_name):
    """
    Process sparsity data from model checkpoints and generate plots:
      - Layer sparsity plots.
      - Accuracy and loss plots from JSON checkpoints.
    """
    pkl_files = glob.glob(os.path.join(output_dir, "*.pkl"))
    if not pkl_files:
        print("No .pkl files found for sparsity processing.")
        return

    with open(pkl_files[0], 'rb') as f:
        pruner = pickle.load(f)
    pruner.logger = None

    paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pth')]

    layer_sparsity_data = defaultdict(lambda: {
        'sparsity': [],
        'zero_weights_in_layer': [],
        'total_weights_in_layer': [],
        'zero_weights_in_model': []
    })

    model = pruner.model
    total_weights_data = []

    for path in tqdm(paths, desc="Processing model checkpoints for sparsity"):
        sparsity = path.split('_')[-1][:-4]
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        names, modules = get_pruneable_named_modules(model, pruner.prunable_layers)
        total_weights_in_model = sum(module.weight.numel() for module in modules)
        total_weights_data.append(total_weights_in_model)
        for name, module in zip(names, modules):
            zero_weights_in_layer = torch.sum(module.weight.data == 0).item()
            total_weights_in_layer = module.weight.numel()
            layer_sparsity_data[name]['sparsity'].append(float(sparsity))
            layer_sparsity_data[name]['zero_weights_in_layer'].append(zero_weights_in_layer)
            layer_sparsity_data[name]['total_weights_in_layer'].append(total_weights_in_layer)
            layer_sparsity_data[name]['zero_weights_in_model'].append(zero_weights_in_layer)

    plot_layer_sparsity(layer_sparsity_data, model_name)

    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if json_files:
        json_data = load_json(json_files[0])
        metrics = plot_accuracy_and_loss(json_data, model_name)
    else:
        print("No JSON checkpoint found for accuracy and loss plotting.")

#############################################
# Main Function
#############################################

def main():
    """Main function to execute the entire post-processing pipeline."""
    model_names = ["LeNet"]#, "ResNet20", "Vgg16"]
    pretrain = "*"
    finetune = "*"
    steps = "21"
    batch = "*"
    
    for model_name in model_names:
        for output_dir in glob.glob(f"/scratch/jgafur/LTH_output/{model_name}_pretrain{pretrain}_finetune{finetune}_steps{steps}_batch{batch}_devicecuda/"):
            neuron_similarity_dir = os.path.join(output_dir, "neuron_similarity")
            process_neuron_similarity(neuron_similarity_dir, model_name)
            process_sparsity_and_accuracy(output_dir, model_name)
            break

if __name__ == "__main__":
    main()
