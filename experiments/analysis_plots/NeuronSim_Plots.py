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

def process_model(output_dir, model_name):
    """Process model data and generate necessary plots."""
    # Find all .pkl files and load the pruner object
    files_found = glob.glob(os.path.join(output_dir, "*.pkl"))
    print(f"Files found: {files_found}")
    if not files_found:
        return

    with open(files_found[0], 'rb') as f:
        pruner = pickle.load(f)

    # Check if the pruner object contains activations information
    if not hasattr(pruner, 'activations_step'):
        print("No 'activations_step' attribute found in pruner.")
        return

    activations_step = pruner.activations_step

    # -------------------------------------------------------------------------
    # 1. Compute and plot cosine similarity between consecutive activation steps
    #    for each layer.
    # -------------------------------------------------------------------------
    cosine_similarities = defaultdict(list)
    # Each key in activations_step is a layer name and the value is a list of
    # [step, activations]. We assume steps are numerical and can be sorted.
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        steps = [item[0] for item in data_sorted]
        acts = [item[1] for item in data_sorted]
        # For every consecutive pair, compute the cosine similarity
        for i in range(1, len(acts)):
            act_prev = np.array(acts[i-1].to('cpu').detach().numpy()).flatten()
            act_curr = np.array(acts[i].to('cpu').detach().numpy()).flatten()
            # Cosine similarity: 1 - cosine distance
            cos_sim = 1 - cosine(act_prev, act_curr)
            cosine_similarities[layer].append((steps[i], cos_sim))

    # Plot cosine similarity across steps for each layer
    plt.figure(figsize=(10, 6))
    for layer, sim_data in cosine_similarities.items():
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        plt.plot(steps, sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity with Previous Step')
    plt.title('Cosine Similarity of Layer Activations Across Steps')
    plt.legend(title='Layer')
    plt.grid(True)
    os.makedirs(f"./plots/{model_name}/activation_similarity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/activation_similarity/cosine_similarity_across_steps.png")
    plt.close()

    # -------------------------------------------------------------------------
    # 2. Plot average similarity per layer over steps using stored metrics.
    #    The pruner.metrics attribute is assumed to be a dictionary where each
    #    key is a step and the value is a dict containing 'average_similarities',
    #    a list of dicts with keys 'layer_name' and 'average_similarity'.
    # -------------------------------------------------------------------------
    if not hasattr(pruner, 'metrics'):
        print("No 'metrics' attribute found in pruner.")
        return

    metrics = pruner.metrics
    avg_sim_by_layer = defaultdict(list)
    for step, step_data in metrics.items():
        for avg_data in step_data['average_similarities']:
            layer = avg_data['layer_name']
            avg_sim = avg_data['average_similarity']
            avg_sim_by_layer[layer].append((step, avg_sim))

    plt.figure(figsize=(10, 6))
    for layer, data in avg_sim_by_layer.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        steps = [x[0] for x in data_sorted]
        avg_sims = [x[1] for x in data_sorted]
        plt.plot(steps, avg_sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step')
    plt.ylabel('Average Cosine Similarity')
    plt.title('Average Cosine Similarity of Neuron Activations per Layer Over Steps')
    plt.legend(title='Layer')
    plt.grid(True)
    os.makedirs(f"./plots/{model_name}/average_similarity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/average_similarity/average_similarity_over_steps.png")
    plt.close()

    # -------------------------------------------------------------------------
    # 3. Plot heatmaps of similarity matrices for each layer at the last recorded step.
    #    The metrics are assumed to contain a key 'similarity_matrices' which is a list
    #    of dicts with 'layer_name' and 'similarity_matrix'.
    # -------------------------------------------------------------------------
    try:
        sorted_steps = sorted(metrics.keys())
    except Exception as e:
        sorted_steps = list(metrics.keys())
    last_step = sorted_steps[-1]
    last_metrics = metrics[last_step]
    for sim_dict in last_metrics['similarity_matrices']:
        layer = sim_dict['layer_name']
        sim_matrix = np.array(sim_dict['similarity_matrix'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(sim_matrix, cmap='viridis')
        plt.title(f"Similarity Matrix Heatmap for {layer} at Step {last_step}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Neuron Index")
        os.makedirs(f"./plots/{model_name}/heatmaps/", exist_ok=True)
        plt.savefig(f"./plots/{model_name}/heatmaps/heatmap_{layer}_step_{last_step}.png")
        plt.close()

    # -------------------------------------------------------------------------
    # 4. Plot histograms showing the distribution of cosine similarities for each layer.
    # -------------------------------------------------------------------------
    for layer, sim_data in cosine_similarities.items():
        sims = [x[1] for x in sim_data]
        plt.figure(figsize=(8, 6))
        plt.hist(sims, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of Cosine Similarities for {layer}")
        os.makedirs(f"./plots/{model_name}/histograms/", exist_ok=True)
        plt.savefig(f"./plots/{model_name}/histograms/histogram_{layer}.png")
        plt.close()

    print(f"Post-processing and plotting complete for model: {model_name}")

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

    axes[0].set_xlabel('Sparsity (%)')
    axes[0].set_ylabel('Zero Weights / Total Weights in Layer')
    axes[0].set_title('Zero Weights / Total Weights in Layer')
    axes[0].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Second subplot: Sparsity vs Zero Weights in Model
    for layer_name, data in layer_sparsity_data.items():
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        sorted_sparsity = [data['sparsity'][i] for i in sorted_indices]
        sorted_zero_weights_in_model = [data['zero_weights_in_model'][i] for i in sorted_indices]
        zero_weights_in_model_ratio = [zero / sum(data['zero_weights_in_layer']) for zero in sorted_zero_weights_in_model]
        axes[1].plot(sorted_sparsity, zero_weights_in_model_ratio, marker='o', linestyle='-', label=layer_name)

    axes[1].set_xlabel('Sparsity (%)')
    axes[1].set_ylabel('Zero Weights / Total Weights in Model')
    axes[1].set_title('Zero Weights / Total Weights in Model')
    axes[1].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    os.makedirs(f"./plots/{model_name}/layer_sparsity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/layer_sparsity/weights_and_sparsity_plots_sorted.png")
    plt.close()

def main():
    """Main function to execute the entire post-processing pipeline."""
    model_names = ["LeNet", "ResNet20", "Vgg16"]
    pretrain = "3"
    finetune = "3"
    steps = "5"
    batch = "128"
    for model_name in model_names:
        output_dir = f"/scratch/jgafur/LTH_output/{model_name}_pretrain{pretrain}_finetune{finetune}_steps{steps}_batch{batch}_devicecuda/neuron_similarity/"
        process_model(output_dir, model_name)
        # If layer sparsity data is available in pruner (e.g., pruner.layer_sparsity_data),
        # you can uncomment and use the following line:
        # plot_layer_sparsity(pruner.layer_sparsity_data, model_name)

if __name__ == "__main__":
    main()
