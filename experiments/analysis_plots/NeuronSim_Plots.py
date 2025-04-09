import os
import torch
import pickle
import glob
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pyPrune.utils import get_pruneable_named_modules, clean_memory
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cosine  # returns the cosine distance (1-sim)
import seaborn as sns
from pyPrune.utils import plot_loss_accuracy_sparsity, set_seed, CustomLambdaLR, lr_lambda

#############################################
# Global Plot Settings for Better Visualization
#############################################
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

#############################################
# Utility Functions for Loading Metrics
#############################################

def load_metric(pkl_file):
    """Load metrics from a pickle file."""
    with open(pkl_file, 'rb') as file:
        return pickle.load(file)

def load_json(json_file):
    """Load metrics from a JSON checkpoint file."""
    print(f"Processing: {json_file}")
    with open(json_file, 'r') as file:
        return json.load(file)


def group_weights_and_sparsity_data(weights_and_sparsity_plots_sorted):
    """
    Groups weights and sparsity data by the base layer name.
    Aggregates weights and sparsity across layers with the same base name by computing the mean.
    """
    grouped_data = defaultdict(lambda: {'sparsity': [], 'zero_weights_in_layer': [], 
                                         'total_weights_in_layer': [], 'zero_weights_in_model': []})
    # Group by base name
    for layer_name, data in weights_and_sparsity_plots_sorted.items():
        if layer_name.count('.') == 0:
            base_name = layer_name
        elif layer_name.count('.') == 1:
            base_name = layer_name.split('.')[-1]
        elif layer_name.count('.') >= 2:
            base_name = layer_name.split('.')[0]
            
        grouped_data[base_name]['sparsity'].append(data['sparsity'])
        grouped_data[base_name]['zero_weights_in_layer'].append(data['zero_weights_in_layer'])
        grouped_data[base_name]['total_weights_in_layer'].append(data['total_weights_in_layer'])
        grouped_data[base_name]['zero_weights_in_model'].append(data['zero_weights_in_model'])

    # Aggregate by computing the mean where necessary
    for base_name, values in grouped_data.items():
        grouped_data[base_name]['sparsity'] = np.mean(values['sparsity'], axis=0) if len(values['sparsity']) > 1 else values['sparsity'][0]
        grouped_data[base_name]['zero_weights_in_layer'] = np.mean(values['zero_weights_in_layer'], axis=0) if len(values['zero_weights_in_layer']) > 1 else values['zero_weights_in_layer'][0]
        grouped_data[base_name]['total_weights_in_layer'] = np.mean(values['total_weights_in_layer'], axis=0) if len(values['total_weights_in_layer']) > 1 else values['total_weights_in_layer'][0]
        grouped_data[base_name]['zero_weights_in_model'] = np.mean(values['zero_weights_in_model'], axis=0) if len(values['zero_weights_in_model']) > 1 else values['zero_weights_in_model'][0]

    return grouped_data

def group_layer_similarity_data(sim_dict):
    """
    Groups layer similarity data by the base name (before the first dot).
    Aggregates the data by averaging similarities at each step.
    """
    grouped_dict = defaultdict(list)
    for layer, sim_list in sim_dict.items():
        if layer.count('.') == 0:
            base_name = layer
        elif layer.count('.') == 1:
            base_name = layer.split('.')[-1]
        elif layer.count('.') >= 2:
            base_name = layer.split('.')[0]
        grouped_dict[base_name].extend(sim_list)

    for base_name, data_list in grouped_dict.items():
        data_sorted = sorted(data_list, key=lambda x: x[0])
        steps = [x[0] for x in data_sorted]
        similarities = [x[1] for x in data_sorted]
        unique_steps = sorted(set(steps))
        aggregated_data = []
        for step in unique_steps:
            step_sims = [sim for s, sim in zip(steps, similarities) if s == step]
            aggregated_data.append((step, np.mean(step_sims)))
        grouped_dict[base_name] = aggregated_data

    for base_name, data_list in grouped_dict.items():
        for i in range(len(data_list)):
            step, similarity = data_list[i]
            if np.isnan(similarity):
                data_list[i] = (step, 0.0)
    return grouped_dict

def aggregate_similarity(sim_dict):
    """
    Aggregates similarity for each step across layers.
    Returns sorted steps, average similarity, 25th percentile, and 75th percentile.
    """
    step_values = defaultdict(list)
    for layer, sim_list in sim_dict.items():
        for step, sim in sim_list:
            step_values[step].append(sim)
    steps_sorted = sorted(step_values.keys())
    avg = [np.mean(step_values[step]) for step in steps_sorted]
    q25 = [np.percentile(step_values[step], 25) for step in steps_sorted]
    q75 = [np.percentile(step_values[step], 75) for step in steps_sorted]
    return steps_sorted, avg, q25, q75

def plot_aggregated_similarity(steps, avg, q25, q75, title, ylabel, save_path):
    """
    Plot aggregated average similarity with a shaded area for the 25th-75th percentiles.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(steps, avg, marker='o', linestyle='-', color='tab:blue', label='Average', linewidth=2)
    plt.fill_between(steps, q25, q75, color='tab:blue', alpha=0.3, label='25th-75th Percentile')
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.ylim(-0.01, 1.01)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300)
    plt.close()

#############################################
# Neuron Similarity Analysis Functions
#############################################
    
def process_neuron_similarity(neuron_similarity_dir, model_name):
    """
    Process neuron similarity data and generate plots:
      - Cosine similarity between consecutive activation steps.
      - Cosine similarity of layer activations compared to the prepruning step.
      - Aggregated plots with average neuron similarity and percentile shading.
    """
    files_found = glob.glob(os.path.join(neuron_similarity_dir, "*.pkl"))
    print(f"Files found for neuron similarity: {files_found}")
 
    checkpoint_dir = os.path.dirname(neuron_similarity_dir)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))

    # Process the first file for per-file plots
    if files_found:
        with open(files_found[0], 'rb') as f:
            pruner = pickle.load(f)
    else:
        print(f"No files found in {neuron_similarity_dir}")
        return        
    # Comment out quit() for full processing
    # quit()
    
    activations_step = pruner.activations_step

    # 1. Cosine similarity between consecutive activation steps for each layer.
    cosine_similarities = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        for i in range(1, len(data_sorted)):
            step = data_sorted[i][0]
            act_prev = np.array(data_sorted[i-1][1].to('cpu').detach().numpy()).flatten()
            act_curr = np.array(data_sorted[i][1].to('cpu').detach().numpy()).flatten()
            cos_sim = 1 - cosine(act_prev, act_curr + 1e-10)
            cosine_similarities[layer].append((step, cos_sim))

    cosine_similarities = group_layer_similarity_data(cosine_similarities)
    save_dir_prev = f"./plots/{checkpoint_name}/activation_similarity/"
    os.makedirs(save_dir_prev, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    # Use a Seaborn color palette and define a set of linestyles
    palette = sns.color_palette("tab10", n_colors=len(cosine_similarities))
    linestyles = ['-', '--', '-.', ':']
    for idx, (layer, sim_data) in enumerate(sorted(cosine_similarities.items())):
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps_layer = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        style = linestyles[idx % len(linestyles)]
        plt.plot(steps_layer, sims, marker='o', linestyle=style, 
                 color=palette[idx], label=layer)
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity with Previous Step')
    plt.title('Cosine Similarity of Layer Activations Across Steps')
    plt.legend(title='Layer')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir_prev, "cosine_similarity_across_steps.svg"), dpi=300)
    plt.close()

    # 2. Cosine similarity of layer activations compared to the prepruning step.
    baseline_cosine_similarity_by_layer = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        baseline_activation = np.array(data_sorted[0][1].to('cpu').detach().numpy()).flatten()
        for step, act in data_sorted:
            current_activation = np.array(act.to('cpu').detach().numpy()).flatten()
            cos_sim = 1 - cosine(baseline_activation, current_activation + 1e-10)
            baseline_cosine_similarity_by_layer[layer].append((step, cos_sim))
    baseline_cosine_similarity_by_layer = group_layer_similarity_data(baseline_cosine_similarity_by_layer)
    save_dir_base = f"./plots/{model_name}/{checkpoint_name}/baseline_similarity/"
    os.makedirs(save_dir_base, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("tab10", n_colors=len(baseline_cosine_similarity_by_layer))
    for idx, (layer, sim_data) in enumerate(sorted(baseline_cosine_similarity_by_layer.items())):
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps_layer = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        style = linestyles[idx % len(linestyles)]
        plt.plot(steps_layer, sims, marker='o', linestyle=style, 
                 color=palette[idx], label=layer)
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity with Prepruning Step')
    plt.title('Cosine Similarity of Layer Activations Compared to Prepruning Step')
    plt.legend(title='Layer')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(save_dir_base, "baseline_cosine_similarity.svg"), dpi=300)
    plt.close()

    # 3. Aggregate and plot average neuron similarity over each pruning step
    steps_prev, avg_prev, q25_prev, q75_prev = aggregate_similarity(cosine_similarities)
    plot_aggregated_similarity(
        steps_prev, avg_prev, q25_prev, q75_prev,
        title='Average Cosine Similarity with Previous Step (Aggregated)',
        ylabel='Cosine Similarity',
        save_path=os.path.join(save_dir_prev, "aggregated_previous_similarity.svg")
    )

    steps_baseline, avg_baseline, q25_baseline, q75_baseline = aggregate_similarity(baseline_cosine_similarity_by_layer)
    plot_aggregated_similarity(
        steps_baseline, avg_baseline, q25_baseline, q75_baseline,
        title='Average Cosine Similarity with Prepruning Step (Aggregated)',
        ylabel='Cosine Similarity',
        save_path=os.path.join(save_dir_base, "aggregated_prepruning_similarity.svg")
    )

    # Aggregate neuron similarity across ALL files in the glob
    all_cosine_similarities = defaultdict(list)
    for file in files_found:
        with open(file, 'rb') as f:
            pruner_file = pickle.load(f)
        if not hasattr(pruner_file, 'activations_step'):
            continue
        for layer, data in pruner_file.activations_step.items():
            data_sorted = sorted(data, key=lambda x: x[0])
            for i in range(1, len(data_sorted)):
                step = data_sorted[i][0]
                act_prev = np.array(data_sorted[i-1][1].to('cpu').detach().numpy()).flatten()
                act_curr = np.array(data_sorted[i][1].to('cpu').detach().numpy()).flatten()
                cos_sim = 1 - cosine(act_prev, act_curr + 1e-10)
                all_cosine_similarities[layer].append((step, cos_sim))

    save_dir_all = os.path.join(save_dir_prev, "all_files_aggregated")
    os.makedirs(save_dir_all, exist_ok=True)

    # (a) Individual plots per layer.
    individual_dir = os.path.join(save_dir_all, "individual_layers")
    os.makedirs(individual_dir, exist_ok=True)
    for layer, sim_list in all_cosine_similarities.items():
        steps, avg, q25, q75 = aggregate_similarity({layer: sim_list})
        title = f'Aggregated Cosine Similarity for {layer} (Individual)'
        save_path = os.path.join(individual_dir, f"aggregated_{layer}_similarity.svg")
        plot_aggregated_similarity(steps, avg, q25, q75, title, 'Cosine Similarity', save_path)

    # (b) Combined figure for all layers.
    plt.figure(figsize=(12, 8))
    palette = sns.color_palette("tab10", n_colors=len(all_cosine_similarities))
    for idx, (layer, sim_list) in enumerate(sorted(all_cosine_similarities.items())):
        steps, avg, q25, q75 = aggregate_similarity({layer: sim_list})
        style = linestyles[idx % len(linestyles)]
        plt.plot(steps, avg, marker='o', linestyle=style, color=palette[idx], label=layer)
        plt.fill_between(steps, q25, q75, color=palette[idx], alpha=0.3)
    plt.xlabel('Step')
    plt.ylabel('Cosine Similarity')
    plt.title('Aggregated Cosine Similarity for All Layers (Combined)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    combined_save_path = os.path.join(save_dir_all, "aggregated_all_layers_similarity.svg")
    plt.savefig(combined_save_path, dpi=300)
    plt.close()

    print(f"Neuron similarity post-processing complete for model: {model_name}, checkpoint: {checkpoint_name}")

def clear_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()

def main():
    """Main function to execute the entire post-processing pipeline."""
    model_names = ["LeNet", "ResNet20", "Vgg16"][::-1]
    for model_name in model_names:
        checkpoint_glob = f"/scratch/jgafur/LTH_output/*{model_name}*"
        for output_dir in glob.glob(checkpoint_glob):
            clear_memory()
            clean_memory()
            print(output_dir)
            neuron_similarity_dir = os.path.join(output_dir, "neuron_similarity")
            process_neuron_similarity(neuron_similarity_dir, '')
            
if __name__ == "__main__":
    main()
