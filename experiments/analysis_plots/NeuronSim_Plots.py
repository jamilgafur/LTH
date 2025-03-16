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


from collections import defaultdict
import numpy as np
import numpy as np
from collections import defaultdict

import numpy as np
from collections import defaultdict

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

#############################################
# Plotting Functions for Accuracy & Loss
#############################################

def plot_accuracy_and_loss(json_data, model_name, checkpoint_name):
    """Plot accuracy and loss over sparsity steps, including separate plots for each metric."""
    metrics = json_data['overall_metrics']
    accuracy = metrics.get('accuracy', [])
    loss = metrics.get('loss', [])
    sparsity = metrics.get('step', [])

    save_dir = f"./plots/{model_name}/{checkpoint_name}/layer_sparsity/"
    os.makedirs(save_dir, exist_ok=True)

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
    combined_path = os.path.join(save_dir, "accuracy_and_loss_plot.svg")
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
    accuracy_path = os.path.join(save_dir, "accuracy_vs_step.svg")
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
    loss_path = os.path.join(save_dir, "loss_vs_step.svg")
    plt.savefig(loss_path, dpi=300)
    plt.close()

    return metrics

#############################################
# Plotting Functions for Layer Sparsity
#############################################

def plot_layer_sparsity(layer_sparsity_data, model_name, checkpoint_name):
    """Generate and save layer sparsity plots."""
    save_dir = f"./plots/{model_name}/{checkpoint_name}/layer_sparsity/"
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    axes[0].grid(True, linestyle='--',alpha=.7)
    axes[1].grid(True, linestyle='--',alpha=.7)
    
    layer_sparsity_data = group_weights_and_sparsity_data(layer_sparsity_data)
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
    axes[0].grid(True, linestyle='--', alpha=.7)

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
    axes[1].grid(True, linestyle='--', alpha=.7)

    # add gridlines to axes[0] and axes[1]
    plt.tight_layout()
    sparsity_path = os.path.join(save_dir, "weights_and_sparsity_plots_sorted.svg")
    plt.savefig(sparsity_path)
    plt.close()

#############################################
# Helper Functions for Aggregated Similarity Plotting
#############################################

from collections import defaultdict
import numpy as np
import pprint
from collections import defaultdict
import numpy as np

from collections import defaultdict
import numpy as np

def group_weights_and_sparsity_data(weights_and_sparsity_plots_sorted):
    """
    This function groups the weights and sparsity data by the base layer name (before the first dot in the layer name).
    It aggregates weights and sparsity across layers with the same base name (by computing the mean) if there are multiple entries.
    If there is only one entry for a base layer, it keeps the original value.

    Args:
    - weights_and_sparsity_plots_sorted: A dictionary where each key is the layer name (e.g. 'conv1.weight') and
      each value is a dictionary containing:
        - 'sparsity': Sparsity of the layer.
        - 'zero_weights_in_layer': Number of zero weights in the layer.
        - 'total_weights_in_layer': Total number of weights in the layer.
        - 'zero_weights_in_model': Total number of zero weights in the model.

    Returns:
    - grouped_data: A dictionary where the keys are base layer names (before the first dot),
      and the values are dictionaries containing the aggregated mean values for each base layer, 
      or the original value if there's only one entry for that base layer.
    """
    grouped_data = defaultdict(lambda: {'sparsity': [], 'zero_weights_in_layer': [], 
                                       'total_weights_in_layer': [], 'zero_weights_in_model': []})

    # Step 1: Group by base name (before the first dot)
    for layer_name, data in weights_and_sparsity_plots_sorted.items():
        # Extract the base name before the first dot if there is a dot
        base_name = layer_name.split('.')[0] if '.' in layer_name else layer_name
        
        # Extract relevant data
        sparsity = data['sparsity']
        zero_weights_in_layer = data['zero_weights_in_layer']
        total_weights_in_layer = data['total_weights_in_layer']
        zero_weights_in_model = data['zero_weights_in_model']

        # Add the data to the grouped dictionary by base name
        grouped_data[base_name]['sparsity'].append(sparsity)
        grouped_data[base_name]['zero_weights_in_layer'].append(zero_weights_in_layer)
        grouped_data[base_name]['total_weights_in_layer'].append(total_weights_in_layer)
        grouped_data[base_name]['zero_weights_in_model'].append(zero_weights_in_model)

    # Step 2: Aggregate the data for each base layer name (compute the mean of each field if there are multiple lists)
    for base_name, values in grouped_data.items():
        # For sparsity, take the mean of each inner list if there are multiple, else retain the original list
        if len(values['sparsity']) > 1:
            grouped_data[base_name]['sparsity'] = np.mean(np.array(values['sparsity']), axis=0)
        else:
            grouped_data[base_name]['sparsity'] = values['sparsity'][0]
        
        # For zero_weights_in_layer, take the mean of each inner list if there are multiple, else retain the original list
        if len(values['zero_weights_in_layer']) > 1:
            grouped_data[base_name]['zero_weights_in_layer'] = np.mean(np.array(values['zero_weights_in_layer']), axis=0)
        else:
            grouped_data[base_name]['zero_weights_in_layer'] = values['zero_weights_in_layer'][0]
        
        # For total_weights_in_layer, take the mean of each inner list if there are multiple, else retain the original list
        if len(values['total_weights_in_layer']) > 1:
            grouped_data[base_name]['total_weights_in_layer'] = np.mean(np.array(values['total_weights_in_layer']), axis=0)
        else:
            grouped_data[base_name]['total_weights_in_layer'] = values['total_weights_in_layer'][0]
        
        # For zero_weights_in_model, take the mean of each inner list if there are multiple, else retain the original list
        if len(values['zero_weights_in_model']) > 1:
            grouped_data[base_name]['zero_weights_in_model'] = np.mean(np.array(values['zero_weights_in_model']), axis=0)
        else:
            grouped_data[base_name]['zero_weights_in_model'] = values['zero_weights_in_model'][0]

    return grouped_data

def group_layer_similarity_data(sim_dict):
    """
    This function groups the layer similarity data by the part before the first dot in the layer name.
    - If the layer name has one dot, it updates the layer name to the part before the first dot.
    - If the layer name has two or more dots, it groups by the part before the first dot and aggregates the data across layers with the same base name (before the first dot).

    Args:
    - sim_dict: A dictionary where keys are layer names and values are lists of (step, similarity) tuples.

    Returns:
    - grouped_dict: A dictionary where the keys are base layer names (before the first dot),
      and the values are lists of (step, similarity) tuples aggregated across the layers with the same base name.
    """
    grouped_dict = defaultdict(list)

    # Step 1: Group by base name (before the first dot)
    for layer, sim_list in sim_dict.items():
        if layer.count('.') == 0:
            base_name = layer
        if layer.count('.') == 1:
            base_name = layer.split('.')[-1]
        if layer.count('.') == 2:
            base_name = layer.split('.')[0]
        grouped_dict[base_name].extend(sim_list)  # Use extend to add the tuples

    # Step 2: Aggregate the data for each base layer name
    for base_name, data_list in grouped_dict.items():
        # Sort the data by step
        data_sorted = sorted(data_list, key=lambda x: x[0])
        steps = [x[0] for x in data_sorted]
        similarities = [x[1] for x in data_sorted]

        # Compute the average similarity at each step
        unique_steps = sorted(set(steps))
        aggregated_data = []
        for step in unique_steps:
            step_similarities = [sim for s, sim in zip(steps, similarities) if s == step]
            avg_similarity = np.mean(step_similarities)
            aggregated_data.append((step, avg_similarity))

        # Update the dictionary with aggregated data
        grouped_dict[base_name] = aggregated_data

    # Step 3: Replace NaN values with zero
    for base_name, data_list in grouped_dict.items():
        for i in range(len(data_list)):
            step, similarity = data_list[i]
            # Check if similarity is NaN and set it to 0 if it is
            if np.isnan(similarity):
                data_list[i] = (step, 0.0)  # Set similarity to 0

    return grouped_dict

def aggregate_similarity(sim_dict):
    """
    Given a dictionary mapping layer names to a list of (step, similarity)
    values, aggregate the similarity for each step across layers.
    Returns sorted steps, average, 25th percentile and 75th percentile.
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
    Plot the aggregated average similarity with a shaded area between the 25th and 75th percentiles.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(steps, avg, marker='o', linestyle='-', color='tab:blue', label='Average')
    plt.fill_between(steps, q25, q75, color='tab:blue', alpha=0.3, label='25th-75th Percentile')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(-.01, 1.01)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(save_path, dpi=300)
    plt.close()

#############################################
# Neuron Similarity Analysis Functions
#############################################

def process_neuron_similarity(neuron_similarity_dir, model_name):
    """
    Process neuron similarity data and generate plots:
      1. Cosine similarity between consecutive activation steps.
      2. Cosine similarity of layer activations compared to the prepruning (first) step.
      3. Aggregated plots: average neuron similarity (with 25th-75th percentile fill)
         over each pruning step for both the previous-step and prepruning comparisons.
      4. Additionally, aggregate similarity values across all files in the directory,
         and plot the results for each individual layer (separately and as one combined figure).
    """
    files_found = glob.glob(os.path.join(neuron_similarity_dir, "*.pkl"))
    print(f"Files found for neuron similarity: {files_found}")
 
    # Extract checkpoint name from the parent folder of neuron_similarity_dir
    checkpoint_dir = os.path.dirname(neuron_similarity_dir)
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))

    # --- Process the first file for per-file plots (as before) ---
    with open(files_found[0], 'rb') as f:
        pruner = pickle.load(f)


    activations_step = pruner.activations_step

    # 1. Compute cosine similarity between consecutive activation steps for each layer.
    cosine_similarities = defaultdict(list)
    for layer, data in activations_step.items():
        data_sorted = sorted(data, key=lambda x: x[0])
        for i in range(1, len(data_sorted)):
            step = data_sorted[i][0]
            act_prev = np.array(data_sorted[i-1][1].to('cpu').detach().numpy()).flatten()
            act_curr = np.array(data_sorted[i][1].to('cpu').detach().numpy()).flatten()
            cos_sim = 1 - cosine(act_prev, act_curr)
            cosine_similarities[layer].append((step, cos_sim))

    cosine_similarities = group_layer_similarity_data(cosine_similarities)
    save_dir_prev = f"./plots/{model_name}/{checkpoint_name}/activation_similarity/"
    os.makedirs(save_dir_prev, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for layer, sim_data in cosine_similarities.items():
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps_layer = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        plt.plot(steps_layer, sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cosine Similarity with Previous Step', fontsize=12)
    plt.title('Cosine Similarity of Layer Activations Across Steps', fontsize=14)
    plt.legend(title='Layer', fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_prev, "cosine_similarity_across_steps.svg"))
    plt.close()

    # 2. Compute cosine similarity of layer activations compared to the prepruning (first) step.
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
    baseline_cosine_similarity_by_layer = group_layer_similarity_data(baseline_cosine_similarity_by_layer)
    save_dir_base = f"./plots/{model_name}/{checkpoint_name}/baseline_similarity/"
    os.makedirs(save_dir_base, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for layer, sim_data in baseline_cosine_similarity_by_layer.items():
        sim_data_sorted = sorted(sim_data, key=lambda x: x[0])
        steps_layer = [x[0] for x in sim_data_sorted]
        sims = [x[1] for x in sim_data_sorted]
        plt.plot(steps_layer, sims, marker='o', linestyle='-', label=layer)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cosine Similarity with Prepruning Step', fontsize=12)
    plt.title('Cosine Similarity of Layer Activations Compared to Prepruning Step', fontsize=14)
    plt.legend(title='Layer', fontsize=10)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir_base, "baseline_cosine_similarity.svg"))
    plt.close()

    # 3. Aggregate and plot average neuron similarity over each pruning step (with 25th-75th percentile fill)
    # For similarity compared to the previous step.
    steps_prev, avg_prev, q25_prev, q75_prev = aggregate_similarity(cosine_similarities)
    plot_aggregated_similarity(
        steps_prev, avg_prev, q25_prev, q75_prev,
        title='Average Cosine Similarity with Previous Step (Aggregated)',
        ylabel='Cosine Similarity',
        save_path=os.path.join(save_dir_prev, "aggregated_previous_similarity.svg")
    )

    # For similarity compared to the prepruning step.
    steps_baseline, avg_baseline, q25_baseline, q75_baseline = aggregate_similarity(baseline_cosine_similarity_by_layer)
    plot_aggregated_similarity(
        steps_baseline, avg_baseline, q25_baseline, q75_baseline,
        title='Average Cosine Similarity with Prepruning Step (Aggregated)',
        ylabel='Cosine Similarity',
        save_path=os.path.join(save_dir_base, "aggregated_prepruning_similarity.svg")
    )

    # --- New Section: Aggregate neuron similarity across ALL files in the glob ---
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
                cos_sim = 1 - cosine(act_prev, act_curr)
                all_cosine_similarities[layer].append((step, cos_sim))

    # Create a new directory for the aggregated results from all files.
    save_dir_all = os.path.join(save_dir_prev, "all_files_aggregated")
    os.makedirs(save_dir_all, exist_ok=True)

    # (a) Plot individual figures per layer.
    individual_dir = os.path.join(save_dir_all, "individual_layers")
    os.makedirs(individual_dir, exist_ok=True)
    for layer, sim_list in all_cosine_similarities.items():
        # Use the aggregate_similarity helper for this individual layer.
        steps, avg, q25, q75 = aggregate_similarity({layer: sim_list})
        title = f'Aggregated Cosine Similarity for {layer} (Individual)'
        save_path = os.path.join(individual_dir, f"aggregated_{layer}_similarity.svg")
        plot_aggregated_similarity(steps, avg, q25, q75, title, 'Cosine Similarity', save_path)

    # (b) Create one combined figure for all layers.
    plt.figure(figsize=(10, 6))
    for layer, sim_list in all_cosine_similarities.items():
        steps, avg, q25, q75 = aggregate_similarity({layer: sim_list})
        plt.plot(steps, avg, marker='o', linestyle='-', label=layer)
        plt.fill_between(steps, q25, q75, alpha=0.3)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Cosine Similarity', fontsize=12)
    plt.title('Aggregated Cosine Similarity for All Layers (Combined)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    combined_save_path = os.path.join(save_dir_all, "aggregated_all_layers_similarity.svg")
    plt.savefig(combined_save_path, dpi=300)
    plt.close()

    print(f"Neuron similarity post-processing complete for model: {model_name}, checkpoint: {checkpoint_name}")

#############################################
# Sparsity and Accuracy/Loss Analysis Functions
#############################################

def process_sparsity_and_accuracy(output_dir, model_name):
    """
    Process sparsity data from model checkpoints and generate plots:
      - Layer sparsity plots.
      - Accuracy and loss plots from JSON checkpoints.
    """
    checkpoint_name = os.path.basename(os.path.normpath(output_dir))
    
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

    plot_layer_sparsity(layer_sparsity_data, model_name, checkpoint_name)

    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    if json_files:
        json_data = load_json(json_files[0])
        metrics = plot_accuracy_and_loss(json_data, model_name, checkpoint_name)
    else:
        print("No JSON checkpoint found for accuracy and loss plotting.")

#############################################
# Main Function
#############################################

def main():
    """Main function to execute the entire post-processing pipeline."""
    model_names = ["LeNet", "ResNet20", "Vgg16"]
    # The glob pattern is used to select the checkpoint directories.    
    for model_name in model_names:
        checkpoint_glob = f"/scratch/jgafur/LTH_output/*{model_name}*"
        for output_dir in glob.glob(checkpoint_glob):
            print(output_dir)
            neuron_similarity_dir = os.path.join(output_dir, "neuron_similarity")
            process_neuron_similarity(neuron_similarity_dir, '')
            process_sparsity_and_accuracy(output_dir, '')
            
            
           
if __name__ == "__main__":
    main()