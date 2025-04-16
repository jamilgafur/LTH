import os
import glob
import pickle
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pyPrune.utils as utils
from pyPrune.utils import plot_loss_accuracy_sparsity, set_seed, lr_lambda



# ----------------------------
# Function to load the pruner pickle
# ----------------------------
def load_pickle(file_path):
    """Load and return the pruner pickle."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    with open(file_path, 'rb') as f:
        pruner = pickle.load(f)
    return pruner


# ----------------------------
# Function to get sorted model paths
# ----------------------------
def get_sorted_model_paths(directory):
    """Return sorted model paths by step number."""
    model_paths = glob.glob(os.path.join(directory, "pruned_model_step_*.pth"))
    model_paths.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
    return model_paths


# ----------------------------
# Correct Layer Aggregation Logic
# ----------------------------
def get_layer_information(pruner, model_path, base_params=None):
    """
    Extracts layer-wise information with proper grouping by base name (first dot segment).

    Args:
        pruner: The pruner object.
        model_path (str): Path to the model checkpoint.
        base_params (dict, optional): Dictionary storing the original total trainable params.

    Returns:
        dict: Dictionary with layer information.
        dict: Base parameters dictionary.
    """
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    pruner.model.load_state_dict(checkpoint['model_state_dict'])
    pruner.model.eval()

    layer_info = defaultdict(lambda: {
        "num_zeros": 0, "num_trainable_params": 0, "total_trainable_params": 0
    })

    if base_params is None:
        base_params = {}

    # Load model parameters
    names, params = utils.get_pruneable_named_parameters(pruner.model, pruner.prunable_layers)

    for name, param in zip(names, params):
        # Group by base name before the first dot
        
        if name.count('.') == 1:
            layer_name = name.split('.')[0]
        if name.count('.') == 2:
            layer_name = name.split('.')[1]
        if name.count('.') == 3:
            layer_name = name.split('.')[1] +"-"+name.split('.')[2]
        print(name, layer_name)
        num_zeros = (param == 0).sum().item()
        num_trainable_params = param.numel()

        # Store base parameters only once
        if layer_name not in base_params:
            base_params[layer_name] = num_trainable_params

        # Aggregate layer information
        layer_info[layer_name]["num_zeros"] += num_zeros
        layer_info[layer_name]["num_trainable_params"] += num_trainable_params
        layer_info[layer_name]["total_trainable_params"] = base_params[layer_name]

    return dict(layer_info), base_params


# ----------------------------
# Plot accuracy and loss
# ----------------------------
def plot_accuracy(pruner, path):
    """Plot accuracy and loss against sparsity."""
    metrics = pruner.metrics
    accuracy = metrics['accuracy']
    loss = metrics['loss']
    sparsity = metrics['step']

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot accuracy vs sparsity
    axs[0].plot(sparsity, accuracy, marker='o', linestyle='-', color='blue', label='Accuracy')
    axs[0].set_title('Accuracy vs Sparsity')
    axs[0].set_xlabel('Sparsity (%)')
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)
    axs[0].legend()

    # Plot loss vs sparsity
    axs[1].plot(sparsity, loss, marker='o', linestyle='-', color='red', label='Loss')
    axs[1].set_title('Loss vs Sparsity')
    axs[1].set_xlabel('Sparsity (%)')
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/accuracy_loss.png")


# ----------------------------
# Plot layer information with improved grouping and colors
# ----------------------------
def plot_layer_information(savedir, data):
    """
    Plots the layer-wise information of pruned models with consistent colors and line styles.

    Args:
        savedir (str): Directory to save the plots.
        data (dict): Dictionary with layer information at different sparsities.
    """
    os.makedirs(savedir, exist_ok=True)
    sparsities = sorted(data.keys())

    # Define color and line style combinations
    colors = plt.cm.viridis(np.linspace(0, 1, len(data[0.0]["layer_info"])))
    line_styles = ['-', '--', '-.', ':']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for idx, (layer, color) in enumerate(zip(data[0.0]["layer_info"].keys(), colors)):
        layer_name = layer

        x1, x2 = [], []

        for sparsity in sparsities:
            info = data[sparsity]['layer_info'][layer]

            total_trainable_params = sum(
                data[sparsity]["layer_info"][l]["total_trainable_params"]
                for l in data[sparsity]["layer_info"]
            )
            total_zeros = sum(
                data[sparsity]["layer_info"][l]["num_zeros"]
                for l in data[sparsity]["layer_info"]
            )

            x1.append(info['num_zeros'] / info['num_trainable_params'])
            x2.append(info['num_zeros'] / total_trainable_params)

        # Apply distinct colors and line styles
        style = line_styles[idx % len(line_styles)]
        axes[0].plot(sparsities, x1, label=f"{layer_name}", color=color, linestyle=style)
        axes[1].plot(sparsities, x2, label=f"{layer_name}", color=color, linestyle=style)

    # Labels and legends
    axes[0].set_title('Zeros in Layer / Trainable Params in Layer')
    axes[0].set_xlabel('Sparsity')
    axes[0].set_ylabel('Ratio')
    axes[0].legend()

    axes[1].set_title('Zeros in Layer / Total Trainable Params')
    axes[1].set_xlabel('Sparsity')
    axes[1].set_ylabel('Ratio')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{savedir}/layer_info.png")


# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == "__main__":

    base_output_dir = "./plots"

    for model_directory in glob.glob("/scratch/jgafur/LTH_output/*cuda*"):
        pruner_pickle_path = f"{model_directory}/pruner.pkl"

        # Load pruner
        pruner = load_pickle(pruner_pickle_path)
        
        # Plot accuracy and loss
        output_dir = os.path.join(base_output_dir, model_directory.split("/")[-1])
        plot_accuracy(pruner, output_dir)

        model_paths = get_sorted_model_paths(model_directory)

        data = {}
        base_params = None

        # Process each model path
        for model_path in model_paths:
            sparsity = float("0." + model_path.split('_')[-1].split('.')[1])

            print(f"Processing Model: {model_path}")
            
            layer_info, base_params = get_layer_information(pruner, model_path, base_params)
            
            data[sparsity] = {
                "layer_info": layer_info,
            }

        # Plot layer information
        plot_layer_information(output_dir, data)
