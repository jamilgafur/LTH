import torch
import os
from scipy.spatial.distance import cosine
import numpy as np
import glob
import matplotlib.pyplot as plt
import pyPrune.utils as utils


# ----------------------------
# Function to load the pruner pickle
# ----------------------------
def load_pickle(file_path):
    """
    Loads and returns the pruner pickle.

    Args:
        file_path (str): Path to the pruner pickle file.

    Returns:
        pruner: Loaded pruner object.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Pickle file not found: {file_path}")

    import pickle
    with open(file_path, 'rb') as f:
        pruner = pickle.load(f)

    return pruner


# ----------------------------
# Function to get model paths
# ----------------------------
def get_sorted_model_paths(directory):
    """
    Returns sorted model paths based on the filenames in a directory.
    
    Args:
        directory (str): The directory containing the model checkpoint files.

    Returns:
        list: Sorted list of model file paths.
    """
    model_paths = glob.glob(os.path.join(directory, "pruned_model_step_*.pth"))
    model_paths.sort(key=lambda x: float(x.split("_")[-1].split(".")[0]))
    return model_paths


# ----------------------------
# Function to get layer information with consistent total_trainable_params
# ----------------------------
import torch
from collections import defaultdict

def get_layer_information(pruner, model_path, base_params=None):
    """
    Gets layer-wise information: # of zeros in the layer, # of trainable parameters in the layer, 
    and the total # of trainable parameters in the model (constant across sparsities).

    Args:
        pruner: The pruner object.
        model_path (str): Path to the model checkpoint.
        base_params (dict, optional): Dictionary storing the original total trainable params for each layer.

    Returns:
        dict: Dictionary containing layer information.
        dict: Base parameters dictionary storing the total_trainable_params.
    """
    # Load model weights into pruner
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    pruner.model.load_state_dict(checkpoint['model_state_dict'])
    pruner.model.eval()

    layer_info = defaultdict(lambda: {"num_zeros": 0, "num_trainable_params": 0, "total_trainable_params": 0})
    
    # Load original layer params only once
    if base_params is None:
        base_params = {}

    names, params = utils.get_pruneable_named_parameters(pruner.model, pruner.prunable_layers)

    for name, param in zip(names, params):
        if name.count(".") == 1:
            name = name.split(".")[0]
        
        if name.count(".") == 2:
            name = name.split(".")[1]
        
        if name.count(".") == 3:
            name = name.split(".")[0]

        num_zeros = (param == 0).sum().item()
        num_trainable_params = param.numel()

        # Store the original total trainable parameters only once
        if name not in base_params:
            base_params[name] = num_trainable_params

        # Aggregate values for layers with the same name
        layer_info[name]["num_zeros"] += num_zeros
        layer_info[name]["num_trainable_params"] += num_trainable_params
        layer_info[name]["total_trainable_params"] = base_params[name]  # Use original total params (constant)

        print(f"Layer: {name}, Zeros: {num_zeros}, Trainable: {num_trainable_params}, Total: {base_params[name]}")

    # Convert defaultdict back to a regular dict before returning
    return dict(layer_info), base_params



# ----------------------------
# Function to plot layer information
# ----------------------------
import os
import matplotlib.pyplot as plt

def plot_layer_information(savedir, data):
    """
    Plots the layer-wise information of pruned models.

    Args:
        data (dict): Dictionary where keys are sparsity values and values are model data
                     containing layer information and cosine similarities.
    """
    os.makedirs(savedir, exist_ok=True)
    sparsities = sorted(data.keys())

    # First figure for axes[0] and axes[1]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for layer in data[0.0]["layer_info"].keys():
        layer_name = layer.split(".")[0]

        x1, x2 = [], []
        
        for sparsity in sparsities:
            info = data[sparsity]['layer_info'][layer]
            
            # Both x1 and x2 should use the same denominator (sum of trainable params)
            total_trainable_params = sum(info['num_trainable_params'] for layer in data[sparsity]['layer_info'])

            # Sum of zeros for all layers
            total_zeros = sum(data[sparsity]["layer_info"][layer]["num_zeros"] for layer in data[sparsity]["layer_info"])
            total_params = sum(data[sparsity]["layer_info"][layer]["total_trainable_params"] for layer in data[sparsity]["layer_info"])

            x1.append(info['num_zeros'] / info['num_trainable_params'])
            x2.append(info['num_zeros'] / total_params)

        # Plot x1 for axes[0] and x2 for axes[1]
        axes[0].plot(sparsities, x1, label=f"{layer_name}")
        axes[1].plot(sparsities, x2, label=f"{layer_name}")

    # Labels and legends for the first figure
    axes[0].set_title('Zeros in Layer / Trainable Params in Layer')
    axes[0].set_xlabel('Sparsity')
    axes[0].set_ylabel('Ratio')
    axes[0].legend()

    axes[1].set_title('Zeros in Layer / Total Trainable Params')
    axes[1].set_xlabel('Sparsity')
    axes[1].set_ylabel('Ratio')
    axes[1].legend()

    # Save the first figure
    plt.tight_layout()
    plt.savefig(f"{savedir}/layer_info_subplot_1_2.png")
    plt.clf()  # Clear the figure after saving

    # Second figure for axes[2]
    fig2, ax2 = plt.subplots(figsize=(14, 5))

    for layer in data[0.0]["layer_info"].keys():
        layer_name = layer.split(".")[0]

        x3 = []
        
        for sparsity in sparsities:
            total_zeros = sum(data[sparsity]["layer_info"][layer]["num_zeros"] for layer in data[sparsity]["layer_info"])
            total_params = sum(data[sparsity]["layer_info"][layer]["total_trainable_params"] for layer in data[sparsity]["layer_info"])
            x3.append(total_zeros / total_params)

        # Plot x3 for axes[2]
        ax2.plot(sparsities, x3, label="Sum of all layers")

    # Labels and legend for the second figure
    ax2.set_title('Sum of Zeros / Sum of Trainable Params')
    ax2.set_xlabel('Sparsity')
    ax2.set_ylabel('Ratio')
    ax2.legend()

    # Save the second figure
    plt.tight_layout()
    plt.savefig(f"{savedir}/layer_info_subplot_3.png")


# ----------------------------
# Main Execution Block
# ----------------------------
if __name__ == "__main__":
    
    for model_directory in  glob.glob("/scratch/jgafur/LTH_output/*cuda*"):
        pruner_pickle_path = f"{model_directory}/pruner.pkl"

        pruner = load_pickle(pruner_pickle_path)
        model_paths = get_sorted_model_paths(model_directory)

        data = {}
        base_params = None

        for model_path in model_paths:
            sparsity = float("0." + model_path.split('_')[-1].split('.')[1])
            
            print(f"Model Path: {model_path}")
            layer_info, base_params = get_layer_information(pruner, model_path, base_params)

            data[sparsity] = {
                "layer_info": layer_info,
            }
            
        plot_layer_information(f"./plots/{model_directory.split("/")[-1].split(".")[0]}", data)