import os
import torch
import matplotlib.pyplot as plt
from pyPrune.models.LeNet import LeNet
from pyPrune.utils import get_pruneable_named_modules
import pickle
import glob
from tqdm import tqdm  # For the progress bar

# Define paths
for model_name in ["LeNet", "ResNet20", "Vgg16"]:
    output_dir = f"/projects/modularai/jgafur/LTH/pruning_checkpoints/{model_name}_pretrain3_finetune1_steps3_batch64_devicecuda/"
    # output_dir = f"/projects/modularai/jgafur/LTH/temp/LeNet_pretrain2_finetune3_steps20_batch64_devicecuda/"
    # Load pruner object from .pkl file
    print(f"{f"{output_dir}*.pkl"}")
    with open(glob.glob(f"{output_dir}*.pkl")[0], 'rb') as f:
        pruner = pickle.load(f)

    pruner.logger = None

    # Get all paths to the .pth files
    paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pth')]
    print(f"Received: {paths}")

    # Dictionary to store sparsity info
    layer_sparsity_data = {}

    # Load the model once
    model = pruner.model

    # Initialize plotting data
    total_weights_data = []  # To store total weights for each model

    # Iterate over paths (models)
    for path in tqdm(paths, desc="Processing models"):
        # Extract sparsity from the filename (assumes the sparsity is before '_steps')
        sparsity = path.split('_')[-1][:-4]
        print(f"Processing file with sparsity: {sparsity} for path: {path}")

        # Load the checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get the pruneable modules
        names, modules = get_pruneable_named_modules(model, pruner.prunable_layers)
        
        # Calculate total number of weights in the model
        total_weights_in_model = sum(module.weight.numel() for module in modules)
        total_weights_data.append(total_weights_in_model)

        # Process layers and store sparsity information
        for name, module in zip(names, modules):
            # Calculate the number of zero weights in the layer
            zero_weights_in_layer = torch.sum(module.weight.data == 0).item() 
            total_weights_in_layer = module.weight.numel()
            
            # Calculate the sparsity of the layer (zero weights in the layer / total weights in the layer)
            layer_sparsity = (zero_weights_in_layer / total_weights_in_layer) * 100  # Sparsity as percentage
            
            # Initialize a list for this layer if not already done
            if name not in layer_sparsity_data:
                layer_sparsity_data[name] = {
                    'sparsity': [],  # Store sparsity values from filenames
                    'zero_weights_in_layer': [],  # Store zero weights in each layer
                    'total_weights_in_layer': [],  # Store total weights in each layer
                    'zero_weights_in_model': []  # Store zero weights in the entire model
                }
            
            # Append the values for this layer
            layer_sparsity_data[name]['sparsity'].append(float(sparsity))  # Sparsity is from the filename
            layer_sparsity_data[name]['zero_weights_in_layer'].append(zero_weights_in_layer)
            layer_sparsity_data[name]['total_weights_in_layer'].append(total_weights_in_layer)
            layer_sparsity_data[name]['zero_weights_in_model'].append(zero_weights_in_layer)

    # Create subplots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # Sort and plot for Layer Sparsity vs Zero Weights in Layer
    for layer_name, data in layer_sparsity_data.items():
        # Sort the data by sparsity
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        sorted_sparsity = [data['sparsity'][i] for i in sorted_indices]
        sorted_zero_weights_in_layer = [data['zero_weights_in_layer'][i] for i in sorted_indices]
        sorted_total_weights_in_layer = [data['total_weights_in_layer'][i] for i in sorted_indices]

        # First subplot: sparsity of layer (x-axis) vs ratio of zero weights to total weights in the layer
        zero_weights_in_layer_ratio = [zero / total for zero, total in zip(sorted_zero_weights_in_layer, sorted_total_weights_in_layer)]
        axes[0].plot(sorted_sparsity, zero_weights_in_layer_ratio, marker='o', linestyle='-', label=layer_name)

    axes[0].set_xlabel('Sparsity (%)')
    axes[0].set_ylabel('Zero Weights / Total Weights in Layer')
    axes[0].set_title('Zero Weights / Total Weights in Layer')
    axes[0].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Sort and plot for Layer Sparsity vs Zero Weights in Model
    for layer_name, data in layer_sparsity_data.items():
        # Sort the data by sparsity
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        sorted_sparsity = [data['sparsity'][i] for i in sorted_indices]
        sorted_zero_weights_in_model = [data['zero_weights_in_model'][i] for i in sorted_indices]

        # Second subplot: sparsity of layer (x-axis) vs ratio of zero weights to total weights in the model
        zero_weights_in_model_ratio = [zero / total_weights_in_model for zero in sorted_zero_weights_in_model]
        axes[1].plot(sorted_sparsity, zero_weights_in_model_ratio, marker='o', linestyle='-', label=layer_name)

    axes[1].set_xlabel('Sparsity (%)')
    axes[1].set_ylabel('Zero Weights / Total Weights in Model')
    axes[1].set_title('Zero Weights / Total Weights in Model')
    axes[1].legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    os.makedirs(f"./plots/{model_name}/layer_sparsity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/layer_sparsity/weights_and_sparsity_plots_sorted.png")

    plt.show()
