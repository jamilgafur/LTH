import os
import torch
import matplotlib.pyplot as plt
from pyPrune.models.LeNet import LeNet
from pyPrune.utils import get_pruneable_named_modules
import pickle
import glob
from tqdm import tqdm  # For the progress bar

# Function to calculate pruning percentage (fraction of zero weights in the layer)
def calculate_pruning_percentage(weights):
    zero_weights = torch.sum(weights == 0).item()  # Number of zero weights
    return zero_weights

# Define paths
for model_name in ["LeNet", "ResNet20", "Vgg16"]:
    # output_dir = f'/scratch/jgafur/LTH_output/{model_name}_pretrain10_finetune10_steps21_batch64_devicecuda/'
    output_dir = f'/projects/modularai/jgafur/LTH/temp/LeNet_pretrain1_finetune1_steps5_batch64_devicecuda/'
    # Load pruner object from .pkl file
    with open(glob.glob(f"{output_dir}*.pkl")[0], 'rb') as f:
        pruner = pickle.load(f)

    pruner.logger = None

    # Get all paths to the .pth files
    paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pth')]
    print(f"Received: {paths}")

    # Dictionary to store sparsity info
    layer_sparsity = {}

    # Load the model once
    model = pruner.model

    # Initialize plotting data
    layer_names = []
    layer_sparsity_data = {}

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
        # total_weights = 0
        # for name, module in zip(names, modules):
        #     total_weights += module.weight.numel()
            
        # Process layers and store sparsity information
        for name, module in zip(names, modules):
            # Sparsity comes from the filename
            # Pruning percentage is calculated from the module's weight
            pruning_percentage = 100*(calculate_pruning_percentage(module.weight.data)/len(module.weight.view(-1)))
            
            # Initialize a list for this layer if not already done
            if name not in layer_sparsity_data:
                layer_sparsity_data[name] = {
                    'sparsity': [],  # Store sparsity values from filenames
                    'pruning_percentage': []  # Store pruning percentages
                }
            
            # Append the values for this layer
            layer_sparsity_data[name]['sparsity'].append(float(sparsity))  # Sparsity is from the filename
            layer_sparsity_data[name]['pruning_percentage'].append(pruning_percentage)

            # Print layer-wise sparsity and pruning information
            print(f"Layer: {name}, Sparsity: {sparsity}, Pruning Percentage: {pruning_percentage:.2f}%")

    # Sort the sparsity values in ascending order (you can change to descending by setting reverse=True)
    for layer_name, data in layer_sparsity_data.items():
        # Sort the sparsity and pruning percentages together based on sparsity
        sorted_indices = sorted(range(len(data['sparsity'])), key=lambda i: data['sparsity'][i])
        layer_sparsity_data[layer_name]['sparsity'] = [data['sparsity'][i] for i in sorted_indices]
        layer_sparsity_data[layer_name]['pruning_percentage'] = [data['pruning_percentage'][i] for i in sorted_indices]

    # Plotting the sparsity vs pruning percentage for each layer
    plt.figure(figsize=(12, 6))

    # For each layer, plot the sparsity vs pruning percentage
    for layer_name, data in layer_sparsity_data.items():
        plt.plot(data['sparsity'], data['pruning_percentage'], label=layer_name)
        plt.scatter(data['sparsity'], data['pruning_percentage'], label=layer_name)

    plt.xlabel('Sparsity (%)')
    plt.ylabel('Percentage of 0 Weight/Total Pruneable Weights')
    plt.title('Sparsity vs Pruning Percentage for Each Layer (Sorted by Sparsity)')
    plt.legend(title="Layer Names", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    os.makedirs(f"./plots/{model_name}/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/sparsity_vs_pruning_percentage_sorted.png")

    # Optionally, summarize the layer sparsity and pruning percentages
    print("\nFinal Layer Sparsity and Pruning Percentages:")
    for name, data in layer_sparsity_data.items():
        print(f"Layer: {name}, Sparsity values: {data['sparsity']}, Pruning Percentages: {data['pruning_percentage']}")
