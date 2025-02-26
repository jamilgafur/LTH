import os
import torch
import pickle
import glob
import logging
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from pyPrune.utils import get_pruneable_named_modules
from collections import defaultdict


# Set up logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_metrics(json_file):
    """Load metrics from a JSON checkpoint file."""
    try:
        with open(json_file, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Failed to load metrics from {json_file}: {e}")
        return {}

def plot_accuracy_and_loss(metrics, model_name):
    """Plot accuracy and loss over sparsity steps."""
    accuracy = metrics.get('accuracy', [])
    loss = metrics.get('loss', [])
    sparsity = metrics.get('step', [])

    logger.info(f"Accuracy len: {len(accuracy)}, Loss len: {len(loss)}, Sparsity len: {len(sparsity)}")

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot accuracy
    ax1.set_xlabel('Step', fontsize=14)
    ax1.set_ylabel('Accuracy', color='tab:blue', fontsize=14)
    ax1.plot(sparsity, accuracy, color='tab:blue', label='Accuracy', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Create a second y-axis to plot loss
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red', fontsize=14)
    ax2.plot(sparsity, loss, color='tab:red', label='Loss', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)

    # Add legends
    ax1.legend(loc='upper left', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)

    # Set title
    plt.title('Accuracy and Loss over Sparsity Steps', fontsize=16, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout()

    # Save the plot
    plot_path = os.path.join(f"./plots/{model_name}/layer_sparsity/", "accuracy_and_loss_plot.png")
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Saved plot to {plot_path}")


def process_model(output_dir, model_name):
    """Process model data and generate necessary plots."""
    try:
        # Find all .pkl files and load the metrics
        files_found = glob.glob(os.path.join(output_dir, "*.pkl"))
        if not files_found:
            logger.warning(f"No .pkl files found in {output_dir}")
            return

        metrics = load_metrics(files_found[0])

        with open(files_found[0], 'rb') as f:
            pruner = pickle.load(f)

        pruner.logger = None
        paths = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.pth')]
        logger.info(f"Received {len(paths)} model checkpoint paths.")

        # Initialize variables
        layer_sparsity_data = defaultdict(lambda: {
            'sparsity': [],
            'zero_weights_in_layer': [],
            'total_weights_in_layer': [],
            'zero_weights_in_model': []
        })

        model = pruner.model
        total_weights_data = []

        # Process models
        for path in tqdm(paths, desc="Processing models"):
            sparsity = path.split('_')[-1][:-4]
            logger.info(f"Processing file with sparsity: {sparsity} for path: {path}")

            checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])

            # Get pruneable modules
            names, modules = get_pruneable_named_modules(model, pruner.prunable_layers)

            # Calculate and store total weights
            total_weights_in_model = sum(module.weight.numel() for module in modules)
            total_weights_data.append(total_weights_in_model)

            # Process sparsity data
            for name, module in zip(names, modules):
                zero_weights_in_layer = torch.sum(module.weight.data == 0).item()
                total_weights_in_layer = module.weight.numel()

                layer_sparsity = (zero_weights_in_layer / total_weights_in_layer) * 100

                layer_sparsity_data[name]['sparsity'].append(float(sparsity))
                layer_sparsity_data[name]['zero_weights_in_layer'].append(zero_weights_in_layer)
                layer_sparsity_data[name]['total_weights_in_layer'].append(total_weights_in_layer)
                layer_sparsity_data[name]['zero_weights_in_model'].append(zero_weights_in_layer)

        # Plot the sparsity data for layers
        plot_layer_sparsity(layer_sparsity_data, model_name)

        # Plot accuracy and loss
        json = load_metrics(glob.glob(os.path.join(output_dir, "*.json"))[0])
        import pdb; pdb.set_trace()
        plot_accuracy_and_loss(metrics, model_name)

    except Exception as e:
        logger.error(f"Error processing {model_name}: {e}")


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

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(f"./plots/{model_name}/layer_sparsity/", exist_ok=True)
    plt.savefig(f"./plots/{model_name}/layer_sparsity/weights_and_sparsity_plots_sorted.png")
    logger.info(f"Saved layer sparsity plot for {model_name}")


def main():
    """Main function to execute the entire process."""
    model_names = ["LeNet", "ResNet20", "Vgg16"]
    
    for model_name in model_names:
        output_dir = f"/scratch/jgafur/LTH_output/{model_name}_pretrain10_finetune10_steps20_batch128_devicecuda/"
        logger.info(f"Processing {model_name} with output directory: {output_dir}")
        process_model(output_dir, model_name)


if __name__ == "__main__":
    main()
