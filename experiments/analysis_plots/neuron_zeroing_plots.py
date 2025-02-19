import os
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np

# Helper function to load checkpoint
def load_checkpoint(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

# Helper function to load metrics from the saved file
def load_metrics(metrics_file):
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    return metrics

# Function to generate figures from the metrics
def generate_figures(metrics, save_dir):
    # Create the necessary directories
    os.makedirs(save_dir, exist_ok=True)

    # Extract relevant metrics
    accuracy_drops = [entry['accuracy_drop'] for entry in metrics['neuron_accuracy_drops']]
    sparsities = [entry['sparsity'] for entry in metrics['sparsity_metrics']]
    loss_changes = [entry['loss_change'] for entry in metrics['loss_changes']]

    # Layer accuracy drops
    layer_accuracy_drops = {}
    for entry in metrics['layer_accuracy_drops']:
        if entry['layer_name'] not in layer_accuracy_drops:
            layer_accuracy_drops[entry['layer_name']] = []
        layer_accuracy_drops[entry['layer_name']].append(entry['accuracy_drop'])

    # Create the figure
    plt.figure(figsize=(14, 12))

    # Plot 1: Accuracy Drop Distribution
    plt.subplot(2, 3, 1)
    plt.hist(accuracy_drops, bins=30)
    plt.title('Impact of Neuron Zeroing on Accuracy')
    plt.xlabel('Accuracy Drop')
    plt.ylabel('Frequency')

    # Plot 2: Sparsity vs Accuracy Drop
    plt.subplot(2, 3, 2)
    plt.scatter(sparsities, accuracy_drops, alpha=0.5)
    plt.title('Sparsity vs Accuracy Drop')
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy Drop')

    # Plot 3: Accuracy Drop per Layer
    plt.subplot(2, 3, 3)
    for layer_name, accuracy in layer_accuracy_drops.items():
        plt.hist(accuracy, bins=20, alpha=0.5, label=layer_name)
    plt.title('Accuracy Drop per Layer')
    plt.xlabel('Accuracy Drop')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 4: Loss Change Distribution
    plt.subplot(2, 3, 4)
    plt.hist(loss_changes, bins=30, color='orange', alpha=0.7)
    plt.title('Loss Change Distribution')
    plt.xlabel('Loss Change')
    plt.ylabel('Frequency')

    # Plot 5: Loss vs Sparsity
    loss_metrics = [entry['loss'] for entry in metrics['loss_metrics']]
    plt.subplot(2, 3, 5)
    plt.scatter(sparsities, loss_metrics, alpha=0.5)
    plt.title('Loss vs Sparsity')
    plt.xlabel('Sparsity')
    plt.ylabel('Loss')

    # Tight layout and save the plot
    plt.tight_layout()
    plot_file = os.path.join(save_dir, 'neuron_zeroing_results.png')
    plt.savefig(plot_file)
    print(f"Figure saved to {plot_file}")
    plt.close()

# Main function to load the checkpoint, load metrics, and generate figures
def main(checkpoint_file, metrics_file, save_dir):
    # Load the checkpoint (for model state, etc.)
    checkpoint = load_checkpoint(checkpoint_file)
    print("Checkpoint loaded successfully.")

    # Load the metrics (accuracy drops, sparsity, etc.)
    metrics = load_metrics(metrics_file)
    print("Metrics loaded successfully.")

    # Generate the figures based on metrics
    generate_figures(metrics, save_dir)
    print("Figures generated and saved.")

if __name__ == "__main__":
    # Specify file paths (update with the correct paths)
    for model_name in ["LeNet"]:
        checkpoint_file = f"/scratch/jgafur/LTH_output/{model_name}_pretrain10_finetune10_steps21_batch64_devicecuda/neuronZeroing_accuracy/neuron_zeroing.pkl"  # Example path to the checkpoint
        metrics_file = f"/scratch/jgafur/LTH_output/{model_name}_pretrain10_finetune10_steps21_batch64_devicecuda/neuronZeroing_accuracy/metrics.json"  # Example path to the saved metrics
        save_dir = f"./plots/{model_name}"  # Path where figures should be saved

        # Run the script
        main(checkpoint_file, metrics_file, save_dir)
