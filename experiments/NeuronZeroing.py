import torch
import torch.nn as nn
import os
import json
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from copy import deepcopy
import random
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm  # Import tqdm for progress bars

class NeuronZeroing:
    def __init__(self, pruner, sample_fraction=0.1, zeroing_metric='accuracy', logger=None):
        """
        Initialize the NeuronZeroing experiment setup.

        Args:
            pruner (IterativeMagnitudePruning): Pruner instance containing model, training info, and dataset.
            sample_fraction (float): Fraction of neurons to sample if the model is too large (default: 0.1).
            zeroing_metric (str): Metric to use for evaluating performance (e.g., 'accuracy', 'f1-score').
            logger (logging.Logger, optional): Logger instance to handle log output. Defaults to None.
            checkpoint_file (str, optional): Path to a checkpoint file for resuming the experiment. Defaults to None.
        """
        self.pruner = pruner
        self.model = deepcopy(pruner.model)  # Make a deep copy to ensure the original model is not affected
        
        # Set up directories and files for experiment results and logs
        self.save_dir = os.path.join(pruner.save_dir, 'neuron_zeroing')
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        self.sample_fraction = sample_fraction
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        
        # Store experiment metrics in a dictionary
        self.metrics = {
            'neuron_accuracy_drops': [],
            'sparsity_metrics': [],
            'precision_recall_fscore': [],
            'loss_metrics': [],
            'layer_accuracy_drops': []
        }

        # Checkpoint file handling
        self.checkpoint_file = self.pruner.save_dir + '/neuron_zeroing.pkl'
        
        if self.checkpoint_file and os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
        else:
            log_file = os.path.join(self.save_dir, 'experiment.log')
            logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

            # Save experiment metadata
            self.metadata = {
                'model_architecture': str(self.model),
                'sample_fraction': self.sample_fraction,
                'zeroing_metric': self.zeroing_metric
            }

    def evaluate_performance(self) -> float:
        """
        Evaluate the model's performance on the test dataset using the specified metric (accuracy).

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.model.eval()  # Set model to evaluation mode (disables dropout, batch normalization, etc.)
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation to speed up evaluation
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)  # Get predicted class index
                total += target.size(0)
                correct += (predicted == target).sum().item()  # Count correct predictions
        accuracy = correct / total  # Compute accuracy as the ratio of correct predictions to total
        return accuracy

    def compute_sparsity(self) -> float:
        """
        Compute the sparsity of the model by calculating the fraction of zero weights.

        Returns:
            float: The sparsity of the model (fraction of zero weights).
        """
        total_params = 0
        zero_params = 0
        for name, param in self.pruner.get_prunable_named_parameters():
            total_params += param.numel()  # Get total number of parameters in the layer
            zero_params += (param == 0).sum().item()  # Count number of zero parameters
        sparsity = zero_params / total_params  # Calculate sparsity
        return sparsity

    def compute_precision_recall_fscore(self) -> dict:
        """
        Compute precision, recall, and F1-score on the test dataset.

        Returns:
            dict: Dictionary with precision, recall, and F1-score for each class.
        """
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())  # Convert target to CPU for easier handling
                y_pred.extend(predicted.cpu().numpy())  # Convert predictions to CPU
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        return {'precision': precision.tolist(), 'recall': recall.tolist(), 'fscore': fscore.tolist()}

    def evaluate_loss(self) -> float:
        """
        Compute the loss on the test dataset.

        Returns:
            float: The average loss on the test dataset.
        """
        self.model.eval()
        criterion = self.pruner.criterion  # Get the loss function
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)  # Multiply loss by batch size
                total_samples += data.size(0)
        return total_loss / total_samples  # Average loss across all samples

    def run_experiment(self):
        """
        Run the neuron zeroing experiment by progressively zeroing out neurons.

        This method tracks accuracy, sparsity, precision-recall metrics, and loss for each zeroing step.
        """
        metrics = {}  # Dictionary to store metrics for each step of the pruning
        for weights, step in zip(self.pruner.weight_history, self.pruner.steps):
            metrics[step] = {}
            self.logger.info(f"Loading weights from step {step}...")  # Log the current step
            self.model.load_state_dict(weights)  # Load model weights for this step
            self.model.eval()
            self.model.to(self.pruner.device)  # Move model to the correct device
            self.logger.info("Starting neuron zeroing experiment...")

            baseline_accuracy = self.evaluate_performance()  # Get baseline accuracy
            baseline_sparsity = self.compute_sparsity()  # Get baseline sparsity
            baseline_metrics = self.compute_precision_recall_fscore()  # Get baseline precision/recall/F1 scores

            # Log baseline values
            self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
            self.logger.info(f"Baseline sparsity: {baseline_sparsity:.4f}")
            self.logger.info(f"Baseline precision: {baseline_metrics['precision']}")
            self.logger.info(f"Baseline recall: {baseline_metrics['recall']}")
            self.logger.info(f"Baseline F1-score: {baseline_metrics['fscore']}")

            total_neurons = 0
            sampled_neurons = []

            # Gather neurons from all fully connected (Linear) layers
            for name, layer in self.pruner.get_pruneable_named_modules():
                total_neurons += layer.out_features
                sampled_neurons.append(layer)  # Add the layer to the list of layers to sample neurons from

            self.logger.info(f"Zeroing neurons from {total_neurons} total neurons.")

            # Progressively zero out neurons and evaluate the impact
            for layer in tqdm(sampled_neurons, desc="Zeroing neurons in layers", unit="layer"):
                original_weights = layer.weight.clone()  # Save original weights for later restoration
                for i in tqdm(range(layer.out_features), desc=f"Zeroing neurons in {layer.__class__.__name__}", unit="neuron", leave=False):
                    layer.weight.data[i, :] = 0  # Zero out the entire row (neuron) in the weight matrix
                    accuracy_after_zeroing = self.evaluate_performance()  # Evaluate performance after zeroing
                    sparsity_after_zeroing = self.compute_sparsity()  # Compute sparsity after zeroing
                    metrics_after_zeroing = self.compute_precision_recall_fscore()  # Get precision/recall/F1 after zeroing

                    accuracy_drop = baseline_accuracy - accuracy_after_zeroing  # Compute accuracy drop after zeroing

                    # Log accuracy drop, sparsity, and other metrics
                    self.metrics['neuron_accuracy_drops'].append({
                        'layer_name': layer.__class__.__name__,
                        'neuron_index': i,
                        'accuracy_drop': accuracy_drop,
                        'original_weights': original_weights[i].tolist()
                    })

                    self.metrics['sparsity_metrics'].append({
                        'layer_name': layer.__class__.__name__,
                        'neuron_index': i,
                        'sparsity': sparsity_after_zeroing
                    })

                    self.metrics['precision_recall_fscore'].append({
                        'layer_name': layer.__class__.__name__,
                        'neuron_index': i,
                        'precision': metrics_after_zeroing['precision'],
                        'recall': metrics_after_zeroing['recall'],
                        'fscore': metrics_after_zeroing['fscore']
                    })

                    # Record loss after zeroing the neuron
                    loss_after_zeroing = self.evaluate_loss()
                    self.metrics['loss_metrics'].append({
                        'layer_name': layer.__class__.__name__,
                        'neuron_index': i,
                        'loss': loss_after_zeroing
                    })

                    # Record layer-wise accuracy drop
                    self.metrics['layer_accuracy_drops'].append({
                        'layer_name': layer.__class__.__name__,
                        'accuracy_drop': accuracy_drop
                    })

                    # Save checkpoint after processing each neuron
                    self.save_checkpoint()

                    layer.weight.data[i, :] = original_weights[i]  # Restore original weights for the neuron
                    self.logger.debug(f"Zeroed neuron {layer.__class__.__name__}, accuracy drop: {accuracy_drop:.4f}, sparsity: {sparsity_after_zeroing:.4f}")

            self.logger.info("Neuron zeroing experiment completed.")

            # Save the experiment metrics and metadata as JSON files
            metrics_file = os.path.join(self.save_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            self.logger.info(f"Metrics saved to {metrics_file}")

            metadata_file = os.path.join(self.save_dir, 'experiment_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=4)
            self.logger.info(f"Metadata saved to {metadata_file}")

            self.plot_results()  # Plot the results of the experiment

            metrics[step] = self.metrics  # Store the metrics for this step

        self.metrics = metrics  # Update metrics with the results of all steps
        return self.metrics
    
    def plot_results(self):
        """Plot the results of the neuron zeroing experiment."""
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        sparsities = [entry['sparsity'] for entry in self.metrics['sparsity_metrics']]
        layer_accuracy_drops = {layer_name: [] for layer_name in set([entry['layer_name'] for entry in self.metrics['layer_accuracy_drops']])}
        
        # Group accuracy drops by layer for plotting
        for entry in self.metrics['layer_accuracy_drops']:
            layer_accuracy_drops[entry['layer_name']].append(entry['accuracy_drop'])

        # Plot histogram of accuracy drops
        plt.figure(figsize=(14, 10))
        plt.subplot(2, 3, 1)
        plt.hist(accuracy_drops, bins=30)
        plt.title('Impact of Neuron Zeroing on Accuracy')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')

        # Plot sparsity vs accuracy drop
        plt.subplot(2, 3, 2)
        plt.scatter(sparsities, accuracy_drops, alpha=0.5)
        plt.title('Sparsity vs Accuracy Drop')
        plt.xlabel('Sparsity')
        plt.ylabel('Accuracy Drop')

        # Plot accuracy drop per layer
        plt.subplot(2, 3, 3)
        for layer_name, accuracy in layer_accuracy_drops.items():
            plt.hist(accuracy, bins=20, alpha=0.5, label=layer_name)
        plt.title('Accuracy Drop per Layer')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot loss vs sparsity
        loss_metrics = [entry['loss'] for entry in self.metrics['loss_metrics']]
        plt.subplot(2, 3, 4)
        plt.scatter(sparsities, loss_metrics, alpha=0.5)
        plt.title('Loss vs Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Loss')

        # Save the plot to the save directory
        plot_file = os.path.join(self.save_dir, 'accuracy_drop_vs_sparsity.png')
        plt.tight_layout()
        plt.savefig(plot_file)
        self.logger.info(f"Plot saved to {plot_file}")
        plt.close()  # Close the plot to avoid displaying it inline

    def load_checkpoint(self):
        """Load the state from the checkpoint file."""
        self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
        with open(self.checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.metrics = checkpoint['metrics']
            self.metadata = checkpoint['metadata']

    def save_checkpoint(self):
        """Save the current state of the experiment to a checkpoint file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'metadata': self.metadata
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info(f"Checkpoint saved to {self.checkpoint_file}")
