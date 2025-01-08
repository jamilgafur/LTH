import torch
import torch.nn as nn
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from copy import deepcopy
import random
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm  # Import tqdm

class NeuronZeroing:
    def __init__(self, pruner, sample_fraction=0.1, zeroing_metric='accuracy', logger=None):
        """
        Initialize the NeuronZeroing experiment.

        Args:
            pruner (IterativeMagnitudePruning): The pruner instance containing the model and training info.
            sample_fraction (float): Fraction of neurons to sample if the model is too large.
            zeroing_metric (str): Metric to use for evaluating performance (e.g., 'accuracy').
            logger (logging.Logger, optional): Logger for logging details.
        """
        self.pruner = pruner
        self.model = deepcopy(pruner.model)
        self.model.load_state_dict(pruner.best_model_weights)
        self.model.eval()
        self.model.to(pruner.device)
        self.save_dir = pruner.save_dir + '/neuron_zeroing'
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure the save directory exists
        self.sample_fraction = sample_fraction
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics = {
            'neuron_accuracy_drops': [],
            'sparsity_metrics': [],
            'precision_recall_fscore': [],
            'loss_metrics': [],
            'layer_accuracy_drops': []
        }

        # Set up logging to file
        log_file = os.path.join(self.save_dir, 'experiment.log')
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Store experiment metadata
        self.metadata = {
            'model_architecture': str(self.model),
            'sample_fraction': self.sample_fraction,
            'zeroing_metric': self.zeroing_metric
        }

    def evaluate_performance(self) -> float:
        """
        Evaluate the model's performance on the test dataset using accuracy.

        Returns:
            float: Accuracy of the model on the test set.
        """
        self.model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients during evaluation
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        return accuracy

    def compute_sparsity(self) -> float:
        """
        Compute the sparsity of the model by counting zero weights.

        Returns:
            float: Fraction of zero weights in the model.
        """
        total_params = 0
        zero_params = 0
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        sparsity = zero_params / total_params
        return sparsity

    def compute_precision_recall_fscore(self) -> dict:
        """
        Compute precision, recall, and F1-score on the test dataset.

        Returns:
            dict: Dictionary containing precision, recall, and F1-score for each class.
        """
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        return {'precision': precision.tolist(), 'recall': recall.tolist(), 'fscore': fscore.tolist()}

    def evaluate_loss(self) -> float:
        """
        Evaluate the model's loss on the test dataset using the criterion.

        Returns:
            float: Loss of the model on the test set.
        """
        self.model.eval()
        criterion = self.pruner.criterion
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
        return total_loss / total_samples
    

    def run_experiment(self):
        """
        Run the neuron zeroing experiment by zeroing out individual neurons (or a subset of neurons).
        """
        self.logger.info("Starting neuron zeroing experiment...")

        baseline_accuracy = self.evaluate_performance()
        baseline_sparsity = self.compute_sparsity()
        baseline_metrics = self.compute_precision_recall_fscore()

        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        self.logger.info(f"Baseline sparsity: {baseline_sparsity:.4f}")
        self.logger.info(f"Baseline precision: {baseline_metrics['precision']}")
        self.logger.info(f"Baseline recall: {baseline_metrics['recall']}")
        self.logger.info(f"Baseline F1-score: {baseline_metrics['fscore']}")

        total_neurons = 0
        sampled_neurons = []

        # Gather all neurons (by considering the layer outputs, i.e., activations)
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):  # Focus on fully connected layers
                total_neurons += layer.out_features
                sampled_neurons.append(layer)

        # No model size threshold, zero out all neurons in the sampled layers
        self.logger.info(f"Zeroing neurons from {total_neurons} total neurons.")

        # Zero out each neuron and evaluate performance with tqdm for progress tracking
        for layer in tqdm(sampled_neurons, desc="Zeroing neurons in layers", unit="layer"):
            original_weights = layer.weight.clone()  # Save original weights
            for i in tqdm(range(layer.out_features), desc=f"Zeroing neurons in {layer.__class__.__name__}", unit="neuron", leave=False):
                layer.weight.data[i, :] = 0  # Zero out the neuron (entire row of weights)
                accuracy_after_zeroing = self.evaluate_performance()
                sparsity_after_zeroing = self.compute_sparsity()
                metrics_after_zeroing = self.compute_precision_recall_fscore()

                accuracy_drop = baseline_accuracy - accuracy_after_zeroing
                self.metrics['neuron_accuracy_drops'].append({
                    'layer_name': layer.__class__.__name__,
                    'neuron_index': i,
                    'accuracy_drop': accuracy_drop,
                    'original_weights': original_weights[i].tolist()
                })

                # Collect sparsity and precision-recall data
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

                # Record loss metrics (assumed to be computed during evaluation)
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

                layer.weight.data[i, :] = original_weights[i]  # Restore neuron
                self.logger.debug(f"Zeroed neuron {layer.__class__.__name__}, accuracy drop: {accuracy_drop:.4f}, sparsity: {sparsity_after_zeroing:.4f}")

        self.logger.info("Neuron zeroing experiment completed.")

        # Save the experiment results as a JSON file
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")

        # Save the experiment metadata
        metadata_file = os.path.join(self.save_dir, 'experiment_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        self.logger.info(f"Metadata saved to {metadata_file}")

        self.plot_results()  # Plot the results

        return self.metrics

    def plot_results(self):
        """Plot the results of the neuron zeroing experiment."""
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        sparsities = [entry['sparsity'] for entry in self.metrics['sparsity_metrics']]
        layer_accuracy_drops = {layer_name: [] for layer_name in set([entry['layer_name'] for entry in self.metrics['layer_accuracy_drops']])}
        
        # Group accuracy drops by layer
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
