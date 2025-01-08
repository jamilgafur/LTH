import torch
import torch.nn as nn
import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pyPrune.utils import plot_loss_accuracy_sparsity
from scipy.stats import ttest_rel
from copy import deepcopy

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
        self.model = deepcopy(pruner.best_model)
        self.model.load_state_dict(pruner.best_model_weights)
        self.model.eval()
        self.model.to(pruner.device)
        self.sample_fraction = sample_fraction
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics = {'neuron_accuracy_drops': []}

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

    def run_experiment(self):
        """
        Run the neuron zeroing experiment by zeroing out individual neurons (or a subset of neurons).
        """
        self.logger.info("Starting neuron zeroing experiment...")
        
        baseline_accuracy = self.evaluate_performance()
        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        total_neurons = 0
        sampled_neurons = []

        # Gather all neurons (by considering the layer outputs, i.e., activations)
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):  # Focus on fully connected layers
                total_neurons += layer.out_features
                sampled_neurons.append(layer)

        # If model is too large, sample neurons to reduce computational burden
        if total_neurons > 1000:  # Arbitrary large model size threshold
            num_sampled_neurons = int(self.sample_fraction * total_neurons)
            sampled_neurons = random.sample(sampled_neurons, num_sampled_neurons)
            self.logger.info(f"Sampling {num_sampled_neurons} neurons from {total_neurons} total neurons.")

        # Zero out each neuron and evaluate performance
        for layer in sampled_neurons:
            original_weights = layer.weight.clone()  # Save original weights
            for i in range(layer.out_features):
                layer.weight.data[i, :] = 0  # Zero out the neuron (entire row of weights)
                accuracy_after_zeroing = self.evaluate_performance()
                accuracy_drop = baseline_accuracy - accuracy_after_zeroing
                self.metrics['neuron_accuracy_drops'].append({
                    'layer_name': layer.__class__.__name__,
                    'neuron_index': i,
                    'accuracy_drop': accuracy_drop,
                    'original_weights': original_weights[i].tolist()
                })
                layer.weight.data[i, :] = original_weights[i]  # Restore neuron
                self.logger.debug(f"Zeroed neuron {layer.__class__.__name__}, accuracy drop: {accuracy_drop:.4f}")

        self.logger.info("Neuron zeroing experiment completed.")
        return self.metrics

    def plot_results(self):
        """Plot the results of the neuron zeroing experiment."""
        import matplotlib.pyplot as plt

        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        plt.hist(accuracy_drops, bins=30)
        plt.title('Impact of Neuron Zeroing on Accuracy')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')
        plt.show()