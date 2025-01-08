import torch
import torch.nn as nn
import random
import logging
import numpy as np
from copy import deepcopy

class WeightZeroing:
    def __init__(self, pruner, sample_fraction=0.1, zeroing_metric='accuracy', logger=None):
        """
        Initialize the WeightZeroing experiment.

        Args:
            pruner (IterativeMagnitudePruning): The pruner instance containing the model and training info.
            sample_fraction (float): Fraction of weights to sample if the model is too large.
            zeroing_metric (str): Metric to use for evaluating performance (e.g., 'accuracy').
            logger (logging.Logger, optional): Logger for logging details.
        """
        self.pruner = pruner
        self.model = deepcopy(pruner.model)
        self.model.load_state_dict(pruner.best_model_weights)
        self.model.eval()
        self.model.to(pruner.device)
        self.sample_fraction = sample_fraction
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics = {'weight_accuracy_drops': []}

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
        Run the weight zeroing experiment by zeroing out individual weights (or a subset of weights).
        """
        self.logger.info("Starting weight zeroing experiment...")
        
        baseline_accuracy = self.evaluate_performance()
        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        total_weights = 0
        sampled_weights = []

        # Gather all weight parameters in the model
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Focus on weights
                total_weights += param.numel()
                sampled_weights.append(param)

        # If model is too large, sample weights to reduce computational burden
        if total_weights > 100000:  # Arbitrary large model size threshold
            num_sampled_weights = int(self.sample_fraction * total_weights)
            sampled_weights = random.sample(sampled_weights, num_sampled_weights)
            self.logger.info(f"Sampling {num_sampled_weights} weights from {total_weights} total weights.")

        # Zero out each weight and evaluate performance
        for param in sampled_weights:
            original_weights = param.clone()  # Save original weights
            with torch.no_grad():  # Disable gradient tracking for in-place operations
                for idx in range(param.numel()):
                    param.data.view(-1)[idx] = 0  # Zero out this weight (in-place)
                    accuracy_after_zeroing = self.evaluate_performance()
                    accuracy_drop = baseline_accuracy - accuracy_after_zeroing
                    
                    # Fix to access a scalar value by flattening the weight tensor
                    original_value = original_weights.view(-1)[idx].item()

                    self.metrics['weight_accuracy_drops'].append({
                        'weight_name': param.name,
                        'accuracy_drop': accuracy_drop,
                        'original_value': original_value
                    })
                    param.data.view(-1)[idx] = original_weights.view(-1)[idx]  # Restore weight
                    self.logger.debug(f"Zeroed weight {param.name}, accuracy drop: {accuracy_drop:.4f}")

        self.plot_results()
        self.logger.info("Weight zeroing experiment completed.")
        return self.metrics

    def plot_results(self):
        """Plot the results of the weight zeroing experiment."""
        import matplotlib.pyplot as plt

        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['weight_accuracy_drops']]
        plt.hist(accuracy_drops, bins=30)
        plt.title('Impact of Weight Zeroing on Accuracy')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')
        plt.savefig(self.pruner.save_dir + '/weight_zeroing_accuracy_drop.png')
