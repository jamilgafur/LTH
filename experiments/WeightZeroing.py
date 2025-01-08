import torch
import torch.nn as nn
import random
import logging
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy

class WeightZeroing:
    def __init__(self, pruner, sample_fraction=0.01, zeroing_metric='accuracy', logger=None, save_plots=True):
        """
        Initialize the WeightZeroing experiment.

        Args:
            pruner (IterativeMagnitudePruning): The pruner instance containing the model and training info.
            sample_fraction (float): Fraction of weights to sample.
            zeroing_metric (str): Metric to use for evaluating performance.
            logger (logging.Logger, optional): Logger for logging details.
            save_plots (bool): Whether to save plots after the experiment.
        """
        self.pruner = pruner
        self.model = deepcopy(pruner.model)  # Avoid deepcopy if large models; could load checkpoint instead
        self.model.load_state_dict(pruner.best_model_weights)
        self.model.eval().to(pruner.device)
        
        self.save_dir = os.path.join(pruner.save_dir, 'weight_zeroing')
        self.sample_fraction = sample_fraction
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics = {'weight_accuracy_drops': [], 'total_accuracy_drops': [], 'zeroed_weights_count': 0, 'step_accuracy': []}
        
        self.save_plots = save_plots
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure directory exists
        self.logger.info(f"WeightZeroing initialized with sample_fraction={sample_fraction}, zeroing_metric={zeroing_metric}")

    def evaluate_performance(self) -> float:
        """Evaluate the model's performance on the test dataset."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in self.pruner.test_loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total
        self.logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        return accuracy

    def zero_weight(self, weight_tensor, idx):
        """Zero out a specific weight tensor at index `idx`."""
        # Detach the tensor to ensure it doesn't require gradients
        weight_tensor = weight_tensor.detach().clone()  # Make a copy of the tensor without tracking gradients
        original_value = weight_tensor[idx].item()
        weight_tensor[idx] = 0  # Zero out the weight
        return original_value

    def run_experiment(self):
        """
        Run the weight zeroing experiment.
        """
        baseline_accuracy = self.evaluate_performance()
        self.logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        total_weights, weight_list = self.collect_weights()

        num_sampled_weights = int(self.sample_fraction * total_weights)
        self.logger.info(f"Sampling {num_sampled_weights} out of {total_weights} total weights.")

        sampled_indices = random.sample(range(len(weight_list)), num_sampled_weights)
        self.logger.info(f"Randomly selected {num_sampled_weights} weight indices.")

        zeroed_weights, step_accuracy = 0, []
        for idx in tqdm(sampled_indices, desc="Zeroing weights", ncols=100):
            param_idx = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    weight_tensor = param.view(-1)
                    if idx < param_idx + weight_tensor.size(0):
                        original_value = self.zero_weight(weight_tensor, idx - param_idx)
                        accuracy_after_zeroing = self.evaluate_performance()
                        accuracy_drop = baseline_accuracy - accuracy_after_zeroing
                        self.track_metrics(name, original_value, accuracy_drop)
                        step_accuracy.append(accuracy_after_zeroing)
                        zeroed_weights += 1
                        break
                    param_idx += weight_tensor.size(0)

        self.logger.info(f"Total {zeroed_weights} weights zeroed.")
        self.save_metrics()
        if self.save_plots:
            self.plot_results()
        return self.metrics

    def collect_weights(self):
        """Collect all weights and return total weight count."""
        total_weights, weight_list = 0, []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_weights += param.numel()
                weight_list.extend(param.view(-1).tolist())
        return total_weights, weight_list

    def track_metrics(self, weight_name, original_value, accuracy_drop):
        """Store the metrics for zeroed weight."""
        self.metrics['weight_accuracy_drops'].append({
            'weight_name': weight_name, 'accuracy_drop': accuracy_drop, 'original_value': original_value
        })
        self.metrics['zeroed_weights_count'] += 1

    def save_metrics(self):
        """Save the weight zeroing experiment metrics to a JSON file."""
        save_path = os.path.join(self.save_dir, 'weight_zeroing_metrics.json')
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f)
        self.logger.info(f"Metrics saved to {save_path}")

    def plot_results(self):
        """Generate and save plots."""
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['weight_accuracy_drops']]
        self.plot_histogram(accuracy_drops, 'Accuracy Drop Distribution', 'Accuracy Drop', 'Frequency', 'accuracy_drop')

        step_accuracy = [entry['accuracy'] for entry in self.metrics['step_accuracy']]
        self.plot_line_plot(range(1, len(step_accuracy) + 1), step_accuracy, 'Accuracy Change Over Steps', 'Step', 'Accuracy', 'accuracy_over_steps')

        original_weights = [entry['original_value'] for entry in self.metrics['weight_accuracy_drops']]
        self.plot_histogram(original_weights, 'Original Weight Distribution', 'Weight Value', 'Frequency', 'original_weights')

    def plot_histogram(self, data, title, xlabel, ylabel, filename):
        """Plot and save a histogram."""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        self.logger.info(f"{title} saved to {self.save_dir}/{filename}.png")

    def plot_line_plot(self, x, y, title, xlabel, ylabel, filename):
        """Plot and save a line plot."""
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        self.logger.info(f"{title} saved to {self.save_dir}/{filename}.png")
