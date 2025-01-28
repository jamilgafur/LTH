import json
import matplotlib.pyplot as plt
import random
import os
import torch
import logging
import pickle
from tqdm import tqdm
from pyPrune.utils import *
from typing import Dict, List, Optional


class WeightZeroing:
    """
    A class that implements the weight zeroing experiment for model pruning.
    
    This experiment evaluates the effect of zeroing individual weights on model accuracy 
    by gradually zeroing a fraction of weights and monitoring the accuracy drop at each step.
    """

    def __init__(self, 
                 sample_fractions: dict,
                 pruner: 'IterativeMagnitudePruning',  
                 zeroing_metric: str = 'accuracy', 
                 logger: Optional[logging.Logger] = None, 
                 save_plots: bool = True):
        """
        Initializes the WeightZeroing experiment.
        
        Args:
            pruner (IterativeMagnitudePruning): The pruner instance containing the model and training info.
            sample_fraction (float): Fraction of weights to sample for zeroing.
            zeroing_metric (str): Metric to use for evaluating performance (default is 'accuracy').
            logger (Optional[logging.Logger]): Logger for logging details. If None, default logger is used.
            save_plots (bool): Whether to save plots after the experiment.
        """
        self.pruner = pruner
        self.experiment_metrics: Dict[str, Dict] = {}

        self.model = pruner.model  # Directly use the pruner's model instead of deepcopy
        self.save_dir = os.path.join(pruner.save_dir, f"weightZeroing_{''.join([str(key) + str(value) for key, value in sample_fractions.items()])}_{zeroing_metric}")
        self.sample_fractions = sample_fractions
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics: Dict[str, List] = {'weight_accuracy_drops': [], 'total_accuracy_drops': [], 'zeroed_weights_count': 0, 'step_accuracy': []}
        self.save_plots = save_plots
        os.makedirs(self.save_dir, exist_ok=True)

        # Check if pruner state already exists
        self.checkpoint_file = os.path.join(self.save_dir, 'weight_zeroing.pkl')
        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()

    def evaluate_performance(self) -> float:
        """
        Evaluates the model's performance on the test dataset.
        
        Returns:
            float: The accuracy of the model on the test dataset.
        """
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
        return accuracy

    def zero_weight(self, weight_tensor: torch.Tensor, idx: int) -> float:
        """
        Zeros out a specific weight in the tensor at index `idx`.
        
        Args:
            weight_tensor (torch.Tensor): The tensor containing the weights.
            idx (int): The index of the weight to zero.
        
        Returns:
            float: The original value of the weight at index `idx`.
        """
        weight_tensor = weight_tensor.detach().clone()  # Avoid gradients
        original_value = weight_tensor[idx].item()
        weight_tensor[idx] = 0  # Zero out the weight
        return original_value

    def run_experiment(self) -> Dict[str, List]:
        """
        Runs the weight zeroing experiment by progressively zeroing a fraction of weights 
        and evaluating the performance after each zeroing.
        
        Returns:
            Dict[str, List]: The collected metrics during the experiment.
        """
        try:
            for weight, step in zip(self.pruner.weight_history, self.pruner.steps):
                self.model.load_state_dict(weight, strict=False)
                self.model.eval().to(self.pruner.device)

                baseline_accuracy = self.evaluate_performance()
                total_weights, weight_list = self.collect_weights()

                zeroed_weights, step_accuracy = 0, []
                for layer_type, fraction in self.sample_fractions.items():
                    # Collect weights only from the layers that match the layer type
                    prunable_names, prunable_params = get_pruneable_named_parameters(self.model, self.pruner.prunable_layers)

                    # Filter out weights that do not belong to the current layer type
                    filtered_weights = []
                    for name, param in zip(prunable_names, prunable_params):
                        if layer_type in name and 'weight' in name:
                            filtered_weights.extend(param.view(-1).tolist())
                    
                    num_sampled_weights = int(fraction * len(filtered_weights))
                    sampled_indices = random.sample(range(len(filtered_weights)), num_sampled_weights)

                    for idx in tqdm(sampled_indices, desc=f"Zeroing weights in {layer_type}"):
                        param_idx = 0
                        for name, param in zip(prunable_names, prunable_params):
                            if layer_type in name and 'weight' in name:
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

                self.metrics['step_accuracy'].append(step_accuracy)
                self.save_metrics()

                self.experiment_metrics[step] = self.metrics
                self.plot_results(step)  # Plot results after each step
                # clear the metrics for the next step
                self.metrics = {'weight_accuracy_drops': [], 'total_accuracy_drops': [], 'zeroed_weights_count': 0, 'step_accuracy': []}

            return self.experiment_metrics
        except Exception as e:
            self.logger.error(f"An error occurred during Weight Zeroing experiment: {e}")
            self.logger.error("Experiment terminated.")
            return {}

    def collect_weights(self) -> (int, List[float]):
        """
        Collects all weights from the model and returns the total weight count 
        along with a flattened list of all weight values.
        
        Returns:
            (int, List[float]): A tuple containing the total number of weights 
            and a list of all weight values.
        """
        total_weights, weight_list = 0, []
        # Use the helper function to get prunable parameters
        prunable_names, prunable_params = get_pruneable_named_parameters(self.model, self.pruner.prunable_layers)
        
        for name, param in zip(prunable_names, prunable_params):
            if 'weight' in name:
                total_weights += param.numel()
                weight_list.extend(param.view(-1).tolist())
        return total_weights, weight_list

    def track_metrics(self, weight_name: str, original_value: float, accuracy_drop: float):
        """
        Stores the metrics for a zeroed weight during the experiment.
        
        Args:
            weight_name (str): The name of the weight parameter.
            original_value (float): The original value of the weight before zeroing.
            accuracy_drop (float): The accuracy drop after zeroing the weight.
        """
        self.metrics['weight_accuracy_drops'].append({'weight_name': weight_name, 'accuracy_drop': accuracy_drop, 'original_value': original_value})
        print("Before incrementing:", self.metrics['zeroed_weights_count'])  # Debug line
        self.metrics['zeroed_weights_count'] += 1
        print("After incrementing:", self.metrics['zeroed_weights_count'])  # Debug line

    def save_metrics(self):
        """
        Saves the metrics from the weight zeroing experiment to a JSON file and the current 
        WeightZeroing object to a pickle file for later use or recovery.
        """
        save_path = os.path.join(self.save_dir, f"weight_zeroing_metrics.json")
        with open(save_path, 'w') as f:
            json.dump(self.experiment_metrics, f, indent=4)

        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(self, f)

    def plot_results(self, step: int):
        """
        Generates and saves plots for the experiment results after each pruning step.
        """
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['weight_accuracy_drops']]
        step_accuracy = self.metrics['step_accuracy'][-1]

        # Accuracy Drop Distribution
        self.plot_histogram(accuracy_drops, f'Accuracy Drop Distribution at Step {step}', 'Accuracy Drop', 'Frequency', f'accuracy_drop_step_{step}')

        # Accuracy Change Over Steps
        self.plot_line_plot(range(1, len(step_accuracy) + 1), step_accuracy, f'Accuracy Change Over Steps at Step {step}', 'Step', 'Accuracy', f'accuracy_over_steps_step_{step}')

        # Original Weight Distribution
        original_weights = [entry['original_value'] for entry in self.metrics['weight_accuracy_drops']]
        self.plot_histogram(original_weights, f'Original Weight Distribution at Step {step}', 'Weight Value', 'Frequency', f'original_weight_distribution_step_{step}')

    def plot_histogram(self, data: List[float], title: str, xlabel: str, ylabel: str, filename: str):
        """
        Plots and saves a histogram of the given data.
        
        Args:
            data (List[float]): The data to plot.
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            filename (str): The filename to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=30, alpha=0.7)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))

    def plot_line_plot(self, x: List[int], y: List[float], title: str, xlabel: str, ylabel: str, filename: str):
        """
        Plots and saves a line plot of the given data.
        
        Args:
            x (List[int]): The x-values for the plot.
            y (List[float]): The y-values for the plot.
            title (str): The title of the plot.
            xlabel (str): The label for the x-axis.
            ylabel (str): The label for the y-axis.
            filename (str): The filename to save the plot.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, marker='o', color='red')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))

    def load_checkpoint(self):
        """Load checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
        with open(self.checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            self.model.load_state_dict(checkpoint.model.state_dict())
            self.metrics = checkpoint.metrics
