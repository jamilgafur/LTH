import json
import os
import pickle
import logging
import copy
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from pyPrune.utils import get_pruneable_named_modules  # Assuming this is defined elsewhere


class NeuronZeroing:
    def __init__(self, pruner: Any, zeroing_metric: str = 'accuracy', logger: Optional[logging.Logger] = None) -> None:
        """
        Initialize the NeuronZeroing experiment.
        
        Args:
            pruner: An object containing the model, device, test_loader, criterion, weight_history, etc.
            zeroing_metric: The metric used for zeroing decisions.
            logger: Optional logger. If not provided, a default logger is created.
        """
        self.pruner = pruner
        self.model = pruner.model  # Use the reference directly
        self.prunerable_layer = self.pruner.prunable_layers
        self.zeroing_metric = zeroing_metric

        # Set up the save directory and logging
        self.save_dir = os.path.join(pruner.save_dir, f"neuronZeroing_{zeroing_metric}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logger if logger else logging.getLogger(__name__)
        log_file = os.path.join(self.save_dir, 'experiment.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Initialize metrics dictionary
        self.metrics: Dict[str, List[Dict[str, Any]]] = {
            'neuron_accuracy_drops': [],
            'sparsity_metrics': [],
            'precision_recall_fscore': [],
            'loss_metrics': [],
            'layer_accuracy_drops': [],  # Added per-layer accuracy drops
            'loss_changes': []           # To track loss changes after zeroing each neuron
        }
        self.checkpoint_file = os.path.join(self.save_dir, 'neuron_zeroing.pkl')
        self.metadata: Dict[str, Any] = {
            'model_architecture': str(self.model),
            'zeroing_metric': self.zeroing_metric,
            'last_step': 0
        }

        # Load checkpoint if it exists
        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
        else:
            self.logger.info("Starting new NeuronZeroing experiment.")

    def evaluate_metrics(self, loader: Optional[torch.utils.data.DataLoader] = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Evaluate performance metrics over the provided loader.
        
        Args:
            loader: DataLoader for evaluation; defaults to pruner.test_loader.
            
        Returns:
            A tuple (accuracy, avg_loss, detailed_metrics) where detailed_metrics includes precision, recall, and fscore.
        """
        loader = loader or self.pruner.test_loader
        self.model.eval()
        y_true, y_pred = [], []
        total_loss, total_samples = 0.0, 0
        correct, total = 0, 0

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                loss = self.pruner.criterion(output, target)
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                correct += (predicted == target).sum().item()
                total += batch_size

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        metrics_detail = {'precision': precision, 'recall': recall, 'fscore': fscore}
        return accuracy, avg_loss, metrics_detail

    def compute_sparsity(self) -> float:
        """
        Compute the overall sparsity of the model (fraction of parameters that are zero).
        
        Returns:
            A float representing the sparsity.
        """
        total_params, zero_params = 0, 0
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0.0

    def zero_and_restore_neurons(self, layer: nn.Module, layer_index: str, idx: int, is_conv_layer: bool = True) -> None:
        original_weights = layer.weight.data.clone()  # Save original weights

        # Zero out the specified neuron/channel
        if is_conv_layer:
            layer.weight.data[idx, :, :, :] = 0
            self.logger.info(f"Zeroed output channel {idx} in {layer.__class__.__name__}_{layer_index}")
        else:
            layer.weight.data[idx, :] = 0
            self.logger.info(f"Zeroed neuron {idx} in {layer.__class__.__name__}_{layer_index}")

        # Evaluate metrics with the modified weights
        accuracy_after, loss_after, metrics_after = self.evaluate_metrics()
        sparsity_after = self.compute_sparsity()

        # Calculate the drop in accuracy and change in loss (normalized)
        accuracy_drop = self.baseline_accuracy - accuracy_after
        loss_change = self.baseline_loss - loss_after

        normalized_accuracy_drop = (accuracy_drop / self.baseline_accuracy) if self.baseline_accuracy else accuracy_drop
        normalized_loss_change = (loss_change / self.baseline_loss) if self.baseline_loss else loss_change

        # Record metrics for this neuron
        record = {
            'layer_name': f"{layer.__class__.__name__}_{layer_index}",
            'neuron_index': idx,
            'accuracy_drop': normalized_accuracy_drop  # Normalized accuracy drop
        }
        self.metrics['neuron_accuracy_drops'].append(record)
        self.metrics['sparsity_metrics'].append({
            'layer_name': f"{layer.__class__.__name__}_{layer_index}",
            'neuron_index': idx,
            'sparsity': sparsity_after
        })
        self.metrics['precision_recall_fscore'].append({
            'layer_name': f"{layer.__class__.__name__}_{layer_index}",
            'neuron_index': idx,
            **metrics_after
        })
        self.metrics['loss_metrics'].append({
            'layer_name': f"{layer.__class__.__name__}_{layer_index}",
            'neuron_index': idx,
            'loss': loss_after
        })
        self.metrics['loss_changes'].append({
            'layer_name': f"{layer.__class__.__name__}_{layer_index}",
            'neuron_index': idx,
            'loss_change': normalized_loss_change  # Normalized loss change
        })

        # Restore the original weights
        layer.weight.data = original_weights

    def run_experiment(self) -> Dict[Any, Dict[str, List[Dict[str, Any]]]]:
        """
        Run the neuron zeroing experiment over all provided weight history steps.
        
        Returns:
            A dictionary mapping each experiment step to the corresponding metrics.
        """
        all_metrics: Dict[Any, Dict[str, List[Dict[str, Any]]]] = {}
        last_step = self.metadata.get('last_step', 0)

        for idx, (weights, step) in enumerate(zip(self.pruner.weight_history, self.pruner.steps)):
            if step <= last_step:
                self.logger.info(f"Skipping step {step} (already processed).")
                continue

            self.logger.info(f"Processing step {step}...")
            self.model.load_state_dict(weights, strict=False)
            self.model.to(self.pruner.device)
            self.baseline_accuracy, self.baseline_loss, baseline_metrics = self.evaluate_metrics()
            baseline_sparsity = self.compute_sparsity()

            self.logger.info(
                f"Baseline metrics for step {step}: Accuracy: {self.baseline_accuracy:.4f}, "
                f"Loss: {self.baseline_loss:.4f}, Sparsity: {baseline_sparsity:.4f}"
            )
            self.logger.info(
                f"Precision: {baseline_metrics['precision']}, "
                f"Recall: {baseline_metrics['recall']}, F1-Score: {baseline_metrics['fscore']}"
            )

            # Get the pruneable layers from the model
            names, layers = get_pruneable_named_modules(self.model, self.prunerable_layer)
            # Only keep layers that have either out_channels or out_features
            sampled_layers = [layer for layer in layers if hasattr(layer, 'out_channels') or hasattr(layer, 'out_features')]

            # For each sampled layer, zero out every neuron/channel
            for layer_index, layer in enumerate(tqdm(sampled_layers, desc="Processing layers", unit="layer")):
                layer_id = str(layer_index)
                if hasattr(layer, 'out_channels'):
                    for i in tqdm(range(layer.out_channels), desc=f"Zeroing channels in {layer.__class__.__name__}_{layer_id}", unit="channel", leave=False):
                        self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=True)
                elif hasattr(layer, 'out_features'):
                    for i in tqdm(range(layer.out_features), desc=f"Zeroing neurons in {layer.__class__.__name__}_{layer_id}", unit="neuron", leave=False):
                        self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=False)

                # Save checkpoint and metrics after processing each layer
                self.save_checkpoint(step)
                self.save_metrics(step)

            # Save a copy of the metrics for this step (avoid later clearing issues)
            all_metrics[step] = copy.deepcopy(self.metrics)
            self.plot_results(step)

            # Clear metrics to prevent memory bloat for the next step
            for key in self.metrics:
                self.metrics[key].clear()

            # Update metadata with the last processed step
            self.metadata['last_step'] = step

        return all_metrics

    def plot_results(self, step: Any) -> None:
        """
        Plot and save results for the given step.
        
        Args:
            step: Identifier for the current experiment step.
        """
        # Extract data for plotting
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        sparsities = [entry['sparsity'] for entry in self.metrics['sparsity_metrics']]
        loss_changes = [entry['loss_change'] for entry in self.metrics['loss_changes']]
        loss_metrics = [entry['loss'] for entry in self.metrics['loss_metrics']]

        # Organize layer-level accuracy drops
        layer_accuracy_drops: Dict[str, List[float]] = {}
        for entry in self.metrics['layer_accuracy_drops']:
            layer_name = entry['layer_name']
            layer_accuracy_drops.setdefault(layer_name, []).append(entry['accuracy_drop'])

        plt.figure(figsize=(14, 12))

        # Plot histogram of neuron-level accuracy drops
        plt.subplot(2, 3, 1)
        plt.hist(accuracy_drops, bins=30)
        plt.title(f'Step {step}: Distribution of Neuron Accuracy Drops')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')

        # Scatter plot: sparsity vs. accuracy drop
        plt.subplot(2, 3, 2)
        plt.scatter(sparsities, accuracy_drops, alpha=0.5)
        plt.title(f'Step {step}: Sparsity vs. Accuracy Drop')
        plt.xlabel('Sparsity')
        plt.ylabel('Accuracy Drop')

        # Histogram for per-layer accuracy drops (if available)
        plt.subplot(2, 3, 3)
        if layer_accuracy_drops:
            for layer_name, drops in layer_accuracy_drops.items():
                plt.hist(drops, bins=20, alpha=0.5, label=layer_name)
            plt.title(f'Step {step}: Accuracy Drop per Layer')
            plt.xlabel('Accuracy Drop')
            plt.ylabel('Frequency')
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No layer accuracy drop data', ha='center')

        # Histogram of loss changes
        plt.subplot(2, 3, 4)
        plt.hist(loss_changes, bins=30, color='orange', alpha=0.7)
        plt.title(f'Step {step}: Loss Change Distribution')
        plt.xlabel('Loss Change')
        plt.ylabel('Frequency')

        # Scatter plot: loss vs. sparsity
        plt.subplot(2, 3, 5)
        plt.scatter(sparsities, loss_metrics, alpha=0.5)
        plt.title(f'Step {step}: Loss vs. Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Loss')

        plt.tight_layout()
        plot_file = os.path.join(self.save_dir, f'accuracy_drop_vs_sparsity_step_{step}.png')
        plt.savefig(plot_file)
        self.logger.info(f"Plot for step {step} saved to {plot_file}")
        plt.close()

    def load_checkpoint(self) -> None:
        """
        Load experiment checkpoint from file.
        """
        self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.metrics = checkpoint['metrics']
                self.metadata = checkpoint['metadata']
        except pickle.UnpicklingError as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            self.logger.error("The checkpoint file might be corrupted.")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading checkpoint: {e}")
            raise

    def save_checkpoint(self, step: Optional[Any] = None) -> None:
        """
        Save a checkpoint of the current experiment state.
        
        Args:
            step: Optional step identifier to store as the last processed step.
        """
        if step is not None:
            self.metadata['last_step'] = step
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'metadata': self.metadata
        }
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info(f"Checkpoint saved to {self.checkpoint_file}")

    def save_metrics(self, step: Optional[Any] = None) -> None:
        """
        Save the current metrics to a JSON file. This method converts any NumPy arrays to lists.
        
        Args:
            step: Optional step identifier (not used in the filename here, but can be extended).
        """
        def convert_ndarray(obj: Any) -> Any:
            """Recursively convert numpy ndarrays to lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_ndarray(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_ndarray(value) for key, value in obj.items()}
            else:
                return obj

        converted_metrics = convert_ndarray(self.metrics)
        metrics_file = os.path.join(self.save_dir, f'metrics_{step}.json')
        with open(metrics_file, 'w') as f:
            json.dump(converted_metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")
