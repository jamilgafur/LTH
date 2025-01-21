import json
import torch
import torch.nn as nn
import os
import logging
import pickle
from pyPrune.utils import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support



class NeuronZeroing:
    def __init__(self, pruner, zeroing_metric='accuracy', logger=None):
        self.pruner = pruner
        self.model = pruner.model  # No need for deepcopy, use the reference directly
        self.prunerable_layer = self.pruner.prunable_layers
        self.save_dir = os.path.join(pruner.save_dir, f"neuronZeroing_{zeroing_metric}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.zeroing_metric = zeroing_metric
        self.logger = logger if logger else logging.getLogger(__name__)
        self.metrics = {
            'neuron_accuracy_drops': [],
            'sparsity_metrics': [],
            'precision_recall_fscore': [],
            'loss_metrics': [],
            'layer_accuracy_drops': [],
            'loss_changes': []  # To track loss changes after zeroing each neuron
        }
        self.checkpoint_file = os.path.join(self.save_dir, 'neuron_zeroing.pkl')

        # Load checkpoint if exists
        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
        else:
            log_file = os.path.join(self.save_dir, 'experiment.log')
            logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            self.metadata = {
                'model_architecture': str(self.model),
                'zeroing_metric': self.zeroing_metric
            }

    def evaluate_metrics(self, loader=None):
        """Evaluate performance metrics in a batch-wise manner to avoid redundant computations."""
        loader = loader or self.pruner.test_loader
        self.model.eval()
        y_true, y_pred = [], []
        total_loss, total_samples = 0, 0
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.pruner.device), target.to(self.pruner.device)
                output = self.model(data)
                loss = self.pruner.criterion(output, target)
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                _, predicted = torch.max(output.data, 1)
                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                correct += (predicted == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        return accuracy, total_loss / total_samples, {'precision': precision, 'recall': recall, 'fscore': fscore}

    def compute_sparsity(self):
        """Compute the sparsity of the model."""
        total_params, zero_params = 0, 0
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        return zero_params / total_params

    def run_experiment(self):
        try:
            metrics = {}  
            last_step = self.metadata.get('last_step', 0)  # Get the last completed step from metadata

            for idx, (weights, step) in enumerate(zip(self.pruner.weight_history, self.pruner.steps)):
                if step <= last_step:
                    continue  # Skip steps that have already been processed

                self.logger.info(f"Processing step {step}...")
                self.model.load_state_dict(weights)
                self.model.to(self.pruner.device)
                baseline_accuracy, baseline_loss, baseline_metrics = self.evaluate_metrics()
                baseline_sparsity = self.compute_sparsity()

                self.logger.info(f"Baseline Accuracy: {baseline_accuracy:.4f}, Loss: {baseline_loss:.4f}, Sparsity: {baseline_sparsity:.4f}")
                self.logger.info(f"Precision: {baseline_metrics['precision']}, Recall: {baseline_metrics['recall']}, F1-Score: {baseline_metrics['fscore']}")

                names, layers = get_pruneable_named_modules(self.model, self.prunerable_layer)
                sampled_neurons = [layer for name, layer in zip(names, layers) if hasattr(layer, 'out_features')]
                
                for layer in tqdm(sampled_neurons, desc="Zeroing neurons in layers", unit="layer"):
                    original_weights = layer.weight.data.clone()  # Use data to avoid creating unnecessary tensor objects
                    for i in tqdm(range(layer.out_features), desc=f"Zeroing neurons in {layer.__class__.__name__}", unit="neuron", leave=False):
                        layer.weight.data[i, :] = 0  # Zero the neuron
                        accuracy_after, loss_after, metrics_after = self.evaluate_metrics()
                        sparsity_after = self.compute_sparsity()

                        accuracy_drop = baseline_accuracy - accuracy_after
                        loss_change = baseline_loss - loss_after  # Calculate the loss change

                        self.metrics['neuron_accuracy_drops'].append({
                            'layer_name': layer.__class__.__name__, 'neuron_index': i, 'accuracy_drop': accuracy_drop
                        })
                        self.metrics['sparsity_metrics'].append({
                            'layer_name': layer.__class__.__name__, 'neuron_index': i, 'sparsity': sparsity_after
                        })
                        self.metrics['precision_recall_fscore'].append({
                            'layer_name': layer.__class__.__name__, 'neuron_index': i, **metrics_after
                        })
                        self.metrics['loss_metrics'].append({
                            'layer_name': layer.__class__.__name__, 'neuron_index': i, 'loss': loss_after
                        })
                        self.metrics['loss_changes'].append({
                            'layer_name': layer.__class__.__name__, 'neuron_index': i, 'loss_change': loss_change
                        })

                        layer.weight.data[i, :] = original_weights[i]  # Restore weights after testing

                    self.save_checkpoint(step)  # Save checkpoint after processing each layer

                metrics[step] = self.metrics
                self.plot_results(step)  # Plot results after each step
                self.save_metrics(step)  # Save metrics after each step

                # Periodically clear metrics to prevent memory bloat
                self.metrics = {key: [] for key in self.metrics}

            return metrics
        except Exception as e:
            self.logger.error(f"An error occurred during Neuron Zeroing experiment: {e}")
            self.logger.error("Experiment terminated.")
            return {}
        
    def plot_results(self, step):
        """Plot the results after each step of the neuron zeroing experiment."""
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        sparsities = [entry['sparsity'] for entry in self.metrics['sparsity_metrics']]
        layer_accuracy_drops = {layer_name: [] for layer_name in set([entry['layer_name'] for entry in self.metrics['layer_accuracy_drops']])}
        loss_changes = [entry['loss_change'] for entry in self.metrics['loss_changes']]  # Track loss changes

        for entry in self.metrics['layer_accuracy_drops']:
            layer_accuracy_drops[entry['layer_name']].append(entry['accuracy_drop'])

        plt.figure(figsize=(14, 12))

        # Plot accuracy drop vs sparsity
        plt.subplot(2, 3, 1)
        plt.hist(accuracy_drops, bins=30)
        plt.title(f'Step {step}: Impact of Neuron Zeroing on Accuracy')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')

        # Plot sparsity vs accuracy drop
        plt.subplot(2, 3, 2)
        plt.scatter(sparsities, accuracy_drops, alpha=0.5)
        plt.title(f'Step {step}: Sparsity vs Accuracy Drop')
        plt.xlabel('Sparsity')
        plt.ylabel('Accuracy Drop')

        # Plot accuracy drop per layer
        plt.subplot(2, 3, 3)
        for layer_name, accuracy in layer_accuracy_drops.items():
            plt.hist(accuracy, bins=20, alpha=0.5, label=layer_name)
        plt.title(f'Step {step}: Accuracy Drop per Layer')
        plt.xlabel('Accuracy Drop')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot loss change histogram
        plt.subplot(2, 3, 4)
        plt.hist(loss_changes, bins=30, color='orange', alpha=0.7)
        plt.title(f'Step {step}: Loss Change Distribution')
        plt.xlabel('Loss Change')
        plt.ylabel('Frequency')

        # Plot loss vs sparsity
        loss_metrics = [entry['loss'] for entry in self.metrics['loss_metrics']]
        plt.subplot(2, 3, 5)
        plt.scatter(sparsities, loss_metrics, alpha=0.5)
        plt.title(f'Step {step}: Loss vs Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Loss')

        # Save the plot after each step
        plot_file = os.path.join(self.save_dir, f'accuracy_drop_vs_sparsity_step_{step}.png')
        plt.tight_layout()
        plt.savefig(plot_file)
        self.logger.info(f"Plot for step {step} saved to {plot_file}")
        plt.close()

    def load_checkpoint(self):
        """Load checkpoint."""
        self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
        with open(self.checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.metrics = checkpoint['metrics']
            self.metadata = checkpoint['metadata']

    def save_checkpoint(self, step=None):
        """Save checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'metrics': self.metrics,
            'metadata': self.metadata
        }
        if step:
            self.metadata['last_step'] = step  # Store the last completed step
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
        self.logger.info(f"Checkpoint saved to {self.checkpoint_file}")

    def save_metrics(self, step=None):
        """Save metrics to a JSON file."""
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")