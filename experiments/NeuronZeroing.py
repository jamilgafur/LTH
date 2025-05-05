import json
import os
import pickle
import logging
from typing import Any, Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from pyPrune.utils import get_pruneable_named_modules


class NeuronZeroing:
    def __init__(self, pruner: Any, zeroing_metric: str = 'accuracy', logger: Optional[logging.Logger] = None) -> None:
        self.pruner = pruner
        self.model = pruner.model
        self.prunerable_layer = pruner.prunable_layers
        self.zeroing_metric = zeroing_metric

        self.save_dir = os.path.join(pruner.save_dir, f"neuronZeroing_{zeroing_metric}")
        os.makedirs(self.save_dir, exist_ok=True)
        self.logger = logger if logger else logging.getLogger(__name__)
        log_file = os.path.join(self.save_dir, 'experiment.log')
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        self.metrics = self.init_metrics()
        self.checkpoint_file = os.path.join(self.save_dir, 'neuron_zeroing.pkl')
        self.metadata: Dict[str, Any] = {
            'model_architecture': str(self.model),
            'zeroing_metric': self.zeroing_metric,
            'last_step': 0
        }

        self.baseline_accuracy = 0.0
        self.baseline_loss = 0.0

        if os.path.exists(self.checkpoint_file):
            self.load_checkpoint()
        else:
            self.logger.info("Starting new NeuronZeroing experiment.")

    def init_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            'neuron_accuracy_drops': [],
            'sparsity_metrics': [],
            'precision_recall_fscore': [],
            'loss_metrics': [],
            'loss_changes': []
        }

    def evaluate_metrics(self, loader: Optional[torch.utils.data.DataLoader] = None) -> Tuple[float, float, Dict[str, Any]]:
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
        total_params, zero_params = 0, 0
        for param in self.model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        return zero_params / total_params if total_params > 0 else 0.0

    def zero_and_restore_neurons(self, layer: nn.Module, layer_index: str, idx: int, is_conv_layer: bool = True) -> None:
        original_weights = layer.weight.data.clone()

        # Zero neuron
        if is_conv_layer:
            layer.weight.data[idx, :, :, :] = 0
        else:
            layer.weight.data[idx, :] = 0

        if layer.bias is not None:
            original_bias = layer.bias.data.clone()
            layer.bias.data[idx] = 0
        else:
            original_bias = None

        layer_name = f"{layer.__class__.__name__}_{layer_index}"
        self.logger.info(f"Zeroed neuron/channel {idx} in {layer_name}")

        # Evaluate
        accuracy_after, loss_after, metrics_after = self.evaluate_metrics()
        sparsity_after = self.compute_sparsity()

        # Compute drop/change
        accuracy_drop = self.baseline_accuracy - accuracy_after
        loss_change = self.baseline_loss - loss_after
        normalized_accuracy_drop = (accuracy_drop / self.baseline_accuracy) if self.baseline_accuracy else accuracy_drop
        normalized_loss_change = (loss_change / self.baseline_loss) if self.baseline_loss else loss_change

        # Record
        self.metrics['neuron_accuracy_drops'].append({
            'layer_name': layer_name, 'neuron_index': idx, 'accuracy_drop': normalized_accuracy_drop
        })
        self.metrics['sparsity_metrics'].append({
            'layer_name': layer_name, 'neuron_index': idx, 'sparsity': sparsity_after
        })
        self.metrics['precision_recall_fscore'].append({
            'layer_name': layer_name, 'neuron_index': idx, **metrics_after
        })
        self.metrics['loss_metrics'].append({
            'layer_name': layer_name, 'neuron_index': idx, 'loss': loss_after
        })
        self.metrics['loss_changes'].append({
            'layer_name': layer_name, 'neuron_index': idx, 'loss_change': normalized_loss_change
        })

        # Restore
        layer.weight.data.copy_(original_weights)
        if original_bias is not None:
            layer.bias.data.copy_(original_bias)

    def run_experiment(self) -> Dict[Any, Dict[str, List[Dict[str, Any]]]]:
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

            self.logger.info(f"Baseline: Acc: {self.baseline_accuracy:.4f}, Loss: {self.baseline_loss:.4f}, Sparsity: {baseline_sparsity:.4f}")
            self.logger.info(f"Precision: {baseline_metrics['precision']}, Recall: {baseline_metrics['recall']}, F1: {baseline_metrics['fscore']}")

            names, layers = get_pruneable_named_modules(self.model, self.prunerable_layer)
            sampled_layers = [layer for layer in layers if isinstance(layer, (nn.Conv2d, nn.Linear))]

            for layer_index, layer in enumerate(tqdm(sampled_layers, desc="Processing layers", unit="layer")):
                layer_id = str(layer_index)
                if isinstance(layer, nn.Conv2d):
                    for i in tqdm(range(layer.out_channels), desc=f"Zeroing {layer.__class__.__name__}_{layer_id}", unit="channel", leave=False):
                        self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=True)
                elif isinstance(layer, nn.Linear):
                    for i in tqdm(range(layer.out_features), desc=f"Zeroing {layer.__class__.__name__}_{layer_id}", unit="neuron", leave=False):
                        self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=False)

                self.save_checkpoint(step)
                self.save_metrics(step)

            all_metrics[step] = self.metrics.copy()
            self.plot_results(step)
            self.metrics = self.init_metrics()
            self.metadata['last_step'] = step

        return all_metrics

    def plot_results(self, step: Any) -> None:
        accuracy_drops = [entry['accuracy_drop'] for entry in self.metrics['neuron_accuracy_drops']]
        sparsities = [entry['sparsity'] for entry in self.metrics['sparsity_metrics']]
        loss_changes = [entry['loss_change'] for entry in self.metrics['loss_changes']]

        plt.figure(figsize=(14, 10))
        plt.subplot(2, 2, 1)
        plt.hist(accuracy_drops, bins=50, color='skyblue')
        plt.title("Histogram of Accuracy Drops")

        plt.subplot(2, 2, 2)
        plt.hist(loss_changes, bins=50, color='salmon')
        plt.title("Histogram of Loss Changes")

        plt.subplot(2, 2, 3)
        plt.scatter(accuracy_drops, sparsities, alpha=0.6)
        plt.xlabel("Accuracy Drop")
        plt.ylabel("Sparsity")
        plt.title("Accuracy Drop vs Sparsity")

        plt.tight_layout()
        plot_file = os.path.join(self.save_dir, f'accuracy_drop_vs_sparsity_step_{step}.png')
        plt.savefig(plot_file)
        self.logger.info(f"Plot for step {step} saved to {plot_file}")
        plt.close()

    def load_checkpoint(self) -> None:
        self.logger.info(f"Resuming from checkpoint: {self.checkpoint_file}")
        try:
            with open(self.checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.metrics = checkpoint['metrics']
                self.metadata = checkpoint['metadata']
        except pickle.UnpicklingError as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading checkpoint: {e}")
            raise

    def save_checkpoint(self, step: Optional[Any] = None) -> None:
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
        def convert_ndarray(obj: Any) -> Any:
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
