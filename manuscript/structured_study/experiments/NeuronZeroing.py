import json
import os
import pickle
import logging
from typing import Any, Dict, Tuple, Optional, List
import concurrent.futures

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from pyPrune.utils import get_pruneable_named_modules


class NeuronZeroing:
    def __init__(self, pruner: Any, zeroing_metric: str = 'accuracy', logger: Optional[logging.Logger] = None, process_Step:int=0) -> None:
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

        self.process_Step = process_Step
        self.metrics = self.init_metrics()
        self.checkpoint_file = os.path.join(self.save_dir, f"neuron_zeroing_{self.process_Step}.pkl")
        self.metadata: Dict[str, Any] = {
            'model_architecture': str(self.model),
            'zeroing_metric': self.zeroing_metric,
            'last_step': 0,
            'last_neuron_index': {},  
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
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in loader:
                data = data.to(self.pruner.device, non_blocking=True)
                target = target.to(self.pruner.device, non_blocking=True)

                output = self.model(data)
                loss = self.pruner.criterion(output, target)

                total_loss += loss.item() * data.size(0)
                total += data.size(0)

                preds = output.argmax(dim=1)
                correct += (preds == target).sum().item()

                y_true.extend(target.cpu().tolist())
                y_pred.extend(preds.cpu().tolist())
                break

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        return accuracy, avg_loss, {'precision': precision, 'recall': recall, 'fscore': fscore}


    def compute_sparsity(self) -> float:
        total, zero = 0, 0
        for p in self.model.parameters():
            total += p.numel()
            zero += torch.count_nonzero(p == 0).item()
        return zero / total if total > 0 else 0.0

    def zero_and_restore_neurons(self, layer: nn.Module, layer_index: str, idx: int, is_conv_layer: bool = True) -> None:
        original_weights = layer.weight.detach().clone()
        original_bias = layer.bias.detach().clone() if layer.bias is not None else None

        with torch.no_grad():
            layer.weight[idx].copy_(torch.zeros_like(layer.weight[idx]))
            if original_bias is not None:
                layer.bias[idx] = 0.0

        acc_after, loss_after, metrics_after = self.evaluate_metrics()
        sparsity_after = self.compute_sparsity()

        acc_drop = self.baseline_accuracy - acc_after
        loss_change = self.baseline_loss - loss_after
        norm_acc_drop = (acc_drop / self.baseline_accuracy) if self.baseline_accuracy else acc_drop
        norm_loss_change = (loss_change / self.baseline_loss) if self.baseline_loss else loss_change

        layer_name = f"{layer.__class__.__name__}_{layer_index}"
        self.metrics['neuron_accuracy_drops'].append({'layer_name': layer_name, 'neuron_index': idx, 'accuracy_drop': norm_acc_drop})
        self.metrics['sparsity_metrics'].append({'layer_name': layer_name, 'neuron_index': idx, 'sparsity': sparsity_after})
        self.metrics['precision_recall_fscore'].append({'layer_name': layer_name, 'neuron_index': idx, **metrics_after})
        self.metrics['loss_metrics'].append({'layer_name': layer_name, 'neuron_index': idx, 'loss': loss_after})
        self.metrics['loss_changes'].append({'layer_name': layer_name, 'neuron_index': idx, 'loss_change': norm_loss_change})

        with torch.no_grad():
            layer.weight.copy_(original_weights)
            if original_bias is not None:
                layer.bias.copy_(original_bias)

    def run_experiment(self) -> Dict[Any, Dict[str, List[Dict[str, Any]]]]:
        all_metrics: Dict[Any, Dict[str, List[Dict[str, Any]]]] = {}
        # Validate the step index
        if self.process_Step >= len(self.pruner.weight_history) or self.process_Step < 0:
            self.logger.error(f"Invalid process_Step: {self.process_Step}")
            return all_metrics

        weights = self.pruner.weight_history[self.process_Step]
        step = self.pruner.steps[self.process_Step]

        self.logger.info(f"Processing step {step}...")
        self.model.load_state_dict(weights, strict=False)
        self.model.to(self.pruner.device)

        self.baseline_accuracy, self.baseline_loss, baseline_metrics = self.evaluate_metrics()
        baseline_sparsity = self.compute_sparsity()

        self.logger.info(f"Baseline: Acc: {self.baseline_accuracy:.4f}, Loss: {self.baseline_loss:.4f}, Sparsity: {baseline_sparsity:.4f}")

        names, layers = get_pruneable_named_modules(self.model, self.prunerable_layer)
        sampled_layers = [layer for layer in layers if isinstance(layer, (nn.Conv2d, nn.Linear))]

        for layer_index, layer in enumerate(tqdm(sampled_layers, desc="Processing layers", unit="layer")):
            layer_id = str(layer_index)

            def run_zeroing_task(i):
                if isinstance(layer, nn.Conv2d):
                    self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=True)
                elif isinstance(layer, nn.Linear):
                    self.zero_and_restore_neurons(layer, layer_id, i, is_conv_layer=False)

            if isinstance(layer, nn.Conv2d):
                neuron_count = layer.out_channels
            else:
                neuron_count = layer.out_features

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                list(tqdm(executor.map(run_zeroing_task, range(neuron_count)),
                        total=neuron_count,
                        desc=f"Zeroing {layer.__class__.__name__}_{layer_id}",
                        leave=False))

            self.save_checkpoint(step)
            self.save_metrics(step)

        all_metrics[step] = self.metrics.copy()
        self.metrics = self.init_metrics()
        self.metadata['last_step'] = step

        return all_metrics


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
