import torch
import torch.nn as nn
import numpy as np
import logging
import os
import glob
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import pickle 
from matplotlib.colors import LinearSegmentedColormap
from pyPrune.utils import get_pruneable_named_modules, clean_memory

class NeuronSimilarity:
    def __init__(self, pruner: 'IterativeMagnitudePruning', sample_fraction: float = 0.1,
                 similarity_metric: str = 'cosine', logger: Optional[logging.Logger] = None, 
                 plot_data: bool = False) -> None:
        self.pruner: 'IterativeMagnitudePruning' = pruner
        self.similarity_metric: str = similarity_metric
        self.sample_fraction: float = sample_fraction
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.metrics = {}
        self.activations_step = {}
        self.plot_data: bool = plot_data
        self.model: nn.Module = self._initialize_model(pruner)
        self.save_dir = self.pruner.save_dir + '/neuron_similarity'
        os.makedirs(self.save_dir, exist_ok=True)

    def _initialize_model(self, pruner: 'IterativeMagnitudePruning') -> nn.Module:
        model = pruner.model
        model.load_state_dict(pruner.best_model_weights)
        model.eval().to(pruner.device)
        return model

    def compute_similarity_matrix(self, activations: torch.Tensor) -> np.ndarray:
        activations = activations.detach().cpu().numpy()
        if self.similarity_metric == 'cosine':
            similarity_matrix = self._cosine_similarity(activations)
        elif self.similarity_metric == 'correlation':
            similarity_matrix = self._correlation_similarity(activations)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        return similarity_matrix

    def _cosine_similarity(self, activations: np.ndarray) -> np.ndarray:
        activations = activations.astype(np.float64)
        if activations.ndim == 4:
            batch_size, num_filters, height, width = activations.shape
            activations = activations.transpose(0, 2, 3, 1).reshape(-1, num_filters)
        activations = activations.T
        activations += 0.00001
        norm_activations = np.linalg.norm(activations, axis=1, keepdims=True)
        normalized_activations = activations / norm_activations
        similarity_matrix = np.dot(normalized_activations, normalized_activations.T)
        similarity_matrix = np.abs(similarity_matrix)
        return similarity_matrix

    def _correlation_similarity(self, activations: np.ndarray) -> np.ndarray:
        return np.corrcoef(activations)

    def evaluate_layer_activations(self, layer_name: str) -> torch.Tensor:
        activations = []

        def hook_fn(module, input, output):
            activations.append(output.detach().cpu())

        hooks = []
        for name, module in self.model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        with torch.no_grad():
            torch.cuda.empty_cache()  # Clear memory before evaluation
            for i, (data, _) in enumerate(self.pruner.test_loader):
                if i >= 5:  # Limit to 5 batches
                    break
                data = data.to(self.pruner.device)
                _ = self.model(data)

        for hook in hooks:
            hook.remove()

        if activations:
            activations = torch.cat(activations, dim=0)
        else:
            activations = torch.Tensor()
        
        torch.cuda.empty_cache()  # Clear memory after evaluation
        return activations

    def run_experiment(self) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        self.logger.info(f"Starting Neuron Similarity experiment for all layers...")
        for model_step, step in zip(self.pruner.weight_history, self.pruner.metrics["step"]):
            metrics = {"similarity_matrices": [], "average_similarities": []}
            clean_memory()
            self.model.load_state_dict(model_step, strict=False)

            names, modules = get_pruneable_named_modules(self.model, self.pruner.prunable_layers)
            for name, module in zip(names, modules):
                if isinstance(module, nn.Module):
                    self.logger.info(f"Evaluating layer: {name}")
                    activations = self.evaluate_layer_activations(name)

                    if name not in self.activations_step.keys():
                        self.activations_step[name] = []
                    self.activations_step[name].append([step, activations])

                    similarity_matrix = self.compute_similarity_matrix(activations)
                    avg_similarity = np.mean(similarity_matrix)

                    metrics['similarity_matrices'].append({
                        'layer_name': name,
                        'similarity_matrix': similarity_matrix.tolist()  
                    })
                    metrics['average_similarities'].append({
                        'layer_name': name,
                        'average_similarity': avg_similarity
                    })

            if self.plot_data:
                self.plot_similarity_matrices(metrics, step)

            self.logger.info("Neuron Similarity experiment completed for all layers.")
            self.metrics[step] = metrics

        with open(f"{self.save_dir}/neuron_similarity.pkl", 'wb') as f:
            pickle.dump(self, f)

        if self.plot_data:
            self.plot_similarity_()

        return self.metrics

    def plot_similarity_matrices(self, metrics, step) -> None:
        cdict = {
            'red': [(0.0, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
            'green': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
            'blue': [(0.0, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
        }
        rb_cmap = LinearSegmentedColormap('RBColormap', cdict)

        for matrix in metrics['similarity_matrices']:
            similarity_matrix = np.array(matrix['similarity_matrix'])
            normed_matrix = (similarity_matrix - np.min(similarity_matrix)) / (np.max(similarity_matrix) - np.min(similarity_matrix))
            
            plt.figure(figsize=(10, 8))
            plt.imshow(normed_matrix, cmap=rb_cmap, interpolation='nearest', vmin=0, vmax=1)
            plt.title(f"Neuron Similarity for Layer: {matrix['layer_name']} - Step {step}")
            plt.colorbar(label='Similarity (Normalized)')
            plt.xlabel('Neuron Index')
            plt.ylabel('Neuron Index')
            
            plot_filename = f'{self.save_dir}/neuron_similarity_{matrix["layer_name"]}_{step}.png'
            plt.savefig(plot_filename)
            plt.close()

    def save_metrics(self) -> None:
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")

    def plot_similarity_(self) -> None:
        pruning_steps = []
        layer_names = [layer_metric['layer_name'] for layer_metric in self.metrics[next(iter(self.metrics))]['average_similarities']]
        non_zero_similarities = {layer_name: [] for layer_name in layer_names}

        for step, metrics in self.metrics.items():
            pruning_steps.append(step)
            for layer_metric in metrics['average_similarities']:
                layer_name = layer_metric['layer_name']
                avg_similarity = layer_metric['average_similarity']
                non_zero_similarities[layer_name].append(avg_similarity)

        for layer_name, similarities in non_zero_similarities.items():
            plt.figure(figsize=(10, 6))
            plt.plot(pruning_steps, similarities, label=layer_name, color='b', marker='o', linestyle='-', markersize=6, linewidth=2)
            plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            plt.title(f"Non-Zero Neuron Similarity for Layer: {layer_name}", fontsize=16)
            plt.xlabel("Pruning Step", fontsize=14)
            plt.ylabel("Non-Zero Similarity", fontsize=14)
            plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.5)
            plt.legend(loc='best', fontsize=12)
            plt.tight_layout()

            plot_filename = f'{self.save_dir}/non_zero_similarity_{layer_name}.png'
            plt.savefig(plot_filename)
            plt.close()
