import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import List, Dict, Optional, Union
import matplotlib.pyplot as plt
import json
from tqdm import tqdm  # Import tqdm

class NeuronSimilarity:
    """
    A class to measure redundancy between neurons in all layers of the neural network during the pruning process.
    Redundancy is measured using similarity matrices based on neuron activations.

    Args:
        pruner (IterativeMagnitudePruning): Pruner instance containing model and training information.
        sample_fraction (float): Fraction of weights to sample for large models.
        similarity_metric (str): Metric to calculate similarity ('cosine', 'correlation', etc.).
        logger (logging.Logger, optional): Logger instance for logging experiment progress.
    """

    def __init__(self, pruner: 'IterativeMagnitudePruning', sample_fraction: float = 0.1,
                 similarity_metric: str = 'cosine', logger: Optional[logging.Logger] = None) -> None:
        self.pruner: 'IterativeMagnitudePruning' = pruner
        self.similarity_metric: str = similarity_metric
        self.sample_fraction: float = sample_fraction
        self.logger: logging.Logger = logger or logging.getLogger(__name__)
        self.metrics = {}

        # Initialize the model and load its state
        self.model: nn.Module = self._initialize_model(pruner)
        self.save_dir = self.pruner.save_dir + '/neuron_similarity'

        # Ensure save directory exists
        os.makedirs(self.save_dir, exist_ok=True)

    def _initialize_model(self, pruner: 'IterativeMagnitudePruning') -> nn.Module:
        """
        Initialize the model by deep copying the pruner's best model weights and setting it to evaluation mode.

        Args:
            pruner (IterativeMagnitudePruning): The pruner instance containing the model.

        Returns:
            nn.Module: The deep-copied model set to evaluation mode.
        """
        model: nn.Module = pruner.model
        model.load_state_dict(pruner.best_model_weights)
        model.eval().to(pruner.device)
        return model

    def compute_similarity_matrix(self, activations: torch.Tensor) -> np.ndarray:
        """
        Compute the similarity matrix for the activations of the neurons in the selected layer.

        Args:
            activations (torch.Tensor): Activations of the neurons in the layer.

        Returns:
            np.ndarray: A similarity matrix that measures the redundancy between neurons.
        """
        activations = activations.detach().cpu().numpy()
        if self.similarity_metric == 'cosine':
            similarity_matrix = self._cosine_similarity(activations)
        elif self.similarity_metric == 'correlation':
            similarity_matrix = self._correlation_similarity(activations)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        return similarity_matrix

    def _cosine_similarity(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute the cosine similarity between neurons' activations.

        Args:
            activations (np.ndarray): Activations of the neurons in the layer.

        Returns:
            np.ndarray: Cosine similarity matrix.
        """
        if activations.ndim == 4: #convolutional layers have more dimensions to flatten
            batch_size, num_filters, height, width = activations.shape
            activations = activations.transpose(0,2,3,1).reshape(-1,num_filters)
        activations = activations.T #batch dimension is first - that needs to change
        norm_activations = np.linalg.norm(activations, axis=1, keepdims=True)
        normalized_activations = activations / norm_activations
        similarity_matrix = np.dot(normalized_activations, normalized_activations.T)
        print(similarity_matrix.shape)
        return similarity_matrix

    def _correlation_similarity(self, activations: np.ndarray) -> np.ndarray:
        """
        Compute the correlation similarity between neurons' activations.

        Args:
            activations (np.ndarray): Activations of the neurons in the layer.

        Returns:
            np.ndarray: Correlation similarity matrix.
        """
        correlation_matrix = np.corrcoef(activations)
        return correlation_matrix

    def evaluate_layer_activations(self, layer_name: str) -> torch.Tensor:
        """
        Evaluate and return the activations from the specified layer during a forward pass.

        Args:
            layer_name (str): Name of the layer to get activations from.

        Returns:
            torch.Tensor: The activations of the neurons in the layer.
        """
        activations: List[torch.Tensor] = []
        
        # Define a hook to capture activations from each layer
        def hook_fn(module, input, output):
            if hasattr(module, 'name') and module.name == layer_name:
                activations.append(output)
        
        # Register the hook on all layers
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Module):
                module.name = name  # Attach name to module for easy identification
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)

        # Run a forward pass
        for data, _ in self.pruner.test_loader:
            data = data.to(self.pruner.device)
            _ = self.model(data)

        # Remove hooks after capturing activations
        for hook in hooks:
            hook.remove()
        
        # Stack activations across all batches
        activations = torch.cat(activations, dim=0)
        
        return activations

    def run_experiment(self) -> Dict[str, List[Dict[str, Union[str, float]]]]:
        """
        Run the Neuron Similarity experiment, computing similarity matrices of neuron activations for all layers.

        Returns:
            Dict[str, List[Dict[str, Union[str, float]]]]: A dictionary containing similarity matrices at each layer.
        """
        self.logger.info(f"Starting Neuron Similarity experiment for all layers...")
        for model_step, step in zip(self.pruner.weight_history, self.pruner.metrics["step"]):
            metrics = {"similarity_matrices" : [], "average_similarities" : []}
            self.model.load_state_dict(model_step, strict=False)
            # Use tqdm to show a progress bar when iterating through layers
            for name, module in tqdm(self.model.named_modules(), desc="Evaluating Layers", ncols=100):
                if isinstance(module, nn.Module):
                    self.logger.info(f"Evaluating layer: {name}")
                    activations = self.evaluate_layer_activations(name)

                    # Compute the similarity matrix for the activations
                    similarity_matrix = self.compute_similarity_matrix(activations)

                    # Calculate the average similarity for the layer
                    avg_similarity = np.mean(similarity_matrix)

                    # Store the similarity matrix and average similarity
                    metrics['similarity_matrices'].append({
                        'layer_name': name,
                        'similarity_matrix': similarity_matrix.tolist()  # Convert tensor to list for easier logging/storage
                    })
                    metrics['average_similarities'].append({
                        'layer_name': name,
                        'average_similarity': avg_similarity
                    })

            # Plot the similarity matrices
            self.plot_similarity_matrices(metrics,step)

            # Save metrics to a JSON file - takes too long to run
            # self.save_metrics() 

            self.logger.info("Neuron Similarity experiment completed for all layers.")
            self.metrics[step] = metrics

        return self.metrics

    def plot_similarity_matrices(self, metrics,step) -> None:
        """
        Plot the similarity matrices for the experiment results.

        Args:
            save_dir (str): Directory to save the similarity matrix plot.
        """
        for matrix in metrics['similarity_matrices']:
            similarity_matrix = np.array(matrix['similarity_matrix'])
            plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
            plt.title(f"Neuron Similarity for Layer: {matrix['layer_name']}")
            plt.colorbar()
            plt.savefig(f'{self.save_dir}/neuron_similarity_{matrix["layer_name"]}_{step}.png')
            plt.close()

    def save_metrics(self) -> None:
        """
        Save the similarity matrices and average similarities to a JSON file.
        """
        metrics_file = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        self.logger.info(f"Metrics saved to {metrics_file}")