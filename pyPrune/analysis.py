import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram
from typing import List, Dict, Any
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

class PruningAnalysis:
    def __init__(self, pruning_instance, log_file: str, save_dir: str, device='cuda', batch_size=64):
        """
        Initialize the pruning analysis process.

        Parameters:
            pruning_instance (IterativeMagnitudePruning): An instance of IterativeMagnitudePruning that has performed pruning.
            log_file (str): Path to the log file that contains pruning details.
            save_dir (str): Directory where the results will be saved.
            device (str): Device to run the analysis on ('cuda' or 'cpu'). Defaults to 'cuda'.
            batch_size (int): The batch size for DataLoader. Default is 64.
        """
        self.pruning_instance = pruning_instance
        self.log_file = log_file
        self.save_dir = save_dir
        self.device = device
        self.batch_size = batch_size

        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Setup logger
        self.logger = self.setup_logger(log_file)
        
        # Reload the model from the pruning instance
        self.model = self.pruning_instance.model.to(self.device)

        # Use DataLoader for training data
        train_data = self.pruning_instance.train_loader.dataset  # Assuming train_loader is a DataLoader
        self.X_train = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        self.y_train = DataLoader(train_data, batch_size=self.batch_size, shuffle=False)
        
        # Ensure target and data are separate for proper training
        self.final_sparsity = self.pruning_instance.final_sparsity
        self.steps = self.pruning_instance.steps
        
        self.logger.info(f"Pruning analysis initialized with model: {self.model} on device: {self.device}")

    def setup_logger(self, log_file: str):
        """
        Setup logger to log information from the analysis process.

        Parameters:
            log_file (str): Path to the log file to store analysis logs.
        """
        logger = logging.getLogger('PruningAnalysis')
        logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def run_analysis(self):
        """
        Run all the analysis computations and return the results.
        """
        self.logger.info(f"Starting pruning analysis...")

        # Initialize empty lists to store results for each step
        similarity_matrices = []
        weight_changes = []
        neuron_changes = []

        # Run the analysis for each pruning step with tqdm
        for step in tqdm(range(self.steps), desc="Pruning Steps", unit="step"):
            self.logger.info(f"Analyzing step {step + 1} of {self.steps}...")

            # Create directory for each step's results
            step_dir = f"{self.save_dir}/experiment_step_{step + 1}"
            if not os.path.exists(step_dir):
                os.makedirs(step_dir)

            # Log model information and statistics
            self.logger.info(f"Model at step {step + 1}: {str(self.model)}")
            
            # Log sparsity (if final_sparsity is a list/array)
            if isinstance(self.final_sparsity, (list, np.ndarray)):
                self.logger.info(f"Sparsity at step {step + 1}: {self.final_sparsity[step]}")
            else:
                self.logger.info(f"Final sparsity: {self.final_sparsity}")

            # Save model weights and masks
            torch.save(self.model.state_dict(), f"{step_dir}/model_step_{step + 1}.pt")
            
            # Compute neuron similarity matrix (Experiment 1)
            similarity_matrix = self.compute_similarity_matrix()
            similarity_matrices.append(similarity_matrix)
            
            # Plot similarity matrices for each layer at each step with tqdm
            for step, similarity_matrix in enumerate(similarity_matrices):
                for layer_name, sim_matrix in similarity_matrix.items():
                    # Plot non-clustered similarity matrix
                    self.plot_similarity_matrix(sim_matrix, step, layer_name)
                    
                    # Plot clustered similarity matrix
                    self.plot_clustered_similarity_matrix(sim_matrix, step, layer_name)

            # Measure weight zeroing effects (Experiment 2)
            weight_change = self.measure_weight_zeroing_effects(step)
            weight_changes.append(weight_change)
            
            
            # Plot weight zeroing effects as a histogram
            for step, weight_changes in enumerate(weight_changes):
                self.plot_weight_zeroing_histogram(weight_changes, step)


            # Measure neuron zeroing effects (Experiment 3)
            neuron_change = self.measure_neuron_zeroing_effects()
            neuron_changes.append(neuron_change)
            
            
            # Plot neuron zeroing effects as a bar plot
            for step, neuron_changes in enumerate(neuron_changes):
                self.plot_neuron_zeroing_barplot(neuron_changes, step)


        self.logger.info("Pruning analysis completed.")
        
        # Return all the results as a dictionary
        return {
            'similarity_matrices': similarity_matrices,
            'weight_changes': weight_changes,
            'neuron_changes': neuron_changes
        }


    def _hook_fn(self, name: str):
        """
        A hook function that stores the activations of each layer.
        """
        def hook(module, input, output):
            self.layer_outputs[name] = output
        return hook

    def compute_similarity_matrix(self):
        """
        Compute the neuron similarity matrix based on activation vectors for each layer.
        """
        self.model.eval()

        # Register hooks for each layer of the model
        hooks = []
        self.layer_outputs = {}

        # Register forward hooks for each layer
        for name, layer in self.model.named_modules():
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):  # Focus on layers that output activations
                hook = layer.register_forward_hook(self._hook_fn(name))
                hooks.append(hook)

        # Dictionary to store similarity matrices for each layer
        layer_similarity_matrices = {}

        with torch.no_grad():
            # Use tqdm to wrap batch iteration
            for batch in tqdm(self.X_train, desc="Computing similarity matrix", unit="batch"):
                batch_x, _ = batch  # Unpack batch (assuming _ is target)
                batch_x = batch_x.to(self.device)  # Move to device
                self.model(batch_x)  # Forward pass to capture activations

                # Compute similarity for each layer based on activation vectors
                for layer_name, activation in self.layer_outputs.items():
                    # Flatten activations (neurons per sample)
                    activation = activation.view(activation.size(0), -1).cpu().numpy()
                    
                    # Compute pairwise cosine similarity
                    sim_matrix = pairwise_distances(activation.T, metric='cosine')
                    np.fill_diagonal(sim_matrix, 0)  # Ignore diagonal (self-similarity)
                    
                    # Store similarity matrix for this layer
                    if layer_name not in layer_similarity_matrices:
                        layer_similarity_matrices[layer_name] = []
                    layer_similarity_matrices[layer_name].append(sim_matrix)

        # Remove hooks after use
        for hook in hooks:
            hook.remove()

        # Aggregate the similarity matrices across all batches
        for layer_name, sim_matrices in layer_similarity_matrices.items():
            layer_similarity_matrices[layer_name] = np.mean(sim_matrices, axis=0)

        return layer_similarity_matrices

    def measure_weight_zeroing_effects(self,step):
        """
        Measure the effect of weight zeroing by evaluating loss change when each weight is zeroed.
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize loss before any zeroing
        loss_before = 0.0
        with torch.no_grad():
            for batch in tqdm(self.X_train, desc="Measuring weight zeroing", unit="batch"):
                batch_x, batch_y = batch  # Unpack batch (data, labels)
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                loss_before += criterion(self.model(batch_x), batch_y).item()

        loss_before /= len(self.X_train)

        weight_changes = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_weights = param.data.clone()
                
                # Plot histogram for the weight of the current layer (pass step number)
                self.plot_weight_histogram(original_weights, name, step)  # Pass 'step' here
                
                # Zero out one weight at a time
                param.data.zero_()

                loss_after = 0.0
                with torch.no_grad():
                    for batch in tqdm(self.X_train, desc=f"Zeroing weights for {name}", unit="batch"):
                        batch_x, batch_y = batch
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        loss_after += criterion(self.model(batch_x), batch_y).item()

                loss_after /= len(self.X_train)
                weight_changes.append(loss_after - loss_before)
                param.data.copy_(original_weights)  # Restore the weight

        return np.array(weight_changes)

    def plot_weight_histogram(self, original_weights, name, step):
        """
        Plot a histogram of the weights for a specific layer and save it in the appropriate directory.
        
        Parameters:
            original_weights (Tensor): The weight tensor of the layer.
            name (str): The name of the layer.
            step (int): The current pruning step.
        """
        plt.figure(figsize=(8, 6))
        plt.hist(original_weights.cpu().numpy().flatten(), bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of Weights for Layer {name}")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.tight_layout()

        # Create directory for the step if it doesn't exist
        step_dir = f"{self.save_dir}/experiment_step_{step + 1}"
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)

        # Save the plot in the corresponding folder for the step
        save_path = f"{step_dir}/weight_histogram_{name}_step_{step + 1}.png"
        plt.savefig(save_path, format="png")
        plt.close()

    def measure_neuron_zeroing_effects(self):
        """
        Measure the effect of neuron zeroing by evaluating loss change when each neuron is zeroed.
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize loss before any zeroing
        loss_before = 0.0
        with torch.no_grad():
            for batch in tqdm(self.X_train, desc="Measuring neuron zeroing", unit="batch"):
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                loss_before += criterion(self.model(batch_x), batch_y).item()

        loss_before /= len(self.X_train)

        neuron_changes = []

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                original_weights = module.weight.data.clone()
                num_neurons = module.out_features if isinstance(module, torch.nn.Linear) else module.out_channels

                for i in tqdm(range(num_neurons), desc=f"Zeroing neurons in {name}", unit="neuron"):
                    # Zero out the neurons one at a time
                    module.weight.data[i].zero_()  # Zero out i-th neuron

                    loss_after = 0.0
                    with torch.no_grad():
                        for batch in self.X_train:
                            batch_x, batch_y = batch
                            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                            loss_after += criterion(self.model(batch_x), batch_y).item()

                    loss_after /= len(self.X_train)
                    neuron_changes.append(loss_after - loss_before)
                    module.weight.data.copy_(original_weights)  # Restore the weight
        
        return np.array(neuron_changes)

    # Plotting functions (no changes needed for DataLoader integration)
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray, step: int, layer_name: str):
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='coolwarm', square=True, cbar=True, annot=False)
        plt.title(f"Neuron Similarity Matrix for Layer {layer_name} at Step {step + 1}")
        plt.xlabel("Neurons")
        plt.ylabel("Neurons")
        plt.tight_layout()

        # Save the plot in the corresponding folder
        save_path = f"{self.save_dir}/experiment_step_{step + 1}/similarity_matrix_{layer_name}_step_{step + 1}.png"
        plt.savefig(save_path, format="png")
        plt.close()

    def plot_clustered_similarity_matrix(self, similarity_matrix, step, layer_name):
        # Perform hierarchical clustering (using the 'ward' method)
        linked = linkage(similarity_matrix, method='ward')
        
        # Create the dendrogram for hierarchical clustering
        plt.figure(figsize=(10, 8))
        dendrogram(linked, labels=np.arange(similarity_matrix.shape[0]), orientation='top', color_threshold=0)
        plt.title(f"Clustered Neuron Similarity at Step {step + 1} for Layer {layer_name}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Similarity")
        plt.tight_layout()
        
        # Save to the correct folder for the step
        step_dir = f"{self.save_dir}/experiment_step_{step + 1}"
        plt.savefig(f"{step_dir}/clustered_similarity_step_{step + 1}_layer_{layer_name}.png", format="png")
        plt.close()

    def plot_weight_zeroing_histogram(self, weight_changes, step):
        plt.figure(figsize=(8, 6))
        plt.hist(weight_changes, bins=50, color='skyblue', edgecolor='black')
        plt.title(f"Histogram of Loss Change vs Weight Zeroing at Step {step + 1}")
        plt.xlabel("Loss Change")
        plt.ylabel("Frequency")
        plt.tight_layout()

        step_dir = f"{self.save_dir}/experiment_step_{step + 1}"
        plt.savefig(f"{step_dir}/weight_zeroing_histogram_step_{step + 1}.png", format="png")
        plt.close()

    def plot_neuron_zeroing_barplot(self, neuron_changes, step):
        plt.figure(figsize=(8, 6))
        plt.bar(np.arange(len(neuron_changes)), neuron_changes, color='salmon')
        plt.title(f"Loss Change vs Neuron Zeroing at Step {step + 1}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Loss Change")
        plt.tight_layout()

        step_dir = f"{self.save_dir}/experiment_step_{step + 1}"
        plt.savefig(f"{step_dir}/neuron_zeroing_barplot_step_{step + 1}.png", format="png")
        plt.close()
