import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from sklearn.metrics import pairwise_distances

class PruningAnalysis:
    def __init__(self, pruning_instance, log_file, save_dir, device='cuda'):
        """
        Initialize the pruning analysis process.

        Parameters:
            pruning_instance (IterativeMagnitudePruning): An instance of IterativeMagnitudePruning that has performed pruning.
            log_file (str): Path to the log file that contains pruning details.
            save_dir (str): Directory where the results will be saved.
            device (str): Device to run the analysis on ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        self.pruning_instance = pruning_instance  # Store the pruning instance
        self.log_file = log_file
        self.save_dir = save_dir
        self.device = device

        # Ensure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Setup logger
        self.logger = self.setup_logger(log_file)
        
        # Reload the model from the pruning instance
        self.model = self.pruning_instance.model.to(self.device)
        self.X_train = self.pruning_instance.X_train.to(self.device)
        self.y_train = self.pruning_instance.y_train.to(self.device)
        self.final_sparsity = self.pruning_instance.final_sparsity
        self.steps = self.pruning_instance.steps

        self.logger.info(f"Pruning analysis initialized with model: {self.model} on device: {self.device}")
        
    def setup_logger(self, log_file):
        """
        Setup logger to log information from the analysis process.
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

    def compute_similarity_matrix(self):
        """
        Compute the neuron similarity matrix based on activation vectors.
        """
        self.model.eval()
        activations = []
        with torch.no_grad():
            for i in range(0, self.X_train.size(0), 64):  # Process in batches
                batch = self.X_train[i:i+64]
                outputs = self.model(batch)
                activations.append(outputs.cpu().numpy())
        
        # Concatenate activations and compute pairwise similarity
        activations = np.concatenate(activations, axis=0)
        sim_matrix = pairwise_distances(activations.T, metric='cosine')  # Cosine similarity as a measure of redundancy
        np.fill_diagonal(sim_matrix, 0)  # Ignore diagonal (same neuron comparison)
        return sim_matrix

    def measure_weight_zeroing_effects(self):
        """
        Measure the effect of weight zeroing by evaluating loss change when each weight is zeroed.
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        loss_before = criterion(self.model(self.X_train), self.y_train)
        weight_changes = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                original_weights = param.data.clone()
                # Zero out one weight at a time
                param.data.zero_()
                loss_after = criterion(self.model(self.X_train), self.y_train)
                weight_changes.append(loss_after.item() - loss_before.item())
                param.data.copy_(original_weights)  # Restore the weight
        
        return np.array(weight_changes)

    def measure_neuron_zeroing_effects(self):
        """
        Measure the effect of neuron zeroing by evaluating loss change when each neuron is zeroed.
        """
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        loss_before = criterion(self.model(self.X_train), self.y_train)
        neuron_changes = []

        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                original_weights = module.weight.data.clone()
                num_neurons = module.out_features if isinstance(module, torch.nn.Linear) else module.out_channels

                for i in range(num_neurons):
                    # Zero out the neurons one at a time
                    module.weight.data[i].zero_()  # Zero out i-th neuron
                    loss_after = criterion(self.model(self.X_train), self.y_train)
                    neuron_changes.append(loss_after.item() - loss_before.item())
                    module.weight.data.copy_(original_weights)  # Restore the weight
        
        return np.array(neuron_changes)

    def run_analysis(self):
        """
        Run all the analysis computations and return the results.
        """
        self.logger.info(f"Starting pruning analysis...")

        # Initialize empty lists to store results for each step
        similarity_matrices = []
        weight_changes = []
        neuron_changes = []

        # Run the analysis for each pruning step
        for step in range(self.steps):
            self.logger.info(f"Analyzing step {step + 1} of {self.steps}...")

            # Compute neuron similarity matrix
            similarity_matrix = self.compute_similarity_matrix()
            similarity_matrices.append(similarity_matrix)

            # Measure weight zeroing effects
            weight_change = self.measure_weight_zeroing_effects()
            weight_changes.append(weight_change)

            # Measure neuron zeroing effects
            neuron_change = self.measure_neuron_zeroing_effects()
            neuron_changes.append(neuron_change)

        self.logger.info("Pruning analysis completed.")
        
        # Return all the results as a dictionary
        return {
            'similarity_matrices': similarity_matrices,
            'weight_changes': weight_changes,
            'neuron_changes': neuron_changes
        }

    def plot_similarity_matrix(self, similarity_matrix, step):
        """
        Plot and save the neuron similarity matrix for a given pruning step as SVG.
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(similarity_matrix, cmap='coolwarm', square=True, cbar=True, annot=False)
        plt.title(f"Neuron Similarity Matrix at Step {step + 1}")
        plt.xlabel("Neurons")
        plt.ylabel("Neurons")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/similarity_step_{step + 1}.svg", format="svg")
        plt.close()

    def plot_weight_zeroing_effects(self, weight_changes, step):
        """
        Plot and save the weight zeroing effect (loss change) as SVG for a given pruning step.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(weight_changes)
        plt.title(f"Loss Change vs Weight Zeroing at Step {step + 1}")
        plt.xlabel("Weight Index")
        plt.ylabel("Loss Change")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/weight_zeroing_effects_step_{step + 1}.svg", format="svg")
        plt.close()

    def plot_neuron_zeroing_effects(self, neuron_changes, step):
        """
        Plot and save the neuron zeroing effect (loss change) as SVG for a given pruning step.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(neuron_changes)
        plt.title(f"Loss Change vs Neuron Zeroing at Step {step + 1}")
        plt.xlabel("Neuron Index")
        plt.ylabel("Loss Change")
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/neuron_zeroing_effects_step_{step + 1}.svg", format="svg")
        plt.close()

    def run_plotting(self, analysis_results):
        """
        Generate and save plots based on the analysis results.
        """
        self.logger.info(f"Starting plotting...")

        # Plot similarity matrices
        for step, similarity_matrix in enumerate(analysis_results['similarity_matrices']):
            self.plot_similarity_matrix(similarity_matrix, step)

        # Plot weight zeroing effects
        for step, weight_changes in enumerate(analysis_results['weight_changes']):
            self.plot_weight_zeroing_effects(weight_changes, step)

        # Plot neuron zeroing effects
        for step, neuron_changes in enumerate(analysis_results['neuron_changes']):
            self.plot_neuron_zeroing_effects(neuron_changes, step)

        self.logger.info(f"Plotting completed.")
