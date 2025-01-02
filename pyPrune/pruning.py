import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from .model_utils import PrunedLinear, PrunedConv2d


class IterativeMagnitudePruning:
    """
    Class for performing Iterative Magnitude Pruning (IMP) on a neural network model.
    """

    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, final_sparsity: float, 
                 steps: int, E: int = 10, pruning_criterion: Optional[Callable[[float, torch.Tensor], torch.Tensor]] = None, 
                 reset_weights: bool = True, is_pretrained: bool = False, pretrain_epochs: int = 20, 
                 device: Optional[str] = None, save_dir: str = 'pruning_checkpoints') -> None:
        """
        Initialize the iterative magnitude pruning process with gradual pruning.

        Parameters:
            model (nn.Module): Neural network model to be pruned.
            train_loader (DataLoader): Training data loader.
            test_loader (DataLoader): Test data loader.
            final_sparsity (float): The final target sparsity.
            steps (int): Number of pruning steps.
            E (int, optional): Number of epochs for pretraining. Defaults to 10.
            pruning_criterion (callable, optional): Function for pruning criterion. Defaults to magnitude pruning.
            reset_weights (bool, optional): Whether to reset weights before pruning. Defaults to True.
            is_pretrained (bool, optional): Whether the model is pretrained. Defaults to False.
            pretrain_epochs (int, optional): Number of epochs for pretraining. Defaults to 20.
            device (str, optional): Device for training (cuda or cpu). Defaults to None.
            save_dir (str, optional): Directory to save logs, metrics, masks, and weights. Defaults to 'pruning_checkpoints'.
        """
        # Initialize logger
        self.save_dir = save_dir  # Custom directory for saving files
        self.setup_save_dir()  # Ensure the save_dir exists
        self.logger = self.setup_logger(os.path.join(self.save_dir, 'logging.txt'))  # Save log inside save_dir

        # Validate inputs
        assert 0 < final_sparsity < 1, "final_sparsity must be between 0 and 1."
        assert steps > 0, "steps must be a positive integer."
        assert isinstance(model, nn.Module), "The model must be a subclass of nn.Module."

        # Set class attributes
        self.final_sparsity = final_sparsity
        self.steps = steps
        self.E = E
        self.pruning_criterion = pruning_criterion or self.magnitude_prune
        self.reset_weights = reset_weights
        self.is_pretrained = is_pretrained
        self.pretrain_epochs = pretrain_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_sparsity = 0.0  # Current sparsity level

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model.to(self.device)  # Move model to device

        # Replace layers with Pruned versions before pruning starts
        self.replace_with_pruned_layers()

        # Save initial weights and zeroed masks for reset during pruning
        self.initial_weights = self.save_initial_weights()
        self.saved_masks: Dict[str, torch.Tensor] = {}  # Store the zeroed masks for each layer
        self.saved_weights: Dict[str, torch.Tensor] = {}  # Store pruned weights after each step

        # Initialize metrics list
        self.metrics: list[Dict[str, float]] = []

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Initialized Iterative Magnitude Pruning with {self.steps} steps.")
        self.logger.info(f"{'='*50}\n")

        # Path to store the final metrics in a Parquet file
        self.metrics_file = os.path.join(self.save_dir, 'metrics.parquet')

    def setup_logger(self, log_file: str) -> logging.Logger:
        """
        Setup the logger to write logs to both the console and a file.

        Parameters:
            log_file (str): Path to the log file.
        """
        logger = logging.getLogger('IMP')
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

    def setup_save_dir(self) -> None:
        """
        Ensure the save directory exists, and create it if necessary.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def run(self) -> nn.Module:
        """
        Perform the iterative magnitude pruning (IMP) with weight resetting and fine-tuning.
        """
        # Step 1: Train or fine-tune the model if it's not pretrained
        if not self.is_pretrained:
            self.logger.info("Pretraining model as it is not pretrained.")
            self.pretrain_model()

        # Gradual pruning and fine-tuning
        self.logger.info("\nStarting gradual pruning and fine-tuning process...\n")
        self.perform_gradual_pruning()

        # Save metrics to a Parquet file
        self.save_metrics_to_parquet()
        
        # save the accuray, sparsity and loss for each step as a csv 
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.save_dir, 'metrics.csv'), index=False)
        # plot the accuracy, sparsity and loss for each step
        df.plot(x='step', y=['accuracy', 'sparsity', 'loss'], secondary_y=['sparsity'], figsize=(10, 6))
        plt.title('Accuracy, Sparsity, and Loss vs. Pruning Step')
        plt.savefig(os.path.join(self.save_dir, 'metrics_plot.png'))
        plt.close()
        
        

    def pretrain_model(self) -> None:
        """
        Pretrain the model if it is not already pretrained.
        """
        self.logger.info(f"\nPretraining the model for {self.pretrain_epochs} epochs...\n")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)
        # Pretrain for the specified number of epochs
        for epoch in  tqdm(range(self.pretrain_epochs), desc='Pretraining', unit='epoch', leave=False):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()

                # Apply zeroed gradients before optimizer step
                self.apply_zeroed_gradients()

                optimizer.step()

                # Log training info
                accuracy = self.calculate_accuracy(outputs, target)
            self.logger.info(f"Epoch {epoch + 1}/{self.pretrain_epochs} - Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
            self.metrics.append({'step': epoch, 'loss': loss.item(), 'accuracy': accuracy, 'sparsity': self.current_sparsity})

    def perform_gradual_pruning(self) -> None:
        """
        Perform gradual pruning from the current sparsity to final sparsity using np.linspace.
        """
        self.logger.info(f"\nStarting gradual pruning: current zeroed weights = {self.current_sparsity * 100:.2f}%")
        self.logger.info(f"Target final non-zero weights = {(1 - self.final_sparsity) * 100:.2f}%")
        self.logger.info(f"Pruning in {self.steps} steps...\n")

        # Generate sparsity values using np.linspace for smooth transitions
        pruning_steps = np.linspace(self.current_sparsity, self.final_sparsity, self.steps)[1:]

        for step, target_sparsity in  tqdm(enumerate(pruning_steps), desc='Pruning', unit='step', leave=False):
            self.logger.info(f"Step {step + 1}/{self.steps}: Pruning to {target_sparsity * 100:.2f}% zeroed weights.")
            self.prune_weights(target_sparsity)
            self.fine_tune_model()

            # Log the metrics after pruning
            accuracy = self.log_accuracy()
            self.metrics.append({'step': step, 'accuracy': accuracy, 'sparsity': self.current_sparsity})

            # Save masks and weights for each pruning step
            self.save_masks_and_weights(step)

    def log_accuracy(self) -> float:
        """
        Log the accuracy of the model after pruning.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                correct += (outputs.argmax(1) == target).sum().item()
                total += target.size(0)

        accuracy = correct / total * 100
        self.logger.info(f"Accuracy after pruning: {accuracy:.2f}%, percentage of zeroed weights: {self.current_sparsity*100:.2f}%")
        return accuracy

    def save_metrics_to_parquet(self) -> None:
        """
        Save the accumulated metrics to a Parquet file for analysis.
        """
        df = pd.DataFrame(self.metrics)
        df.to_parquet(self.metrics_file, index=False)
        self.logger.info(f"Metrics saved to {self.metrics_file}")

    def save_initial_weights(self) -> Dict[str, torch.Tensor]:
        """
        Save the initial weights of the model to restore later.
        """
        initial_weights = {}
        for name, param in self.model.named_parameters():
            initial_weights[name] = param.clone()
        return initial_weights

    def replace_with_pruned_layers(self) -> None:
        """
        Replace layers with their pruned counterparts, if necessary.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.model._modules[name] = PrunedLinear(module.in_features, module.out_features)
            elif isinstance(module, nn.Conv2d):
                self.model._modules[name] = PrunedConv2d(module.in_channels, module.out_channels, module.kernel_size)

    def prune_weights(self, target_sparsity: float) -> None:
        """
        Perform pruning by zeroing out the weights with smallest magnitude.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    num_zeroed = int(param.numel() * target_sparsity)
                    threshold = torch.topk(param.abs().flatten(), num_zeroed, largest=False).values[-1]
                    param.data[param.abs() <= threshold] = 0
                    self.saved_masks[name] = param.abs() <= threshold  # Save zeroed mask

        self.current_sparsity = target_sparsity

    def save_masks_and_weights(self, step: int) -> None:
        """
        Save the current masks and pruned model to disk.
        """
        masks_file = os.path.join(self.save_dir, f'masks_step_{step}.pt')
        weights_file = os.path.join(self.save_dir, f'weights_step_{step}.pt')

        # Save masks and weights
        torch.save(self.saved_masks, masks_file)
        torch.save(self.model.state_dict(), weights_file)

        self.logger.info(f"Saved masks to {masks_file}")
        self.logger.info(f"Saved weights to {weights_file}")
        
    def get_model_zeroed_weight_percentage(self) -> float:
        """
        Calculate the percentage of zeroed weights in the model.
        """
        zeroed_count = 0
        total_count = 0
        for param in self.model.parameters():
            zeroed_count += (param == 0).sum().item()
            total_count += param.numel()

        return zeroed_count / total_count * 100

    def get_layer_sparsities(self) -> Dict[str, float]:
        """
        Get the sparsity (percentage of zeroed weights) for each layer.
        """
        sparsities = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                sparsities[name] = (param == 0).sum().item() / param.numel() * 100
        return sparsities

    def magnitude_prune(self, param: torch.Tensor, sparsity: float) -> torch.Tensor:
        """
        Apply magnitude pruning to a tensor by zeroing the smallest magnitudes.
        """
        num_zeroed = int(param.numel() * sparsity)
        threshold = torch.topk(param.abs().flatten(), num_zeroed, largest=False).values[-1]
        param.data[param.abs() <= threshold] = 0
        return param

    def apply_zeroed_gradients(self):
        """
        Apply zeroed gradients to pruned weights (those corresponding to zeroed masks).
        """
        for name, param in self.model.named_parameters():
            if name in self.saved_masks:
                mask = self.saved_masks[name]
                if param.grad is not None:
                    param.grad.mul_(mask)  # Mask gradients
                
    def fine_tune_model(self) -> None:
        """
        Fine-tune the model after pruning.
        """
        self.logger.info(f"\nFine-tuning the model for {self.E} epochs...\n")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.to(self.device)
        for epoch in  tqdm(range(self.E), desc='Fine-tuning', unit='epoch', leave=False):
            self.model.train()
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()

                # Apply zeroed gradients before optimizer step
                self.apply_zeroed_gradients()

                optimizer.step()

            accuracy = self.log_accuracy()
            self.metrics.append({'step': epoch, 'loss': loss.item(), 'accuracy': accuracy, 'sparsity': self.current_sparsity})

    def calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate accuracy given the outputs and targets.
        """
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / len(targets)
        return accuracy
