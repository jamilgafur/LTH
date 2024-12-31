import torch
import torch.nn as nn
import numpy as np
import logging
import pandas as pd
import os
from tqdm import tqdm  # For progress bars
from datetime import datetime
from .model_utils import PrunedLinear, PrunedConv2d

class IterativeMagnitudePruning:
    """
    Class for performing Iterative Magnitude Pruning (IMP) on a neural network model.
    """

    def __init__(self, model, X_train, y_train, final_sparsity, steps, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, pretrain_epochs=20, device=None, 
                 save_dir='pruning_checkpoints'):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.
        
        Parameters:
            model (nn.Module): Neural network model to be pruned.
            X_train (torch.Tensor): Training input data.
            y_train (torch.Tensor): Training labels.
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
        assert isinstance(X_train, torch.Tensor), "X_train must be a torch tensor."
        assert isinstance(y_train, torch.Tensor), "y_train must be a torch tensor."
        
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
        self.X_train = X_train
        self.y_train = y_train
        self.model.to(self.device)  # Move model to device

        # Replace layers with Pruned versions before pruning starts
        self.replace_with_pruned_layers()

        # Save initial weights and zeroed masks for reset during pruning
        self.initial_weights = self.save_initial_weights()
        self.saved_masks = {}  # Store the zeroed masks for each layer
        self.saved_weights = {}  # Store pruned weights after each step

        # Initialize metrics list
        self.metrics = []

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Initialized Iterative Magnitude Pruning with {self.steps} steps.")
        self.logger.info(f"{'='*50}\n")

        # Path to store the final metrics in a Parquet file
        self.metrics_file = os.path.join(self.save_dir, 'metrics.parquet')

    def setup_logger(self, log_file):
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

    def setup_save_dir(self):
        """
        Ensure the save directory exists, and create it if necessary.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def run(self):
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

        return self.model

    def pretrain_model(self):
        """
        Pretrain the model if it is not already pretrained.
        """
        self.logger.info(f"\nPretraining the model for {self.pretrain_epochs} epochs...\n")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # move everything to device
        self.model.to(self.device)
        self.X_train = self.X_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        
        for epoch in range(self.pretrain_epochs):  # Pretrain for the specified number of epochs
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()

            # Apply zeroed gradients before optimizer step
            self.apply_zeroed_gradients()

            optimizer.step()

            # Calculate accuracy
            accuracy = self.calculate_accuracy(outputs, self.y_train)
            
            # Log training info
            self.logger.info(f"Epoch {epoch + 1}/{self.pretrain_epochs} - Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
            self.metrics.append({'step': epoch, 'loss': loss.item(), 'accuracy': accuracy, 'sparsity': self.current_sparsity})

    def perform_gradual_pruning(self):
        """
        Perform gradual pruning from the current sparsity to final sparsity using np.linspace.
        """
        self.logger.info(f"\nStarting gradual pruning: current zeroed weights = {self.current_sparsity * 100:.2f}%")
        self.logger.info(f"Target final non-zero weights = {(1 - self.final_sparsity) * 100:.2f}%")
        self.logger.info(f"Pruning in {self.steps} steps...\n")

        # Generate sparsity values using np.linspace for smooth transitions
        pruning_steps = np.linspace(self.current_sparsity, self.final_sparsity, self.steps)[1:]
        
        for step, target_sparsity in enumerate(pruning_steps):
            self.logger.info(f"Step {step + 1}/{self.steps}: Pruning to {target_sparsity * 100:.2f}% zeroed weights.")
            self.prune_weights(target_sparsity)
            self.fine_tune_model()

            # Log the metrics after pruning
            accuracy = self.log_accuracy()
            layer_sparsities = self.get_layer_sparsities()
            self.metrics.append({'step': step, 'accuracy': accuracy, 'sparsity': self.current_sparsity,
                                 'layer_sparsities': layer_sparsities})

            # Save masks and weights for each pruning step
            self.save_masks_and_weights(step)

    def log_accuracy(self):
        """
        Log the accuracy of the model after pruning.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_train)
            accuracy = self.calculate_accuracy(outputs, self.y_train)
            self.logger.info(f"Accuracy after pruning: {accuracy:.2f}%")
        return accuracy

    def save_metrics_to_parquet(self):
        """
        Save the accumulated metrics to a Parquet file for analysis.
        """
        df = pd.DataFrame(self.metrics)
        df.to_parquet(self.metrics_file, index=False)
        self.logger.info(f"Metrics saved to {self.metrics_file}")

    def save_initial_weights(self):
        """
        Save the initial weights of the model to restore later.
        """
        initial_weights = {}
        for name, param in self.model.named_parameters():
            initial_weights[name] = param.data.clone()
        return initial_weights

    def replace_with_pruned_layers(self):
        """
        Replace the standard Linear and Conv2d layers with Pruned versions.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.logger.info(f"Replacing Linear layer {name} with PrunedLinear")
                setattr(self.model, name, PrunedLinear(module.in_features, module.out_features))
            elif isinstance(module, nn.Conv2d):
                self.logger.info(f"Replacing Conv2d layer {name} with PrunedConv2d")
                setattr(self.model, name, PrunedConv2d(module.in_channels, module.out_channels, module.kernel_size))

    def prune_weights(self, prune_percentage):
        """
        Perform pruning based on the specified percentage of weights to prune.
        """
        self.logger.info(f"Pruning weights to {prune_percentage * 100:.2f}% zeroed weights.")
        all_weights = []
        all_masks = []
        all_names = []
        start_idx = 0  # Start index to track the flat weight vector position
        weight_shapes = {}  # Dictionary to store the shapes of each layer's weights

        # Flatten all weights from the model layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                all_weights.append(weight.view(-1))  # Flatten the weight tensor
                weight_shapes[name] = weight.shape  # Store the shape of each layer
                all_names.append(name)  # Store the name for layer tracking

        # Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)

        # Call the pruning criterion (default is magnitude_prune)
        prune_mask = self.pruning_criterion(prune_percentage, all_weights)

        # Apply the mask to the weights in each layer and store the mask
        new_start_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                num_params = weight.numel()  # Number of parameters in this layer
                # Create mask for this layer based on the flat weight vector
                mask = prune_mask[new_start_idx:new_start_idx + num_params].view(weight.shape)
                mask = mask.to(self.device)
                module.weight.data.mul_(mask)  # Apply mask to weights
                module.weight.requires_grad = False
                self.saved_masks[name] = mask  # Save the mask for this layer
                new_start_idx += num_params

        # After pruning, ensure that we report the sparsity correctly
        self.current_sparsity = self.get_model_zeroed_weight_percentage()
        self.logger.info(f"Zeroed weights after pruning: {self.current_sparsity:.2f}%")

    def save_masks_and_weights(self, step):
        """
        Save the masks and weights at each pruning step.
        """
        step_dir = os.path.join(self.save_dir, f"step_{step + 1}")
        os.makedirs(step_dir, exist_ok=True)

        # Save masks and pruned weights
        for name, mask in self.saved_masks.items():
            torch.save(mask, os.path.join(step_dir, f"{name}_mask.pth"))
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                torch.save(param.data, os.path.join(step_dir, f"{name}_weights.pth"))
        
        self.logger.info(f"Saved masks and weights for step {step + 1}.")

    def get_model_zeroed_weight_percentage(self):
        """
        Get the percentage of zeroed weights in the model.
        """
        zeroed_weights = 0
        total_weights = 0
        for param in self.model.parameters():
            total_weights += param.numel()
            zeroed_weights += torch.sum(param == 0).item()

        return zeroed_weights / total_weights * 100

    def get_layer_sparsities(self):
        """
        Get the sparsity for each individual layer.
        """
        layer_sparsities = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                zeroed_weights = torch.sum(module.weight.data == 0).item()
                total_weights = module.weight.data.numel()
                sparsity = zeroed_weights / total_weights * 100
                layer_sparsities[name] = sparsity
        return layer_sparsities

    def magnitude_prune(self, prune_percentage, all_weights):
        """
        Prune weights by magnitude, removing the smallest values based on their absolute value.
        """
        num_to_prune = int(prune_percentage * all_weights.numel())
        _, indices = torch.topk(all_weights.abs(), num_to_prune, largest=False)
        
        mask = torch.ones(all_weights.numel(), device=all_weights.device)
        mask[indices] = 0
        return mask

    def apply_zeroed_gradients(self):
        """
        Apply zeroed gradients to pruned weights (those corresponding to zeroed masks).
        """
        for name, param in self.model.named_parameters():
            if name in self.saved_masks:
                mask = self.saved_masks[name]
                if param.grad is not None:
                    param.grad.mul_(mask)  # Mask gradients

    def fine_tune_model(self):
        """
        Fine-tune the model after pruning.
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(self.X_train)
        loss = criterion(outputs, self.y_train)
        loss.backward()

        self.apply_zeroed_gradients()
        optimizer.step()

        accuracy = self.calculate_accuracy(outputs, self.y_train)
        self.logger.info(f"Fine-Tune Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

    def calculate_accuracy(self, outputs, targets):
        """
        Calculate the accuracy of the model.
        """
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / targets.size(0)
        return accuracy

