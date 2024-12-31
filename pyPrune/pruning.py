import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

class IterativeMagnitudePruning:
    def __init__(self, model, X_train, y_train, final_sparsity, steps, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, pretrain_epochs=20, device=None):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.

        Args:
            model (torch.nn.Module): The model to prune.
            X_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.
            final_sparsity (float): Target sparsity level (final sparsity percentage, e.g., 0.99 for 99%).
            steps (int): Number of pruning steps to gradually reach final_sparsity.
            E (int): Number of fine-tuning epochs after each pruning step (default: 10).
            pruning_criterion (function): Custom pruning criterion (default: magnitude pruning).
            reset_weights (bool): Whether to reset remaining weights after pruning (default: True).
            is_pretrained (bool): Whether the model is pretrained or not (default: False).
            pretrain_epochs (int): Number of epochs to pretrain if the model is not pretrained.
            device (torch.device): The device to run the model on (default: None, auto-detect).
        """
        assert 0 < final_sparsity < 1, "final_sparsity must be between 0 and 1."
        assert steps > 0, "steps must be a positive integer."

        self.final_sparsity = final_sparsity
        self.prune_sparsity = final_sparsity / steps
        self.steps = steps
        self.E = E
        self.pruning_criterion = pruning_criterion or self.magnitude_pruning
        self.reset_weights = reset_weights
        self.is_pretrained = is_pretrained
        self.pretrain_epochs = pretrain_epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        self.model.to(self.device)  # Move model to device

        # Save initial weights for reset during pruning
        self.initial_weights = self.save_initial_weights()

        self.current_sparsity = 0.0  # Current sparsity level
        # Dictionary to store checkpoints during pruning
        self.checkpoints = {}

    def run(self):
        """
        Perform the iterative magnitude pruning (IMP) with weight resetting and fine-tuning.

        Returns:
            torch.nn.Module: The pruned model after all iterations.
        """
        # Step 1: Train or fine-tune the model if it's not pretrained
        if not self.is_pretrained:
            self.pretrain_model()

        # Gradual pruning and fine-tuning
        self.perform_gradual_pruning()

        return self.model

    def pretrain_model(self):
        """
        Pretrain the model if it is not already pretrained.
        """
        print(f"Pretraining the model for {self.pretrain_epochs} epochs.")
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
            optimizer.step()
            print(f"Pretrain Epoch {epoch + 1}, Loss: {loss.item()}")

    def perform_gradual_pruning(self):
        """
        Perform gradual pruning from the current sparsity to final sparsity.
        """
        
        prune_percentage_per_step = self.prune_sparsity  # Derived pruning percentage per step

        print(f"Starting gradual pruning: current sparsity = {self.current_sparsity:.2f}%")
        print(f"Target final sparsity = {self.final_sparsity * 100:.2f}%")
        print(f"Pruning in {self.steps} steps, each pruning {prune_percentage_per_step * 100:.2f}% of weights.")

        for step in range(self.steps):
            self.current_sparsity += prune_percentage_per_step

            if self.current_sparsity >= self.final_sparsity:
                print(f"Target sparsity reached: {self.final_sparsity * 100:.2f}%")
                break  # Stop pruning if we've reached the final sparsity
            
            print(f"Step {step + 1} of {self.steps}:")
            self.prune_weights(self.current_sparsity)

            self.fine_tune_model()

            # Save checkpoint for this iteration
            self.save_checkpoint(step)

    def save_initial_weights(self):
        """
        Save the initial weights of the model to restore later.
        """
        initial_weights = {}
        for name, param in self.model.named_parameters():
            initial_weights[name] = param.data.clone()
        return initial_weights

    def prune_weights(self, prune_percentage):
        """
        Prune weights based on the selected criterion (e.g., magnitude).
        """
        with torch.no_grad():
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):  # Apply pruning to specific layers
                    self.pruning_criterion(module, name, prune_percentage)

    def magnitude_pruning(self, module, name, p):
        """
        Prune weights based on their magnitude.
        """
        # Flatten the weights and compute the magnitude
        weight = module.weight.data
        flattened_weights = weight.view(-1)
        sorted_indices = torch.argsort(torch.abs(flattened_weights))  # Sort by absolute magnitude

        # Calculate number of weights to prune
        num_to_prune = int(p * flattened_weights.numel())

        # Create a mask that prunes the smallest weights
        mask = torch.ones_like(flattened_weights)
        mask[sorted_indices[:num_to_prune]] = 0  # Set smallest weights to 0

        # Apply the mask to the weights
        weight.view(-1).data *= mask

        # Set the gradients for pruned weights to False
        for param in module.parameters():
            if param.requires_grad:
                param.grad = None

    def fine_tune_model(self):
        """
        Fine-tune the model after pruning for a given number of epochs.
        """
        print(f"Fine-tuning the model for {self.E} epochs.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.E):  # Fine-tune for E epochs
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
            print(f"Fine-tune Epoch {epoch + 1}, Loss: {loss.item()}")

    def get_model_sparsity(self):
        """
        Calculate the current sparsity of the model (percentage of zeroed-out weights).
        """
        total_params = 0
        total_zeros = 0
        for param in self.model.parameters():
            total_params += param.numel()
            total_zeros += torch.sum(param == 0).item()
        sparsity = 100 * total_zeros / total_params
        return sparsity

    def save_checkpoint(self, step):
        """
        Save checkpoint after each pruning step (weights, masks, loss, etc.).
        """
        self.checkpoints[step] = {
            'weights': {name: param.data.clone() for name, param in self.model.named_parameters()},
            'masks': {name: param == 0 for name, param in self.model.named_parameters()},
            'sparsity': self.get_model_sparsity(),
        }
        print(f"Checkpoint {step + 1} saved. Sparsity: {self.checkpoints[step]['sparsity']:.2f}%, Mask percentage: {sum([torch.sum(mask).item() for mask in self.checkpoints[step]['masks'].values()]) / sum([mask.numel() for mask in self.checkpoints[step]['masks'].values()]) * 100:.2f}%")
