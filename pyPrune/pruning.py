import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import copy

class IterativeMagnitudePruning:
    def __init__(self, model, X_train, y_train, final_sparsity, prune_sparsity, steps, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, device=None):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.

        Args:
            model (torch.nn.Module): The model to prune.
            X_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.
            final_sparsity (float): Target sparsity level (final sparsity percentage, e.g., 0.99 for 99%).
            prune_sparsity (float): Percentage of weights to prune in each step (e.g., 0.1 for 10%).
            steps (int): Number of pruning steps to gradually reach final_sparsity.
            E (int): Number of fine-tuning epochs after each pruning step (default: 10).
            pruning_criterion (function): Custom pruning criterion (default: magnitude pruning).
            reset_weights (bool): Whether to reset remaining weights after pruning (default: True).
            is_pretrained (bool): Whether the model is pretrained or not (default: False).
            device (torch.device): The device to run the model on (default: None, auto-detect).
        """
        assert 0 < final_sparsity < 1, "final_sparsity must be between 0 and 1."
        assert 0 < prune_sparsity < 1, "prune_sparsity must be between 0 and 1."
        assert steps > 0, "steps must be a positive integer."

        self.final_sparsity = final_sparsity
        self.prune_sparsity = prune_sparsity
        self.steps = steps
        self.E = E
        self.pruning_criterion = pruning_criterion or self.magnitude_pruning
        self.reset_weights = reset_weights
        self.is_pretrained = is_pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        self.model.to(self.device)  # Move model to device

        # Save initial weights for reset during pruning
        self.initial_weights = self.save_initial_weights()

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
            self.train_model()

        # Gradual pruning and fine-tuning
        self.perform_gradual_pruning()

        return self.model

    def perform_gradual_pruning(self):
        """
        Perform gradual pruning from the current sparsity to final sparsity.
        """
        current_sparsity = self.get_model_sparsity()
        total_steps = self.steps
        prune_percentage_per_step = self.prune_sparsity  # This will be fixed across steps

        print(f"Starting gradual pruning: current sparsity = {current_sparsity:.2f}%")
        print(f"Target final sparsity = {self.final_sparsity * 100:.2f}%")
        print(f"Pruning in {total_steps} steps, each pruning {prune_percentage_per_step * 100:.2f}% of weights.")

        for step in range(self.steps):
            current_sparsity = self.get_model_sparsity()

            if current_sparsity >= self.final_sparsity:
                print(f"Target sparsity reached: {self.final_sparsity * 100:.2f}%")
                break  # Stop pruning if we've reached the final sparsity
            
            # Step 2: Reset weights to their initial state before pruning
            self.reset_weights_to_initial()

            # Step 3: Prune weights based on the pruning criterion (e.g., magnitude pruning)
            self.prune_weights(prune_percentage_per_step)

            # Step 4: Optionally reset the remaining weights
            if self.reset_weights:
                self.reset_remaining_weights()

            # Step 5: Fine-tune the model after pruning
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

    def reset_weights_to_initial(self):
        """
        Reset the model weights to their initial values before pruning.
        """
        for name, param in self.model.named_parameters():
            param.data = self.initial_weights[name].clone()

    def prune_weights(self, prune_percentage):
        """
        Prune weights based on the selected criterion (e.g., magnitude).
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):  # Apply pruning to specific layers
                self.pruning_criterion(module, name, prune_percentage)

    def magnitude_pruning(self, module, name, p):
        """
        Prune weights based on their magnitude.
        """
        prune.l1_unstructured(module, name=name + '.weight', amount=p)

    def reset_remaining_weights(self):
        """
        Reset non-pruned weights to small random values.
        """
        for name, param in self.model.named_parameters():
            if 'mask' not in name:  # Skip pruning masks
                # Apply small random reset to non-pruned weights
                param.data.normal_(mean=0, std=0.01)

    def train_model(self):
        """
        Train the model from scratch.
        """
        self.model.to(self.device)  # Move the model to the selected device
        self.X_train, self.y_train = self.X_train.to(self.device), self.y_train.to(self.device)  # Move data to device

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(20):  # Train for 20 epochs
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    def fine_tune_model(self):
        """
        Fine-tune the model after pruning for a given number of epochs.
        """
        self.model.to(self.device)  # Move the model to the selected device
        self.X_train, self.y_train = self.X_train.to(self.device), self.y_train.to(self.device)  # Move data to device

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
            'sparsity': self.get_model_sparsity(),
        }
        print(f"Checkpoint {step + 1} saved. Sparsity: {self.checkpoints[step]['sparsity']:.2f}%")
