import torch
import torch.nn as nn
import torch.optim as optim

class IterativeMagnitudePruning:
    def __init__(self, model, X_train, y_train, final_sparsity, prune_step, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, device=None, pretrain_epochs=20):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.

        Args:
            model (torch.nn.Module): The model to prune.
            X_train (torch.Tensor): The training data.
            y_train (torch.Tensor): The training labels.
            final_sparsity (float): Target sparsity level (final sparsity percentage, e.g., 0.99 for 99%).
            prune_step (int): Number of pruning steps to reach the final sparsity (e.g., 5).
            E (int): Number of fine-tuning epochs after each pruning step (default: 10).
            pruning_criterion (function): Custom pruning criterion (default: magnitude pruning).
            reset_weights (bool): Whether to reset remaining weights after pruning (default: True).
            is_pretrained (bool): Whether the model is pretrained or not (default: False).
            device (torch.device): The device to run the model on (default: None, auto-detect).
            pretrain_epochs (int): Number of pretraining epochs (default: 20).
        """
        assert 0 < final_sparsity < 1, "final_sparsity must be between 0 and 1."
        assert 0 < prune_step, "prune_step must be a positive integer."
        assert isinstance(model, nn.Module), "model must be a PyTorch nn.Module."

        self.final_sparsity = final_sparsity
        self.prune_step = prune_step
        self.E = E
        self.pruning_criterion = pruning_criterion or self.magnitude_pruning
        self.reset_weights = reset_weights
        self.is_pretrained = is_pretrained
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pretrain_epochs = pretrain_epochs
        
        self.model = model.to(self.device)  # Move model to device
        self.X_train = X_train.to(self.device)  # Move input data to device
        self.y_train = y_train.to(self.device)  # Move labels to device
        
        # Save initial weights for reset during pruning
        self.initial_weights = self.save_initial_weights()

        # Initialize current sparsity
        self.current_sparsity = self.get_model_sparsity()

        # Initialize a list to store checkpoints after each pruning step
        self.checkpoints = []

    def run(self):
        """
        Perform the iterative magnitude pruning (IMP) with weight resetting and fine-tuning.
        
        Returns:
            torch.nn.Module: The pruned model after all iterations.
        """
        print("Starting pruning process.")
        
        # Step 1: Train or fine-tune the model if it's not pretrained
        if not self.is_pretrained:
            print(f"Pretraining the model for {self.pretrain_epochs} epochs.")
            self.train_model(self.pretrain_epochs)  # Pretrain the model before pruning

        # Gradual pruning and fine-tuning
        self.perform_gradual_pruning()

        return self.model
    
    def train_model(self, epochs):
        """
        Train the model for a specified number of epochs.

        Args:
            epochs (int): Number of training epochs.
        """
        print(f"Training the model for {epochs} epochs.")
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = criterion(output, self.y_train)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    def perform_gradual_pruning(self):
        """
        Perform gradual pruning of the model weights in steps.
        """
        print(f"Starting gradual pruning: current sparsity = {self.current_sparsity * 100:.2f}%")
        print(f"Target final sparsity = {self.final_sparsity * 100:.2f}%")

        # Calculate the amount of pruning per step based on the number of steps
        prune_per_step = (self.final_sparsity - self.current_sparsity) / self.prune_step
        print(f"Pruning {prune_per_step * 100:.2f}% at each step")

        for step in range(1, self.prune_step + 1):
            print(f"\nStep {step} of {self.prune_step}:")
            
            # Prune the model weights for this step
            self.current_sparsity = self.prune_weights(prune_per_step)
            
            # Fine-tune the model after pruning
            self.fine_tune_model()

            # Save checkpoint for this step
            self.checkpoints.append(self.save_checkpoint(step))
            print(f"Saving checkpoint for step {step}...\n")

            # Skip sparsity check after each step, except the last step
            if step == self.prune_step:
                self.assert_mask_sparsity(self.final_sparsity)
    
    def prune_weights(self, prune_percentage):
        """
        Prune the model weights based on the specified pruning percentage.
        """
        print(f"Pruning weights: {prune_percentage * 100:.2f}%")
        
        # Flatten all the weight parameters by iterating over named parameters
        flattened_weights = torch.cat([param.view(-1) for name, param in self.model.named_parameters() if 'weight' in name])
        
        total_weights = flattened_weights.numel()
        num_pruned = int(prune_percentage * total_weights)

        # Find the threshold to prune based on the smallest magnitude weights
        abs_weights = flattened_weights.abs()
        threshold = torch.kthvalue(abs_weights, num_pruned).values.item()

        # Create a mask based on the threshold
        mask = abs_weights > threshold
        self.global_mask = mask  # Store the mask

        # Apply the mask to each weight parameter
        idx = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                num_elements = param.numel()
                mask_layer = self.global_mask[idx:idx + num_elements]
                mask_layer = mask_layer.view_as(param.data)
                param.data *= mask_layer  # Apply mask to the weights
                idx += num_elements

        # Calculate current sparsity
        current_sparsity = 1.0 - float(mask.sum()) / float(total_weights)
        print(f"After pruning: current sparsity = {current_sparsity * 100:.2f}%")

        return current_sparsity

    def apply_global_mask(self):
        """
        Apply the global mask to the model's parameters.
        """
        idx = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                num_elements = param.numel()
                param.data *= self.global_mask[idx:idx + num_elements].view_as(param.data)
                idx += num_elements

    def fine_tune_model(self):
        """
        Fine-tune the model for E epochs after pruning.
        """
        print(f"Fine-tuning the model for {self.E} epochs.")
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.E):
            self.model.train()
            optimizer.zero_grad()
            output = self.model(self.X_train)
            loss = criterion(output, self.y_train)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{self.E}, Loss: {loss.item()}")

    def assert_mask_sparsity(self, target_sparsity, tolerance=0.01):
        """
        Assert that the current sparsity is close to the target sparsity.
        """
        current_sparsity = self.get_model_sparsity()
        assert abs(current_sparsity - target_sparsity) < tolerance, \
            f"Sparsity mismatch! Target: {target_sparsity * 100}%, Actual: {current_sparsity * 100}%"

    def get_model_sparsity(self):
        """
        Calculate the current sparsity of the model (percentage of zero weights).
        """
        total_params = 0
        total_zeros = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                total_params += param.numel()
                total_zeros += torch.sum(param == 0).item()

        sparsity = total_zeros / total_params
        return sparsity

    def save_initial_weights(self):
        """
        Save the initial weights of the model.
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters() if 'weight' in name}

    def magnitude_pruning(self, weights, pruning_percentage):
        """
        Magnitude-based pruning: remove the smallest weights (by magnitude).
        
        Args:
            weights (torch.Tensor): The tensor of model weights to prune.
            pruning_percentage (float): The percentage of weights to prune (e.g., 0.5 for 50%).
        
        Returns:
            torch.Tensor: The pruned weight tensor.
        """
        num_weights = weights.numel()
        threshold_index = int(pruning_percentage * num_weights)
        threshold_value = torch.kthvalue(weights.abs().view(-1), threshold_index).values.item()
        pruned_weights = torch.where(weights.abs() > threshold_value, weights, torch.zeros_like(weights))
        return pruned_weights

    def save_checkpoint(self, step):
        """
        Save a checkpoint of the model after a pruning step.
        
        Args:
            step (int): The current pruning step.
        
        Returns:
            dict: Checkpoint data containing model weights.
        """
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
        }
        return checkpoint

