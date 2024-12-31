import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm  # For progress bars

class IterativeMagnitudePruning:
    """
    Class for performing Iterative Magnitude Pruning (IMP) on a neural network model.
    """
    
    def __init__(self, model, X_train, y_train, final_sparsity, steps, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, pretrain_epochs=20, device=None):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.
        """
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

        # Dictionary to store checkpoints during pruning
        self.checkpoints = {}

        print(f"\n{'='*50}")
        print(f"Initialized Iterative Magnitude Pruning with {self.steps} steps.")
        print(f"{'='*50}\n")
    
    def run(self):
        """
        Perform the iterative magnitude pruning (IMP) with weight resetting and fine-tuning.
        """
        # Step 1: Train or fine-tune the model if it's not pretrained
        if not self.is_pretrained:
            print("Pretraining model as it is not pretrained.")
            self.pretrain_model()

        # Gradual pruning and fine-tuning
        print("\nStarting gradual pruning and fine-tuning process...\n")
        self.perform_gradual_pruning()

        return self.model

    def pretrain_model(self):
        """
        Pretrain the model if it is not already pretrained.
        """
        print(f"\nPretraining the model for {self.pretrain_epochs} epochs...\n")
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
            
            # Print training info
            print(f"Epoch {epoch + 1}/{self.pretrain_epochs} - Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

    def perform_gradual_pruning(self):
        """
        Perform gradual pruning from the current sparsity to final sparsity using np.linspace.
        """
        print(f"\nStarting gradual pruning: current zeroed weights = {self.current_sparsity * 100:.2f}%")
        print(f"Target final non-zero weights = {(1 - self.final_sparsity) * 100:.2f}%")
        print(f"Pruning in {self.steps} steps...\n")

        # Generate sparsity values using np.linspace for smooth transitions
        pruning_steps = np.linspace(self.current_sparsity, self.final_sparsity, self.steps)[1:]
        
        # Display the steps visually
        print(f"Pruning steps (zeroed weights): {pruning_steps}\n")
        for step, target_sparsity in enumerate(pruning_steps):
            print(f"Step {step + 1}/{self.steps}: Pruning to {target_sparsity * 100:.2f}% zeroed weights.")
            self.prune_weights(target_sparsity)
            # Fine-tune the model after pruning
            print(f"before fine_tune_model === {self.get_model_zeroed_weight_percentage()}")
            self.fine_tune_model()
            print(f"after fine_tune_model=== {self.get_model_zeroed_weight_percentage()}")
            
            # Recalculate the actual zeroed weights after pruning and fine-tuning
            actual_zeroed_weights = self.get_model_zeroed_weight_percentage()
            print(f"\nZeroed weights after pruning and fine-tuning: {actual_zeroed_weights:.2f}%\n")
            # Log the accuracy after pruning
            self.log_accuracy()


    def log_accuracy(self):
        """
        Log the accuracy of the model after pruning.
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(self.X_train)
            accuracy = self.calculate_accuracy(outputs, self.y_train)
            print(f"Accuracy after pruning: {accuracy:.2f}%")
            
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
        Replace the standard Linear and Conv2d layers with Pruned versions before pruning starts.
        """
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                print(f"Replacing Linear layer {name} with PrunedLinear")
                setattr(self.model, name, PrunedLinear(module.in_features, module.out_features))
            elif isinstance(module, nn.Conv2d):
                print(f"Replacing Conv2d layer {name} with PrunedConv2d")
                setattr(self.model, name, PrunedConv2d(module.in_channels, module.out_channels, module.kernel_size))
            # No recursive call here unless you handle submodules carefully

    def prune_weights(self, prune_percentage):
        """
        Perform pruning based on the specified percentage of weights to prune.
        This ensures that we prune exactly the desired percentage of weights
        across all layers of the model.
        """
        print(f"Pruning weights to {prune_percentage * 100:.2f}% zeroed weights.")

        # Collect all weights and their corresponding masks
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
                # move mask to device
                mask = mask.to(self.device)
                module.weight.data.mul_(mask)  # Apply mask to weights
                module.weight.requires_grad = False
                self.saved_masks[name] = mask  # Save the mask for this layer
                new_start_idx += num_params

        # After pruning, ensure that we report the sparsity correctly
        self.current_sparsity = self.get_model_zeroed_weight_percentage()
        print(f"Zeroed weights after pruning: {self.current_sparsity:.2f}%")

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

    def magnitude_prune(self, prune_percentage, all_weights):
        """
        Prune weights by magnitude, removing the smallest values based on their absolute value.
        """
        num_to_prune = int(prune_percentage * all_weights.numel())
        # Get the indices of the smallest weights
        _, indices = torch.topk(all_weights.abs(), num_to_prune, largest=False)
        
        # Create a mask that sets the smallest weights to zero
        mask = torch.ones(all_weights.numel(), device=all_weights.device)
        mask[indices] = 0  # Set smallest weights to zero
        
        return mask

    def apply_zeroed_gradients(self):
        """
        Apply zeroed gradients to pruned weights (those corresponding to zeroed masks).
        This ensures that pruned weights are ignored during optimization.
        """
        for name, param in self.model.named_parameters():
            if name in self.saved_masks:
                mask = self.saved_masks[name]
                
                # Check if gradients exist for this parameter
                if param.grad is not None:
                    # Zero out the gradients for pruned weights by applying the mask
                    param.grad.mul_(mask)  # Mask gradients

                    # Alternatively, set gradients to None for pruned weights entirely
                    if mask.sum().item() == 0:  # All weights in this layer are pruned
                        param.grad = None  # Ignore this weight during optimization

    def fine_tune_model(self):
        """
        Fine-tune the model after pruning, ensuring pruned weights stay zeroed out 
        and gradients are zeroed for pruned weights.
        """
        print(f"Inside fine_tune_model === {self.get_model_zeroed_weight_percentage()}")
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(self.X_train)
        loss = criterion(outputs, self.y_train)
        
        print(f"before loss.backward === {self.get_model_zeroed_weight_percentage()}")
        loss.backward()
        print(f"after loss.backward === {self.get_model_zeroed_weight_percentage()}")
        
        print(f"before optimizer step === {self.get_model_zeroed_weight_percentage()}")
        
        # Apply gradients masking before optimizer step
        self.apply_zeroed_gradients()
        optimizer.step()
        
        print(f"after optimizer step === {self.get_model_zeroed_weight_percentage()}")
        import pdb; pdb.set_trace()
        # Log accuracy after fine-tuning
        accuracy = self.calculate_accuracy(outputs, self.y_train)
        print(f"Fine-Tune Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")

    def calculate_accuracy(self, outputs, targets):
        """
        Calculate the accuracy of the model.
        """
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        accuracy = 100 * correct / targets.size(0)
        return accuracy

class PrunedLinear(nn.Linear):
    def forward(self, x):
        if hasattr(self, 'mask'):
            # Debug: print mask application
            print(f"Applying mask to Linear layer: {self.mask.sum()} weights are pruned.")
            self.weight.data *= self.mask.view(self.weight.data.shape)
        return super().forward(x)

class PrunedConv2d(nn.Conv2d):
    def forward(self, x):
        if hasattr(self, 'mask'):
            # Debug: print mask application
            print(f"Applying mask to Conv2d layer: {self.mask.sum()} weights are pruned.")
            self.weight.data *= self.mask.view(self.weight.data.shape)
        return super().forward(x)
