import torch
import torch.nn as nn
import numpy as np

class IterativeMagnitudePruning:
    def __init__(self, model, X_train, y_train, final_sparsity, steps, 
                 E=10, pruning_criterion=None, reset_weights=True, 
                 is_pretrained=False, pretrain_epochs=20, device=None):
        """
        Initialize the iterative magnitude pruning process with gradual pruning.
        """
        assert 0 < final_sparsity < 1, "final_sparsity must be between 0 and 1."
        assert steps > 0, "steps must be a positive integer."
        assert isinstance(model, nn.Module), "The model must be a subclass of nn.Module."
        assert isinstance(X_train, torch.Tensor), "X_train must be a torch tensor."
        assert isinstance(y_train, torch.Tensor), "y_train must be a torch tensor."
        
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

        # Save initial weights and masks for reset during pruning
        self.initial_weights = self.save_initial_weights()
        self.saved_masks = {}  # Store the masks for each layer

        # Dictionary to store checkpoints during pruning
        self.checkpoints = {}

        print(f"Initialized Iterative Magnitude Pruning with {self.steps} steps.")
    
    def run(self):
        """
        Perform the iterative magnitude pruning (IMP) with weight resetting and fine-tuning.
        """
        # Step 1: Train or fine-tune the model if it's not pretrained
        if not self.is_pretrained:
            print("Pretraining model as it is not pretrained.")
            self.pretrain_model()

        # Gradual pruning and fine-tuning
        print("Starting gradual pruning and fine-tuning process.")
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
        Perform gradual pruning from the current sparsity to final sparsity using np.arange.
        """
        print(f"Starting gradual pruning: current sparsity = {self.current_sparsity * 100:.2f}%")
        print(f"Target final sparsity = {self.final_sparsity * 100:.2f}%")
        print(f"Pruning in {self.steps} steps.")

        # Generate sparsity values using np.linspace for smooth transitions
        pruning_steps = np.linspace(0, self.final_sparsity, self.steps + 1)  
        print(f"Pruning steps: {pruning_steps[1::-1]}")

        for step, target_sparsity in enumerate(pruning_steps[1:], 1):  # Skip the 0% sparsity (initial state)
            print(f"Step {step} of {self.steps}: Pruning to {target_sparsity * 100:.2f}% sparsity.")
            self.prune_weights(target_sparsity)

            # Fine-tune the model after pruning
            self.fine_tune_model()

            # Recalculate the actual model sparsity after pruning and fine-tuning
            actual_sparsity = self.get_model_sparsity()
            print(f"Sparsity after pruning and fine-tuning: {actual_sparsity:.2f}%")

            # Save checkpoint for this iteration
            self.save_checkpoint(step)

            # Stop pruning if we've reached the final sparsity
            if target_sparsity >= self.final_sparsity:
                print(f"Target sparsity reached: {self.final_sparsity * 100:.2f}%")
                break

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
        Replace the standard Linear and Conv2d layers with PrunedLinear and PrunedConv2d.
        """
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                print(f"Replacing Linear layer {name} with PrunedLinear")
                setattr(self.model, name, PrunedLinear(module.in_features, module.out_features))
            elif isinstance(module, nn.Conv2d):
                print(f"Replacing Conv2d layer {name} with PrunedConv2d")
                setattr(self.model, name, PrunedConv2d(module.in_channels, module.out_channels, module.kernel_size))
            else:
                self.replace_with_pruned_layers(module)  # Recursively handle submodules

    def prune_weights(self, prune_percentage):
        """
        Concatenate all model weights, calculate pruning mask, and prune weights.
        """
        print(f"Pruning weights with {prune_percentage * 100:.2f}% sparsity.")
        self.magnitude_prune(prune_percentage)

    def magnitude_prune(self, prune_percentage):
        """
        Perform pruning based on magnitude (absolute values) of weights.
        This method will flatten all weights, sort them by magnitude, and prune the smallest weights.
        """
        # Step 1: Collect all weights from the model layers
        all_weights = []
        all_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                all_weights.append(weight.view(-1))  # Flatten and add to list
                all_names.append(name)  # Store the name for mask tracking
        
        # Step 2: Concatenate all weights into a single tensor
        all_weights = torch.cat(all_weights)  # Concatenate all the weights
        all_weights_abs = torch.abs(all_weights)  # Take the absolute value for magnitude pruning

        # Step 3: Sort weights by magnitude (absolute values)
        sorted_indices = torch.argsort(all_weights_abs)

        # Step 4: Calculate how many weights to prune
        num_weights = all_weights.size(0)
        num_to_prune = int(num_weights * prune_percentage)

        # Step 5: Create a mask for pruning
        prune_mask = torch.ones(num_weights).to(self.device)
        prune_mask[sorted_indices[:num_to_prune]] = 0  # Set smallest `num_to_prune` weights to 0

        # Step 6: Reapply the mask to the weights
        start_idx = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                num_params = weight.numel()  # Number of parameters in this layer
                mask = prune_mask[start_idx:start_idx + num_params]
                # Apply the mask to the weight and store it
                module.mask = mask.clone()  # Set the mask as an attribute
                weight.mul_(mask.view(weight.shape))  # Apply the mask to the weight
                start_idx += num_params
                
                # Save the mask for future use
                self.saved_masks[name] = mask.clone()

    def load_weights(self):
        """
        Load the pruned weights back into the model using the masks.
        """
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                mask = self.saved_masks.get(name)
                if mask is not None:
                    weight.mul_(mask.view(weight.shape))  # Reapply the saved mask to the weights
                else:
                    print(f"Warning: No mask found for {name}, skipping weight reload.")
    
    def fine_tune_model(self):
        """
        Fine-tune the model after pruning.
        """
        print("Fine-tuning the model for 5 epochs.")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Fine-tuning loop
        self.model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = self.model(self.X_train)
            loss = criterion(outputs, self.y_train)
            loss.backward()
            optimizer.step()
            print(f"Fine-tune Epoch {epoch + 1}, Loss: {loss.item()}")

    def get_model_sparsity(self):
        """
        Calculate the current sparsity of the model based on the masks.
        """
        total_params = 0
        pruned_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                weight = module.weight.data
                mask = self.saved_masks.get(name)
                if mask is not None:
                    pruned_params += torch.sum(mask == 0).item()
                    total_params += weight.numel()
        sparsity = (pruned_params / total_params) * 100
        return sparsity

    def save_checkpoint(self, step):
        """
        Save checkpoint after each pruning step (weights, masks, sparsity, etc.).
        """
        self.checkpoints[step] = {
            'weights': {name: param.data.clone() for name, param in self.model.named_parameters()},
            'masks': {name: getattr(module, 'mask', None) for name, module in self.model.named_modules()},
            'sparsity': self.get_model_sparsity()
        }

        # Check for None masks and avoid operations on them
        total_masked_weights = 0
        total_weights = 0
        for mask in self.checkpoints[step]['masks'].values():
            if mask is not None:
                total_masked_weights += mask.sum().item()
                total_weights += mask.numel()


        # Print checkpoint information
        print(f"\tCheckpoint {step} saved.")
        print(f"\tSparsity: {self.checkpoints[step]['sparsity']:.2f}%")
        print(f"\tTotal masked weights: {total_masked_weights}")
        print(f"\tTotal weights: {total_weights}")
        print(f"\tTotal masked percentage: {(1-(total_masked_weights/total_weights)) * 100:.2f}%")
        


class PrunedLinear(nn.Linear):
    def forward(self, x):
        if hasattr(self, 'mask'):
            # Ensure the mask is applied to the weights correctly
            # Reshape the mask to match the weight dimensions
            self.weight.data *= self.mask.view(self.weight.data.shape)  # Apply mask during forward pass (keeping original weight shape)
        return super().forward(x)


class PrunedConv2d(nn.Conv2d):
    def forward(self, x):
        if hasattr(self, 'mask'):
            # Ensure the mask is applied to the weights correctly
            # Reshape the mask to match the weight dimensions
            self.weight.data *= self.mask.view(self.weight.data.shape)  # Apply mask during forward pass (keeping original weight shape)
        return super().forward(x)
