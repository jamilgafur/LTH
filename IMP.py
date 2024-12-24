import torch
import os
import numpy as np
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt

class IterativeMagnitudePruning:
    def __init__(self, model, optimizer, criterion, pruning_rate, target_sparsity, device, trainloader, testloader, validloader, experiment_dir, pretrain_epochs=1, finetune_epochs=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.pruning_rate = pruning_rate
        self.target_sparsity = target_sparsity
        self.device = device
        self.trainloader = trainloader
        self.testloader = testloader
        self.validloader = validloader
        self.experiment_dir = experiment_dir  # Folder to save results
        self.pretrain_epochs = pretrain_epochs  # Number of pretraining epochs
        self.finetune_epochs = finetune_epochs  # Number of fine-tuning epochs

        # Initialize current sparsity
        self.current_sparsity = 0.0

        # Ensure experiment folder exists
        os.makedirs(experiment_dir, exist_ok=True)

        # Track accuracy vs sparsity
        self.accuracy_vs_sparsity = []

        # Prepare logs
        self.log_file = os.path.join(self.experiment_dir, "training_log.txt")
        with open(self.log_file, 'w') as log:
            log.write(f"Training Log for pruning experiment\n")
            log.write(f"Pruning rate: {self.pruning_rate}\n")
            log.write(f"Target sparsity: {self.target_sparsity}\n")
            log.write(f"Pretrain epochs: {self.pretrain_epochs}\n")
            log.write(f"Fine-tune epochs: {self.finetune_epochs}\n")
            log.write(f"=========================\n")

    def prune_and_retrain(self):
        """Main method to prune and retrain the model."""
        # Save the initial model before pruning
        self._save_initial_model()

        # Pretrain the model before pruning
        self.pretrain()

        # Perform incremental pruning
        while self.current_sparsity < self.target_sparsity:
            new_sparsity = min(self.current_sparsity + self.pruning_rate, self.target_sparsity)
            self.prune_step(new_sparsity)

        # Save masks for each layer after pruning
        self._save_masks()

        # Fine-tune the model after pruning
        self.fine_tune()

        # Save the final model after retraining
        self._save_final_model()

        # Generate and save accuracy vs sparsity plot
        self._plot_accuracy_vs_sparsity()

        # Save pruning logs
        self._save_pruning_log()

    def pretrain(self):
        """Pretrain the model before pruning."""
        print(f"Pretraining the model for {self.pretrain_epochs} epochs...")

        self.model.train()

        for epoch in range(self.pretrain_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients, backprop, optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * correct / total
            print(f"Pretraining Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.2f}%")

            # Track accuracy vs sparsity during pretraining
            sparsity = self.current_sparsity  # Sparsity is 0 during pretraining
            self.accuracy_vs_sparsity.append((epoch_accuracy, sparsity))

            # Log accuracy and sparsity
            self._log_accuracy_sparsity(epoch_accuracy, sparsity)

    def fine_tune(self):
        """Fine-tune the model after pruning."""
        print(f"Fine-tuning the model for {self.finetune_epochs} epochs...")

        self.model.train()

        for epoch in range(self.finetune_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero gradients, backprop, optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Track accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_accuracy = 100 * correct / total
            print(f"Fine-tuning Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {epoch_accuracy:.2f}%")

            # Track accuracy vs sparsity during fine-tuning
            sparsity = self.current_sparsity
            self.accuracy_vs_sparsity.append((epoch_accuracy, sparsity))

            # Log accuracy and sparsity
            self._log_accuracy_sparsity(epoch_accuracy, sparsity)

    def prune_step(self, new_sparsity):
        """
        Incrementally prune the model to a new sparsity level.
        Args:
            new_sparsity (float): The target sparsity level to reach in this step (e.g., 0.1 for 10% sparsity).
        """
        # Ensure new sparsity is higher than the current sparsity
        if new_sparsity <= self.current_sparsity:
            print(f"Sparsity already at or above the requested level ({new_sparsity}). No pruning applied.")
            return

        # Calculate the number of weights to prune in this step
        all_weights = []
        all_params = []

        # Collect all weight parameters and flatten them
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.cpu().numpy().flatten())
                all_params.append(param)

        # Concatenate all weights into a single vector
        all_weights = np.concatenate(all_weights)
        total_weights = all_weights.size

        # Number of weights to prune in this step (based on the difference in sparsity)
        num_weights_to_prune = int((new_sparsity - self.current_sparsity) * total_weights)

        # Identify the smallest weights and create a mask for pruning
        prune_indices = np.argsort(np.abs(all_weights))[:num_weights_to_prune]
        mask = np.ones_like(all_weights)
        mask[prune_indices] = 0

        # Apply the pruning mask to the model parameters
        mask_idx = 0
        for param in all_params:
            num_param_weights = param.data.numel()
            param_mask = mask[mask_idx:mask_idx + num_param_weights].reshape(param.data.shape)

            # Apply mask to the weights and disable gradients for pruned weights
            param_mask_tensor = torch.tensor(param_mask, device=self.device)
            param.data.mul_(param_mask_tensor)  # Apply mask to the weights
            param.requires_grad = False  # Disable gradient calculation for pruned weights

            mask_idx += num_param_weights

            # Save the mask for future use, ensuring it's moved to CPU before saving
            mask_file = os.path.join(self.experiment_dir, f"mask_{param.name}.pth")
            torch.save(param_mask_tensor.cpu(), mask_file)

        # Update current sparsity after pruning step
        self.current_sparsity = new_sparsity

        # Log accuracy and sparsity after this pruning step
        model_accuracy = self._get_model_accuracy()
        self.accuracy_vs_sparsity.append((model_accuracy, self.current_sparsity))
        self._log_accuracy_sparsity(model_accuracy, self.current_sparsity)

        print(f"Pruned to sparsity of {new_sparsity:.4f}, sparsity step: {self.current_sparsity:.4f}, End sparsity: {self.target_sparsity:.4f}")

        # Optionally, fine-tune after pruning
        self.fine_tune()

    def _save_initial_model(self):
        """Save the initial model weights (before pruning)."""
        torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'model_initial.pth'))

    def _save_masks(self):
        """Save the pruning masks for each layer."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Save mask only for weight parameters
                mask = param.data.eq(0).cpu().float()  # Create mask (0 for pruned weights)
                mask_file = os.path.join(self.experiment_dir, f"mask_{name}.pth")
                torch.save(mask, mask_file)

    def _save_final_model(self):
        """Save the final model weights (after pruning and retraining)."""
        torch.save(self.model.state_dict(), os.path.join(self.experiment_dir, 'final_model.pth'))

    def _save_pruning_log(self):
        """Save logs for pruning steps."""
        with open(self.experiment_dir + '/pruning_log.txt', 'w') as f:
            f.write(f"Pruning rate: {self.pruning_rate}\n")
            f.write(f"Target sparsity: {self.target_sparsity}\n")
            f.write(f"Pretrain epochs: {self.pretrain_epochs}\n")
            f.write(f"Fine-tune epochs: {self.finetune_epochs}\n")

    def _plot_accuracy_vs_sparsity(self):
        """Plot accuracy vs sparsity during training and save it."""
        accuracies, sparsities = zip(*self.accuracy_vs_sparsity)

        plt.figure(figsize=(10, 6))
        plt.plot(sparsities, accuracies, label='Accuracy vs Sparsity', marker='o', color='b')
        plt.title('Accuracy vs Sparsity')
        plt.xlabel('Sparsity')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        plt.legend()
        plot_file = os.path.join(self.experiment_dir, 'accuracy_vs_sparsity.png')
        plt.savefig(plot_file)
        plt.close()

    def _get_model_accuracy(self):
        """Compute the accuracy of the model."""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return accuracy

    def _log_accuracy_sparsity(self, accuracy, sparsity):
        """Log accuracy and sparsity at each step."""
        with open(self.log_file, 'a') as log:
            log.write(f"Accuracy: {accuracy:.2f}%, Sparsity: {sparsity:.4f}\n")
