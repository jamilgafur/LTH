import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from typing import Optional, Callable
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class IterativeMagnitudePruning:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, final_sparsity: float, 
                 steps: int, optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                 pruning_criterion: Optional[Callable[[float, torch.Tensor], torch.Tensor]] = None,
                 device: Optional[str] = None, save_dir: str = 'pruning_checkpoints', finetune_epochs: int = 0, 
                 pretrain_epochs: int = 0, learning_rate: float = 0.01) -> None:
        self.save_dir = save_dir
        self.setup_save_dir()

        self.final_sparsity = final_sparsity
        self.steps = steps
        self.pruning_criterion = pruning_criterion or self.magnitude_prune
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.finetune_epochs = finetune_epochs
        self.pretrain_epochs = pretrain_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.current_sparsity = 0.0

        self.initial_weights = self.save_initial_weights()
        self.metrics = []

        logger.info("IterativeMagnitudePruning initialized.")
        logger.info(f"Device: {self.device}, Final sparsity: {self.final_sparsity}, Steps: {self.steps}")

    def setup_save_dir(self) -> None:
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Created directory: {self.save_dir}")
        else:
            logger.info(f"Save directory {self.save_dir} already exists.")

    def save_initial_weights(self) -> dict:
        initial_weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                initial_weights[name] = param.data.clone()
        logger.info(f"Initial weights saved for {len(initial_weights)} weight parameters.")
        return initial_weights


    def unroll(self, percentage: float = 0) -> None:
        logger.debug(f"Unrolling model at {percentage * 100:.2f}% sparsity")

        all_weights = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                all_weights.append(param.data.flatten())

        all_weights = torch.cat(all_weights)
        sorted_weights = torch.abs(all_weights).sort()[0]

        num_prune = max(1, int(all_weights.numel() * percentage))
        threshold_value = sorted_weights[num_prune - 1] if num_prune > 0 else float('inf')

        # Apply pruning: directly zero out weights below the threshold
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = torch.abs(param.data) >= threshold_value
                param.data.mul_(mask.float())  # Prune weights

                # Detach pruned weights from gradients
                param.grad = None  # Zero out any existing gradient for the pruned weights
                param.requires_grad = not param.data.eq(0).all()  # Set requires_grad to False if all weights are zero

        logger.debug(f"Pruning applied at {percentage * 100:.2f}% sparsity with threshold {threshold_value:.6f}")

    def update_optimizer(self) -> None:
        # Get the parameters that are still trainable after pruning
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]

        # Update the optimizer to reflect the current parameters
        self.optimizer = torch.optim.Adam(params_to_optimize, lr=self.learning_rate)

        # Log the number of parameters being passed to the optimizer
        total_params = sum(p.numel() for p in params_to_optimize)
        logger.info(f"Optimizer updated to reflect {len(params_to_optimize)} parameters, Total: {total_params} parameters.")

        # Log the parameters (optional, can be verbose for large models)
        logger.debug(f"Optimizer parameters: {[p.shape for p in params_to_optimize]}")

    def save_checkpoint(self, step: int, file_path: str) -> None:
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(checkpoint, file_path)
            logger.info(f"Checkpoint saved at {file_path}")

        except KeyError as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise

    def magnitude_prune(self, percentage: float) -> None:
        logger.info(f"Pruning model at {percentage * 100:.2f}% sparsity.")
        self.unroll(percentage)
        self.update_optimizer()

    def reset_weights(self) -> None:
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                param.data = self.initial_weights[name].clone()
        logger.info("Weights reset to initial values.")

    def train(self, type: str = "train") -> None:
        if type == "train":
            self.model.train()
            for data, target in tqdm(self.train_loader, desc="Training", unit="batch"):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Mask the gradients for zeroed-out weights
                for name, param in self.model.named_parameters():
                    if 'weight' in name and param.requires_grad:
                        mask = param.data != 0  # Mask for non-zero weights
                        param.grad *= mask.float()  # Zero out gradients for pruned weights

                self.optimizer.step()
            logger.info(f"Training step complete, Loss: {loss.item()}")

    def run(self) -> None:
        self.initial_weights = self.save_initial_weights()

        if self.pretrain_epochs > 0:
            logger.info("Starting pretraining...")
            self.train("train")
        
        steps = np.linspace(0, self.final_sparsity, self.steps)
        logger.info(f"Starting pruning with {self.steps} steps for sparsity levels: {steps}")

        for step in tqdm(steps, desc="Pruning Steps", unit="step"):
            logger.info(f"Starting pruning step: {step * 100:.2f}% sparsity")
            self.magnitude_prune(step)

            logger.info("Fine-tuning the model...")
            self.train("train")

            logger.info("Updating optimizer to reflect pruned weights...")
            self.save_checkpoint(step, os.path.join(self.save_dir, f"pruned_model_step_{int(step * 100)}.pth"))
            self.metrics.append({'sparsity': step, 'loss': self.metrics[-1]['loss'] if self.metrics else 0.0})
            
            self.train("eval")
            print("\n\n\n")
            self.assert_sparsity(step)

        logger.info("Pruning complete.")

    def assert_sparsity(self, sparsity: float) -> None:
        total_params_model = 0
        pruned_params_model = 0
        for name, param in self.model.named_parameters():
            if 'weight' in name:  # Ensure we are only considering weight parameters
                total_params_model += param.numel()
                pruned_params_model += torch.sum(param == 0).item()

        current_sparsity_model = pruned_params_model / total_params_model

        # Allow for a small tolerance in the sparsity difference
        assert np.isclose(current_sparsity_model, sparsity, atol=1e-2), \
            f"Model sparsity mismatch: {current_sparsity_model} vs {sparsity}"

        logger.info(f"Sparsity assertion passed: {current_sparsity_model * 100:.2f}% model.")
