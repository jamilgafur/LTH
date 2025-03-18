import os
import json
import pickle
import datetime
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Callable, List, Tuple, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyPrune.utils import (
    get_pruneable_modules,
    clean_memory,
)

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IterativeMagnitudePruning:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        steps: List[float],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        pruning_criterion: Optional[Callable[[float, torch.Tensor], torch.Tensor]] = None,
        device: Optional[str] = None,
        save_dir: str = 'pruning_checkpoints',
        finetune_epochs: int = 0,
        pretrain_epochs: int = 0,
        learning_rate: float = 0.01,
        file_handler: str = "logger.log",
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear)
    ) -> None:
        """
        Initializes the IterativeMagnitudePruning class to perform iterative magnitude pruning on a neural network model.
        """
        self.model = model.to(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.steps = [0]+steps
        self.optimizer = optimizer
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.pruning_criterion = pruning_criterion or self.magnitude_prune
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.finetune_epochs = finetune_epochs
        self.pretrain_epochs = pretrain_epochs
        self.prunable_layers = prunable_layers

        # File for pickle saving
        self.pickle_name = os.path.join(self.save_dir, "pruner.pkl")
        self.current_sparsity = 0.0
        self.current_finetune_epoch = 0
        self.best_model_weights = None

        # To store detailed metrics for each pruning step (for scientific logging)
        self.step_details: List[Dict] = []

        self.setup_save_dir()
        self.initial_parameters = self.save_initial_parameters()
        self.weight_history = [self.initial_parameters]

        # Overall metrics tracked over all epochs (train and eval)
        self.metrics: Dict[str, List] = {
            'sparsity': [],
            'loss': [],
            'accuracy': [],
            'gradients': [],
            'optimizer': [],
            'step': [],
        }

        # If the first step is 0, assume pretraining is needed even if pretrain_epochs==0.
        if self.steps and self.steps[0] == 0:
            if self.pretrain_epochs == 0:
                self.pretrain_epochs = 1
            self.steps = self.steps[1:]

        self._setup_logger(file_handler)
        self.complete = False

        logger.info("IterativeMagnitudePruning initialized.")
        logger.info(f"Device: {self.device}, Target final sparsity: {self.steps[-1] if self.steps else 'N/A'}, Steps: {self.steps}")

    def _setup_logger(self, file_handler: str) -> None:
        """Configure a file handler for logging."""
        log_path = os.path.join(self.save_dir, file_handler)
        # If the log file exists, append a timestamp
        if os.path.exists(log_path):
            base, ext = os.path.splitext(file_handler)
            file_handler = f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            log_path = os.path.join(self.save_dir, file_handler)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        self.logger = logger

    def setup_save_dir(self) -> None:
        """Ensure the save directory exists and initialize the pruner pickle file."""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"Created directory: {self.save_dir}")
        else:
            logger.info(f"Save directory {self.save_dir} already exists.")

        print(f"saving pickle to {self.pickle_name}")
        with open(self.pickle_name, 'wb') as f:
            pickle.dump(self, f)
        logger.info("Initial pruner state saved as pickle.")

    def save_initial_parameters(self) -> Dict[str, torch.Tensor]:
        """Save the initial weights of the model (used for rewinding)."""
        init_params = {name: param.data.clone() for name, param in self.model.named_parameters()}
        logger.info(f"Saved initial parameters for {len(init_params)} parameters.")
        return init_params

    def unroll(self, percentage: float) -> Tuple[int, torch.Tensor]:
        """
        Flatten and concatenate weights from prunable layers.
        """
        logger.debug(f"Unrolling model weights for {percentage * 100:.2f}% sparsity.")
        weights_list = []
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            weights_list.append(module.weight.data.flatten())
        all_weights = torch.cat(weights_list)
        num_prune = max(1, int(all_weights.numel() * percentage))
        return num_prune, all_weights

    def update_optimizer(self) -> None:
        """Reinitialize the optimizer to reset any internal states."""
        self.optimizer = self.optimizer.__class__(self.model.parameters(), lr=self.learning_rate)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Optimizer updated with {len(list(self.model.parameters()))} parameters (Total: {total_params}).")
        logger.debug(f"Parameter shapes: {[p.shape for p in self.model.parameters()]}")

    def save_checkpoint(self, step: float, file_path: str) -> None:
        """
        Save a checkpoint of the current model and optimizer state.
        """
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
        """
        Perform magnitude-based pruning by zeroing out weights below a computed threshold.
        Updates the pruning mask so that pruned weights remain zero.
        """
        logger.info(f"Pruning model to {percentage * 100:.2f}% sparsity.")
        num_prune, all_weights = self.unroll(percentage)
        threshold_value = np.partition(np.abs(all_weights.cpu().numpy()), num_prune - 1)[num_prune - 1]
        if num_prune == 0:
            threshold_value = float('inf')

        for module in get_pruneable_modules(self.model, self.prunable_layers):
            # Compute current mask: True for weights to keep
            current_mask = torch.abs(module.weight.data) >= threshold_value
            # If a mask already exists, accumulate (keep previously pruned weights zeroed)
            if hasattr(module, 'mask'):
                module.mask = module.mask & current_mask
            else:
                module.mask = current_mask
            # Apply the mask so that pruned weights remain zero
            module.weight.data.mul_(module.mask.float())
            module.weight.grad = None
            # Optionally, you might not want to change requires_grad globally
            # module.weight.requires_grad = not module.weight.data.eq(0).all()

        logger.debug(f"Applied pruning with threshold {threshold_value:.6f}.")
        self.update_optimizer()

    def reset_weights(self) -> None:
        """
        Reset model weights to the saved initial (rewind) parameters,
        then reapply the cumulative pruning mask so that pruned weights remain zero.
        """
        for name, param in self.model.named_parameters():
            param.data = self.initial_parameters[name].clone()
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            if hasattr(module, 'mask'):
                module.weight.data.mul_(module.mask.float())
        logger.info("Model weights reset to initial parameters with pruning mask reapplied.")

    def update_metrics(self, loss: float, accuracy: float, gradients: Optional[torch.Tensor]) -> None:
        """Record current metrics into the internal metrics dictionary."""
        self.metrics['sparsity'].append(self.current_sparsity)
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['gradients'].append(self.convert_tensor(gradients) if gradients is not None else None)

        # Update best model if current accuracy is higher
        if self.current_sparsity > 0 and accuracy > max(self.metrics['accuracy']):
            logger.info(f"Best model updated at {self.current_sparsity * 100:.2f}% sparsity with accuracy {accuracy:.2f}%.")
            self.best_model_weights = self.model.state_dict()

    def epoch(self, mode: str = "train") -> Optional[Dict[str, float]]:
        """
        Run one epoch in training or evaluation mode.
        """
        if mode == "train":
            self.model.train()
            epochs_without_improvement = 0
            best_val_loss = float("inf")
            for data, target in tqdm(self.train_loader, desc="Training", unit="batch"):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()

                # Mask gradients for pruned weights
                for module in get_pruneable_modules(self.model, self.prunable_layers):
                    mask = module.weight.data != 0
                    if module.weight.grad is not None:
                        module.weight.grad.data.mul_(mask.float())
                self.optimizer.step()

            logger.info("Training epoch complete.")
            clean_memory()
            return None

        elif mode == "eval":
            self.model.eval()
            total_loss, correct = 0.0, 0
            with torch.no_grad():
                for data, target in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    total_loss += self.criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            total_loss /= len(self.test_loader.dataset)
            accuracy = 100. * correct / len(self.test_loader.dataset)
            self.update_metrics(total_loss, accuracy, None)
            logger.info(f"Evaluation complete, Average Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")
            clean_memory()
            return {"eval_loss": total_loss, "eval_accuracy": accuracy}
        else:
            logger.error("Epoch mode must be either 'train' or 'eval'.")
            return None

    def update_pickle(self) -> None:
        """Update the pickle file with the current pruner state."""
        with open(self.pickle_name, 'wb') as f:
            pickle.dump(self, f)
        logger.info("Pruner state updated and saved as pickle.")

    def run(self) -> None:
        """
        Run the complete iterative pruning process:
          1. Pretrain (if specified)
          2. Iteratively prune and fine-tune
          3. Save checkpoints and metrics
        """
        try:
            # Pretraining phase
            if self.pretrain_epochs > 0:
                logger.info("Starting pretraining...")
                for epoch_num in range(self.pretrain_epochs):
                    logger.info(f"Pretraining epoch {epoch_num + 1}/{self.pretrain_epochs}")
                    self.epoch("train")
                # Update initial parameters after pretraining (rewinding effect)
                self.initial_parameters = self.save_initial_parameters()
                self.weight_history[0] = self.initial_parameters 
                self.best_model_weights = self.model.state_dict()

            logger.info(f"Starting iterative pruning with steps: {self.steps}")
            for step in self.steps:
                self.current_finetune_epoch = step
                logger.info(f"--- Pruning step: Target sparsity = {step * 100:.2f}% ---")

                # Fine-tuning phase (if any)
                if self.finetune_epochs > 0:
                    for ft_epoch in range(self.finetune_epochs):
                        logger.info(f"Fine-tuning epoch {ft_epoch + 1}/{self.finetune_epochs} at {step * 100:.2f}% sparsity")
                        self.epoch("train")

                # Apply pruning
                logger.info(f"Pruning the model to {step * 100:.2f}% sparsity...")
                self.magnitude_prune(step)
                self.assert_sparsity(step)

                checkpoint_path = os.path.join(self.save_dir, f"pruned_model_step_{step:.4f}.pth")
                self.save_checkpoint(step, checkpoint_path)
                self.metrics['step'].append(step)

                # Evaluate after pruning and fine-tuning
                eval_metrics = self.epoch("eval")
                self.weight_history.append(self.model.state_dict())

                # Record step-level details for scientific logging
                step_detail = {
                    "pruning_step": step,
                    "eval_loss": eval_metrics["eval_loss"] if eval_metrics else None,
                    "eval_accuracy": eval_metrics["eval_accuracy"] if eval_metrics else None,
                    "sparsity": self.current_sparsity,
                    "checkpoint": checkpoint_path,
                }
                self.step_details.append(step_detail)
                logger.info(f"Step metrics recorded: {step_detail}")

                # Reset weights to initial (rewinding) state for next pruning step
                logger.info("Resetting weights to initial state for next pruning step.")
                self.reset_weights()
                self.update_pickle()
                print("\n" + "=" * 50 + "\n")

            logger.info("Pruning process complete. Saving overall metrics...")
            self.save_metrics()
            self.update_pickle()
            logger.info("Pruner state and metrics saved successfully.")
            self.complete = True

        except Exception as e:
            logger.error(f"Error during pruning: {str(e)}")
            logger.error("Deleting pickle file due to error...")
            self.delete_pickle()

    def save_metrics(self) -> None:
        """Save overall pruning metrics and step details to a JSON file."""
        all_metrics = {
            "overall_metrics": {
                key: [self.convert_tensor(val) if isinstance(val, (torch.Tensor, torch.nn.Parameter)) else val
                      for val in values]
                for key, values in self.metrics.items()
            },
            "step_details": self.step_details
        }
        metrics_path = os.path.join(self.save_dir, 'pruning_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

    def convert_tensor(self, t):
        """Convert a tensor or parameter to a serializable Python type."""
        if isinstance(t, torch.Tensor):
            return t.item() if t.numel() == 1 else t.tolist()
        elif isinstance(t, torch.nn.Parameter):
            return self.convert_tensor(t.data)
        return t

    def assert_sparsity(self, target_sparsity: float) -> None:
        """
        Verify that the model's sparsity is close to the target value.
        """
        total_params, pruned_params = 0, 0
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            total_params += module.weight.numel()
            pruned_params += torch.sum(module.weight == 0).item()
        current_sparsity = pruned_params / total_params
        assert np.isclose(current_sparsity, target_sparsity, atol=.01), \
            f"Model sparsity mismatch: {current_sparsity:.4f} vs {target_sparsity:.4f}"
        self.current_sparsity = current_sparsity
        logger.info(f"Sparsity assertion passed: {current_sparsity * 100:.2f}%")

    def delete_pickle(self) -> None:
        """Delete the pickle file if it exists."""
        if os.path.exists(self.pickle_name):
            # os.remove(self.pickle_name)
            logger.info(f"(deprecated) Deleted pickle file: {self.pickle_name}")
        else:
            logger.info(f"No pickle file found at: {self.pickle_name}")
