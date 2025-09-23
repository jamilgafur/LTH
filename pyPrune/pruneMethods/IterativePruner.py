import logging
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import glob

from pyPrune.pruneMethods.Trainer import BaseTrainer
from pyPrune.pruneMethods.Pruner import BasePruner
from pyPrune.strategies.PruningStrategy import PruningStrategy
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy
from pyPrune.utils import clean_memory

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

CHECKPOINT_FMT = "checkpoint_Finetuned_{:.6f}.pth"
PRUNED_FMT = "Pruned_{:.6f}"

class IterativePruner(BasePruner):
    """
    High-level pruner that composes training logic from BaseTrainer (via BasePruner)
    with dynamic pruning strategies injected at runtime.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strategy: Optional[PruningStrategy] = None,
        steps: List[float] = [],
        device: Optional[str] = None,
        save_dir: str = 'pruning_checkpoints',
        finetune_epochs: int = 0,
        pretrain_epochs: int = 0,
        learning_rate: float = 0.01,
        file_handler: str = "logger.log",
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
        early_stopping: int = 0,
        finish_training_epochs: int = 0,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            strategy=strategy,
            steps=steps,
            save_dir=save_dir,
            finetune_epochs=finetune_epochs,
            pretrain_epochs=pretrain_epochs,
            learning_rate=learning_rate,
            file_handler=file_handler,
            prunable_layers=prunable_layers,
            device=device
        )
        self.initial_state = self._save_model_state()
        self.weight_history = [self.initial_state]
        self.finish_training_epochs = finish_training_epochs

        if not strategy:
            self.strategy = MagnitudePruningStrategy(device=self.device)

        logger.info(f"IterativePruner initialized with strategy: {self.strategy.__class__.__name__}")

    def save_and_log(self, step: float, prefix: str, acc: float, loss: float, label: str = "original") -> None:
        checkpoint_name = f"{prefix}_{step:.6f}"
        self.save_checkpoint(checkpoint_name)
        logger.info(f"[{prefix}] Accuracy at sparsity {step:.4f}: {acc:.6f}, Loss: {loss:.6f}")
        self.update_metrics(loss, acc, label=label)

    def run(self) -> None:
        self._maybe_pretrain()
        clean_memory()

        for step in self.steps:
            self._process_step(step)

        self._final_evaluation()

    def _maybe_pretrain(self) -> None:
        self.pretrain()
       
    def _process_step(self, step: float) -> None:
        logger.info(f"[Step {step:.6f}] Starting pruning iteration.")
        if self._checkpoint_exists(step):
            return

        self._evaluate_and_save_original_model(step)
        self._finetune_and_log(step)
        self._prune_model_and_train(step)

        clean_memory()
        logger.info(f"[Step {step:.6f}] Completed.")

    def _checkpoint_exists(self, step: float) -> bool:
        checkpoint_path = os.path.join(self.save_dir, CHECKPOINT_FMT.format(step))
        if os.path.exists(checkpoint_path):
            logger.info(f"[Step {step:.6f}] Checkpoint already exists. Skipping...")
            return True
        return False

    def _evaluate_and_save_original_model(self, step: float) -> None:
        acc, loss = self.evaluate()
        self._assign_memory_tag("Original_memory")
        self.save_and_log(step, "Original", acc, loss)

    def _finetune_and_log(self, step: float) -> None:
        self.current_sparsity = step
        self.finetune()
        acc_ft, loss_ft = self.evaluate()
        self._assign_memory_tag("Finetuned_memory")
        self.save_and_log(step, "Finetuned", acc_ft, loss_ft, label="finetune")

    def _prune_model_and_train(self, step: float) -> None:
        model_state_dict = self.prune_step()
        if model_state_dict is None:
            logger.warning(f"[Step {step:.6f}] prune_step returned None. Skipping pruning.")
            return

        for _ in range(self.finish_training_epochs):
            self._epoch(train=True)
            torch.cuda.empty_cache()  # Frees unused memory after each epoch

        acc, loss = self.evaluate()
        loss = loss.item() if isinstance(loss, torch.Tensor) else loss  # Detach if needed
        acc = acc.item() if isinstance(acc, torch.Tensor) else acc

        self._assign_memory_tag("Pruned_memory")
        self.step_details.append({'sparsity': step, 'loss': float(loss), 'accuracy': float(acc)})
        self.save_and_log(step, "Trained", acc, loss)

        # Restore the model to pre-pruning weights
        self.model.load_state_dict(model_state_dict, strict=False)
        self.save_checkpoint(PRUNED_FMT.format(step))
        self.assert_sparsity(step)

        # Store a *copy* of the pruned weights to avoid holding reference to computation graph
        pruned_weights_copy = {k: v.detach().clone().cpu() for k, v in model_state_dict.items()}
        self.weight_history.append(pruned_weights_copy)
        
        self.metrics["step"].append(step)
        self.reset_weights()
        self.update_pickle()

        # Cleanup to avoid holding unnecessary GPU memory
        del model_state_dict
        del pruned_weights_copy
        torch.cuda.empty_cache()

    def _final_evaluation(self) -> None:
        acc, loss = self.evaluate()
        self._assign_memory_tag("Final_memory")
        logger.info(f"Final evaluation at {self.current_sparsity * 100:.6f}% sparsity - Accuracy: {acc:.6f}%, Loss: {loss:.4f}")
        self.save_and_log(self.current_sparsity, "Finetuned", acc, loss, label="finetune")
        self.best_model_weights = self.best_model_weights[1]
        self.save_metrics()
        logger.info("Pruning run complete.")

    def _assign_memory_tag(self, tag: str):
        """
        Takes the last entry from self.metrics["memory"] and appends it to a named list.
        """
        if "memory" not in self.metrics or not self.metrics["memory"]:
            logger.warning(f"Cannot assign memory tag '{tag}': no memory data available.")
            return

        if tag not in self.metrics:
            self.metrics[tag] = []

        self.metrics[tag].append(self.metrics["memory"][-1])
