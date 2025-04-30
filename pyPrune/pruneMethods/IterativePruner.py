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

CHECKPOINT_FMT = "checkpoint_Finetuned_{:.2f}.pth"
PRUNED_FMT = "Pruned_{:.2f}"

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
        self.finish_training_epochs = finish_training_epochs

        if not strategy:
            self.strategy = MagnitudePruningStrategy(device=self.device)

        logger.info(f"IterativePruner initialized with strategy: {self.strategy.__class__.__name__}")

    def save_and_log(self, step: float, prefix: str, acc: float, loss: float, label: str = "original") -> None:
        checkpoint_name = f"{prefix}_{step:.2f}"
        self.save_checkpoint(checkpoint_name)
        logger.info(f"[{prefix}] Accuracy at sparsity {step:.4f}: {acc:.2f}, Loss: {loss:.2f}")
        self.update_metrics(loss, acc, label=label)

    def run(self) -> None:
        self._maybe_pretrain()
        clean_memory()

        for step in self.steps:
            self._process_step(step)

        self._final_evaluation()

    def _maybe_pretrain(self) -> None:
        if not glob.glob(os.path.join(self.save_dir, "*.pkl")) and self.pretrain_epochs > 0:
            logger.info("No existing pickle file found. Starting pre-training.")
            self.pretrain()
        else:
            logger.info("Pickle file exists, skipping pre-training.")

    def _process_step(self, step: float) -> None:
        logger.info(f"[Step {step:.2f}] Starting pruning iteration.")
        checkpoint_path = os.path.join(self.save_dir, CHECKPOINT_FMT.format(step))

        if os.path.exists(checkpoint_path):
            logger.info(f"[Step {step:.2f}] Checkpoint already exists. Skipping...")
            return

        # Evaluate and save original model
        acc, loss = self.evaluate()
        self.save_and_log(step, "Original", acc, loss)

        # Finetune before pruning
        self.current_sparsity = step
        self.finetune()
        acc_ft, loss_ft = self.evaluate()
        self.save_and_log(step, "Finetuned", acc_ft, loss_ft, label="finetune")

        # Prune model
        model_state_dict = self.prune_step()
        if model_state_dict is None:
            logger.warning(f"[Step {step:.2f}] prune_step returned None. Skipping pruning.")
            return

        # Optional extra training after pruning
        for _ in range(self.finish_training_epochs):
            self._epoch(train=True)

        # Final evaluation
        acc, loss = self.evaluate()
        self.step_details.append({'sparsity': step, 'loss': loss, 'accuracy': acc})
        self.save_and_log(step, "Trained", acc, loss)

        # Save pruned model
        self.model.load_state_dict(model_state_dict, strict=False)
        self.save_checkpoint(PRUNED_FMT.format(step))
        self.assert_sparsity(step)

        # Bookkeeping
        self.weight_history.append(model_state_dict)
        self.metrics["step"].append(step)
        self.reset_weights()
        self.update_pickle()
        clean_memory()
        logger.info(f"[Step {step:.2f}] Completed.")

    def _final_evaluation(self) -> None:
        acc, loss = self.evaluate()
        logger.info(f"Final evaluation at {self.current_sparsity * 100:.2f}% sparsity - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        self.save_and_log(self.current_sparsity, "Finetuned", acc, loss, label="finetune")
        self.best_model_weights = self.best_model_weights[1]
        self.save_metrics()
        logger.info("Pruning run complete.")
