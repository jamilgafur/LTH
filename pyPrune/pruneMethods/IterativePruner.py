import logging
import os
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


from pyPrune.pruneMethods.Trainer import BaseTrainer
from pyPrune.pruneMethods.Pruner import BasePruner
from pyPrune.strategies.PruningStrategy import PruningStrategy
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy
from pyPrune.strategies.OptimalBrainDamageStrategy import OptimalBrainDamageStrategy
from pyPrune.utils import clean_memory
# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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
        # Initialize BasePruner, which itself extends BaseTrainer
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
        # Fallback to magnitude if no strategy provided
        if not strategy:
            self.strategy = MagnitudePruningStrategy(device=self.device)
        logger.info(f"IterativePruner initialized with strategy: {self.strategy.__class__.__name__}")

    def save_and_log(self, step: float, prefix: str, acc: float, loss: float, label: str = "original"):
        """
        Helper method to save checkpoints and log accuracy/loss.
        """
        checkpoint_name = f"{prefix}_{step:.2f}"
        self.save_checkpoint(checkpoint_name)
        logger.info(f"{prefix} - Accuracy at sparsity {step:.4f}: {acc:.2f}, Loss: {loss:.2f}")
        self.update_metrics(loss, acc, label=label)


    def run(self):
        if self.pretrain_epochs > 0:
            self.pretrain()

        clean_memory()
        
        for step in self.steps:
            # Step 1: Save the original state
            acc, loss = self.evaluate()
            self.save_and_log(step, "Original", acc, loss)

            # Step 2: Skip if checkpoint already exists
            checkpoint_path = os.path.join(self.save_dir, f"checkpoint_Finetuned_{step:.2f}.pth")
            if os.path.exists(checkpoint_path):
                logger.info(f"Checkpoint for sparsity {step:.2f}% already exists. Skipping...")
                continue
            
            # Step 3: Perform pruning and finetuning
            self.current_sparsity = step
            self.finetune()
            self.save_and_log(step, "Finetuned", *self.evaluate(), label="finetune")

            model_state_dict = self.prune_step()

            # Step 4: Finish training
            for _ in range(self.finish_training_epochs):
                self._epoch(train=True)

            # Step 5: Log trained accuracy
            acc, loss = self.evaluate()
            self.step_details.append({'sparsity': step, 'loss': loss, 'accuracy': acc})
            self.save_and_log(step, "Trained", acc, loss)

            # Step 6: Load the pruned model and apply pruning
            self.model.load_state_dict(model_state_dict, strict=False)
            self.save_checkpoint(f"Pruned_{step:.2f}")
            self.assert_sparsity(step)

            # Step 7: Update history and reset weights
            self.weight_history.append(model_state_dict)
            self.metrics["step"].append(step)
            self.reset_weights()
            self.update_pickle()

            # Step 8: Clean up and move to next step
            logger.info("-" * 40)
            clean_memory()

        # Final evaluation and logging
        acc, loss = self.evaluate()
        logger.info(f"Final evaluation at {self.current_sparsity * 100:.2f}% sparsity - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        self.save_and_log(self.current_sparsity, "Finetuned", acc, loss, label="finetune")
        self.save_metrics()
        logger.info("Pruning run complete.")
