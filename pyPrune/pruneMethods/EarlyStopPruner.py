import logging
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyPrune.pruneMethods.Trainer import BaseTrainer
from pyPrune.pruneMethods.Pruner import BasePruner
from pyPrune.PruningStrategy import PruningStrategy
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy
from pyPrune.strategies.OptimalBrainDamageStrategy import OptimalBrainDamageStrategy

# Configure module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EarlyStopPruner(BasePruner):
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
        # Fallback to magnitude if no strategy provided
        if not strategy:
            self.strategy = MagnitudePruningStrategy(device=self.device)
        logger.info(f"IterativePruner initialized with strategy: {self.strategy.__class__.__name__}")


    def run(self):
        