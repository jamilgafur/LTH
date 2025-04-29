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
from abc import ABC, abstractmethod

from pyPrune.utils import get_pruneable_modules, get_pruneable_named_parameters, clean_memory
from pyPrune.strategies.PruningStrategy import PruningStrategy
from pyPrune.pruneMethods.Trainer import BaseTrainer
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BasePruner(BaseTrainer, ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        strategy: PruningStrategy = None,
        steps: List[float] = [],
        save_dir: str = 'pruning_checkpoints',
        finetune_epochs: int = 0,
        pretrain_epochs: int = 0,
        learning_rate: float = 0.01,
        file_handler: str = "logger.log",
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
        early_stopping: int = 0,
        device: Optional[str] = None
    ):
        super().__init__(model, train_loader, test_loader, optimizer, criterion, scheduler, device, prunable_layers, learning_rate)
        self.strategy = strategy or MagnitudePruningStrategy(device=self.device)
        self.steps = [0.0] + steps 
        self.save_dir = save_dir
        self.finetune_epochs = finetune_epochs
        self.pretrain_epochs = pretrain_epochs
        self.learning_rate = learning_rate
        self.prunable_layers = prunable_layers
        self.scheduler = scheduler

        self.total_weight_count = sum(
            p.numel() for m in get_pruneable_modules(model, prunable_layers) for p in [m.weight]
        )
        self.current_sparsity = 0.0
        self.best_model_weights = None
        self.step_details: List[Dict] = []
        self.metrics: Dict[str, List] = {
            'sparsity': [],
            'loss': [],
            'accuracy': [],
            'gradients': [],
            'optimizer': [],
            'step': [],
        }

        self._setup_directory(file_handler)
        self.initial_state = self._save_initial_state()
        self.weight_history: List[Dict] = [self.initial_state]

        # If first step is 0, pretraining is required
        if self.steps and self.steps[0] == 0:
            if self.pretrain_epochs == 0:
                self.pretrain_epochs = 1
            self.steps = self.steps[1:]
            
        logger.info("Pruner initialized.")

    def pretrain(self):
        logger.info("Starting pretraining...")   
        acc, loss = self._train_with_early_stopping(self.pretrain_epochs, phase="pretrain")
        self.initial_state = self._save_initial_state()
        self.weight_history[0] = self.initial_state
        self.best_model_weights = (acc, self.model.state_dict())

    def finetune(self):
        logger.info(f"Finetuning at {self.current_sparsity * 100:.2f}% sparsity...")
        acc, loss = self._train_with_early_stopping(self.finetune_epochs, phase="finetune")
        return acc, loss

    @abstractmethod
    def run(self):
        pass
