import os
import json
import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
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
        self.best_model_weights = (-1, None)
        self.step_details: List[Dict] = []
        self.metrics: Dict[str, List] = {
            'step': [],
        }

        self._setup_directory(file_handler)
        self.initial_state = self._save_model_state()
        self.weight_history: List[Dict] = [self.initial_state]

        # If first step is 0, pretraining is required
        if self.steps and self.steps[0] == 0:
            if self.pretrain_epochs == 0:
                self.pretrain_epochs = 1
            self.steps = self.steps[1:]
            
        logger.info("Pruner initialized.")

    def save_metrics(self):
        all_metrics = {
            "overall_metrics": {
                key: [self.convert_tensor(val) for val in values]
                for key, values in self.metrics.items()
            },
            "step_details": self.step_details
        }
        metrics_path = os.path.join(self.save_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_path}")

    def update_metrics(self, loss: float, accuracy: float, gradients: Optional[torch.Tensor] = None, label: str = ""):
        if f"step_{label}" not in self.metrics:
            self.metrics[f"step_{label}"] = []
        if f"loss_{label}" not in self.metrics:
            self.metrics[f"loss_{label}"] = []
        if f"accuracy_{label}" not in self.metrics:
            self.metrics[f"accuracy_{label}"] = []
            
        self.metrics[f"step_{label}"].append(self.current_sparsity)
        self.metrics[f"loss_{label}"].append(loss)
        self.metrics[f"accuracy_{label}"].append(accuracy)
        if not self.metrics[f"accuracy_{label}"] or accuracy > max(self.metrics[f"accuracy_{label}"]):
            logger.info(f"Best model updated at {self.current_sparsity * 100:.6f}% sparsity with accuracy {accuracy:.6f}%.")
            self.best_model_weights = (accuracy, deepcopy(self.model.state_dict()))

    def pretrain(self):
        logger.info("Starting pretraining...")   
        acc, loss = self.train(self.pretrain_epochs, phase="pretrain")
        self.initial_state = self._save_model_state()
        self.weight_history[0] = self.initial_state
    
    def finetune(self):
        logger.info(f"Finetuning at {self.current_sparsity * 100:.6f}% sparsity...")
        acc, loss = self.train(self.finetune_epochs, phase="finetune")
        return acc, loss

    def assert_sparsity(self, expected_sparsity: float):
        total = 0
        zero = 0
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            total += module.weight.data.numel()
            zero += (module.weight.data == 0).sum().item()
        actual_sparsity = zero / total
        logger.info(f"Sparsity check: actual = {actual_sparsity * 100:.6f}%, expected = {expected_sparsity * 100:.6f}%")
        # update the current sparsity to be the actual sparsity
        self.current_sparsity = actual_sparsity
        
        # assert abs(actual_sparsity - expected_sparsity) < 0.1, f"Sparsity mismatch actural {actual_sparsity * 100:.6f}% vs expected {expected_sparsity * 100:.6f}%"

    def reset_weights(self):
        names, params = get_pruneable_named_parameters(self.model, self.prunable_layers)
        for name, param in zip(names, params):
            param.data = self.initial_state[name].clone()
        for m in get_pruneable_modules(self.model, self.prunable_layers):
            if hasattr(m, 'mask'):
                m.weight.data.mul_(m.mask.float())
        logger.info("Weights reset to initial state with pruning mask reapplied.")
        self.optimizer = type(self.optimizer)(self.model.parameters(), lr=self.learning_rate)
        if self.scheduler:
            self.scheduler.last_epoch = self.finetune_epochs

    def prune_step(self) -> nn.Module:
        logger.info(f"Pruning to {self.current_sparsity * 100:.6f}% sparsity")
        model_state_dict = self.strategy.apply(
            self.model,
            self.optimizer,
            self.current_sparsity,
            prunable_layers=self.prunable_layers,
            total_weight_count=self.total_weight_count
        )
        return model_state_dict
    
    @abstractmethod
    def run(self):
        pass
