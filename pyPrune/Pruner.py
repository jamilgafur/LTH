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

from pyPrune.utils import get_pruneable_modules, clean_memory
from pyPrune.strategies.base import PruningStrategy

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BasePruner(BaseTrainer):
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
        device: Optional[str] = None
    ):
        super().__init__(model, train_loader, test_loader, optimizer, criterion, scheduler, device, prunable_layers, learning_rate)
        self.strategy = strategy or MagnitudePruningStrategy(device=self.device)
        self.steps = [0.0] + steps if steps and steps[0] != 0 else steps
        self.save_dir = save_dir
        self.finetune_epochs = finetune_epochs
        self.pretrain_epochs = pretrain_epochs

        self.total_weight_count = sum(
            p.numel() for m in get_pruneable_modules(model, prunable_layers) for p in [m.weight]
        )
        self.current_sparsity = 0.0
        self.weight_history: List[Dict] = []
        self.step_details: List[Dict] = []
        self.metrics: Dict[str, List] = {k: [] for k in ['sparsity','loss','accuracy','gradients','step']}

        self._setup_directory(file_handler)
        self.initial_state = self._save_initial_state()
        self.weight_history.append(self.initial_state)
        logger.info("Pruner initialized.")

    def _setup_directory(self, file_handler: str):
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, file_handler)
        if os.path.exists(path):
            base, ext = os.path.splitext(file_handler)
            path = os.path.join(self.save_dir, f"{base}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        with open(os.path.join(self.save_dir, 'pruner.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def _save_initial_state(self) -> Dict[str, torch.Tensor]:
        state = {n: p.data.clone() for n, p in self.model.named_parameters()}
        logger.info(f"Saved {len(state)} initial params.")
        return state

    def pretrain(self):
        logger.info("Starting pretraining...")
        for epoch in range(self.pretrain_epochs):
            acc = self._epoch(train=True)
            logger.info(f"Pretrain epoch {epoch+1}/{self.pretrain_epochs}, acc={acc:.2f}%")
        self.initial_state = self._save_initial_state()

    def finetune(self):
        logger.info(f"Finetuning at {self.current_sparsity*100:.2f}% sparsity")
        for epoch in range(self.finetune_epochs):
            acc = self._epoch(train=True)
            logger.info(f" Finetune epoch {epoch+1}/{self.finetune_epochs}, acc={acc:.2f}%")

    def prune_step(self):
        logger.info(f"Pruning to {self.current_sparsity*100:.2f}% sparsity")
        self.strategy.apply(
            self.model,
            self.optimizer,
            self.current_sparsity,
            prunable_layers=self.prunable_layers,
            total_weight_count=self.total_weight_count
        )

    def evaluate(self) -> float:
        loss = self._epoch(train=False)
        logger.info(f"Evaluated at {self.current_sparsity*100:.2f}% sparsity: loss={loss:.4f}")
        self.metrics['sparsity'].append(self.current_sparsity)
        self.metrics['loss'].append(loss)
        return loss

    def reset_weights(self):
        for name, param in self.model.named_parameters():
            param.data = self.initial_state[name].clone()
        for m in get_pruneable_modules(self.model, self.prunable_layers):
            if hasattr(m, 'mask'):
                m.weight.data.mul_(m.mask.float())
        logger.info("Weights reset with mask applied.")
        self.optimizer = type(self.optimizer)(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, suffix: str):
        path = os.path.join(self.save_dir, f"checkpoint_{suffix}.pth")
        torch.save({'model': self.model.state_dict()}, path)
        logger.info(f"Checkpoint saved: {path}")

    def run(self):
        if self.pretrain_epochs > 0:
            self.pretrain()
        for step in self.steps:
            self.current_sparsity = step
            self.finetune()
            self.prune_step()
            self.save_checkpoint(f"sparsity_{step:.2f}")
            loss = self.evaluate()
            self.step_details.append({'sparsity': step, 'loss': loss})
            self.reset_weights()
            with open(os.path.join(self.save_dir, 'pruner.pkl'), 'wb') as f:
                pickle.dump(self, f)

        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump({'metrics': self.metrics, 'steps': self.step_details}, f, indent=4)
        logger.info("Pruning run complete.")
