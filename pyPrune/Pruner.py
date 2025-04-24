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
from pyPrune.PruningStrategy import PruningStrategy
from pyPrune.Trainer import BaseTrainer
from pyPrune.strategies.MagnitudePruningStrategy import MagnitudePruningStrategy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        names, parameters = get_pruneable_named_parameters(self.model, self.prunable_layers)
        state = {n: p.data.clone() for n, p in zip(names, parameters)}
        logger.info(f"Saved {len(state)} initial parameters.")
        return state

    def pretrain(self):
        logger.info("Starting pretraining...")
        for epoch in range(self.pretrain_epochs):
            acc, loss = self._epoch(train=True)
            logger.info(f"Pretrain epoch {epoch+1}/{self.pretrain_epochs} - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        self.initial_state = self._save_initial_state()
        self.weight_history[0] = self.initial_state
        self.best_model_weights = (acc, self.model.state_dict())

    def finetune(self):
        logger.info(f"Finetuning at {self.current_sparsity * 100:.2f}% sparsity...")
        for epoch in range(self.finetune_epochs):
            acc, loss = self._epoch(train=True)
            logger.info(f"Finetune epoch {epoch+1}/{self.finetune_epochs} - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")

    def prune_step(self):
        logger.info(f"Pruning to {self.current_sparsity * 100:.2f}% sparsity")
        model_state_dict = self.strategy.apply(
            self.model,
            self.optimizer,
            self.current_sparsity,
            prunable_layers=self.prunable_layers,
            total_weight_count=self.total_weight_count
        )
        # load the model_state_dict
        self.model.load_state_dict(model_state_dict, strict=False)

    def evaluate(self) -> Tuple[float, float]:
        acc, loss = self._epoch(train=False)
        self.update_metrics(loss, acc)
        logger.info(f"Evaluated at {self.current_sparsity * 100:.2f}% sparsity - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        if acc > self.best_model_weights[0]:
            logger.info(f"New best model at sparsity {self.current_sparsity * 100:.2f}% with accuracy {acc:.2f}%")
            self.best_model_weights = (acc, self.model.state_dict())
        return acc, loss

    def update_metrics(self, loss: float, accuracy: float, gradients: Optional[torch.Tensor] = None):
        self.metrics['step'].append(self.current_sparsity)
        self.metrics['loss'].append(loss)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['gradients'].append(self.convert_tensor(gradients) if gradients is not None else None)
        if not self.metrics['accuracy'] or accuracy > max(self.metrics['accuracy']):
            logger.info(f"Best model updated at {self.current_sparsity * 100:.2f}% sparsity with accuracy {accuracy:.2f}%.")
            self.best_model_weights = self.model.state_dict()
        print(self.metrics)


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

    def save_checkpoint(self, suffix: str):
        path = os.path.join(self.save_dir, f"checkpoint_{suffix}.pth")
        torch.save({'model': self.model.state_dict()}, path)
        logger.info(f"Checkpoint saved: {path}")

    def update_pickle(self):
        with open(os.path.join(self.save_dir, 'pruner.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info("Pruner state updated and saved.")

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

    def convert_tensor(self, t):
        if isinstance(t, torch.Tensor):
            return t.item() if t.numel() == 1 else t.tolist()
        elif isinstance(t, torch.nn.Parameter):
            return self.convert_tensor(t.data)
        return t

    def assert_sparsity(self, expected_sparsity: float):
        total = 0
        zero = 0
        for module in get_pruneable_modules(self.model, self.prunable_layers):
            total += module.weight.data.numel()
            zero += (module.weight.data == 0).sum().item()
        actual_sparsity = zero / total
        logger.info(f"Sparsity check: actual = {actual_sparsity * 100:.2f}%, expected = {expected_sparsity * 100:.2f}%")
        assert abs(actual_sparsity - expected_sparsity) < 0.1, f"Sparsity mismatch actural {actual_sparsity * 100:.2f}% vs expected {expected_sparsity * 100:.2f}%"

    def run(self):
        if self.pretrain_epochs > 0:
            self.pretrain()
        
        for step in self.steps:
            self.current_sparsity = step
            self.finetune()
            self.prune_step()
            self.assert_sparsity(step)
            self.save_checkpoint(f"sparsity_{step:.2f}")
            acc, loss = self.evaluate()
            self.weight_history.append(self._save_initial_state())
            self.step_details.append({'sparsity': step, 'loss': loss, 'accuracy': acc})
            self.reset_weights()
            self.update_pickle()
            logger.info("-" * 40)
        acc, loss = self.evaluate()
        self.save_metrics()
        logger.info("Pruning run complete.")
