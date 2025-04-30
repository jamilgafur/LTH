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

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseTrainer(ABC):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[str] = None,
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
        learning_rate: float = 0.01, 
        early_stopping: int = 0,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.prunable_layers = prunable_layers
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.best_model_weights = (-1, None)
        

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

    def _save_model_state(self) -> Dict[str, torch.Tensor]:
        names, parameters = get_pruneable_named_parameters(self.model, self.prunable_layers)
        state = {n: p.data.clone() for n, p in zip(names, parameters)}
        logger.info(f"Saved {len(state)} initial parameters.")
        return state

    def _epoch(self, train: bool) -> float:
        loader = self.train_loader if train else self.test_loader
        self.model.train() if train else self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(loader, desc="Train" if train else "Eval", unit="batch"):
                x, y = x.to(self.device), y.to(self.device)
                if train:
                    self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                if train:
                    loss.backward()
                    for m in get_pruneable_modules(self.model, self.prunable_layers):
                        if m.weight.grad is not None:
                            m.weight.grad.mul_(m.weight.data != 0)
                    self.optimizer.step()
                else:
                    clean_memory()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()
        if not train and self.scheduler:
            self.scheduler.step()
        avg_loss = total_loss / total
        acc = 100. * correct / total
        return acc, avg_loss

    def _train_with_early_stopping(
        self,
        epochs: int,
        phase: str = "pretrain",
        tolerance: float = 0.2,  
        eval: bool = False
    ) -> Tuple[float, float]:
        best_metric = float('-inf')
        best_weights = None
        patience_counter = 0

        for epoch in range(epochs):
            acc, loss = self._epoch(train=True)
            logger.info(f"{phase.capitalize()} epoch {epoch + 1}/{epochs} - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")

            # Only reset patience if accuracy improves by more than `tolerance`
            if acc - best_metric > tolerance:
                best_metric = acc
                best_weights = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if self.early_stopping > 0 and patience_counter >= self.early_stopping:
                logger.info(f"Early stopping triggered in {phase} phase at epoch {epoch + 1}")
                break
            
            acc, loss = self._epoch(train=False)
            print(f"Evaluation - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")

        if best_weights:
            self.model.load_state_dict(best_weights)

        return acc, loss

    def evaluate(self) -> Tuple[float, float]:
        acc, loss = self._epoch(train=False)
        logger.info(f"Evaluated at {self.current_sparsity * 100:.2f}% sparsity - Accuracy: {acc:.2f}%, Loss: {loss:.4f}")
        if acc > self.best_model_weights[0]:
            logger.info(f"New best model at sparsity {self.current_sparsity * 100:.2f}% with accuracy {acc:.2f}%")
            self.best_model_weights = (acc, self.model.state_dict())
        return acc, loss

    def save_checkpoint(self, suffix: str):
        path = os.path.join(self.save_dir, f"checkpoint_{suffix}.pth")
        torch.save({'model': self.model.state_dict()}, path)
        logger.info(f"Checkpoint saved: {path}")

    def convert_tensor(self, t):
        if isinstance(t, torch.Tensor):
            return t.item() if t.numel() == 1 else t.tolist()
        elif isinstance(t, torch.nn.Parameter):
            return self.convert_tensor(t.data)
        return t

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
            logger.info(f"Best model updated at {self.current_sparsity * 100:.2f}% sparsity with accuracy {accuracy:.2f}%.")
            self.best_model_weights = (accuracy, self.model.state_dict())

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

    def run(self):
        logger.info("Starting training...")
        acc, loss = self._train_with_early_stopping(self.pretrain_epochs, phase="train")
        self.initial_state = self._save_model_state()
        self.weight_history[0] = self.initial_state
        self.best_model_weights = (acc, self.model.state_dict())
        logger.info("Training completed.")