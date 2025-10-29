import os
import time
import json
import pickle
import datetime
import logging
import numpy as np
import psutil
from tqdm import tqdm
from copy import deepcopy
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
        save_dir: str = ".",
        finish_training_epochs = 0
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
        self.save_dir = save_dir
        self.metrics = {}
        self.initial_state = self._save_model_state()
        self.weight_history =  [self.initial_state]
        self.finish_training_epochs = finish_training_epochs
        
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
        state = {n: p.detach().clone() for n, p in zip(names, parameters)}
        logger.info(f"Saved {len(state)} initial parameters.")
        return state

    def _epoch(self, train: bool) -> Tuple[float, float]:
        clean_memory()
        loader = self.train_loader if train else self.test_loader
        dataset_size = len(loader.dataset)
        self.model.train() if train else self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        phase = "train" if train else "eval"

        start_time = time.time()

        if self.device and "cuda" in self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
        process = psutil.Process(os.getpid())
        cpu_mem_before = process.memory_info().rss

        context = torch.enable_grad() if train else torch.no_grad()
        if not train and self.scheduler:
            self.scheduler.step()
        with context:
            for x, y in tqdm(loader, desc=phase.capitalize(), unit="batch"):
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

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

                preds = out.detach().argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                total_loss += loss.item()

                # Clean up batch tensors
                del x, y, out, loss, preds
                torch.cuda.empty_cache()



        time_elapsed = time.time() - start_time

        if not train:
            cpu_mem_after = process.memory_info().rss
            cpu_mem_diff = (cpu_mem_after - cpu_mem_before) / 1e6
            mem_per_sample_MB = cpu_mem_diff / dataset_size if dataset_size > 0 else 0.0

            gpu_peak_mem = (
                torch.cuda.max_memory_allocated(self.device) / 1e6
                if self.device and "cuda" in self.device
                else None
            )

            if "memory" not in self.metrics:
                self.metrics["memory"] = []

            self.metrics["memory"].append({
                "cpu_mem": mem_per_sample_MB,
                "gpu_mem": gpu_peak_mem,
                "passthrough": time_elapsed
            })

            logger.info(
                f"[{phase.capitalize()} Epoch] CPU ΔMem/sample: {mem_per_sample_MB:.4f} MB | "
                f"GPU Peak: {gpu_peak_mem:.6f} MB | Time: {time_elapsed:.6f}s"
            )

        avg_loss = total_loss / total
        acc = 100. * correct / total

        clean_memory()
        torch.cuda.empty_cache()
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
        acc, loss = 0.0, 0.0
        for epoch in range(epochs):
            acc, loss = self._epoch(train=True)
            logger.info(f"{phase.capitalize()} epoch {epoch + 1}/{epochs} - Accuracy: {acc:.6f}%, Loss: {loss:.4f}")

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
            print(f"Evaluation - Accuracy: {acc:.6f}%, Loss: {loss:.4f}")

        if best_weights:
            self.model.load_state_dict(best_weights)

        return acc, loss

    def train(
        self,
        epochs: int,
        phase: str = "pretrain",
        eval: bool = False
    ) -> Tuple[float, float]:
        acc, loss = 0.0, 0.0
        for epoch in range(epochs):
            acc, loss = self._epoch(train=True)
            logger.info(f"{phase.capitalize()} epoch {epoch + 1}/{epochs} - Accuracy: {acc:.6f}%, Loss: {loss:.4f}")
            
        return acc, loss

    def evaluate(self) -> Tuple[float, float]:
        clean_memory()
        acc, loss = self._epoch(train=False)
        if acc > self.best_model_weights[0]:
            self.best_model_weights = (acc, deepcopy(self.model.state_dict()))
        clean_memory()
        logger.info(f"Evaluation - Accuracy: {acc:.6f}%, Loss: {loss:.4f}")
        return acc, loss

    def save_checkpoint(self, suffix: str):
        path = os.path.join(self.save_dir, f"checkpoint_{suffix}.pth")
        torch.save({'model': deepcopy(self.model.state_dict())}, path)
        logger.info(f"Checkpoint saved: {path}")

    def convert_tensor(self, t):
        if isinstance(t, torch.Tensor):
            return t.item() if t.numel() == 1 else t.tolist()
        elif isinstance(t, torch.nn.Parameter):
            return self.convert_tensor(t.data)
        return t

    def update_pickle(self):
        with open(os.path.join(self.save_dir, 'pruner.pkl'), 'wb') as f:
            pickle.dump(self, f)
        logger.info("Pruner state updated and saved.")

    @abstractmethod
    def run(self):
        pass