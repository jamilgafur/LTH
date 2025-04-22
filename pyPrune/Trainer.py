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
        learning_rate: float = 0.01
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
        return acc if train else avg_loss
