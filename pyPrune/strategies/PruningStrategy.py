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

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PruningStrategy(ABC):
    """
    Defines the interface for pruning methods. New strategies extend this without changing IterativePruner (OCP).
    returns the models state_dict
    """
    @abstractmethod
    def apply(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_sparsity: float,
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
        total_weight_count: Optional[int] = None
    ) -> nn.Module:
        pass
