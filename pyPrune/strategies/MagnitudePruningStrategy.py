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
from pyPrune.strategies.PruningStrategy import PruningStrategy
# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class MagnitudePruningStrategy(PruningStrategy):
    """
    Global magnitude-based pruning.
    """
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def _unroll(self, model: nn.Module, prunable_layers: Tuple):
        weights = []
        for module in get_pruneable_modules(model, prunable_layers):
            if hasattr(module, 'mask'):
                w = module.weight.data[module.mask.bool()].flatten()
            else:
                w = module.weight.data.flatten()
            if w.numel():
                weights.append(w)
        if not weights:
            raise ValueError("No weights to prune.")
        return torch.cat(weights)

    def apply(self,
              model: nn.Module,
              optimizer: torch.optim.Optimizer,
              target_sparsity: float,
              prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
              total_weight_count: Optional[int] = None) -> None:
        logger.info(f"[Magnitude] Target sparsity: {target_sparsity*100:.2f}%")
        all_weights = self._unroll(model, prunable_layers)
        total = total_weight_count or all_weights.numel()
        
        num_prune = int(total * target_sparsity) - (total - all_weights.numel())
        if num_prune <= 0:
            logger.info("Already at or above target sparsity.")
            return model.state_dict()
        abs_w = np.abs(all_weights.cpu().numpy())
        thresh = np.partition(abs_w, num_prune - 1)[num_prune - 1]
        for module in get_pruneable_modules(model, prunable_layers):
            w = module.weight.data
            mask = torch.abs(w) >= thresh
            module.mask = mask if not hasattr(module, 'mask') else module.mask & mask
            module.weight.data.mul_(module.mask.float())
            if module.weight.grad is not None:
                module.weight.grad.zero_()
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is module.weight and 'momentum_buffer' in optimizer.state[p]:
                        optimizer.state[p]['momentum_buffer'].mul_(module.mask.float())
        logger.debug(f"Applied magnitude pruning at threshold={thresh:.6f}.")
        
        # return the updated weights
        return model.state_dict()
