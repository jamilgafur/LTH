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
from pyPrune.PruningStrategy import PruningStrategy

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimalBrainDamageStrategy(PruningStrategy):
    """
    Optimal Brain Damage: prunes by approximating saliency = 0.5 * w^2 * H_ii.
    Prunes based on the lowest percentage of Hessian values.
    """
    def __init__(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        device: str = 'cpu'
    ):
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device

    def _compute_hessian_diag(
        self,
        model: nn.Module,
        prunable_layers: Tuple
    ) -> Dict[nn.Module, torch.Tensor]:
        """Estimate diagonal Hessian for each prunable module weight."""
        modules = list(get_pruneable_modules(model, prunable_layers))
        hessian = {m: torch.zeros_like(m.weight.data) for m in modules}
        data, target = next(iter(self.train_loader))
        data, target = data.to(self.device), target.to(self.device)
        model.zero_grad()
        output = model(data)
        loss = self.criterion(output, target)
        grads = torch.autograd.grad(
            loss,
            [m.weight for m in modules],
            create_graph=True
        )
        for m, g in zip(modules, grads):
            hessian[m] = g.pow(2)  # Hessian diagonal approximation: H_ii = g^2
        return hessian

    def apply(self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            target_sparsity: float,
            prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
            total_weight_count: Optional[int] = None) -> None:
        logger.info(f"[OBD] Target sparsity: {target_sparsity * 100:.2f}%")
        modules = list(get_pruneable_modules(model, prunable_layers))
        total = total_weight_count or sum(m.weight.numel() for m in modules)

        # Estimate Hessian diagonals
        hess = self._compute_hessian_diag(model, prunable_layers)

        # Collect Hessian values for active weights only
        hessian_values = []
        active_weights = 0
        for m in modules:
            w = m.weight.data
            hess_values = hess[m].to(self.device)

            if hasattr(m, 'mask'):
                active_mask = m.mask
            else:
                active_mask = torch.ones_like(w, dtype=torch.bool)

            hess_values = hess_values[active_mask]
            hessian_values.append(hess_values)
            active_weights += active_mask.sum().item()

        all_hessian_values = torch.cat(hessian_values)

        # Compute how many weights to prune to hit target sparsity
        target_nonzero = int(total * (1 - target_sparsity))
        num_to_prune = total - target_nonzero
        logger.info(f"Target nonzero weights: {target_nonzero}, \n"
                    f"Current nonzero weights: {active_weights}, \n"
                    f"Pruning {num_to_prune} weights. \n"
                    f"Current sparsity: {1 - (active_weights / total) * 100:.2f}\n%"
                    f"Target sparsity: {target_sparsity * 100:.2f}%\n")

        if num_to_prune <= 0:
            logger.info("Already at or below target sparsity.")
            return

        # Sort Hessian values to prune the lowest
        sorted_hessian_values, sorted_indices = torch.sort(all_hessian_values)
        thresh_idx = sorted_indices[num_to_prune - 1].item()  # Get the index of the threshold value
        thresh = sorted_hessian_values[thresh_idx]

        # Apply mask based on the threshold
        for m in modules:
            w = m.weight.data
            hess_values = hess[m].to(self.device)
            current_mask = getattr(m, 'mask', torch.ones_like(w, dtype=torch.bool))

            # New mask: keep weights with Hessian greater than threshold
            new_mask = (hess_values >= thresh) & current_mask
            m.mask = new_mask
            m.weight.data.mul_(new_mask.float())

            if m.weight.grad is not None:
                m.weight.grad.zero_()

            # Update optimizer state for the pruned weights
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is m.weight and 'momentum_buffer' in optimizer.state[p]:
                        optimizer.state[p]['momentum_buffer'].mul_(new_mask.float())

        # Check actual sparsity
        nonzero = sum((m.weight.data != 0).sum().item() for m in modules)
        actual_sparsity = 1 - (nonzero / total)
        logger.debug(f"Actual sparsity after pruning: {actual_sparsity * 100:.2f}%")

        return model.state_dict()
