# Standard library imports
import os
import json
import pickle
import datetime
import logging

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List

# Local application/library imports
from pyPrune.utils import get_pruneable_modules, clean_memory
from pyPrune.strategies.PruningStrategy import PruningStrategy

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OptimalBrainDamageStrategy(PruningStrategy):
    """
    Optimal Brain Damage: uses second-order diagonal Hessian approximation to prune
    weights with smallest saliency: saliency = 0.5 * h_ii * w_i^2.
    """

    def __init__(self, train_loader: DataLoader, criterion: nn.Module, device: str = 'cpu'):
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device

    def compute_diagonal_hessian(self, model: nn.Module, prunable_layers: Tuple) -> Dict[nn.Module, torch.Tensor]:
        """
        Approximates the diagonal of the Hessian matrix by accumulating squared gradients
        of the loss with respect to model parameters over the training data.
        """
        logger.info("Computing diagonal Hessian approximation...")
        model.train().to(self.device)
        model.zero_grad()

        hessian_diag: Dict[nn.Module, torch.Tensor] = {}
        for module in get_pruneable_modules(model, prunable_layers):
            hessian_diag[module] = torch.zeros_like(module.weight.data, device=self.device)

        for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc="Diagonal Hessian Pass")):
            x, y = x.to(self.device), y.to(self.device)
            output = model(x)
            loss = self.criterion(output, y)

            grads = torch.autograd.grad(loss, [module.weight for module in hessian_diag], create_graph=True)

            for module, g in zip(hessian_diag, grads):
                h_diag = torch.autograd.grad(g, module.weight, grad_outputs=torch.ones_like(g), retain_graph=True)[0]
                hessian_diag[module] += h_diag ** 2  # Accumulate squared curvature

            model.zero_grad()

        # Normalize
        for module in hessian_diag:
            hessian_diag[module] /= len(self.train_loader)

        logger.info("Hessian diagonal computed.")
        clean_memory()
        return hessian_diag

    def compute_saliency(self, model: nn.Module, prunable_layers: Tuple) -> Dict[nn.Module, torch.Tensor]:
        logger.info("Computing saliency via OBD (Hessian diagonal * weight^2)...")
        hessian_diag = self.compute_diagonal_hessian(model, prunable_layers)
        saliency = {}
        for module in hessian_diag:
            saliency[module] = 0.5 * hessian_diag[module] * module.weight.data.pow(2)
        return saliency

    def apply(self,
              model: nn.Module,
              optimizer: torch.optim.Optimizer,
              target_sparsity: float,
              prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
              total_weight_count: Optional[int] = None) -> None:

        logger.info(f"[OBD] Target sparsity: {target_sparsity * 100:.2f}%")

        saliency = self.compute_saliency(model, prunable_layers)

        modules: List[Tuple[nn.Module, int]] = []
        for module in get_pruneable_modules(model, prunable_layers):
            if hasattr(module, 'mask'):
                cnt = int(module.mask.sum().item())
            else:
                cnt = module.weight.data.numel()
            modules.append((module, cnt))

        flat_sal = []
        for module, cnt in modules:
            s = saliency[module]
            if hasattr(module, 'mask'):
                s = s[module.mask.bool()]
            flat_sal.append(s.flatten())
        flat_sal = torch.cat(flat_sal).cpu()

        total = total_weight_count or sum(cnt for _, cnt in modules)
        remaining = int(total * (1 - target_sparsity))
        num_prune = max(0, flat_sal.numel() - remaining)

        logger.debug(f"Total: {total}, Active: {flat_sal.numel()}, Pruning: {num_prune}")
        if num_prune <= 0:
            logger.info("Already at or below target sparsity. Skipping pruning.")
            return model.state_dict()

        prune_flat_idx = torch.argsort(flat_sal)[:num_prune]

        offset = 0
        pruned = 0
        for module, cnt in modules:
            local_idx = prune_flat_idx[(prune_flat_idx >= offset) & (prune_flat_idx < offset + cnt)] - offset

            if hasattr(module, 'mask'):
                mask = module.mask.clone().flatten()
            else:
                mask = torch.ones(cnt, dtype=torch.bool, device=self.device)

            mask[local_idx.to(self.device)] = False
            module.mask = mask.view_as(module.weight.data)
            module.weight.data.mul_(module.mask.float())

            pruned += len(local_idx)

            if module.weight.grad is not None:
                module.weight.grad.zero_()

            for group in optimizer.param_groups:
                for p in group['params']:
                    if id(p) == id(module.weight):
                        if 'momentum_buffer' in optimizer.state[p]:
                            optimizer.state[p]['momentum_buffer'].mul_(module.mask.float())

            offset += cnt

        actual_sparsity = pruned / total
        logger.info(
            f"[OBD] Pruned {pruned}/{total} weights "
            f"-> Sparsity {actual_sparsity * 100:.2f}% (Target: {target_sparsity * 100:.2f}%)"
        )

        return model.state_dict()
