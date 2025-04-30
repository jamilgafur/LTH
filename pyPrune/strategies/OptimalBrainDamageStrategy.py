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
    Target sparsity is defined as fraction of zeroed weights (existing & new) over all prunable weights.
    """

    def __init__(self, train_loader: DataLoader, criterion: nn.Module, device: str = 'cpu'):
        super().__init__()
        self.train_loader = train_loader
        self.criterion = criterion
        self.device = device

    def compute_diagonal_hessian(
        self, model: nn.Module, prunable_layers: Tuple
    ) -> Dict[nn.Module, torch.Tensor]:
        """
        Approximates the diagonal of the Hessian matrix by accumulating second derivatives
        of the loss wrt model parameters over training data.
        """
        logger.info("Computing diagonal Hessian approximation...")
        model.train().to(self.device)

        # Initialize storage for each module's diagonal
        hessian_diag: Dict[nn.Module, torch.Tensor] = {}
        for module in get_pruneable_modules(model, prunable_layers):
            hessian_diag[module] = torch.zeros_like(module.weight, device=self.device)

        # Loop over data to accumulate second derivatives
        for x, y in tqdm(self.train_loader, desc="Diagonal Hessian Pass"):
            x, y = x.to(self.device), y.to(self.device)
            model.zero_grad()
            output = model(x)
            loss = self.criterion(output, y)

            # First derivative
            grads = torch.autograd.grad(loss, [m.weight for m in hessian_diag], create_graph=True)

            # Second derivative (diagonal)
            for module, grad in zip(hessian_diag, grads):
                second_deriv = torch.autograd.grad(
                    grad,
                    module.weight,
                    grad_outputs=torch.ones_like(grad),
                    retain_graph=True,
                    create_graph=False
                )[0]
                hessian_diag[module] += second_deriv

        # Average over batches
        num_batches = float(len(self.train_loader))
        for module in hessian_diag:
            hessian_diag[module] /= num_batches

        logger.info("Hessian diagonal computed.")
        clean_memory()
        return hessian_diag

    def compute_saliency(
        self, model: nn.Module, prunable_layers: Tuple
    ) -> Dict[nn.Module, torch.Tensor]:
        """
        Compute saliency = 0.5 * h_ii * w_i^2 for each weight.
        """
        logger.info("Computing saliency via OBD (Hessian diagonal * weight^2)...")
        hessian_diag = self.compute_diagonal_hessian(model, prunable_layers)
        saliency: Dict[nn.Module, torch.Tensor] = {}
        for module, h_diag in hessian_diag.items():
            saliency[module] = 0.5 * h_diag * (module.weight ** 2)
        return saliency

    def apply(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        target_sparsity: float,
        prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
        total_weight_count: Optional[int] = None
    ) -> None:
        """
        Prune weights with smallest saliency to achieve target sparsity.
        Target sparsity includes existing zeros: total zero weights / total prunable weights.
        """
        logger.info(f"[OBD] Target sparsity: {target_sparsity * 100:.2f}%")

        # Ensure every module has a mask
        modules = get_pruneable_modules(model, prunable_layers)
        for module in modules:
            if not hasattr(module, 'mask'):
                module.mask = torch.ones_like(module.weight, dtype=torch.bool, device=self.device)

        # Count total weights and current zeros
        total = total_weight_count or sum(m.weight.numel() for m in modules)
        current_zeros = sum((m.mask.numel() - m.mask.sum().item()) for m in modules)
        target_zeros = int(total * target_sparsity)
        to_prune = target_zeros - current_zeros

        logger.debug(f"Total weights: {total}, current zeros: {current_zeros}, to prune: {to_prune}")
        if to_prune <= 0:
            logger.info("Already at or above target sparsity. Skipping pruning.")
            return

        # Compute saliency and flatten active entries
        saliency = self.compute_saliency(model, prunable_layers)
        flat_sal = []
        for module in modules:
            active_sal = saliency[module][module.mask]
            flat_sal.append(active_sal.flatten())
        flat_sal = torch.cat(flat_sal).cpu()

        # Select smallest-saliency indices to prune
        prune_flat_idx = torch.argsort(flat_sal)[:to_prune]

        # Apply pruning masks
        offset = 0
        pruned = 0
        for module in modules:
            mask_flat = module.mask.clone().view(-1)
            num = mask_flat.numel()
            local = prune_flat_idx[(prune_flat_idx >= offset) & (prune_flat_idx < offset + num)] - offset
            if local.numel() > 0:
                mask_flat[local.to(self.device)] = False
                module.mask = mask_flat.view_as(module.weight)
                module.weight.data.mul_(module.mask.float())
                pruned += local.numel()

                # Zero gradients and momentum buffers
                if module.weight.grad is not None:
                    module.weight.grad.zero_()
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if p is module.weight and 'momentum_buffer' in optimizer.state[p]:
                            optimizer.state[p]['momentum_buffer'].mul_(module.mask.float())

            offset += num

        logger.info(f"[OBD] Pruned {pruned}/{total} new weights -> Achieved zeros: {(current_zeros+pruned)/total*100:.2f}%")
