import os
import json
import pickle
import datetime
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pyPrune.utils import get_pruneable_modules, clean_memory
from pyPrune.PruningStrategy import PruningStrategy

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Tuple, Dict, List
from torch.utils.data import DataLoader

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
    Optimal Brain Damage: prunes by approximating saliency = w^2 * (grad)^2.
    Performs true global rank-based pruning to remove exactly num_prune weights.
    """

    def __init__(self, train_loader: DataLoader, criterion: nn.Module, device: str = 'cpu'):
        self.device = device
        self.train_loader = train_loader
        self.criterion = criterion

    def compute_saliency(self, model: nn.Module, prunable_layers: Tuple) -> Dict[nn.Module, torch.Tensor]:
        logger.info("Starting saliency computation for OBD strategy...")
        model.train().to(self.device)
        model.zero_grad()

        # Initialize per-module accumulator
        saliency: Dict[nn.Module, torch.Tensor] = {}
        for module in get_pruneable_modules(model, prunable_layers):
            saliency[module] = torch.zeros_like(module.weight.data, device=self.device)

        # One epoch of gradient accumulation
        for batch_idx, (x, y) in enumerate(tqdm(self.train_loader, desc="Computing OBD saliency")):
            x, y = x.to(self.device), y.to(self.device)
            out = model(x)
            loss = self.criterion(out, y)
            loss.backward()

            for module in saliency:
                if module.weight.grad is not None:
                    # saliency ~ w^2 * g^2
                    saliency[module] += (module.weight.data ** 2) * (module.weight.grad.data ** 2)
                else:
                    logger.warning(f"No grad for {module} on batch {batch_idx}")

            model.zero_grad()

        # Average over batches and report stats
        batches = len(self.train_loader)
        for module, score in saliency.items():
            score.div_(batches)
            logger.debug(
                f"Saliency[{module}]: min={score.min():.4e}, max={score.max():.4e}, "
                f"mean={score.mean():.4e}, std={score.std():.4e}"
            )

        clean_memory()
        logger.info("Completed saliency computation.")
        return saliency

    def _unroll(self, model: nn.Module, prunable_layers: Tuple) -> torch.Tensor:
        """
        Flatten all currently *active* (unmasked) weights into one vector.
        """
        flat_weights = []
        for module in get_pruneable_modules(model, prunable_layers):
            w = module.weight.data
            if hasattr(module, 'mask'):
                w = w[module.mask.bool()]
            flat_weights.append(w.flatten())
        if not flat_weights:
            raise ValueError("No weights to prune.")
        return torch.cat(flat_weights)

    def apply(self,
              model: nn.Module,
              optimizer: torch.optim.Optimizer,
              target_sparsity: float,
              prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
              total_weight_count: Optional[int] = None) -> None:
        """
        Apply OBD pruning by globally ranking active weights by saliency
        and zeroing out exactly num_prune smallest-saliency weights.
        """
        logger.info(f"[OBD] Target sparsity: {target_sparsity*100:.2f}%")

        # 1) Compute per-module saliency
        saliency = self.compute_saliency(model, prunable_layers)

        # 2) Gather modules and counts of active weights
        modules: List[Tuple[nn.Module,int]] = []
        for module in get_pruneable_modules(model, prunable_layers):
            # count active weights
            if hasattr(module, 'mask'):
                cnt = int(module.mask.sum().item())
            else:
                cnt = module.weight.data.numel()
            modules.append((module, cnt))

        # 3) Flatten all saliency scores for active weights
        flat_sal = []
        for module, cnt in modules:
            s = saliency[module]
            if hasattr(module, 'mask'):
                s = s[module.mask.bool()]
            flat_sal.append(s.flatten())
        flat_sal = torch.cat(flat_sal).cpu()

        # 4) Determine how many to prune
        total = total_weight_count or flat_sal.numel()
        num_prune = int(total * target_sparsity) - (total - flat_sal.numel())
        logger.debug(f"Total weights: {total}, Active: {flat_sal.numel()}, To prune: {num_prune}")
        if num_prune <= 0:
            logger.info("Already at or above target sparsity. Skipping.")
            return model.state_dict()

        # 5) Get indices of lowest-saliency weights
        sorted_idx = torch.argsort(flat_sal)
        prune_flat_idx = sorted_idx[:num_prune]

        # 6) Build a global mask and apply per-module
        pruned = 0
        offset = 0
        for module, cnt in modules:
            # indices in [offset, offset+cnt)
            local_idx = prune_flat_idx[(prune_flat_idx >= offset) & (prune_flat_idx < offset + cnt)] - offset

            # Build local mask
            if hasattr(module, 'mask'):
                mask = module.mask.clone().flatten()
            else:
                mask = torch.ones(cnt, dtype=torch.bool, device=self.device)

            # Zero out exactly those indices
            mask[local_idx.to(self.device)] = False
            module.mask = mask.view_as(module.weight.data)

            # Zero the weights
            before = module.weight.data.nonzero().size(0)
            module.weight.data.mul_(module.mask.float())
            after = module.weight.data.nonzero().size(0)
            pruned += (before - after)

            logger.debug(
                f"Module {module}: pruned {before - after}/{cnt} weights"
            )

            # Clear grads & momentum
            if module.weight.grad is not None:
                module.weight.grad.zero_()
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is module.weight and 'momentum_buffer' in optimizer.state[p]:
                        optimizer.state[p]['momentum_buffer'].mul_(module.mask.float())

            offset += cnt

        actual_sparsity = pruned / total
        logger.info(
            f"[OBD] Pruned {pruned}/{total} -> actual sparsity {actual_sparsity*100:.2f}% "
            f"(target {target_sparsity*100:.2f}%)"
        )

        return model.state_dict()
