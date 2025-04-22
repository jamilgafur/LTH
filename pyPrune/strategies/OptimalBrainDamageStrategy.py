
class OptimalBrainDamageStrategy(PruningStrategy):
    """
    Optimal Brain Damage: prunes by approximated saliency = 0.5 * w^2 * H_ii.
    Approximates diagonal Hessian via a single backward pass.
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
            hessian[m] = hessian[m] + g.pow(2)
        return hessian

    def apply(self,
              model: nn.Module,
              optimizer: torch.optim.Optimizer,
              target_sparsity: float,
              prunable_layers: Tuple = (nn.Conv2d, nn.Linear),
              total_weight_count: Optional[int] = None) -> None:
        logger.info(f"[OBD] Target sparsity: {target_sparsity*100:.2f}%")
        modules = list(get_pruneable_modules(model, prunable_layers))
        total = total_weight_count or sum(m.weight.numel() for m in modules)

        # Estimate Hessian diagonals
        hess = self._compute_hessian_diag(model, prunable_layers)
        # Compute saliency for each weight
        saliency_list = []
        for m in modules:
            w = m.weight.data
            sal = 0.5 * w.pow(2) * hess[m].to(self.device)
            saliency_list.append(sal.flatten())
        all_sal = torch.cat(saliency_list)
        num_prune = int(total * target_sparsity) - (total - all_sal.numel())
        if num_prune <= 0:
            logger.info("Already at or above target sparsity.")
            return
        thresh = torch.kthvalue(all_sal.cpu(), num_prune).values.item()
        # Apply mask based on saliency
        for m in modules:
            sal = 0.5 * m.weight.data.pow(2) * hess[m].to(self.device)
            mask = sal > thresh
            m.mask = mask if not hasattr(m, 'mask') else m.mask & mask
            m.weight.data.mul_(m.mask.float())
            if m.weight.grad is not None:
                m.weight.grad.zero_()
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p is m.weight and 'momentum_buffer' in optimizer.state[p]:
                        optimizer.state[p]['momentum_buffer'].mul_(m.mask.float())
        logger.debug(f"Applied OBD pruning at threshold={thresh:.6f}.")

