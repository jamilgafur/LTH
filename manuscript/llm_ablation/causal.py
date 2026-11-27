# causal.py -- does the identified region *cause* the undesired output, or
# does it merely correlate with it? Intervention: block routing to the suspect
# experts (their contribution becomes zero) and re-measure the poison score.
# If the score drops more than for random same-size expert sets, the region is
# causally involved. Works on both expert layouts:
#   classic: zero-output forward hooks on the per-expert modules
#   fused:   a pre-hook on the experts container remaps blocked experts out of
#            top-k and zeroes their routing weight
import random
from typing import Callable, Dict, List

import torch

from model_io import find_moe_blocks


class ExpertBlocker:
    """Context manager that zeroes the contribution of chosen experts."""

    def __init__(self, model: torch.nn.Module, suspects: Dict[str, List[int]]):
        self.model = model
        self.suspects = suspects  # {block_name: [expert_idx, ...]}
        self._handles = []

    def __enter__(self):
        for blk in find_moe_blocks(self.model):
            idx = self.suspects.get(blk.name)
            if not idx:
                continue
            if not blk.fused:  # classic: zero each blocked expert's output
                for i in idx:
                    def zero(module, args, output):
                        out = output[0] if isinstance(output, tuple) else output
                        z = torch.zeros_like(out)
                        return (z,) + output[1:] if isinstance(output, tuple) else z
                    self._handles.append(blk.experts[i].register_forward_hook(zero))
            else:  # fused: reroute blocked experts out of top-k, weight -> 0
                blocked = torch.tensor(sorted(idx))
                fallback = next(i for i in range(blk.num_experts) if i not in idx)

                def mask_routing(module, args, kwargs):
                    hidden, topk_idx, topk_weight = args[0], args[1], args[2]
                    hit = torch.isin(topk_idx.cpu(), blocked).to(topk_idx.device)
                    idx2 = topk_idx.clone()
                    w2 = topk_weight.clone()
                    idx2[hit] = fallback
                    w2[hit] = 0.0
                    return (hidden, idx2, w2), kwargs

                self._handles.append(
                    blk.experts_container.register_forward_pre_hook(mask_routing,
                                                                    with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        return False


def causal_check(model: torch.nn.Module, suspects: Dict[str, List[int]],
                 score_fn: Callable[[torch.nn.Module], float],
                 n_controls: int = 5, seed: int = 0) -> dict:
    """Compare the poison-score drop from blocking `suspects` against random
    same-size expert sets.

    score_fn: average poison score over the flagged queries for the model as
    it currently stands (interventions apply automatically).
    """
    all_slots = [(b.name, i) for b in find_moe_blocks(model)
                 for i in range(b.num_experts)]
    suspect_slots = [(n, i) for n, idx in suspects.items() for i in idx]
    free_slots = [s for s in all_slots if s not in suspect_slots]
    rng = random.Random(seed)

    base = score_fn(model)
    with ExpertBlocker(model, suspects):
        suspect_score = score_fn(model)
    suspect_drop = base - suspect_score

    control_drops = []
    for _ in range(n_controls):
        picks = rng.sample(free_slots, min(len(suspect_slots), len(free_slots)))
        ctrl: Dict[str, List[int]] = {}
        for n, i in picks:
            ctrl.setdefault(n, []).append(i)
        with ExpertBlocker(model, ctrl):
            control_drops.append(base - score_fn(model))

    mean_ctrl = sum(control_drops) / len(control_drops) if control_drops else 0.0
    return {
        "base_score": base,
        "suspect_score": suspect_score,
        "suspect_drop": suspect_drop,
        "control_drops": control_drops,
        "mean_control_drop": mean_ctrl,
        "causal_ratio": suspect_drop / (abs(mean_ctrl) + 1e-9),
        "beats_controls": suspect_drop > max(control_drops) if control_drops else True,
    }
