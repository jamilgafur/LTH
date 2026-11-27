# ablate.py -- replace regions of the MoE LLM with smaller ones, the LLM
# analogue of the block collapse in ../Tranfer/collapse.py. Units of removal,
# finest to coarsest (never inside an expert's weights):
#   1. ablate_experts()   -- a whole expert (or several) in every layer
#   2. collapse_layers()  -- one set of contiguous transformer layers
#   3. collapse_region()  -- multiple layers x multiple experts: a contiguous
#                            layer span whose layers each lose their
#                            lowest-relevance experts
# Selection is driven by the domain/activation correlation analysis.
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from model_io import MoEBlock, TOP_K_ATTRS, find_moe_blocks, router_width

# layer-span surgery lives in the package (the LLM counterpart of the CNN
# block collapse in ../Tranfer/collapse.py); the model_io import above puts the
# repo root on sys.path so pyPrune imports without installation.
from pyPrune.collapseMethods import find_decoder_stack, remove_layers

# int attributes that must shrink along with the router
ROUTER_SIZE_ATTRS = ("n_routed_experts", "num_experts", "n_experts")


# ------------------------------------------------------------------- experts
def _shrink_router(router: nn.Module, keep_idx: List[int]) -> nn.Module:
    """Return a router restricted to the kept expert rows.

    Handles both nn.Linear routers (older Mixtral/OLMoE/DBRX) and gate modules
    that keep a raw [num_experts, hidden] Parameter (e.g. MixtralTopKRouter,
    Qwen2MoeGate).
    """
    n_keep = len(keep_idx)
    idx = torch.tensor(keep_idx, dtype=torch.long)
    if isinstance(router, nn.Linear):
        new = nn.Linear(router.in_features, n_keep,
                        bias=router.bias is not None,
                        device=router.weight.device, dtype=router.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(router.weight.index_select(0, idx))
            if router.bias is not None:
                new.bias.copy_(router.bias.index_select(0, idx))
        return new
    with torch.no_grad():
        router.weight = nn.Parameter(router.weight.index_select(0, idx),
                                     requires_grad=router.weight.requires_grad)
        bias = getattr(router, "bias", None)
        if isinstance(bias, nn.Parameter):
            router.bias = nn.Parameter(bias.index_select(0, idx),
                                       requires_grad=bias.requires_grad)
    for attr in ROUTER_SIZE_ATTRS:
        if isinstance(getattr(router, attr, None), int):
            setattr(router, attr, n_keep)
    if isinstance(getattr(router, "top_k", None), int):
        router.top_k = min(router.top_k, n_keep)
    return router


def _shrink_experts(container: nn.Module, keep_idx: List[int]) -> None:
    """Keep only the selected experts, for either layout (see model_io)."""
    if isinstance(container, nn.ModuleList):
        drop = [i for i in range(len(container)) if i not in keep_idx]
        for i in reversed(drop):
            del container[i]
        return
    idx = torch.tensor(keep_idx, dtype=torch.long)  # fused: params are [E, ...]
    with torch.no_grad():
        for name, p in list(container.named_parameters(recurse=False)):
            setattr(container, name,
                    nn.Parameter(p.index_select(0, idx), requires_grad=p.requires_grad))
        for name, b in list(container.named_buffers(recurse=False)):
            container.register_buffer(name, b.index_select(0, idx))
    if isinstance(getattr(container, "num_experts", None), int):
        container.num_experts = len(keep_idx)


def _shrink_block(blk: MoEBlock, keep_idx: List[int]) -> dict:
    n = blk.num_experts
    _shrink_experts(blk.experts_container, keep_idx)
    setattr(blk.block, blk.router_attr, _shrink_router(blk.router, keep_idx))
    for attr in TOP_K_ATTRS:
        if isinstance(getattr(blk.block, attr, None), int):
            setattr(blk.block, attr, min(getattr(blk.block, attr), len(keep_idx)))
    if isinstance(getattr(blk.block, "num_experts", None), int):
        blk.block.num_experts = len(keep_idx)
    return {"experts_before": n, "kept": sorted(keep_idx),
            "dropped": [i for i in range(n) if i not in keep_idx]}


def _keep_count(n: int, keep_fraction: float, keep_experts: Optional[int]) -> int:
    k = keep_experts if keep_experts is not None else max(1, round(n * keep_fraction))
    return max(1, min(k, n))


def ablate_experts(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    keep_fraction: float = 0.5,
    keep_experts: Optional[int] = None,
) -> Dict[str, dict]:
    """Drop the lowest-scoring experts in every MoE block (whole experts only)."""
    report = {}
    for blk in find_moe_blocks(model):
        k = _keep_count(blk.num_experts, keep_fraction, keep_experts)
        keep_idx = sorted(torch.topk(scores[blk.name], k).indices.tolist())
        report[blk.name] = _shrink_block(blk, keep_idx)
    _sync_config(model)
    return report


# -------------------------------------------------------------------- layers
def _layer_name(block_name: str) -> str:
    """Longest prefix of a MoE block name ending in a digit: '<stack>.<idx>'."""
    parts = block_name.split(".")
    while parts and not parts[-1].isdigit():
        parts.pop()
    return ".".join(parts)


def _layer_index(block_name: str) -> int:
    return int(_layer_name(block_name).split(".")[-1])


def find_transformer_layers(model: nn.Module) -> Tuple[str, nn.ModuleList]:
    """Locate the decoder-layer stack that contains the MoE blocks.

    Stack discovery is pyPrune's find_decoder_stack; we only add the check
    that the MoE blocks live in that one stack.
    """
    name, layers = find_decoder_stack(model)
    container_names = {".".join(_layer_name(b.name).split(".")[:-1])
                       for b in find_moe_blocks(model)}
    if container_names != {name}:
        raise ValueError(f"MoE layers are not in one container: {container_names}")
    return name, layers


def best_layer_window(layer_scores: Dict[str, float], width: int) -> Tuple[int, float]:
    """Contiguous span of `width` layers with the lowest total relevance.

    Spans are contiguous in layer index, mirroring the Transfer study's
    contiguous block collapse.
    """
    items = sorted(((_layer_index(name), s) for name, s in layer_scores.items()))
    if not 1 <= width <= len(items):
        raise ValueError(f"collapse_layers={width} out of range for {len(items)} layers")
    best = min(range(len(items) - width + 1),
               key=lambda i: sum(s for _, s in items[i:i + width]))
    return items[best][0], sum(s for _, s in items[best:best + width])


def collapse_layers(model: nn.Module, start: int, count: int) -> dict:
    """Remove the contiguous layer span [start, start+count) from the stack.

    The surgery itself (deletion, layer_idx renumbering for KV caches, config
    sync) is pyPrune.collapseMethods.remove_layers; here we only keep the
    MoE-stack validation before it runs.
    """
    find_transformer_layers(model)
    return remove_layers(model, start, count)


def collapse_region(
    model: nn.Module,
    scores: Dict[str, torch.Tensor],
    layer_scores: Dict[str, float],
    collapse_layers_width: int = 2,
    keep_fraction: float = 0.5,
    keep_experts: Optional[int] = None,
    start: Optional[int] = None,
) -> Dict[str, dict]:
    """Remove a multi-layer x multi-expert region: a contiguous layer span in
    which every layer keeps only its highest-relevance experts. Layers outside
    the span are untouched. start=None picks the least-relevant span.
    """
    if start is None:
        start, _ = best_layer_window(layer_scores, collapse_layers_width)
    span = set(range(start, start + collapse_layers_width))
    report = {"span": sorted(span), "layers": {}}
    for blk in find_moe_blocks(model):
        idx = _layer_index(blk.name)
        if idx not in span:
            continue
        k = _keep_count(blk.num_experts, keep_fraction, keep_experts)
        keep_idx = sorted(torch.topk(scores[blk.name], k).indices.tolist())
        report["layers"][blk.name] = _shrink_block(blk, keep_idx)
    _sync_config(model)
    return report


# ------------------------------------------------------------------ reporting
def _sync_config(model: nn.Module) -> None:
    """Keep model.config consistent with the shrunk expert sets."""
    n_after = find_moe_blocks(model)[0].num_experts
    for attr in ("num_local_experts", "num_experts"):
        if isinstance(getattr(model.config, attr, None), int):
            setattr(model.config, attr, n_after)


def expert_footprint(model: nn.Module) -> dict:
    """Parameter accounting over experts vs. the rest of the model."""
    blocks = find_moe_blocks(model)
    expert_params = sum(p.numel() for blk in blocks
                        for p in blk.experts_container.parameters())
    total = sum(p.numel() for p in model.parameters())
    _, layers = find_transformer_layers(model)
    return {
        "moe_layers": len(blocks),
        "transformer_layers": len(layers),
        "experts_per_layer": [b.num_experts for b in blocks],
        "expert_params": expert_params,
        "total_params": total,
    }
