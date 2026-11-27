# api.py -- public collapse API: collapse one span (collapse_block), remove one
# span (remove_layers), or collapse every span of a compression_set in a single
# pass before retraining (collapse_only).
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .merge import MERGE_MODES, _merge_blocks
from .stack import (_renumber_layer_idx, _resolve_index, _sync_config,
                    find_decoder_stack)

logger = logging.getLogger(__name__)


def count_params(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def collapse_block(
    model: nn.Module,
    start: int,
    end: int,
    merge: str = "mean",
    debug: bool = False,
    stack_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Collapse the inclusive layer span [start, end] of the decoder stack into
    a single collapsed layer. Returns a report dict (like the CNN collapse)."""
    if merge not in MERGE_MODES:
        raise ValueError(f"merge must be one of {MERGE_MODES}, got {merge!r}")
    stack_name, stack = find_decoder_stack(model, stack_name)
    n = len(stack)
    start, end = _resolve_index(stack, start, n), _resolve_index(stack, end, n)
    if start > end:
        start, end = end, start

    pre_params = count_params(model)
    logger.info("collapsing %s.[%d..%d] (%d layers -> 1, merge=%s) params=%d",
                stack_name, start, end, end - start + 1, merge, pre_params)
    blocks = list(stack)[start:end + 1]

    collapsed = _merge_blocks(blocks, merge)
    stack[start] = collapsed
    for i in reversed(range(start + 1, end + 1)):
        del stack[i]
    _renumber_layer_idx(stack)
    _sync_config(model, len(stack))

    post_params = count_params(model)
    report = {
        "stack": stack_name,
        "layers_before": n,
        "layers_after": len(stack),
        "collapsed": list(range(start, end + 1)),
        "merge": merge,
        "params_before": pre_params,
        "params_after": post_params,
    }
    if debug:
        logger.info("collapse report: %s", report)
    return report


def remove_layers(
    model: nn.Module,
    start: int,
    count: int,
    stack_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Remove the contiguous layer span [start, start+count) outright (no
    replacement). Used by the removal ablations in manuscript/llm_ablation."""
    stack_name, stack = find_decoder_stack(model, stack_name)
    n = len(stack)
    start = _resolve_index(stack, start, n)
    if count < 1 or start + count > n:
        raise ValueError(f"span [{start}, {start + count}) outside 0..{n}")
    if n - count < 1:
        raise ValueError(f"would remove all {n} layers")
    drop = list(range(start, start + count))
    for i in reversed(drop):
        del stack[i]
    _renumber_layer_idx(stack)
    _sync_config(model, len(stack))
    return {"layers_before": n, "layers_after": len(stack),
            "removed_layers": drop, "stack": stack_name}


# -----------------------------------------------------------------------------
# Top-level multi-span collapse function (flexible API, like collapse_only)
# -----------------------------------------------------------------------------

def _normalize_compression_set(compression_set) -> List[Tuple[Union[int, str], Union[int, str]]]:
    if isinstance(compression_set, dict):
        items = list(compression_set.values())
    else:
        items = list(compression_set)
    spans = []
    for i, pair in enumerate(items):
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(f"compression_set entry #{i} must be a (start, end) pair, got {pair!r}")
        spans.append((pair[0], pair[1]))
    return spans


def collapse_only(
    model: Optional[nn.Module] = None,
    compression_set: Optional[Sequence] = None,
    model_weights_1: Optional[str] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    merge: str = "mean",
    stack_name: Optional[str] = None,
    inplace: bool = False,
    device: str = "cpu",
    safe_param_reduction: bool = True,
    debug: bool = True,
    dry_run: bool = False,
) -> nn.Module:
    """
    Top-level API to collapse multiple contiguous layer spans with the simple
    replacement policy. Contract mirrors manuscript/Tranfer/collapse.py::
    collapse_only: compression_set is a dict or sequence of (start, end) pairs
    (inclusive; ints or names like 'model.layers.3'), the parameter counts are
    reported around the operation, and the model is deep-copied unless
    inplace=True (set it for large models, where the copy alone doubles peak
    memory).
    """
    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("provide either `model` or (`model_weights_1` + `model_class`)")
        model = model_class(**(model_kwargs or {}))
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get("model", chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)

    model = model.to(device) if inplace else deepcopy(model).to(device)
    model.eval()

    if not compression_set:
        logger.warning("compression_set is None or empty; skipping collapse")
        return model

    spans = _normalize_compression_set(compression_set)
    stack_name, stack = find_decoder_stack(model, stack_name)
    n = len(stack)
    resolved = sorted(
        (_resolve_index(stack, s, n), _resolve_index(stack, e, n)) for s, e in spans)
    resolved = [(min(s, e), max(s, e)) for s, e in resolved]
    for (a1, b1), (a2, b2) in zip(resolved, resolved[1:]):
        if b1 >= a2:
            raise ValueError(f"overlapping collapse spans ({a1},{b1}) and ({a2},{b2})")

    pre_total = count_params(model)
    reports = []
    if dry_run:
        logger.info("dry_run: would collapse %s in %s (%d -> %d layers)",
                    resolved, stack_name, n, n - sum(b - a for a, b in resolved))
        model._collapsed_blocks = resolved
        return model

    # collapse from the end so earlier indices stay valid
    for start, end in reversed(resolved):
        reports.append(collapse_block(model, start, end, merge=merge,
                                      debug=debug, stack_name=stack_name))

    post_total = count_params(model)
    logger.info("collapse summary: params %d -> %d (%+d), layers %d -> %d",
                pre_total, post_total, post_total - pre_total, n,
                len(find_decoder_stack(model, stack_name)[1]))
    if safe_param_reduction and post_total > pre_total:
        logger.warning("parameter count increased after collapsing - investigate merge policy")
    model._collapsed_blocks = resolved
    model._collapse_reports = reports
    return model
