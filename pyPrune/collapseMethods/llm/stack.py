# stack.py -- locating the decoder layer stack and keeping its bookkeeping
# (layer_idx KV-cache attributes, config.num_hidden_layers) consistent when
# layers are collapsed or removed.
from typing import Optional, Tuple, Union

import torch.nn as nn


def find_decoder_stack(model: nn.Module, stack_name: Optional[str] = None
                       ) -> Tuple[str, nn.ModuleList]:
    """Locate the decoder layer stack: the largest ModuleList/Sequential whose
    children are parameter-bearing blocks.

    Works structurally, so HF causal LMs (model.model.layers, GPT-2's
    transformer.h), the study's toy models (layers) and plain custom stacks are
    all found without class checks. For encoder-decoder or multi-tower models
    (T5, BART, LLaVA, ...) pass `stack_name` (e.g. 'decoder.block') to pick the
    tower explicitly. The stack may hold a single layer (after earlier
    collapses) and its blocks need not be identical (e.g. after expert
    ablations); merging a span only requires the span itself to be mergeable
    (_merge_blocks checks that).
    """
    if stack_name is not None:
        mod = model.get_submodule(stack_name)
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)) or not list(mod.children()):
            raise ValueError(f"stack {stack_name!r} is not a ModuleList/Sequential of blocks")
        return stack_name, mod
    best_name, best_mod, best_params = None, None, -1
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.ModuleList, nn.Sequential)):
            continue
        kids = list(mod.children())
        if not kids:
            continue
        n_params = sum(p.numel() for k in kids for p in k.parameters())
        if n_params == 0 or n_params <= best_params:
            continue
        best_name, best_mod, best_params = name, mod, n_params
    if best_mod is None:
        raise ValueError(
            "no decoder stack found (expected an nn.ModuleList/nn.Sequential "
            "of parameter-bearing blocks)")
    return best_name, best_mod


def _resolve_index(stack: nn.ModuleList, ref: Union[int, str], n: int) -> int:
    """Accept layer indices as ints, numeric strings, or module names whose
    last component is the index ('model.layers.3')."""
    if isinstance(ref, int):
        idx = ref
    else:
        tail = str(ref).split(".")[-1]
        if not tail.lstrip("-").isdigit():
            raise ValueError(f"cannot resolve layer reference {ref!r} to an index")
        idx = int(tail)
    if not 0 <= idx < n:
        raise ValueError(f"layer index {idx} out of range for {n} layers")
    return idx


def _renumber_layer_idx(stack: nn.ModuleList) -> None:
    """Keep `layer_idx` attributes (HF KV-cache bookkeeping) consistent after
    the stack changed length."""
    for new_i, layer in enumerate(stack):
        for m in layer.modules():
            if isinstance(getattr(m, "layer_idx", None), int):
                m.layer_idx = new_i


def _sync_config(model: nn.Module, n: int) -> None:
    cfg = getattr(model, "config", None)
    try:
        if cfg is not None and isinstance(getattr(cfg, "num_hidden_layers", None), int):
            cfg.num_hidden_layers = n
    except AttributeError:  # read-only config properties (e.g. T5's alias)
        pass
