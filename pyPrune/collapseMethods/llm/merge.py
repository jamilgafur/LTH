# merge.py -- merging a contiguous span of decoder layers into a single
# collapsed layer (weight averaging, or picking one span member).
from copy import deepcopy
from typing import Sequence

import torch
import torch.nn as nn

MERGE_MODES = ("mean", "middle", "first", "last")


def _merge_blocks(blocks: Sequence[nn.Module], merge: str) -> nn.Module:
    """Build the one collapsed layer that replaces `blocks`.

    "mean": element-wise mean of the span's shared floating-point tensors (the
    LLM analogue of folding a conv block into one convolution); parameters
    only some blocks own (e.g. T5's shared relative_attention_bias, kept by the
    first block) are carried over from the first layer, and non-float buffers
    are taken from the first layer. The other modes copy one span member.
    """
    if merge == "mean":
        merged = deepcopy(blocks[0])
        states = [b.state_dict() for b in blocks]
        keys0 = set(states[0])
        extra = set().union(*(set(s) for s in states[1:])) - keys0
        if extra:
            raise ValueError(
                "cannot merge: the first layer of the span lacks parameters of "
                f"later layers (e.g. {sorted(extra)[:3]})")
        with torch.no_grad():
            out = {}
            for k, v in states[0].items():
                shared = [s[k] for s in states if k in s]
                if any(tuple(c.shape) != tuple(v.shape) for c in shared):
                    raise ValueError(f"cannot merge: parameter {k} has mismatched "
                                     "shapes across the span")
                out[k] = (torch.stack(shared).mean(dim=0)
                          if v.is_floating_point() and len(shared) > 1 else v.clone())
        merged.load_state_dict(out)
        return merged
    pick = {"middle": len(blocks) // 2, "first": 0, "last": len(blocks) - 1}[merge]
    return deepcopy(blocks[pick])
