# CNNCollapse.py -- collapse contiguous Conv2d/Linear blocks into one cheap
# layer, the CNN counterpart of the transformer-layer collapse
# (LLMCollapse.py). The block is replaced by a pointwise/depthwise conv +
# BatchNorm + adaptive pool with the same interface (the collapse mechanics
# formerly in manuscript/Tranfer/collapse.py); skip connections are
# shape-aligned and pooling layers are wrapped defensively afterwards.
#
# This module is the public facade; the implementation lives in the `cnn`
# subpackage: `containers` (module-tree editing), `capture` (stats and
# activation hooks), `analysis` (locating and measuring the span),
# `replacement` (building and installing the collapsed block),
# `postprocess` (skip/ReLU/pool hardening), `api` (_collapse_block,
# collapse_only).
from .cnn.analysis import (_analyze_block_output, _find_next_linear,
                           _locate_and_prepare_block)
from .cnn.api import _collapse_block, collapse_only
from .cnn.capture import (count_trainable_params, count_zeros, layer_stats,
                          _capture_preblock_activation, _simulate_input_hook)
from .cnn.containers import (SmartIdentity, get_layer,
                             _get_container_and_subname, _is_int_str,
                             _replace_layers, _update_container)
from .cnn.postprocess import (_SafePool, _wrap_pools_safe,
                              disable_inplace_relu, patch_skip_connections)
from .cnn.replacement import (_build_and_replace_block,
                              _insert_corrective_pool, _validate_downstream)

__all__ = [
    "SmartIdentity",
    "collapse_only",
    "count_trainable_params",
    "count_zeros",
    "disable_inplace_relu",
    "get_layer",
    "layer_stats",
    "patch_skip_connections",
]
