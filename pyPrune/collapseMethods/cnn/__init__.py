# cnn -- contiguous Conv2d/Linear block collapse for CNNs (the collapse
# mechanics of the manuscript/Tranfer study). Public entry point:
# pyPrune.collapseMethods.CNNCollapse (facade) or pyPrune.collapseMethods.
from .analysis import (_analyze_block_output, _find_next_linear,
                       _locate_and_prepare_block)
from .api import _collapse_block, collapse_only
from .capture import (count_trainable_params, count_zeros, layer_stats,
                      _capture_preblock_activation, _simulate_input_hook)
from .containers import (SmartIdentity, get_layer, _get_container_and_subname,
                         _is_int_str, _replace_layers, _update_container)
from .postprocess import (_SafePool, _wrap_pools_safe, disable_inplace_relu,
                          patch_skip_connections)
from .replacement import (_build_and_replace_block, _insert_corrective_pool,
                          _validate_downstream)

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
