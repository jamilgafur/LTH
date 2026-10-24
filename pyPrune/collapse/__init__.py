"""Structural layer collapse and variance-based collapse-region discovery.

Public API:
    collapse_only                  -- collapse (start, end) layer spans into surrogate blocks
    discover_regions               -- find collapsible regions from activation variances
    get_dynamic_experiment_config  -- deprecated alias of discover_regions
    get_layer_variances            -- per-layer activation variance probe
    calibrate_layer_variances      -- multi-batch variance calibration (mean + drift)
    wrap_pools_safe                -- wrap pooling layers with underflow-safe wrappers
    patch_skip_connections         -- patch residual forwards to tolerate shape drift
    disable_inplace_relu           -- replace inplace ReLUs with out-of-place ones
    SmartIdentity                  -- pass-through module used for excised layers
    get_layer                      -- dotted-path module access
    count_trainable_params         -- trainable parameter count
    layer_stats                    -- per-layer zero-weight statistics
"""

from .core import collapse_only
from .discovery import (
    calibrate_layer_variances,
    discover_regions,
    get_dynamic_experiment_config,
    get_layer_variances,
)
from .graph_utils import (
    count_trainable_params,
    disable_inplace_relu,
    get_layer,
    layer_stats,
)
from .identity import (
    SmartIdentity,
    _SafePool,
    _wrap_pools_safe,
    wrap_pools_safe,
)
from .patching import patch_skip_connections

__all__ = [
    "collapse_only",
    "discover_regions",
    "get_dynamic_experiment_config",
    "get_layer_variances",
    "calibrate_layer_variances",
    "wrap_pools_safe",
    "_wrap_pools_safe",
    "patch_skip_connections",
    "disable_inplace_relu",
    "SmartIdentity",
    "_SafePool",
    "get_layer",
    "count_trainable_params",
    "layer_stats",
]
