# analysis.py -- understanding a span before collapsing it: locating the
# contiguous block between two layers, finding the next Linear downstream,
# and measuring the block's output characteristics.
import logging
import math

import torch
import torch.nn as nn

from .containers import _get_container_and_subname, get_layer

logger = logging.getLogger(__name__)


def _locate_and_prepare_block(model, start_layer_name, end_layer_name):
    logger.debug(f"Locating and preparing block: start='{start_layer_name}', end:'{end_layer_name}'")

    # --- LCA resolution ---
    start_parts = start_layer_name.split('.') if start_layer_name else []
    end_parts = end_layer_name.split('.') if end_layer_name else []
    common_parts = []
    
    # NEW FIX: If start and end are identical, the LCA is their parent!
    if start_layer_name == end_layer_name and len(start_parts) > 0:
        lca_path = '.'.join(start_parts[:-1])
    else:
        for a, b in zip(start_parts, end_parts):
            if a == b:
                common_parts.append(a)
            else:
                break
        lca_path = '.'.join(common_parts)

    # FIX 1: Allow root to be the container
    if lca_path == "":
        start_container_name, _ = _get_container_and_subname(start_layer_name)
        end_container_name, _ = _get_container_and_subname(end_layer_name)
        
        # If they share a parent (even root ""), use it.
        if start_container_name == end_container_name:
            lca_path = start_container_name
            container = get_layer(model, lca_path)
        else:
            # Fallback: If layers are in different top-level branches, use root as LCA
            logger.debug(f"Layers in different branches or direct children; using root as LCA.")
            lca_path = ""
            container = model
    else:
        container = get_layer(model, lca_path)

    logger.debug(f"Containers resolved → chosen LCA container: '{lca_path}'")

    named_layers = list(container.named_children())
    logger.debug(f"Found {len(named_layers)} children in container '{lca_path or '<root>'}'")

    # --- locate child indices ---
    start_idx = end_idx = None
    for i, (child_name, _) in enumerate(named_layers):
        full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
        
        # Check start
        if start_idx is None and (
            start_layer_name == full_child_prefix
            or start_layer_name.startswith(full_child_prefix + ".")
        ):
            start_idx = i
        
        # Check end
        if end_idx is None and (
            end_layer_name == full_child_prefix
            or end_layer_name.startswith(full_child_prefix + ".")
        ):
            end_idx = i
        
        if start_idx is not None and end_idx is not None:
            break

    if start_idx is None or end_idx is None:
        raise ValueError(f"[ERROR] Could not map start/end layers into LCA container '{lca_path}'.")

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    full_block = named_layers[start_idx:end_idx + 1]
    logger.debug(f"Block slice length: {len(full_block)}")

    # --- collect collapsible layers (mixed allowed) ---
    conv_layers = []
    for _, mod in full_block:
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            conv_layers.append(mod)
        for sub_name, sub_mod in mod.named_modules():
            if sub_name and isinstance(sub_mod, (nn.Conv2d, nn.Linear)):
                conv_layers.append(sub_mod)

    if not conv_layers:
        raise ValueError("[ERROR] No Conv2d or Linear layers found in block.")

    has_conv = any(isinstance(l, nn.Conv2d) for l in conv_layers)
    collapse_mode = "conv" if has_conv else "linear"

    logger.debug(f"Block composition → {sum(isinstance(l, nn.Conv2d) for l in conv_layers)} Conv2d, {sum(isinstance(l, nn.Linear) for l in conv_layers)} Linear (collapse_mode={collapse_mode})")
    layer_type = nn.Conv2d if collapse_mode == "conv" else nn.Linear

    # FIX 2: Return names (strings) for first/last layer to avoid naming errors
    return {
        "container": container,
        "container_name": lca_path,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "layer_type": layer_type,
        "conv_layers": conv_layers,
        "collapse_mode": collapse_mode,
        "first_layer_name": full_block[0][0],  # Changed from object to name string
        "last_layer_name": full_block[-1][0],   # Changed from object to name string
    }



def _find_next_linear(model, end_layer_name, debug):
    if debug:
        logger.debug(f"Searching for next nn.Linear after '{end_layer_name}'...")
    modules_list = list(model.named_modules())
    idx_end_global = None

    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            if debug:
                logger.debug(f"Exact layer match found at index {i}: {n} ({type(m).__name__})")
            break

    if idx_end_global is None:
        for i, (n, m) in enumerate(modules_list):
            if n.endswith(end_layer_name):
                idx_end_global = i
                if debug:
                    logger.debug(f"Fallback: found partial match at index {i}: {n}")
                break

    next_linear_name = None
    next_linear_mod = None
    if idx_end_global is not None:
        if debug:
            logger.debug(f"Scanning forward from index {idx_end_global + 1} for next Linear...")
        for j in range(idx_end_global + 1, len(modules_list)):
            n, m = modules_list[j]
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    logger.debug(f"Found Linear layer ahead: {n} ({m})")
                break

    if next_linear_mod is None:
        if debug:
            logger.debug(f"No Linear found after {end_layer_name}. Searching globally...")
        for n, m in modules_list:
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    logger.debug(f"Global fallback Linear found: {n} ({m})")
                break

    if next_linear_mod is None:
        logger.warning(f"No Linear layer found in model after '{end_layer_name}'")

    if debug:
        logger.info(f"Next Linear detected → name='{next_linear_name}', module={next_linear_mod}")

    return next_linear_name, next_linear_mod


def _analyze_block_output(
    model,
    full_block,
    conv_layers,
    named_layers,
    end_idx,
    layer_type,
    x,
    next_linear_mod,
    debug,
):
    if debug:
        logger.debug(f"Analyzing output of collapsed block ({len(full_block)} layers)...")
        logger.debug(f"Input tensor shape before block: {tuple(x.shape)}")
        logger.debug(f"Running forward pass through block layers:")

    with torch.no_grad():
        y = x.clone()
        for idx, (name, layer) in enumerate(full_block):
            if debug:
                logger.debug(f"Layer {idx+1}/{len(full_block)}: {name} ({layer.__class__.__name__}) input={tuple(y.shape)}")
            y = layer(y)
            if debug:
                logger.debug(f"└── output shape: {tuple(y.shape)}")

    # --- Ground-truth output characteristics ---
    out_shape = tuple(y.shape)

    # 🔧 CRITICAL FIX:
    # Infer channels from actual tensor, not from layer attributes
    out_channels = y.shape[1] if y.ndim >= 2 else None

    if debug:
        logger.debug(f"Final block output shape: {out_shape}")
        logger.debug(f"Determined out_channels={out_channels}")

    # --- Detect pooling inside original block (informational) ---
    pool_layer = next(
        (
            m for _, m in reversed(full_block)
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        ),
        None,
    )

    if debug:
        if pool_layer is not None:
            logger.debug(f"Detected pool in original block: {type(pool_layer).__name__}")
        else:
            logger.debug(f"No explicit pooling layer detected inside block.")

    # --- Linear compatibility heuristic ---
    linear_in_features_heuristic = next_linear_mod.in_features if next_linear_mod else None
    if debug:
        logger.debug(f"Heuristic next linear in_features = {linear_in_features_heuristic}")

    adaptive_pool_to_use = None
    if (
        layer_type == nn.Conv2d
        and linear_in_features_heuristic is not None
        and out_channels is not None
        and len(out_shape) >= 4
    ):
        expected_hw = max(1, linear_in_features_heuristic // out_channels)
        cur_H, cur_W = out_shape[-2], out_shape[-1]
        cur_hw = cur_H * cur_W

        if debug:
            logger.debug(f"Comparing spatial dims: expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})")

        if cur_hw != expected_hw:
            target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
            target_W = max(1, expected_hw // target_H)
            adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))

            if debug:
                logger.debug(f"Suggest AdaptiveAvgPool2d({target_H},{target_W}) to reconcile linear in_features mismatch.")

    # --- Shortcut detection (unchanged) ---
    shortcut_out_channels = None
    for nm, mod in model.named_modules():
        if hasattr(mod, "shortcut") and isinstance(mod.shortcut, nn.Module):
            first_conv = next(
                (m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)),
                None,
            )
            if first_conv is not None:
                shortcut_out_channels = first_conv.out_channels
                if debug:
                    logger.debug(f"Found shortcut conv → out_channels={shortcut_out_channels}")
                break

    if debug:
        logger.info(f"Block analysis complete:")
        logger.debug(f"out_shape={out_shape}")
        logger.debug(f"out_channels={out_channels}")
        logger.debug(f"has_pool={pool_layer is not None}")
        logger.debug(f"adaptive_pool_to_use={adaptive_pool_to_use}")
        logger.debug(f"shortcut_out_channels={shortcut_out_channels}")

    return {
        "out_shape": out_shape,
        "out_channels": out_channels,
        "pool_layer": pool_layer,
        "adaptive_pool_to_use": adaptive_pool_to_use,
        "shortcut_out_channels": shortcut_out_channels,
        "linear_in_features_heuristic": linear_in_features_heuristic,
    }

