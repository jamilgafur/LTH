# api.py -- orchestration and public API: collapse one span
# (_collapse_block), or every span of a compression_set in one pass before
# retraining (collapse_only).
import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .analysis import (_analyze_block_output, _find_next_linear,
                       _locate_and_prepare_block)
from .capture import (count_trainable_params, layer_stats,
                      _capture_preblock_activation)
from .postprocess import (disable_inplace_relu, patch_skip_connections,
                          _wrap_pools_safe)
from .replacement import _build_and_replace_block

logger = logging.getLogger(__name__)


def _collapse_block(
    model: nn.Module,
    start_layer_name: str,
    end_layer_name: str,
    input_shape: tuple,
    device='cpu',
    debug: bool = False
) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive)
    by replacing them with: Conv2d(1x1) -> AdaptiveAvgPool2d(H_last,W_last).
    """
    logger.info(f"===== Collapsing block: {start_layer_name} → {end_layer_name} =====")

    # Step 1: Locate block
    logger.debug(f"[STEP 1] Locating block boundaries...")
    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name)
    if debug:
        logger.debug(f"Located block with {len(info['full_block'])} layers")
        for n, l in info["full_block"]:
            logger.debug(f"[LAYER] {n}: {l}")

    # Step 2: Capture activation entering the start layer
    logger.debug(f"[STEP 2] Capturing activation before start layer '{start_layer_name}'...")
    x, pre_params = _capture_preblock_activation(
        model, start_layer_name, input_shape, info["conv_layers"], info["layer_type"], device, debug
    )
    if debug:
        logger.debug(f"Input activation shape entering block: {tuple(x.shape)}")

    # Step 3: Find next linear
    logger.debug(f"[STEP 3] Searching for next linear layer after '{end_layer_name}'...")
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)
    if debug:
        logger.debug(f"Next linear layer: {next_linear_name} -> {type(next_linear_mod).__name__ if next_linear_mod else 'None'}")

    # Step 4: Analyze block output
    logger.debug(f"[STEP 4] Analyzing block output characteristics...")
    block_analysis = _analyze_block_output(
        model,
        info["full_block"],
        info["conv_layers"],
        info["named_layers"],
        info["end_idx"],
        info["layer_type"],
        x,
        next_linear_mod,
        debug
    )
    if debug:
        logger.debug(f"Block output analysis result:")
        for k, v in block_analysis.items():
            logger.debug(f"{k}: {v}")

    # Step 5: Replace block
    logger.debug(f"[STEP 5] Rebuilding and replacing collapsed block...")
    model = _build_and_replace_block(
        model,
        start_layer_name,
        input_shape,
        info,
        x,
        pre_params,
        next_linear_name,
        next_linear_mod,
        block_analysis,
        device,
        debug
    )
    logger.info(f"✅ Collapse complete for block '{start_layer_name}' → '{end_layer_name}'")

    return model


def collapse_only(
    model: Optional[nn.Module] = None,
    model_weights_1: Optional[str] = None,
    compression_set: Optional[Sequence[Tuple[str, str]]] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = 'cpu',
    safe_param_reduction: bool = True,
    handle_skips: bool = True,
    debug: bool = True,
    dry_run: bool = False
) -> nn.Module:
    """
    Top-level API to collapse multiple blocks with the simple replacement policy.
    """
    logger.debug(f"===== Starting collapse_only process =====")
    logger.debug(f"Device={device}, dry_run={dry_run}, handle_skips={handle_skips}, safe_param_reduction={safe_param_reduction}")

    # Load or use provided model
    if model is None:
        logger.debug(f"Loading model from disk...")
        if not (model_weights_1 and model_class):
            raise ValueError("[ERROR] Must provide either an instantiated `model` or (`model_weights_1` + `model_class`).")

        model_kwargs = model_kwargs or {}
        logger.info(f"Instantiating model from class '{model_class.__name__}' with kwargs={model_kwargs}")
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to instantiate model class {model_class}: {e}")

        logger.info(f"Loading weights from file: {model_weights_1}")
        try:
            chk = torch.load(model_weights_1, map_location=device)
            state = chk.get('model', chk) if isinstance(chk, dict) else chk
            model.load_state_dict(state)
            logger.info(f"Weights successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model weights: {e}")
    else:
        logger.debug(f"Using provided in-memory model instance ({model.__class__.__name__})")

    # Inspect model
    if debug:
        try:
            logger.debug("Model layer statistics before collapse:\n%s", layer_stats(model))
        except Exception as e:
            logger.warning(f"layer_stats() failed: {e}")

    # Make a deep copy to avoid modifying the original model
    logger.debug(f"Creating deepcopy of model for safe modification...")
    model = deepcopy(model).to(device)
    model.eval()

    # Normalize compression_set
    logger.debug(f"Parsing compression set...")
    if compression_set is None:
        logger.warning("compression_set is None or empty; skipping collapse.")
        return model

    collapse_map = {}
    if isinstance(compression_set, dict):
        if debug:
            logger.debug(f"Detected compression_set as dict with {len(compression_set)} entries.")
        for k, v in compression_set.items():
            start, end = v
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[k] = (start, end)
            if debug:
                logger.debug(f"Added mapping: {k} = ({start} → {end})")
    else:
        if debug:
            logger.debug(f"Detected compression_set as sequence with {len(compression_set)} pairs.")
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)
            if debug:
                logger.debug(f"Added mapping: collapse_{i} = ({start} → {end})")

    # Store collapsed blocks for reference
    model._collapsed_blocks = list(collapse_map.values())
    if debug:
        logger.debug(f"Total collapse targets: {len(model._collapsed_blocks)}")
        for idx, (s, e) in enumerate(model._collapsed_blocks):
            logger.debug(f"[BLOCK {idx}] {s} → {e}")

    # Track parameters
    pre_total = count_trainable_params(model)
    logger.info(f"Model parameter count before collapsing: {pre_total:,}")

    # Process each block in sequence
    logger.debug(f"Beginning block-wise collapsing...")
    for name, (start, end) in collapse_map.items():
        logger.info(f"Processing collapse task '{name}': {start} → {end}")
        if dry_run:
            logger.info("dry_run enabled; skipping actual modification for this block.")
            continue

        try:
            logger.debug(f"Calling _collapse_block('{start}', '{end}')")
            model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)
            logger.info(f"✅ Successfully collapsed block '{name}' ({start} → {end})")
        except Exception as e:
            raise RuntimeError(f"collapse failed for block '{name}' ({start} → {end})") from e

        if handle_skips:
            logger.debug(f"Patching skip connections (if any)...")
            try:
                patch_skip_connections(model)
                if debug:
                    logger.debug(f"Skip connections patched successfully.")
            except Exception as e:
                logger.warning(f"Failed to patch skip connections: {e}")

        logger.debug(f"Disabling in-place ReLUs for autograd safety...")
        try:
            disable_inplace_relu(model)
            if debug:
                logger.debug(f"In-place ReLUs converted to out-of-place versions.")
        except Exception as e:
            logger.warning(f"Failed to disable in-place ReLUs: {e}")

    # Safe wrapping of pooling layers
    logger.debug(f"Wrapping pooling layers safely...")
    try:
        _wrap_pools_safe(model)
        if debug:
            logger.debug("All pooling layers wrapped with _SafePool to prevent underflow errors.")
    except Exception as e:
        logger.warning(f"Failed to wrap pools safely: {e}")

    # Post-collapse summary
    post_total = count_trainable_params(model)
    logger.debug(f"===== Collapse summary =====")
    logger.info(f"Parameters before: {pre_total:,}")
    logger.info(f"Parameters after : {post_total:,}")
    delta = pre_total - post_total
    logger.info(f"ΔParams = {delta:+,} (expected ≤ 0)")

    if post_total > pre_total:
        logger.warning(f"⚠ Model gained parameters after collapsing! Investigate collapse policy or replacement logic.")

    if safe_param_reduction and delta < 0:
        logger.warning(f"⚠ Parameter count increased when safe_param_reduction=True — collapse may have failed silently.")

    logger.info(f"✅ collapse_only complete. Total collapsed blocks: {len(collapse_map)}")
    return model

