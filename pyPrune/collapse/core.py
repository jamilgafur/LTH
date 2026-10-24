"""Core block-collapse pipeline and public collapse_only entry point."""

import time
from copy import deepcopy
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .graph_utils import (
    _capture_preblock_activation,
    _log_exception,
    count_trainable_params,
    disable_inplace_relu,
    execution_order,
    layer_stats,
)
from .identity import _wrap_pools_safe
from .patching import patch_skip_connections
from .replace import (
    _analyze_block_output,
    _build_and_replace_block,
    _find_next_linear,
    _locate_and_prepare_block,
)

def _collapse_block(
    model: nn.Module,
    start_layer_name: str,
    end_layer_name: str,
    input_shape: tuple,
    device='cpu',
    debug: bool = False,
    init: str = 'random',
) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive)
    by replacing them with: Conv2d(1x1) -> AdaptiveAvgPool2d(H_last,W_last).
    """
    print(f"\n[INFO] {'='*20} Collapsing Block {'='*20}")
    print(f"[INFO] Start Target : {start_layer_name}")
    print(f"[INFO] End Target   : {end_layer_name}")
    if debug:
        print(f"[DEBUG] Target Device Allocation: {device}")

    # Step 1: Locate block
    print(f"\n[STEP 1] Locating block boundaries and Lowest Common Ancestor (LCA)...")
    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name, debug=debug)
    
    # ISSUE #3 FIX: Cross-branch filtering at collapse level
    if info is None:
        raise ValueError(f"[ERROR] Could not map start/end layers into LCA container. This may be a cross-branch collapse.")
    
    if debug:
        print(f"[DEBUG] --- LCA Resolution Results ---")
        print(f"[DEBUG] Container Path  : '{info.get('container_name', '<root>')}'")
        print(f"[DEBUG] Container Type  : {type(info['container']).__name__}")
        print(f"[DEBUG] Slice Indices   : {info['start_idx']} to {info['end_idx']}")
        print(f"[DEBUG] Captured Layers : {len(info['full_block'])} total")
        for n, l in info["full_block"]:
            print(f"    [LAYER MAP] {n} -> {type(l).__name__}")

    
    # =========================================================
    # THE CRITICAL FIX: Align capture hooks to the LCA boundaries
    # =========================================================
    is_sequential = isinstance(info['container'], (nn.Sequential, nn.ModuleList))
    
    if not is_sequential and info['container_name'] != "":
        # It's a complex/parallel block (e.g., InceptionBlock). Hook the container 
        # itself to capture the true global tensor sizes entering and exiting the block.
        lca_start_name = info['container_name']
        lca_end_name = info['container_name']
    else:
        # Standard sequential layout. Hook the explicit child boundaries.
        lca_start_name = f"{info['container_name']}.{info['first_layer_name']}" if info['container_name'] else info['first_layer_name']
        lca_end_name = f"{info['container_name']}.{info['last_layer_name']}" if info['container_name'] else info['last_layer_name']

    # Step 2: Capture activation entering the LCA boundaries
    print(f"\n[STEP 2] Simulating Forward Pass & Capturing LCA Activations...")
    if debug:
        print(f"[DEBUG] Hooking Start Node : '{lca_start_name}'")
        print(f"[DEBUG] Hooking End Node   : '{lca_end_name}'")
        print(f"[DEBUG] Dummy Input Shape  : {input_shape}")
        
    x, y_out, pre_params = _capture_preblock_activation(
        model, lca_start_name, lca_end_name, input_shape, info["conv_layers"], info["layer_type"], device, debug
    )
    # =========================================================# =========================================================

    # Topology guard: the surrogate replacement is defined for 4D (NCHW)
    # feature maps. Anything else (sequence models, 1D/3D tensors, flattened
    # activations) is skipped instead of silently corrupted.
    if x.ndim != 4 or y_out.ndim != 4:
        raise ValueError(
            f"unsupported topology for collapse: captured x.ndim={x.ndim}, "
            f"y_out.ndim={y_out.ndim} (expected 4D NCHW feature maps)."
        )

    if debug:
        print(f"[DEBUG] --- Activation Capture Success ---")
        print(f"[DEBUG] Input (x) shape entering LCA   : {tuple(x.shape)}")
        print(f"[DEBUG] Output (y) shape exiting LCA   : {tuple(y_out.shape)}")
        print(f"[DEBUG] Pre-collapse Parameter Count   : {pre_params:,}")

    # Step 3: Find next linear (prefer true execution order over DFS name order)
    print(f"\n[STEP 3] Probing graph for downstream classifier after '{end_layer_name}'...")
    try:
        exec_order = execution_order(model, input_shape, device=device)
    except Exception as e:
        if debug:
            print(f"[DEBUG] Execution-order probe failed ({e}); falling back to name-order scan.")
        exec_order = None
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug, order=exec_order)
    
    if debug:
        print(f"[DEBUG] --- Downstream Probe Results ---")
        print(f"[DEBUG] Next Linear Name : '{next_linear_name}'")
        print(f"[DEBUG] Next Linear Type : {type(next_linear_mod).__name__ if next_linear_mod else 'None'}")

    # Step 4: Analyze block output
    print(f"\n[STEP 4] Analyzing topological characteristics for replacement...")
    block_analysis = _analyze_block_output(
        model,
        info["full_block"],
        info["conv_layers"],
        info["named_layers"],
        info["end_idx"],
        info["layer_type"],
        x,
        y_out,  
        next_linear_mod,
        debug
    )
    
    if debug:
        print(f"[DEBUG] --- Feature Map Strategy ---")
        for k, v in block_analysis.items():
            print(f"    - {k:<25}: {v}")

    # Step 5: Replace block
    print(f"\n[STEP 5] Synthesizing surrogate block and fusing into computational graph...")
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
        debug,
        y_out=y_out,
        init=init,
    )
    
    print(f"\n[INFO] ✅ Framework successfully merged '{start_layer_name}' → '{end_layer_name}'")

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
    dry_run: bool = False,
    init: str = 'random'
) -> nn.Module:
    """
    Top-level API to collapse multiple blocks with the simple replacement policy.
    """
    print(f"\n[STEP] ===== Starting collapse_only process =====")
    print(f"[DEBUG] Device={device}, dry_run={dry_run}, handle_skips={handle_skips}, safe_param_reduction={safe_param_reduction}, init={init}")
    overall_start_ts = time.perf_counter()

    if init not in ('random', 'lsq'):
        raise ValueError(f"Unknown init mode '{init}'. Expected 'random' or 'lsq'.")

    if model is None:
        print(f"[STEP] Loading model from disk...")
        if not (model_weights_1 and model_class):
            raise ValueError("[ERROR] Must provide either an instantiated `model` or (`model_weights_1` + `model_class`).")

        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating model from class '{model_class.__name__}' with kwargs={model_kwargs}")
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to instantiate model class {model_class}: {e}")

        print(f"[INFO] Loading weights from file: {model_weights_1}")
        try:
            chk = torch.load(model_weights_1, map_location=device)
            state = chk.get('model', chk) if isinstance(chk, dict) else chk
            model.load_state_dict(state)
            print(f"[INFO] Weights successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model weights: {e}")
    else:
        print(f"[STEP] Using provided in-memory model instance ({model.__class__.__name__})")

    if debug:
        try:
            print(f"[DEBUG] Model layer statistics before collapse:\n{layer_stats(model)}")
        except Exception as e:
            print(f"[WARN] layer_stats() failed: {e}")

    print(f"[STEP] Creating deepcopy of model for safe modification...")
    model = deepcopy(model).to(device)
    model.eval()

    print(f"[STEP] Parsing compression set...")
    if compression_set is None:
        print("[WARN] compression_set is None or empty; skipping collapse.")
        return model

    collapse_map = {}
    if isinstance(compression_set, dict):
        if debug:
            print(f"[DEBUG] Detected compression_set as dict with {len(compression_set)} entries.")
        for k, v in compression_set.items():
            start, end = v
            if isinstance(start, tuple): start = start[0]
            if isinstance(end, tuple): end = end[0]
            collapse_map[k] = (start, end)
    else:
        if debug:
            print(f"[DEBUG] Detected compression_set as sequence with {len(compression_set)} pairs.")
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple): start = start[0]
            if isinstance(end, tuple): end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)

    model._collapsed_blocks = list(collapse_map.values())
    if debug:
        print(f"[DEBUG] Total collapse targets: {len(model._collapsed_blocks)}")

    pre_total = count_trainable_params(model)
    print(f"[INFO] Model parameter count before collapsing: {pre_total:,}")

    print(f"[STEP] Beginning block-wise collapsing...")
    success_count = 0
    failed_count = 0
    skipped_count = 0
    for name, (start, end) in collapse_map.items():
        block_start_ts = time.perf_counter()
        
        # [CRITICAL FIX] Guarantee single-layers are safely bypassed
        if start == end:
            print(f"[WARN] Skipping collapse task '{name}': Group contains only one layer ({start}). Must span at least two layers.")
            skipped_count += 1
            continue

        print(f"\n[INFO] Processing collapse task '{name}': {start} → {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping actual modification for this block.")
            continue

        try:
            print(f"[STEP] Calling _collapse_block('{start}', '{end}')")
            model = _collapse_block(model, start, end, input_shape, device=device, debug=debug, init=init)
            elapsed = time.perf_counter() - block_start_ts
            print(f"[INFO] ✅ Successfully collapsed block '{name}' ({start} → {end}) in {elapsed:.2f}s")
            success_count += 1
        except ValueError as e:
            # ISSUE #3 FIX: Filter out cross-branch collapse candidates
            if "Could not map start/end layers" in str(e):
                print(f"[WARN] ⚠ Collapse candidate '{name}' appears to be cross-branch (parallel paths).")
                print(f"[WARN]   Filtering out and skipping. Re-run region discovery to exclude these.")
                skipped_count += 1
                continue
            elif "unsupported topology" in str(e):
                # Topology guard: non-4D blocks cannot be surrogated; skip cleanly.
                print(f"[WARN] ⚠ Collapse candidate '{name}' skipped: {e}")
                skipped_count += 1
                continue
            else:
                print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
                if debug:
                    _log_exception(
                        f"Collapse task '{name}' failed",
                        e,
                        debug=True,
                        extra={"start": start, "end": end},
                    )
                failed_count += 1
        except Exception as e:
            print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
            if debug:
                _log_exception(
                    f"Collapse task '{name}' failed",
                    e,
                    debug=True,
                    extra={"start": start, "end": end},
                )
            failed_count += 1
               
        if handle_skips:
            print(f"[STEP] Patching skip connections (if any)...")
            try:
                patch_skip_connections(model)
                if debug: print(f"[DEBUG] Skip connections patched successfully.")
            except Exception as e:
                print(f"[WARN] Failed to patch skip connections: {e}")

        print(f"[STEP] Disabling in-place ReLUs for autograd safety...")
        try:
            disable_inplace_relu(model)
            if debug: print(f"[DEBUG] In-place ReLUs converted to out-of-place versions.")
        except Exception as e:
            print(f"[WARN] Failed to disable in-place ReLUs: {e}")

    print(f"\n[STEP] Wrapping pooling layers safely...")
    try:
        _wrap_pools_safe(model)
        if debug: print("[DEBUG] All pooling layers wrapped with _SafePool to prevent underflow errors.")
    except Exception as e:
        print(f"[WARN] Failed to wrap pools safely: {e}")

    post_total = count_trainable_params(model)
    total_elapsed = time.perf_counter() - overall_start_ts
    print(f"\n[STEP] ===== Collapse summary =====")
    print(f"[INFO] Parameters before: {pre_total:,}")
    print(f"[INFO] Parameters after : {post_total:,}")
    delta = pre_total - post_total
    print(f"[INFO] ΔParams = {delta:+,} (expected ≤ 0)")
    print(f"[INFO] Collapse counts: success={success_count}, failed={failed_count}, skipped={skipped_count}")
    print(f"[INFO] Total collapse runtime: {total_elapsed:.2f}s")

    if post_total > pre_total:
        print(f"[WARN] ⚠ Model gained parameters after collapsing! Investigate collapse policy or replacement logic.")

    if safe_param_reduction and delta < 0:
        print(f"[WARN] ⚠ Parameter count increased when safe_param_reduction=True — collapse may have failed silently.")

    print(f"[RESULT] ✅ collapse_only complete. Total targets: {len(collapse_map)} | success={success_count} | failed={failed_count} | skipped={skipped_count}")
    return model
