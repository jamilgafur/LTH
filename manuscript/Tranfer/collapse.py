# collapse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from uuid import uuid4
from typing import Optional, Sequence, Tuple, Dict, Any
from copy import deepcopy
import copy
from utils import count_trainable_params, layer_stats
import math

def _locate_and_prepare_block(model, start_layer_name, end_layer_name):
    print(f"[DEBUG] Locating and preparing block: start='{start_layer_name}', end:'{end_layer_name}'")

    # --- LCA resolution (unchanged) ---
    start_parts = start_layer_name.split('.') if start_layer_name else []
    end_parts = end_layer_name.split('.') if end_layer_name else []
    common_parts = []
    for a, b in zip(start_parts, end_parts):
        if a == b:
            common_parts.append(a)
        else:
            break
    lca_path = '.'.join(common_parts)

    if lca_path == "":
        start_container_name, _ = _get_container_and_subname(start_layer_name)
        end_container_name, _ = _get_container_and_subname(end_layer_name)
        if start_container_name == end_container_name and start_container_name != "":
            lca_path = start_container_name
            container = get_layer(model, lca_path)
        else:
            raise ValueError(
                "[ERROR] Start and end layers do not share a non-root common ancestor."
            )
    else:
        container = get_layer(model, lca_path)

    print(f"[DEBUG] Containers resolved → chosen LCA container: '{lca_path}'")

    named_layers = list(container.named_children())
    print(f"[DEBUG] Found {len(named_layers)} children in container '{lca_path or '<root>'}'")

    # --- locate child indices ---
    start_idx = end_idx = None
    for i, (child_name, _) in enumerate(named_layers):
        full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
        if start_idx is None and (
            start_layer_name == full_child_prefix
            or start_layer_name.startswith(full_child_prefix + ".")
        ):
            start_idx = i
        if end_idx is None and (
            end_layer_name == full_child_prefix
            or end_layer_name.startswith(full_child_prefix + ".")
        ):
            end_idx = i
        if start_idx is not None and end_idx is not None:
            break

    if start_idx is None or end_idx is None:
        raise ValueError("[ERROR] Could not map start/end layers into LCA container.")

    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    full_block = named_layers[start_idx:end_idx + 1]
    print(f"[DEBUG] Block slice length: {len(full_block)}")

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
    has_linear = any(isinstance(l, nn.Linear) for l in conv_layers)

    collapse_mode = "conv" if has_conv else "linear"

    print(
        f"[DEBUG] Block composition → "
        f"{sum(isinstance(l, nn.Conv2d) for l in conv_layers)} Conv2d, "
        f"{sum(isinstance(l, nn.Linear) for l in conv_layers)} Linear "
        f"(collapse_mode={collapse_mode})"
    )

    return {
        "container": container,
        "container_name": lca_path,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "conv_layers": conv_layers,
        "collapse_mode": collapse_mode,
        "first_layer": conv_layers[0],
        "last_layer": conv_layers[-1],
    }


# def _locate_and_prepare_block(model, start_layer_name, end_layer_name):
#     print(f"[DEBUG] Locating and preparing block: start='{start_layer_name}', end:'{end_layer_name}'")
#     # Compute lowest common ancestor (LCA) container path for the two layers.
#     start_parts = start_layer_name.split('.') if start_layer_name else []
#     end_parts = end_layer_name.split('.') if end_layer_name else []
#     common_parts = []
#     for a, b in zip(start_parts, end_parts):
#         if a == b:
#             common_parts.append(a)
#         else:
#             break
#     lca_path = '.'.join(common_parts)  # may be empty string if root
#     if lca_path == "":
#         # Fallback: if both layers are within the same immediate container (old behaviour),
#         # use that container. Otherwise refuse (collapsing across absolute root isn't supported here).
#         start_container_name, _ = _get_container_and_subname(start_layer_name)
#         end_container_name, _ = _get_container_and_subname(end_layer_name)
#         if start_container_name == end_container_name and start_container_name != "":
#             lca_path = start_container_name
#             if lca_path == "":
#                 container = model
#             else:
#                 container = get_layer(model, lca_path)
#         else:
#             raise ValueError(
#                 f"[ERROR] Start and end layers do not share a non-root common ancestor. "
#                 f"LCA would be root (''), which is not supported by this collapse routine. "
#                 f"start_container='{start_container_name}', end_container='{end_container_name}'"
#             )
#     else:
#         container = get_layer(model, lca_path)

#     print(f"[DEBUG] Containers resolved → chosen LCA container: '{lca_path}'")

#     named_layers = list(container.named_children())
#     print(f"[DEBUG] Found {len(named_layers)} children in container '{lca_path or '<root>'}'")

#     # Find which child indices contain the start and end targets (they may be nested inside children)
#     start_idx = end_idx = None
#     for i, (child_name, child_mod) in enumerate(named_layers):
#         # full prefix to compare with full layer paths in model.named_modules()
#         full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
#         # if target matches the child itself or is a descendant of the child
#         if start_idx is None and (start_layer_name == full_child_prefix or start_layer_name.startswith(full_child_prefix + ".")):
#             start_idx = i
#         if end_idx is None and (end_layer_name == full_child_prefix or end_layer_name.startswith(full_child_prefix + ".")):
#             end_idx = i
#         if start_idx is not None and end_idx is not None:
#             break

#     if start_idx is None or end_idx is None:
#         raise ValueError(f"[ERROR] Could not map start/end layers into children of LCA container '{lca_path}'. start_idx={start_idx}, end_idx={end_idx}")

#     if start_idx > end_idx:
#         # swap to ensure ordering
#         start_idx, end_idx = end_idx, start_idx

#     print(f"[DEBUG] Child indices in container '{lca_path}': start={start_idx}, end={end_idx}")

#     # Build full_block as the sequence of child modules (these child modules may themselves be blocks)
#     full_block = named_layers[start_idx:end_idx + 1]
#     print(f"[DEBUG] Block slice length (children count): {len(full_block)}")

#     # Collect any Conv2d/Linear modules inside these children (flattened, in order)
#     conv_layers = []
#     for nm, mod in full_block:
#         # if the child itself is a Conv2d/Linear, count it first
#         if isinstance(mod, (nn.Conv2d, nn.Linear)):
#             conv_layers.append(mod)
#         # then search deeper to preserve execution order (depth-first)
#         for sub_name, sub_mod in mod.named_modules():
#             # skip root (the child module itself) already considered
#             if sub_name == "":
#                 continue
#             if isinstance(sub_mod, (nn.Conv2d, nn.Linear)):
#                 conv_layers.append(sub_mod)

#     print(f"[DEBUG] Found {len(conv_layers)} Conv2d/Linear layers inside selected child range")

#     if not conv_layers:
#         raise ValueError("[ERROR] No Conv2d/Linear layers found in block to collapse.")

#     layer_type = type(conv_layers[0])
#     if not all(isinstance(l, layer_type) for l in conv_layers):
#         raise ValueError("[ERROR] Mixed layer types detected within block.")

#     print(f"[DEBUG] Uniform layer type across block: {layer_type.__name__}")

#     # return both the module and the path string (container_name) so downstream code can update correctly
#     return {
#         "container": container,
#         "container_name": lca_path,
#         "named_layers": named_layers,
#         "start_idx": start_idx,
#         "end_idx": end_idx,
#         "full_block": full_block,
#         "conv_layers": conv_layers,
#         "layer_type": layer_type,
#     }

def _build_and_replace_block(
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
):
    if debug:
        print(f"\n[STEP] Building replacement for collapsed block '{start_layer_name}'")
        print(f"[DEBUG] Analyzing info dict keys: {list(info.keys())}")
        print(f"[DEBUG] Target device: {device}")

    named_layers = info["named_layers"]
    # Prefer the container determined by the locator (LCA). If not present, fall back to original start container.
    start_container_name = info.get("container_name") or _get_container_and_subname(start_layer_name)[0]
    start_idx, end_idx = info["start_idx"], info["end_idx"]
    out_shape = block_analysis.get("out_shape")
    out_channels = block_analysis.get("out_channels")

    if out_shape is None or out_channels is None:
        raise RuntimeError("[ERROR] block_analysis missing required out_shape/out_channels")

    target_H, target_W = (out_shape[-2], out_shape[-1]) if len(out_shape) >= 4 else (1, 1)
    if debug:
        print(f"[DEBUG] Replacement target spatial size (HxW): {target_H}x{target_W}")

    if x is None or x.ndim < 2:
        raise RuntimeError("[ERROR] Invalid captured activation `x`.")
    in_channels = int(x.shape[1])
    if debug:
        print(f"[DEBUG] Replacement conv in_channels={in_channels}, out_channels={out_channels}")

    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    pool = nn.AdaptiveAvgPool2d((target_H, target_W))
    replacement = nn.Sequential(OrderedDict([
        ("conv_1x1", conv),
        ("adaptive_pool", pool),
    ]))
    if debug:
        print(f"[DEBUG] Built replacement Sequential:\n{replacement}")

    if debug:
        print(f"[DEBUG] Replacing children indices {start_idx}..{end_idx} in container '{start_container_name}'")
    updated_container = _replace_layers(named_layers, start_idx, end_idx, replacement)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Params before collapse: {pre_params:,}")
        print(f"[DEBUG] Params after  collapse: {post_params:,}")
        print(f"[INFO] ΔParams = {pre_params - post_params:+,}")

    try:
        dev = next((p.device for p in model.parameters()), torch.device('cpu'))
        rep_module = get_layer(model, start_container_name)
        child = None
        for nm, m in rep_module.named_children():
            if nm.startswith("collapsed_") or (isinstance(m, nn.Sequential) and "conv_1x1" in dict(m.named_children())):
                child = m
                break
        if child is None and isinstance(updated_container, nn.Sequential):
            child = updated_container[start_idx]

        if child is not None:
            with torch.no_grad():
                test_x = x.clone().to(dev)
                out = child(test_x)
                if debug:
                    print(f"[DEBUG] Replacement validation OK — output shape {tuple(out.shape)}")
        else:
            print(f"[WARN] Could not find inserted collapsed module for validation.")
    except Exception as e:
        print(f"[WARN] Replacement forward validation failed: {e}")
        quit()
    try:
        if debug:
            print(f"[STEP] Validating downstream after replacement...")
        _validate_downstream(model, start_container_name, start_idx, x, input_shape, next_linear_name, next_linear_mod, device, debug)
    except Exception as e:
        print(f"[WARN] Downstream validation failed: {e}")
        quit()
    try:
        if debug:
            print(f"[STEP] Performing corrective pooling (if needed)...")
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug)
    except Exception as e:
        print(f"[WARN] Corrective pool insertion failed: {e}")
        quit()

    if post_params > pre_params:
        print(f"[WARN] ⚠ Collapsed block increased parameter count — investigate collapse policy.")
        quit()

    if debug:
        print(f"[RESULT] Block replacement complete for '{start_layer_name}'.")

    return model


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _is_int_str(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Access layer via dot-separated path (supports Sequential indices). Empty -> model."""
    if layer_name == "":
        return model
    layer = model
    for part in layer_name.split('.'):
        layer = layer[int(part)] if _is_int_str(part) else getattr(layer, part)
    return layer


def _get_container_and_subname(layer_name: str) -> Tuple[str, str]:
    """Return (container_path, subname) from layer_name."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]

def _replace_layers(named_layers: Sequence[Tuple[str, nn.Module]], start_idx: int, end_idx: int, new_block: nn.Module) -> nn.Sequential:
    """Replace layers start_idx..end_idx inclusive in named_layers with new_block (Sequential)."""
    new_layers = []
    unique_suffix = uuid4().hex[:8]
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_layers.append((f"collapsed_{unique_suffix}", new_block))
        elif start_idx < i <= end_idx:
            continue
        else:
            new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))


def _update_container(model: nn.Module, container_path: str, new_container: nn.Module):
    """Replace the module at `container_path` in `model` with `new_container`."""
    if container_path == "":
        raise ValueError("Refusing to replace root module with container.")
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_container
    else:
        setattr(parent, last, new_container)


def disable_inplace_relu(model: nn.Module):
    """Replace inplace ReLU with out-of-place ReLU to avoid in-place autograd issues."""
    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU) and getattr(module, "inplace", False):
            container, subname = _get_container_and_subname(name)
            parent = get_layer(model, container) if container != "" else model
            new_relu = nn.ReLU(inplace=False)
            if _is_int_str(subname):
                parent[int(subname)] = new_relu
            else:
                setattr(parent, subname, new_relu)
            replaced += 1
    if replaced:
        print(f"[INFO] Replaced {replaced} in-place ReLU(s) with out-of-place variants.")


def _simulate_input_hook(model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward with a dummy input and capture the activation that is
    the *input to* the target layer (registered as forward hook on the target layer).
    Returns (dummy_input, captured_activation).
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    target_module = get_layer(model, target_layer_path)

    captured = {}
    def hook(module, inp, out):
        # store a detached copy of the input to the target module
        captured['in'] = inp[0].detach()
    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        handle.remove()
    if 'in' not in captured:
        raise RuntimeError(f"Failed to capture activation at {target_layer_path}.")
    return dummy_input, captured['in']


# -----------------------------------------------------------------------------
# Skip connection patcher
# -----------------------------------------------------------------------------

def patch_skip_connections(model: nn.Module):
    """
    Patch module forwards for blocks that include `shortcut` so that if shapes
    mismatch the shortcut is safely ignored instead of raising on addition.
    """
    for name, module in model.named_modules():
        # treat modules that look like residual blocks with .block and .shortcut
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module) and hasattr(module, 'block'):
            orig_forward = getattr(module, 'forward', None)
            if orig_forward is None:
                continue

            def make_patched_forward(orig_fwd):
                def new_forward(self, x):
                    # compute main path
                    out = self.block(x)
                    # attempt shortcut; if fails or shape mismatch, skip it
                    try:
                        sc = self.shortcut(x)
                        if out.shape != sc.shape:
                            # shapes differ: skip addition
                            return F.relu(out)
                        return F.relu(out + sc)
                    except Exception:
                        return F.relu(out)
                return new_forward

            module.forward = make_patched_forward(orig_forward).__get__(module)
            print(f"[PATCH] Patched residual block forward: {name}")


# -----------------------------------------------------------------------------
# Core collapse of a single block (simple replacement)
# -----------------------------------------------------------------------------


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
    print(f"\n[INFO] ===== Collapsing block: {start_layer_name} → {end_layer_name} =====")

    # Step 1: Locate block
    print(f"[STEP 1] Locating block boundaries...")
    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name)
    if debug:
        print(f"[DEBUG] Located block with {len(info['full_block'])} layers")
        for n, l in info["full_block"]:
            print(f"    [LAYER] {n}: {l}")

    # Step 2: Capture activation entering the start layer
    print(f"[STEP 2] Capturing activation before start layer '{start_layer_name}'...")
    x, pre_params = _capture_preblock_activation(
        model, start_layer_name, input_shape, info["conv_layers"], info["layer_type"], device, debug
    )
    if debug:
        print(f"[DEBUG] Input activation shape entering block: {tuple(x.shape)}")

    # Step 3: Find next linear
    print(f"[STEP 3] Searching for next linear layer after '{end_layer_name}'...")
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)
    if debug:
        print(f"[DEBUG] Next linear layer: {next_linear_name} -> {type(next_linear_mod).__name__ if next_linear_mod else 'None'}")

    # Step 4: Analyze block output
    print(f"[STEP 4] Analyzing block output characteristics...")
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
        print(f"[DEBUG] Block output analysis result:")
        for k, v in block_analysis.items():
            print(f"    {k}: {v}")

    # Step 5: Replace block
    print(f"[STEP 5] Rebuilding and replacing collapsed block...")
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
    print(f"[INFO] ✅ Collapse complete for block '{start_layer_name}' → '{end_layer_name}'")

    return model

def _capture_preblock_activation(model, start_layer_name, input_shape, conv_layers, layer_type, device, debug):
    print(f"[DEBUG] Attempting to capture activation before '{start_layer_name}' using simulation hook...")
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        if debug:
            print(f"[DEBUG] Successfully captured activation → shape: {tuple(x.shape)}")
    except Exception as e:
        print(f"[WARN] Hook failed: {e}")
        print(f"[WARN] Falling back to dummy tensor initialization.")
        if layer_type == nn.Conv2d:
            in_ch = conv_layers[0].in_channels if hasattr(conv_layers[0], 'in_channels') else input_shape[1]
            H, W = input_shape[-2], input_shape[-1]
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            in_feat = conv_layers[0].in_features if hasattr(conv_layers[0], 'in_features') else input_shape[1]
            x = torch.randn(1, in_feat, device=device)
        print(f"[DEBUG] Created fallback tensor of shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Total trainable parameters before collapse: {pre_params:,}")

    print(f"[DEBUG] Activation capture complete.")
    return x, pre_params

def _find_next_linear(model, end_layer_name, debug):
    if debug:
        print(f"\n[STEP] Searching for next nn.Linear after '{end_layer_name}'...")
    modules_list = list(model.named_modules())
    idx_end_global = None

    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            if debug:
                print(f"[DEBUG] Exact layer match found at index {i}: {n} ({type(m).__name__})")
            break

    if idx_end_global is None:
        for i, (n, m) in enumerate(modules_list):
            if n.endswith(end_layer_name):
                idx_end_global = i
                if debug:
                    print(f"[DEBUG] Fallback: found partial match at index {i}: {n}")
                break

    next_linear_name = None
    next_linear_mod = None
    if idx_end_global is not None:
        if debug:
            print(f"[DEBUG] Scanning forward from index {idx_end_global + 1} for next Linear...")
        for j in range(idx_end_global + 1, len(modules_list)):
            n, m = modules_list[j]
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Found Linear layer ahead: {n} ({m})")
                break

    if next_linear_mod is None:
        if debug:
            print(f"[DEBUG] No Linear found after {end_layer_name}. Searching globally...")
        for n, m in modules_list:
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Global fallback Linear found: {n} ({m})")
                break

    if next_linear_mod is None:
        print(f"[WARN] No Linear layer found in model after '{end_layer_name}'")

    if debug:
        print(f"[RESULT] Next Linear detected → name='{next_linear_name}', module={next_linear_mod}")

    return next_linear_name, next_linear_mod


def _analyze_block_output(model, full_block, conv_layers, named_layers, end_idx, layer_type, x, next_linear_mod, debug):
    if debug:
        print(f"\n[STEP] Analyzing output of collapsed block ({len(full_block)} layers)...")
        print(f"[DEBUG] Input tensor shape before block: {tuple(x.shape)}")
        print(f"[DEBUG] Running forward pass through block layers:")

    with torch.no_grad():
        y = x.clone()
        last_conv = None
        for idx, (name, layer) in enumerate(full_block):
            if debug:
                print(f"    [DEBUG] Layer {idx+1}/{len(full_block)}: {name} ({layer.__class__.__name__}) input={tuple(y.shape)}")
            y = layer(y)
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
            if debug:
                print(f"        └── output shape: {tuple(y.shape)}")

    out_shape = tuple(y.shape)
    out_channels = (
        last_conv.out_channels if last_conv is not None
        else (conv_layers[-1].out_channels if layer_type == nn.Conv2d else None)
    )

    if debug:
        print(f"[DEBUG] Final block output shape: {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")

    pool_layer = next((m for _, m in reversed(full_block)
                       if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)
    if debug:
        if pool_layer is not None:
            print(f"[DEBUG] Detected pool in original block: {type(pool_layer).__name__}")
        else:
            print(f"[DEBUG] No explicit pooling layer detected inside block.")

    linear_in_features_heuristic = next_linear_mod.in_features if next_linear_mod else None
    if debug:
        print(f"[DEBUG] Heuristic next linear in_features = {linear_in_features_heuristic}")

    adaptive_pool_to_use = None
    if layer_type == nn.Conv2d and linear_in_features_heuristic is not None and out_channels:
        expected_hw = max(1, linear_in_features_heuristic // out_channels)
        cur_H, cur_W = out_shape[-2], out_shape[-1]
        cur_hw = cur_H * cur_W
        if debug:
            print(f"[DEBUG] Comparing spatial dims: expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})")
        if cur_hw != expected_hw:
            target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
            target_W = max(1, expected_hw // target_H)
            adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))
            if debug:
                print(f"[DEBUG] Suggest AdaptiveAvgPool2d({target_H},{target_W}) to reconcile linear in_features mismatch.")

    shortcut_out_channels = None
    for nm, mod in model.named_modules():
        if hasattr(mod, 'shortcut') and isinstance(mod.shortcut, nn.Module):
            first_conv = next((m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)), None)
            if first_conv is not None:
                shortcut_out_channels = first_conv.out_channels
                if debug:
                    print(f"[DEBUG] Found shortcut conv → out_channels={shortcut_out_channels}")
                break

    if debug:
        print(f"[RESULT] Block analysis complete:")
        print(f"         out_shape={out_shape}")
        print(f"         out_channels={out_channels}")
        print(f"         has_pool={pool_layer is not None}")
        print(f"         adaptive_pool_to_use={adaptive_pool_to_use}")
        print(f"         shortcut_out_channels={shortcut_out_channels}")

    return {
        "out_shape": out_shape,
        "out_channels": out_channels,
        "pool_layer": pool_layer,
        "adaptive_pool_to_use": adaptive_pool_to_use,
        "shortcut_out_channels": shortcut_out_channels,
        "linear_in_features_heuristic": linear_in_features_heuristic,
    }


def _validate_downstream(
    model: nn.Module,
    start_container_name: str,
    start_idx: int,
    pre_activation: torch.Tensor,
    input_shape: Tuple[int, ...],
    next_linear_name: Optional[str] = None,
    next_linear_mod: Optional[nn.Module] = None,
    device: str = 'cpu',
    debug: bool = False
) -> None:
    """
    Validates downstream modules immediately after inserting a collapsed replacement block.
    If a downstream module fails during a forward pass (e.g. pooling mismatch),
    wraps it in _SafePool or replaces it with Identity to preserve model functionality.
    """
    print(f"\n[STEP] ===== Starting _validate_downstream for '{start_container_name}' =====")
    print(f"[DEBUG] start_idx={start_idx}, device={device}, has_next_linear={next_linear_name is not None}")

    # Retrieve target container
    try:
        container = get_layer(model, start_container_name)
    except Exception as e:
        print(f"[WARN] Could not access container '{start_container_name}': {e}")
        return

    children = list(container.named_children())
    if not children:
        print(f"[DEBUG] Container '{start_container_name}' has no children; skipping downstream validation.")
        return

    # Find collapsed/inserted child index
    collapsed_idx = None
    print(f"[STEP] Searching for inserted collapsed module within '{start_container_name}'...")
    for i, (nm, m) in enumerate(children):
        if nm.startswith("collapsed_") or (
            isinstance(m, nn.Sequential)
            and any(k in dict(m.named_children()) for k in ("conv_1x1", "adaptive_pool"))
        ):
            collapsed_idx = i
            print(f"[DEBUG] Found inserted block candidate at index {i}: '{nm}'")
            break

    if collapsed_idx is None:
        # fallback search: direct 1x1 conv
        for i, (nm, m) in enumerate(children):
            if isinstance(m, nn.Conv2d) and getattr(m, "kernel_size", None) == (1, 1):
                collapsed_idx = i
                print(f"[DEBUG] Found replacement candidate (Conv2d 1x1) at index {i}: '{nm}'")
                break

    if collapsed_idx is None:
        print(f"[WARN] Could not identify inserted module inside '{start_container_name}'. Skipping validation.")
        return

    inserted_mod = children[collapsed_idx][1]
    print(f"[STEP] Validating inserted module '{inserted_mod.__class__.__name__}' at index {collapsed_idx}...")

    # Run through inserted module
    try:
        with torch.no_grad():
            dev = next((p.device for p in model.parameters()), torch.device('cpu'))
            t = pre_activation.clone().to(dev)
            t = inserted_mod(t)
        print(f"[DEBUG] Inserted module forward successful, output shape: {tuple(t.shape)}")
    except Exception as e:
        print(f"[WARN] Forward pass through inserted module failed: {e}")
        return

    # Validate next modules downstream
    print(f"[STEP] Scanning immediate downstream modules for shape or runtime errors...")
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            t = mod(t)
            if debug:
                print(f"[DEBUG] Downstream '{start_container_name}.{nm}' executed successfully, output shape: {tuple(t.shape)}")
        except Exception as e:
            print(f"[WARN] Downstream module '{start_container_name}.{nm}' raised exception: {e}")
            print(f"[STEP] Replacing problematic module '{nm}' with safe alternative...")

            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
                print(f"[INFO] Replaced with _SafePool wrapper.")
            else:
                safe = nn.Identity()
                print(f"[INFO] Replaced with Identity() to bypass invalid operation.")

            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)

            print(f"[DEBUG] Replacement applied to '{start_container_name}.{nm}' ({safe.__class__.__name__}).")
            return  # stop after first fix

        # detect zero-spatial output
        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            print(f"[WARN] Module '{start_container_name}.{nm}' produced zero spatial dimensions. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            print(f"[DEBUG] Zero-dimension fix applied to '{start_container_name}.{nm}' with _SafePool.")
            return

    print(f"[RESULT] ✅ Downstream validation for '{start_container_name}' completed successfully.")



def _insert_corrective_pool(
    model: nn.Module,
    next_linear_name: str,
    input_shape: Tuple[int, ...],
    debug: bool = False
) -> nn.Module:
    """
    If a Linear layer’s expected input size does not match the actual flattened activation
    shape feeding it, replace it with a corrected Linear layer using the actual size.
    """
    print(f"\n[STEP] ===== Starting _insert_corrective_pool =====")
    print(f"[DEBUG] next_linear_name='{next_linear_name}', input_shape={input_shape}")

    if next_linear_name is None:
        print(f"[DEBUG] No next_linear_name provided; skipping corrective pooling step.")
        return model

    # Locate the linear layer
    try:
        next_linear_mod = get_layer(model, next_linear_name)
        print(f"[INFO] Located next linear layer: '{next_linear_name}' ({next_linear_mod.__class__.__name__})")
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not locate next linear '{next_linear_name}': {e}")

    expected = next_linear_mod.in_features
    dev = next((p.device for p in model.parameters()), torch.device('cpu'))

    # Capture the activation feeding that linear
    print(f"[STEP] Capturing activation entering '{next_linear_name}' to compute true flattened size...")
    try:
        probe_shape = (1,) + tuple(input_shape[1:])
        _, cap = _simulate_input_hook(
            model,
            next_linear_name,
            probe_shape,
            device=str(dev)
        )

        print(f"[DEBUG] Activation capture succeeded. Activation shape: {tuple(cap.shape)}")
    except Exception as e:
        print(f"[WARN] Failed to capture activation for '{next_linear_name}': {e}")
        raise

    flat_actual = cap.view(cap.size(0), -1).size(1)
    print(f"[INFO] Linear layer expected in_features={expected}, actual flattened={flat_actual}")

    if flat_actual == expected:
        print(f"[DEBUG] Linear input size matches expected. No corrective action needed.")
        return model

    # Replace mismatched Linear
    print(f"[STEP] Replacing mismatched Linear '{next_linear_name}' ({expected} → {flat_actual})...")
    parent_path, child_name = (
        next_linear_name.rsplit('.', 1)
        if '.' in next_linear_name
        else ("", next_linear_name)
    )

    if parent_path == "":
        raise RuntimeError("[ERROR] Cannot safely replace a top-level Linear without parent container context.")

    new_linear = nn.Linear(
        flat_actual,
        next_linear_mod.out_features,
        bias=(next_linear_mod.bias is not None)
    )

    parent_mod = get_layer(model, parent_path)
    if isinstance(parent_mod, nn.Sequential):
        new_od = OrderedDict()
        for n, m in parent_mod.named_children():
            new_od[n] = new_linear if n == child_name else m
        _update_container(model, parent_path, nn.Sequential(new_od))
    else:
        setattr(parent_mod, child_name, new_linear)

    print(f"[WARN] ⚠ Corrected Linear '{next_linear_name}' in_features updated from {expected} to {flat_actual}")
    print(f"[RESULT] ✅ Corrective pool insertion completed for '{next_linear_name}'")
    return model

# -----------------------------------------------------------------------------
# Top-level multi-block collapse function (flexible API)
# -----------------------------------------------------------------------------
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
    print(f"\n[STEP] ===== Starting collapse_only process =====")
    print(f"[DEBUG] Device={device}, dry_run={dry_run}, handle_skips={handle_skips}, safe_param_reduction={safe_param_reduction}")

    # Load or use provided model
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

    # Inspect model
    if debug:
        try:
            print(f"[DEBUG] Model layer statistics before collapse:\n{layer_stats(model)}")
        except Exception as e:
            print(f"[WARN] layer_stats() failed: {e}")

    # Make a deep copy to avoid modifying the original model
    print(f"[STEP] Creating deepcopy of model for safe modification...")
    model = deepcopy(model).to(device)
    model.eval()

    # Normalize compression_set
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
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[k] = (start, end)
            if debug:
                print(f"    [DEBUG] Added mapping: {k} = ({start} → {end})")
    else:
        if debug:
            print(f"[DEBUG] Detected compression_set as sequence with {len(compression_set)} pairs.")
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)
            if debug:
                print(f"    [DEBUG] Added mapping: collapse_{i} = ({start} → {end})")

    # Store collapsed blocks for reference
    model._collapsed_blocks = list(collapse_map.values())
    if debug:
        print(f"[DEBUG] Total collapse targets: {len(model._collapsed_blocks)}")
        for idx, (s, e) in enumerate(model._collapsed_blocks):
            print(f"    [BLOCK {idx}] {s} → {e}")

    # Track parameters
    pre_total = count_trainable_params(model)
    print(f"[INFO] Model parameter count before collapsing: {pre_total:,}")

    # Process each block in sequence
    print(f"[STEP] Beginning block-wise collapsing...")
    for name, (start, end) in collapse_map.items():
        print(f"\n[INFO] Processing collapse task '{name}': {start} → {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping actual modification for this block.")
            continue

        try:
            print(f"[STEP] Calling _collapse_block('{start}', '{end}')")
            model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)
            print(f"[INFO] ✅ Successfully collapsed block '{name}' ({start} → {end})")
        except Exception as e:
            print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
            quit()

        if handle_skips:
            print(f"[STEP] Patching skip connections (if any)...")
            try:
                patch_skip_connections(model)
                if debug:
                    print(f"[DEBUG] Skip connections patched successfully.")
            except Exception as e:
                print(f"[WARN] Failed to patch skip connections: {e}")

        print(f"[STEP] Disabling in-place ReLUs for autograd safety...")
        try:
            disable_inplace_relu(model)
            if debug:
                print(f"[DEBUG] In-place ReLUs converted to out-of-place versions.")
        except Exception as e:
            print(f"[WARN] Failed to disable in-place ReLUs: {e}")

    # Safe wrapping of pooling layers
    print(f"\n[STEP] Wrapping pooling layers safely...")
    try:
        _wrap_pools_safe(model)
        if debug:
            print("[DEBUG] All pooling layers wrapped with _SafePool to prevent underflow errors.")
    except Exception as e:
        print(f"[WARN] Failed to wrap pools safely: {e}")

    # Post-collapse summary
    post_total = count_trainable_params(model)
    print(f"\n[STEP] ===== Collapse summary =====")
    print(f"[INFO] Parameters before: {pre_total:,}")
    print(f"[INFO] Parameters after : {post_total:,}")
    delta = pre_total - post_total
    print(f"[INFO] ΔParams = {delta:+,} (expected ≤ 0)")

    if post_total > pre_total:
        print(f"[WARN] ⚠ Model gained parameters after collapsing! Investigate collapse policy or replacement logic.")

    if safe_param_reduction and delta < 0:
        print(f"[WARN] ⚠ Parameter count increased when safe_param_reduction=True — collapse may have failed silently.")

    print(f"[RESULT] ✅ collapse_only complete. Total collapsed blocks: {len(collapse_map)}")
    return model


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Safe pooling wrapper (prevents underflow crashes)
# -----------------------------------------------------------------------------

class _SafePool(nn.Module):
    """
    Wrapper that attempts to apply the wrapped pooling module; if the input
    spatial dimensions are too small or the pool raises, we fall back safely:
      - For non-adaptive pools: if kernel > input dim, uses AdaptiveAvgPool2d to
        produce a minimal valid output (>=1).
      - If anything else fails, returns the input (identity).
    This avoids 'Output size is too small' runtime errors.
    """
    def __init__(self, pool_module: nn.Module):
        super().__init__()
        self.pool = pool_module

    def forward(self, x):
        # guard shape sanity
        try:
            H, W = x.shape[-2], x.shape[-1]
        except Exception:
            # not a 4D tensor (some unexpected case) -> try to apply pool and catch exceptions
            try:
                return self.pool(x)
            except Exception:
                return x

        try:
            # For standard pools, check kernel size
            if isinstance(self.pool, (nn.MaxPool2d, nn.AvgPool2d)):
                k = self.pool.kernel_size
                if isinstance(k, tuple):
                    kh, kw = k
                else:
                    kh = kw = k
                # if kernel/stride would underflow, use adaptive avg pool to safe size
                if kh > H or kw > W or H <= 0 or W <= 0:
                    # choose a safe target HxW (at least 1)
                    target_H = max(1, min(H, kh) if H > 0 else 1)
                    target_W = max(1, min(W, kw) if W > 0 else 1)
                    return F.adaptive_avg_pool2d(x, (target_H, target_W))

            # Try to apply original pool
            out = self.pool(x)

            # post-check: if shape became invalid, return identity
            if out.shape[-2] < 1 or out.shape[-1] < 1:
                return x
            return out
        except Exception:
            # Any failure -> safe fallback
            return x


def _wrap_pools_safe(module: nn.Module):
    """
    Recursively replace pooling modules in `module` with _SafePool wrappers.
    This mutates the module in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
            safe = _SafePool(child)
            parent = module
            try:
                setattr(parent, name, safe)
            except Exception:
                try:
                    idx = int(name)
                    parent[idx] = safe
                except Exception:
                    setattr(parent, name, safe)
        else:
            _wrap_pools_safe(child)
