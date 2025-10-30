# collapse.py
import copy
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_trainable_params, layer_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
    """Access layer via dot-separated path (supports Sequential indices)."""
    if layer_name == "":
        return model
    layer = model
    for part in layer_name.split("."):
        layer = layer[int(part)] if _is_int_str(part) else getattr(layer, part)
    return layer


def _set_module_by_path(model: nn.Module, module_path: str, new_module: nn.Module):
    """Replace module at dot-separated path (supports Sequential indices)."""
    if module_path == "":
        raise ValueError("Cannot replace root module")
    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _get_container_and_subname(layer_name: str) -> Tuple[str, str]:
    """Return (container_path, subname) from layer_name."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split(".")
    return ".".join(parts[:-1]), parts[-1]


def _find_layer_indices(
    named_layers: Sequence[Tuple[str, nn.Module]],
    start_layer_name: str,
    end_layer_name: str,
) -> Tuple[Optional[int], Optional[int]]:
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx


def _replace_layers(
    named_layers: Sequence[Tuple[str, nn.Module]],
    start_idx: int,
    end_idx: int,
    new_block: nn.Module,
) -> nn.Sequential:
    """Replace layers start_idx..end_idx inclusive in named_layers with new_block."""
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
        raise ValueError("Cannot replace root module")
    parts = container_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_container
    else:
        setattr(parent, last, new_container)


# -----------------------------------------------------------------------------
# ReLU / skip helpers
# -----------------------------------------------------------------------------


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


def patch_skip_connections(model: nn.Module):
    """Patch residual block forwards to safely ignore mismatched shortcut shapes."""
    for name, module in model.named_modules():
        if hasattr(module, "shortcut") and isinstance(module.shortcut, nn.Module) and hasattr(module, "block"):
            orig_forward = getattr(module, "forward", None)
            if orig_forward is None:
                continue

            def make_patched_forward(orig_fwd):
                def new_forward(self, x):
                    out = self.block(x)
                    try:
                        sc = self.shortcut(x)
                        if out.shape != sc.shape:
                            return F.relu(out)
                        return F.relu(out + sc)
                    except Exception:
                        return F.relu(out)

                return new_forward

            module.forward = make_patched_forward(orig_forward).__get__(module)
            print(f"[PATCH] Patched residual block forward: {name}")


# -----------------------------------------------------------------------------
# Forward hook / shape tracing
# -----------------------------------------------------------------------------


def _simulate_input_hook( model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device="cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Capture the input activation to a target layer using a dummy input."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    target_module = get_layer(model, target_layer_path)
    captured = {}

    def hook(module, inp, out):
        captured["in"] = inp[0].detach()

    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        handle.remove()

    if "in" not in captured:
        raise RuntimeError(f"Failed to capture activation at {target_layer_path}.")
    return dummy_input, captured["in"]


def _trace_block_shapes(named_layers, input_tensor_or_shape, device, debug):
    """Trace shapes through layers (include initial input shape)."""
    if isinstance(input_tensor_or_shape, torch.Tensor):
        x = input_tensor_or_shape.to(device)
    else:
        x = torch.zeros(input_tensor_or_shape).to(device)

    shapes = []
    # record initial input shape BEFORE applying any layers
    shapes.append(("input", tuple(x.shape)))
    if debug:
        print("[DEBUG] Forwarding through block layers for shape tracing:")
        print(f"   -> Initial input shape = {tuple(x.shape)}")

    for name, layer in named_layers:
        try:
            x = layer(x)
            shapes.append((name, tuple(x.shape)))
            if debug:
                print(f"   -> After {layer.__class__.__name__:<22}: shape = {tuple(x.shape)}")
        except Exception as e:
            print(f"[ERROR] Shape tracing failed at {name}: {e}")
            raise
    return {"final": tuple(x.shape), "list": shapes}


def _get_submodule(model: nn.Module, target: str):
    """Retrieve a submodule by dotted path. Returns model if target==''."""
    if not target:
        return model
    current = model
    for attr in target.split("."):
        if attr.isdigit():
            current = current[int(attr)]
        else:
            current = getattr(current, attr)
    return current


# -----------------------------------------------------------------------------
# Collapsed block builders
# -----------------------------------------------------------------------------

def fix_conv_in_channels(layers):
    """
    Adjusts Conv2d layers so that in_channels match the actual input channels.
    """
    prev_out_ch = None
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Conv2d):
            if prev_out_ch is not None and layer.in_channels != prev_out_ch:
                print(f"[DEBUG] Adjusting Conv2d layer {i}: {layer.in_channels} → {prev_out_ch}")
                layer.in_channels = prev_out_ch
                # Recreate weight to match new in_channels
                layer.weight = nn.Parameter(torch.randn(layer.out_channels, layer.in_channels, *layer.kernel_size))
            prev_out_ch = layer.out_channels
        elif isinstance(layer, nn.BatchNorm2d):
            prev_out_ch = layer.num_features
    return layers


def _absorb_pools_if_needed(named_layers, traced_shapes=None, debug=False):
    """
    Checks for pool layers that can be absorbed or moved.
    Returns:
        adjusted_layers: list of modules (or (name,module)) after absorption
        pool_used: the pool layer absorbed (if any)
    """
    adjusted_layers = []
    pool_used = None
    current_out_ch = None  # Track output channels for Conv/BN
    
    for i, item in enumerate(named_layers):
        if isinstance(item, tuple) and len(item) == 2:
            name, module = item
        else:
            name, module = None, item

        # Track Conv output channels
        if isinstance(module, nn.Conv2d):
            current_out_ch = module.out_channels
            if debug:
                print(f"[DEBUG][_absorb_pools] Conv2d {name or i} out_channels={current_out_ch}")

        # Track BN and ensure it matches current channels
        elif isinstance(module, nn.BatchNorm2d):
            if current_out_ch is not None and module.num_features != current_out_ch:
                if debug:
                    print(f"[WARN][_absorb_pools] BN {name or i} num_features={module.num_features} != expected {current_out_ch}")
            current_out_ch = module.num_features

        # Handle pooling layers
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            pool_used = module
            if debug:
                print(f"[DEBUG][_absorb_pools] Found pool layer {name or i}: {module}")
            # Optionally, modify or absorb pool logic here
            # For now, just track it and add to adjusted_layers

        adjusted_layers.append(module if name is None else (name, module))

    if debug:
        print(f"[DEBUG][_absorb_pools] final adjusted_layers count = {len(adjusted_layers)}")
        if pool_used is not None:
            print(f"[DEBUG][_absorb_pools] pool_used = {pool_used}")

    return adjusted_layers, pool_used


import torch
import torch.nn as nn
from collections import OrderedDict


# ----------------------------------------------------------------------------- 
# Updated methods with fixes and more debugging
# -----------------------------------------------------------------------------

def _perform_collapse(block_layers, runtime_input, traced_shapes=None, device="cpu", debug=True):
    """
    Collapse a block of layers while preserving input/output channels and spatial size.
    Skips pooling layers that would reduce H/W below 1.
    """
    collapsed_layers = []
    x = runtime_input.clone().to(device)
    in_channels = x.shape[1]

    if debug:
        print(f"[DEBUG][_perform_collapse] Starting collapse: {len(block_layers)} layers")
        print(f"[DEBUG][_perform_collapse] Input shape: {tuple(x.shape)}")

    for i, (name, layer) in enumerate(block_layers):
        if debug:
            print(f"[DEBUG][_perform_collapse] Layer {i}: {name} ({type(layer).__name__}) input_ch={in_channels}")

        # Skip pooling if it would reduce spatial size to zero
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            H, W = x.shape[2], x.shape[3]
            if H <= 1 or W <= 1:
                if debug:
                    print(f"[DEBUG] Skipping pool '{name}' to avoid zero-size output (H={H}, W={W})")
                continue

        # Adjust Conv2d in_channels if needed
        if isinstance(layer, nn.Conv2d) and layer.in_channels != in_channels:
            if debug:
                print(f"[WARN] Adjusting Conv2d '{name}' in_channels {layer.in_channels} -> {in_channels}")
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None),
                padding_mode=layer.padding_mode
            ).to(device)
            # Copy overlapping weights
            min_ic = min(layer.weight.shape[1], in_channels)
            new_conv.weight.data[:, :min_ic, :, :] = layer.weight.data[:, :min_ic, :, :]
            if layer.bias is not None:
                new_conv.bias.data = layer.bias.data.clone()
            layer = new_conv
            in_channels = layer.out_channels

        # Adjust BatchNorm2d features if needed
        elif isinstance(layer, nn.BatchNorm2d) and layer.num_features != in_channels:
            if debug:
                print(f"[WARN] Adjusting BatchNorm2d '{name}' num_features {layer.num_features} -> {in_channels}")
            new_bn = nn.BatchNorm2d(in_channels).to(device)
            min_feat = min(layer.num_features, in_channels)
            new_bn.weight.data[:min_feat] = layer.weight.data[:min_feat]
            new_bn.bias.data[:min_feat] = layer.bias.data[:min_feat]
            new_bn.running_mean[:min_feat] = layer.running_mean[:min_feat]
            new_bn.running_var[:min_feat] = layer.running_var[:min_feat]
            layer = new_bn

        # Forward dummy input to track shape
        with torch.no_grad():
            x = layer(x)

        collapsed_layers.append(layer)
        if debug:
            print(f"[DEBUG][_perform_collapse] After '{name}': shape = {tuple(x.shape)}")

    collapsed_block = nn.Sequential(*collapsed_layers)
    if debug:
        print(f"[INFO][_perform_collapse] Collapse complete. Out channels={in_channels}, Out shape={tuple(x.shape)}")

    return collapsed_block, in_channels


def _build_collapsed_block_with_checks(named_layers, input_activation, device="cpu", debug=True):
    """
    Build collapsed block safely with first Conv2d and pooling checks.
    """
    layers = []
    x = input_activation.clone().to(device)

    for i, (name, layer) in enumerate(named_layers):
        # Adjust first Conv2d in_channels
        if isinstance(layer, nn.Conv2d) and len(layers) == 0 and layer.in_channels != x.shape[1]:
            if debug:
                print(f"[DEBUG] Adjusting first Conv2d '{name}' in_channels {layer.in_channels} -> {x.shape[1]}")
            layer = nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None)
            ).to(device)
        else:
            layer = copy.deepcopy(layer).to(device)

        # Prevent pooling from reducing spatial size to zero
        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            H, W = x.shape[2], x.shape[3]
            if H <= 1 or W <= 1:
                if debug:
                    print(f"[DEBUG] Skipping pool '{name}' to avoid zero-size output (H={H}, W={W})")
                continue

        layers.append((name, layer))
        with torch.no_grad():
            x = layer(x)
        if debug:
            print(f"[DEBUG] After '{name}' ({type(layer).__name__}): shape = {tuple(x.shape)}")

    return nn.Sequential(OrderedDict(layers))


def _replace_block_in_container(container, named_layers, collapsed_block):
    """
    Replace layers in container with collapsed block.
    """
    start_name, end_name = named_layers[0][0], named_layers[-1][0]

    if isinstance(container, nn.Sequential):
        children = list(container.named_children())
    elif isinstance(container, nn.ModuleList):
        children = [(str(i), container[i]) for i in range(len(container))]
    else:
        children = list(container.named_children())

    name_to_idx = {n: i for i, (n, _) in enumerate(children)}
    start_idx = name_to_idx[start_name]
    end_idx = name_to_idx[end_name]

    updated_container = _replace_layers(children, start_idx, end_idx, collapsed_block)
    # Assign back safely
    for i, (name, module) in enumerate(updated_container.named_children()):
        if isinstance(container, nn.Sequential) or isinstance(container, nn.ModuleList):
            container[i] = module
        else:
            setattr(container, name, module)



# -----------------------------------------------------------------------------
# Block-level collapse
# -----------------------------------------------------------------------------


def _get_block_layers(model, start, end):
    """Extract container and list of layers to collapse."""
    def split_path(p):
        parts = p.split(".")
        return ".".join(parts[:-1]), parts[-1]

    start_container, start_name = split_path(start)
    end_container, end_name = split_path(end)
    if start_container != end_container:
        raise ValueError(f"Start and end must be in same container: '{start}' vs '{end}'")
    container = _get_submodule(model, start_container)
    if isinstance(container, nn.Sequential):
        children = list(container.named_children())
    elif isinstance(container, nn.ModuleList):
        children = [(str(i), container[i]) for i in range(len(container))]
    else:
        children = list(container.named_children())
    name_to_idx = {n: i for i, (n, _) in enumerate(children)}
    start_idx = name_to_idx[start_name]
    end_idx = name_to_idx[end_name]
    named_layers = children[start_idx : end_idx + 1]
    return start_container, named_layers


def _collapse_block(model, start, end, input_shape, device="cpu", debug=True):
    """
    Collapse a block of layers from `start` to `end` in `model` with detailed debug.

    Args:
        model       : nn.Module containing the block
        start       : str, start layer name
        end         : str, end layer name
        input_shape : tuple, input shape to block
        device      : str, device for collapsed block
        debug       : bool, enable debug prints

    Returns:
        model with collapsed block replaced
    """
    # Extract container and block layers
    start_container_name, named_layers = _get_block_layers(model, start, end)
    if debug:
        print(f"[DEBUG] Collapsing block '{start}' → '{end}'")
        print(f"[DEBUG] Layers in block: {[name for name, _ in named_layers]}")

    # Capture input activation using dummy input
    try:
        hook_input_shape = (1,) + tuple(input_shape[1:]) if isinstance(input_shape, (tuple, list)) else input_shape
        _, captured_activation = _simulate_input_hook(model, start, hook_input_shape, device)
        if debug:
            print(f"[DEBUG] Captured activation before '{start}': {tuple(captured_activation.shape)}")
        traced_shapes = _trace_block_shapes(named_layers, captured_activation, device, debug)
    except Exception as e:
        if debug:
            print(f"[WARN] Failed to capture activation for '{start}': {e}. Using global input_shape.")
        captured_activation = torch.zeros((1,) + tuple(input_shape[1:])).to(device)
        traced_shapes = _trace_block_shapes(named_layers, captured_activation, device, debug)

    # Count parameters before collapse
    pre_params = sum(p.numel() for p in model.parameters())
    if debug:
        print(f"[DEBUG] Parameters before collapse: {pre_params:,}")

    # Build collapsed block
    collapsed_seq, out_channels = _perform_collapse(
        named_layers,
        runtime_input=captured_activation,
        traced_shapes=traced_shapes,
        device=device,
        debug=debug
    )

    # Count parameters of collapsed block
    collapsed_params = sum(p.numel() for p in collapsed_seq.parameters())
    if debug:
        print(f"[DEBUG] Collapsed block parameters: {collapsed_params:,}")

    # Replace block in container
    container = _get_submodule(model, start_container_name)
    _replace_block_in_container(container, named_layers, collapsed_seq)

    # Forward test through collapsed block
    if debug:
        try:
            collapsed_seq.eval()
            with torch.no_grad():
                test_out = collapsed_seq(captured_activation)
            print(f"[DEBUG] Forward test output shape: {tuple(test_out.shape)}")
            print(f"[INFO] Collapse completed. Output channels: {out_channels}, Output shape: {tuple(test_out.shape)}")
        except Exception as e:
            print(f"[WARN] Forward test failed: {e}")

    # Count parameters after replacement
    post_params = sum(p.numel() for p in model.parameters())
    if debug:
        print(f"[DEBUG] Parameters after collapse: {post_params:,} (Δ = {post_params - pre_params:+,})")

    return model


# -----------------------------------------------------------------------------
# Top-level collapse function
# -----------------------------------------------------------------------------


def collapse_only(
    model: Optional[nn.Module] = None,
    model_weights_1: Optional[str] = None,
    compression_set: Optional[Sequence[Tuple[str, str]]] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = "cpu",
    safe_param_reduction: bool = True,
    handle_skips: bool = True,
    debug: bool = True,
    dry_run: bool = False,
) -> nn.Module:
    """Collapse multiple layer blocks in a model with debug info."""
    import time

    t_global = time.time()

    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("Provide either `model` or (`model_weights_1`, `model_class`).")
        model_kwargs = model_kwargs or {}
        print(f"[INFO] Loading model {model_class.__name__} from {model_weights_1}")
        model = model_class(**model_kwargs)
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get("model", chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)

    model = copy.deepcopy(model).to(device)
    model.eval()

    if compression_set is None:
        print("[WARN] No compression set provided — skipping collapse.")
        return model

    if isinstance(compression_set, dict):
        collapse_map = compression_set
    else:
        collapse_map = {f"collapse_{i}": tuple(pair) for i, pair in enumerate(compression_set)}

    model._collapsed_blocks = list(collapse_map.values())
    pre_total = count_trainable_params(model)
    print(f"[INFO] Starting collapse_only; params before = {pre_total:,}")
    print(f"[INFO] Blocks to collapse: {len(collapse_map)}")

    for idx, (name, (start, end)) in enumerate(collapse_map.items(), 1):
        print(f"\n[INFO] ---- ({idx}/{len(collapse_map)}) Processing collapse '{name}': {start} → {end} ----")
        t0 = time.time()
        if dry_run:
            print("[INFO] Dry-run: skipping actual collapse.")
            continue

        model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)

        if handle_skips:
            patch_skip_connections(model)
        disable_inplace_relu(model)

        print(f"[INFO] Collapse '{name}' completed in {time.time() - t0:.3f}s")

    post_total = count_trainable_params(model)
    t_elapsed = time.time() - t_global
    print("\n[INFO] === Collapse Summary ===")
    print(f"   Parameters before: {pre_total:,}")
    print(f"   Parameters after : {post_total:,}")
    print(f"   ΔParams          : {pre_total - post_total:+,}")
    print(f"   Time total       : {t_elapsed:.2f}s")
    print("===============================")

    if post_total > pre_total:
        print("[WARN] ⚠ Model has MORE parameters after collapse — check collapse policy.")

    return model
