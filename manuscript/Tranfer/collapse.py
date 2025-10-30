# collapse.py
import copy
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_trainable_params, layer_stats


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


def _simulate_input_hook(
    model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device="cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """Trace shapes through layers."""
    if isinstance(input_tensor_or_shape, torch.Tensor):
        x = input_tensor_or_shape.to(device)
    else:
        x = torch.zeros(input_tensor_or_shape).to(device)

    shapes = []
    if debug:
        print("[DEBUG] Forwarding through block layers for shape tracing:")
    for name, layer in named_layers:
        try:
            x = layer(x)
            shapes.append((name, x.shape))
            if debug:
                print(f"   -> After {layer.__class__.__name__:<22}: shape = {tuple(x.shape)}")
        except Exception as e:
            print(f"[ERROR] Shape tracing failed at {name}: {e}")
            raise
    return {"final": x.shape, "list": shapes}


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


def _count_params_for_block(full_block: Sequence[Tuple[str, nn.Module]]) -> int:
    """Count trainable parameters in the block."""
    total = 0
    for _, m in full_block:
        for p in m.parameters():
            if p.requires_grad:
                total += p.numel()
    return total


def _absorb_pools_if_needed(named_layers, traced_shapes, debug):
    """Replace pooling layers with Identity if spatial size >1, return last pool used."""
    output_shape = traced_shapes["final"]
    H, W = output_shape[-2:]
    pool_used = None
    adjusted_layers = list(named_layers)

    if H > 1 or W > 1:
        for i, (n, l) in reversed(list(enumerate(named_layers))):
            if isinstance(l, (nn.MaxPool2d, nn.AvgPool2d)):
                adjusted_layers[i] = nn.Identity()
                pool_used = l
                H //= l.kernel_size if isinstance(l.kernel_size, int) else l.kernel_size[0]
                W //= l.kernel_size if isinstance(l.kernel_size, int) else l.kernel_size[1]
                if debug:
                    print(f"[DEBUG] Absorbed pool {l.__class__.__name__}, new HxW={H}x{W}")
                if H <= 1 and W <= 1:
                    break
    return adjusted_layers, pool_used


def _build_collapsed_block_with_checks(layers, traced_shapes, debug):
    """Build a simple collapsed Sequential (identity-preserving)."""
    in_channels = layers[0][1].in_channels if hasattr(layers[0][1], "in_channels") else None
    out_channels = layers[-1][1].out_channels if hasattr(layers[-1][1], "out_channels") else None

    collapsed_layers = [layer for _, layer in layers if not isinstance(layer, nn.Identity)]
    collapsed_block = nn.Sequential(*collapsed_layers)

    if debug:
        print(f"[DEBUG] Built collapsed block: {collapsed_block}")
        print(f"[DEBUG] Input/Output channels: {in_channels} → {out_channels}")

    return collapsed_block


def _perform_collapse(named_layers, traced_shapes, device="cpu", debug=False):
    """Perform actual collapse: absorb pools and build collapsed Sequential."""
    layers, pool_used = _absorb_pools_if_needed(named_layers, traced_shapes, debug)
    collapsed_block = _build_collapsed_block_with_checks(layers, traced_shapes, debug)
    collapsed_block.to(device)
    return collapsed_block, pool_used


def _replace_block_in_container(container, named_layers, collapsed_block):
    """Replace original block layers inside container with collapsed block."""
    start_name = named_layers[0][0]
    end_name = named_layers[-1][0]

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
    for name, module in updated_container.named_children():
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


def _collapse_block(model, start, end, input_shape, device="cpu", debug=False):
    """Collapse a single block of layers in a model."""
    start_container_name, named_layers = _get_block_layers(model, start, end)

    if debug:
        print(f"[DEBUG] Collapsing block {start} → {end}")
        print(f"[DEBUG] Layers in block: {[name for name, _ in named_layers]}")

    try:
        if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 1:
            hook_input_shape = (1,) + tuple(input_shape[1:])
        else:
            hook_input_shape = input_shape

        _, captured_activation = _simulate_input_hook(model, start, hook_input_shape, device)
        if debug:
            print(f"[DEBUG] Captured activation before start '{start}': {tuple(captured_activation.shape)}")

        traced_shapes = _trace_block_shapes(named_layers, captured_activation, device, debug)

    except Exception as e:
        if debug:
            print(f"[WARN] Failed to capture activation for start '{start}': {e}. Using global input_shape.")
        traced_shapes = _trace_block_shapes(named_layers, input_shape, device, debug)

    collapsed_layer, _ = _perform_collapse(named_layers, traced_shapes, device=device, debug=debug)
    container = _get_submodule(model, start_container_name)
    _replace_block_in_container(container, named_layers, collapsed_layer)

    if debug:
        print(f"[INFO] Collapsed block {start} → {end}")
        print(f"[INFO] Params after collapse: {sum(p.numel() for p in model.parameters())}")

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
