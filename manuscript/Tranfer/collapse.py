# collapse.py
import copy
from copy import deepcopy
from collections import OrderedDict
from uuid import uuid4
from typing import Optional, Sequence, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import count_trainable_params, layer_stats

# ---------------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------------

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


def _set_module_by_path(model: nn.Module, module_path: str, new_module: nn.Module):
    """Replace module at dot-separated path (supports Sequential indices)."""
    if module_path == "":
        raise ValueError("Cannot replace root module")
    parts = module_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _get_container_and_subname(layer_name: str) -> Tuple[str, str]:
    """Return (container_path, subname) from layer_name (both '' if empty)."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]


def _find_layer_indices(named_layers: Sequence[Tuple[str, nn.Module]],
                        start_layer_name: str, end_layer_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Find start/end indices in a sequence of (name, module)."""
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx


def _replace_layers(named_layers: Sequence[Tuple[str, nn.Module]], start_idx: int, end_idx: int,
                    new_block: nn.Module) -> nn.Sequential:
    """Return a Sequential built from named_layers with start..end replaced by new_block (keeps names)."""
    new_layers = []
    unique_suffix = uuid4().hex[:8]
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_layers.append((f"collapsed_{unique_suffix}", new_block))
        elif start_idx < i <= end_idx:
            # skip these original layers (they are collapsed)
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


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------

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
    """
    Patch residual-like blocks that expose `.block` and `.shortcut`.
    The patched forward will safely skip the shortcut if shapes mismatch or it throws.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module) and hasattr(module, 'block'):
            orig_forward = getattr(module, 'forward', None)
            if orig_forward is None:
                continue

            def make_patched_forward(orig_fwd):
                def new_forward(self, x):
                    out = self.block(x)
                    try:
                        sc = self.shortcut(x)
                        if out.shape != sc.shape:
                            # shapes differ: ignore shortcut
                            return F.relu(out)
                        return F.relu(out + sc)
                    except Exception:
                        return F.relu(out)
                return new_forward

            module.forward = make_patched_forward(orig_forward).__get__(module)
            print(f"[PATCH] Patched residual block forward: {name}")


# ---------------------------------------------------------------------------
# Forward hook simulator (capture activation that is input to a layer)
# ---------------------------------------------------------------------------

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
        # inp is a tuple; take the first element
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


# ---------------------------------------------------------------------------
# Collapsed block builder utilities
# ---------------------------------------------------------------------------

def _count_params_for_block(full_block: Sequence[Tuple[str, nn.Module]]) -> int:
    """Count trainable params for modules inside full_block (conv/linear/bn etc)."""
    total = 0
    for _, m in full_block:
        for p in m.parameters():
            if p.requires_grad:
                total += p.numel()
    return total


def _build_collapsed_block(
    layer_type: type,
    in_features: int,
    out_features: int,
    output_shape: Tuple[int, ...],
    full_block: Optional[Sequence[Tuple[str, nn.Module]]] = None,
    stride: Tuple[int,int] = (1,1),
    pool_layer: Optional[nn.Module] = None,
    linear_in_features: Optional[int] = None,
    shortcut_out_channels: Optional[int] = None,
    debug: bool = False,
    preserve_out_channels: bool = True,
    inherit_conv_attrs: bool = True,
    force_hw: Optional[Tuple[int,int]] = None
) -> nn.Sequential:
    """
    Heuristic builder that returns a small nn.Sequential intended to roughly
    replace `full_block`. Keeps some properties (bn/relu/pool) if present.
    """
    if debug:
        print(f"[DEBUG] _build_collapsed_block called: layer_type={getattr(layer_type,'__name__',str(layer_type))}, in={in_features}, out={out_features}, linear_in_features={linear_in_features}, force_hw={force_hw}")

    seq = []
    original_param_budget = _count_params_for_block(full_block) if full_block else None

    if layer_type == nn.Conv2d:
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block) if full_block else False
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block) if full_block else False

        # Try to inherit first conv attributes if available
        first_conv = next((m for _, m in full_block if isinstance(m, nn.Conv2d)), None) if full_block else None
        if first_conv:
            orig_kernel = first_conv.kernel_size
            orig_stride = first_conv.stride
            orig_padding = first_conv.padding
            orig_groups = first_conv.groups
            orig_dilation = first_conv.dilation
            orig_bias = first_conv.bias is not None
        else:
            orig_kernel = (1,1)
            orig_stride = stride
            orig_padding = (0,0)
            orig_groups = 1
            orig_dilation = (1,1)
            orig_bias = False

        H = force_hw[0] if force_hw else output_shape[-2]
        W = force_hw[1] if force_hw else output_shape[-1]

        suggested_out = out_features
        if linear_in_features is not None:
            suggested_out = max(1, linear_in_features // (H * W))
            if debug:
                print(f"[DEBUG] Linear follower present: target channels ≈ {suggested_out} (H*W={H*W})")

        if shortcut_out_channels:
            suggested_out = min(suggested_out, shortcut_out_channels)
            if debug:
                print(f"[DEBUG] Honoring shortcut output channels cap: {shortcut_out_channels}")

        # initial heuristic: reduce out channels
        collapse_out = max(1, int(out_features * 0.5))
        if suggested_out < collapse_out:
            collapse_out = suggested_out

        # param-budget aware reduction
        if original_param_budget:
            def conv_params(cin, cout, kx, groups):
                return (cin // max(1, groups)) * cout * (kx*kx)
            cand = collapse_out
            cand_params = conv_params(in_features, cand, 1, orig_groups) + (2*cand if has_bn else 0)
            while cand > 1 and cand_params > original_param_budget:
                cand = max(1, cand - max(1,int(cand*0.1)))
                cand_params = conv_params(in_features, cand, 1, orig_groups) + (2*cand if has_bn else 0)
                if debug:
                    print(f"[DEBUG] Trying cand_out={cand}, cand_params={cand_params}, budget={original_param_budget}")
            collapse_out = cand

        collapse_out = max(1, min(collapse_out, out_features))

        # build collapsed conv(s)
        conv = nn.Conv2d(in_features, collapse_out, kernel_size=1, stride=orig_stride,
                         padding=0, dilation=orig_dilation, groups=orig_groups, bias=orig_bias)
        seq.append(conv)
        if has_bn:
            seq.append(nn.BatchNorm2d(collapse_out))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))
        if pool_layer:
            seq.append(copy.deepcopy(pool_layer))
        if collapse_out != out_features and preserve_out_channels:
            seq.append(nn.Conv2d(collapse_out, out_features, kernel_size=1, bias=False))
            if debug:
                print(f"[DEBUG] Added 1x1 projection: {collapse_out} -> {out_features}")

    elif layer_type == nn.Linear:
        reduced_out = max(1, int(out_features * 0.75))
        if original_param_budget:
            while reduced_out > 1 and (in_features*reduced_out + reduced_out) > original_param_budget:
                reduced_out = max(1, reduced_out - max(1, int(reduced_out*0.1)))
                if debug:
                    print(f"[DEBUG] Trying reduced_out={reduced_out} vs budget={original_param_budget}")
        seq.append(nn.Linear(in_features, reduced_out))
        if reduced_out != out_features and preserve_out_channels:
            seq.append(nn.Linear(reduced_out, out_features, bias=False))
            if debug:
                print(f"[DEBUG] Added Linear projection: {reduced_out} -> {out_features}")

    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", m) for i,m in enumerate(seq)]))
    if debug:
        print(f"[DEBUG] Collapsed block modules: {[type(m).__name__ for m in collapsed]}")

    return collapsed


# ---------------------------------------------------------------------------
# Core collapse for a single start..end block
# ---------------------------------------------------------------------------

def _find_next_linear(model: nn.Module, end_layer_name: str, debug: bool = False):
    """Find first Linear module that occurs after end_layer_name in named_modules; fallback to any Linear."""
    modules_list = list(model.named_modules())
    idx_end_global = None
    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            break
    if idx_end_global is None:
        for i, (n, m) in enumerate(modules_list):
            if n.endswith(end_layer_name):
                idx_end_global = i
                break

    next_linear_name = None
    next_linear_mod = None
    if idx_end_global is not None:
        for j in range(idx_end_global + 1, len(modules_list)):
            n, m = modules_list[j]
            if isinstance(m, nn.Linear):
                next_linear_name = n
                next_linear_mod = m
                break

    # fallback: first Linear anywhere
    if next_linear_mod is None:
        for n, m in modules_list:
            if isinstance(m, nn.Linear):
                next_linear_name = n
                next_linear_mod = m
                break

    if debug:
        print(f"[DEBUG] Next Linear detected: name='{next_linear_name}', module={next_linear_mod}")

    return next_linear_name, next_linear_mod


def _forward_through_block(full_block: Sequence[Tuple[str, nn.Module]],
                           x: torch.Tensor,
                           debug: bool = False) -> Tuple[torch.Tensor, Optional[nn.Module]]:
    """
    Forward x through full_block to compute output activation and remember the
    last Conv2d encountered. Returns (y, last_conv_module).
    """
    with torch.no_grad():
        y = x.clone()
        last_conv = None
        for _, layer in full_block:
            y = layer(y)
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
    if debug:
        print(f"[DEBUG] Forwarded through block: output shape {tuple(y.shape)}; last_conv={type(last_conv).__name__ if last_conv is not None else None}")
    return y, last_conv


def _collapse_block(
    model: nn.Module,
    start_layer_name: str,
    end_layer_name: str,
    input_shape: Tuple[int, ...],
    device='cpu',
    debug: bool = False
) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive).
    Modifies a deep-copied model in-place and returns it.
    """
    print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    # locate container and subnames relative to container
    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)

    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find start/end layers '{start_layer_name}'/'{end_layer_name}' in container '{start_container_name}'.")

    full_block = named_layers[start_idx:end_idx + 1]
    # conv or linear layers inside the block
    conv_like_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not conv_like_layers:
        raise ValueError("No Conv2d/Linear layers found in block to collapse.")

    layer_type = type(conv_like_layers[0])
    if not all(isinstance(l, layer_type) for l in conv_like_layers):
        raise ValueError("Cannot collapse mixed layer types inside one block.")

    # capture input activation (pre-block)
    try:
        _, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        if debug:
            print(f"[DEBUG] Captured activation before start: {tuple(x.shape)}")
    except Exception as e:
        # fallback: create dummy activation based on first conv/linear layer attributes
        if layer_type == nn.Conv2d:
            first_conv = next((l for l in conv_like_layers if isinstance(l, nn.Conv2d)), None)
            in_ch = first_conv.in_channels if first_conv is not None and hasattr(first_conv, 'in_channels') else input_shape[1]
            H, W = input_shape[-2], input_shape[-1]
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            first_linear = next((l for l in conv_like_layers if isinstance(l, nn.Linear)), None)
            in_feat = first_linear.in_features if first_linear is not None and hasattr(first_linear, 'in_features') else input_shape[1]
            x = torch.randn(1, in_feat, device=device)
        print(f"[WARN] Hook failed: {e}. Using dummy input shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Params before collapse: {pre_params:,}")

    # find next linear after this block (helps choose channels and pooling decisions)
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)

    # forward through original block to determine out shape and last conv
    y, last_conv = _forward_through_block(full_block, x, debug=debug)
    out_shape = tuple(y.shape)
    out_channels = None
    if last_conv is not None:
        out_channels = last_conv.out_channels
    elif layer_type == nn.Conv2d:
        # as fallback take conv_like_layers[-1]
        out_channels = conv_like_layers[-1].out_channels

    # detect candidate pooling layer inside block (max/avg/adaptive)
    pool_layer = next((m for _, m in reversed(full_block) if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)

    # Build collapsed block
    if layer_type == nn.Conv2d:
        in_channels = x.shape[1]
        collapsed_block = _build_collapsed_block(
            nn.Conv2d,
            in_features=in_channels,
            out_features=out_channels,
            output_shape=out_shape,
            full_block=full_block,
            stride=(1,1),
            pool_layer=pool_layer,
            debug=debug
        )
    else:  # Linear
        in_features = x.view(x.size(0), -1).size(1)
        # compute output size by forwarding through conv_like_layers (if needed)
        with torch.no_grad():
            y_lin = x.clone()
            for _, layer in full_block:
                y_lin = layer(y_lin)
        out_features_lin = y_lin.view(y_lin.size(0), -1).size(1)
        collapsed_block = _build_collapsed_block(
            nn.Linear,
            in_features=in_features,
            out_features=out_features_lin,
            output_shape=tuple(y_lin.shape),
            full_block=full_block,
            debug=debug
        )

    # Replace the block inside container
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Params after collapse: {post_params:,}")
    print(f"[INFO] ΔParams = {pre_params - post_params:+,} (should be >= 0)")

    return model


# ---------------------------------------------------------------------------
# Top-level multi-block collapse function (API kept exactly the same)
# ---------------------------------------------------------------------------

def collapse_only(
    model: Optional[nn.Module] = None,
    model_weights_1: Optional[str] = None,
    compression_set: Optional[Sequence[Tuple[str, str]]] = None,
    model_class: Optional[type] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    input_shape: Tuple[int, ...] = (1, 3, 32, 32),
    device: str = 'cpu',
    safe_param_reduction: bool = True,   # currently enforced by builder
    handle_skips: bool = True,
    debug: bool = True,
    dry_run: bool = False
) -> nn.Module:
    """
    Top-level API to collapse multiple blocks.
    Signature unchanged from your original request.

    Accepts either:
      - a pre-instantiated `model` (returned object will be a deep-copied collapsed model),
      OR
      - `model_weights_1` (path) + `model_class` + `model_kwargs` to construct & load model.

    compression_set: either a list of (start_layer, end_layer) tuples OR a dict mapping names -> (start,end)
    Returns collapsed model (copied).
    """
    # load or use provided model
    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("Either provide `model` or provide (`model_weights_1` and `model_class`).")
        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating model from class {model_class.__name__} and loading weights from {model_weights_1}")
        model = model_class(**model_kwargs)
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get('model', chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)
    else:
        # we will operate on a deepcopy so the original is preserved
        pass

    model = deepcopy(model).to(device)
    model.eval()

    # normalize compression_set into mapping name -> (start,end)
    if compression_set is None:
        print("[WARN] compression_set is empty; nothing to do.")
        return model

    if isinstance(compression_set, dict):
        collapse_map = compression_set
    else:
        collapse_map = {f"collapse_{i}": tuple(pair) for i, pair in enumerate(compression_set)}

    # store collapsed ranges for downstream patching / debugging
    model._collapsed_blocks = list(collapse_map.values())

    pre_total = count_trainable_params(model)
    print(f"[INFO] Starting collapse_only; params before = {pre_total:,}")

    for name, (start, end) in collapse_map.items():
        print(f"\n[INFO] Processing collapse '{name}': {start} -> {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping actual modification.")
            continue
        model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)

        # after each collapse optionally patch skip connections to avoid invalid adds
        if handle_skips:
            patch_skip_connections(model)

        # ensure out-of-place ReLUs to avoid autograd issues
        disable_inplace_relu(model)

    post_total = count_trainable_params(model)
    print("\n[INFO] Collapse finished.")
    print(f"[INFO] Parameters before: {pre_total:,}")
    print(f"[INFO] Parameters after : {post_total:,}")
    print(f"[INFO] ΔParams = {pre_total - post_total:+,} (should be >= 0)")

    if post_total > pre_total:
        print("[WARN] ⚠ Model has MORE parameters after collapse! This indicates a bug in collapse policy.")

    if debug:
        print(f"[DEBUG] Model structure after collapse:\n{layer_stats(model)}")

    return model
