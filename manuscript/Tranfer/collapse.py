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

# =============================================================================
# Utility helpers
# =============================================================================

def _is_int_str(s: str) -> bool:
    """Check if a string represents an integer index (for Sequential paths)."""
    try:
        int(s)
        return True
    except Exception:
        return False


def get_layer(model: nn.Module, layer_name: str) -> nn.Module:
    """Access a layer in a model via dot-separated path. Empty path returns model itself."""
    if layer_name == "":
        return model
    layer = model
    for part in layer_name.split('.'):
        layer = layer[int(part)] if _is_int_str(part) else getattr(layer, part)
    return layer


def _set_module_by_path(model: nn.Module, module_path: str, new_module: nn.Module):
    """Replace module at dot-separated path."""
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
    """Split a layer path into (container_path, subname)."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]


def _find_layer_indices(named_layers: Sequence[Tuple[str, nn.Module]],
                        start_layer_name: str,
                        end_layer_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Find the indices of the start and end layers in a named layer list."""
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx


def _replace_layers(named_layers: Sequence[Tuple[str, nn.Module]],
                    start_idx: int,
                    end_idx: int,
                    new_block: nn.Module) -> nn.Sequential:
    """Replace a range of layers with a single collapsed block."""
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
    """Replace a sub-container (Sequential or Module) inside model."""
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


# =============================================================================
# Safety & skip-connection helpers
# =============================================================================

def disable_inplace_relu(model: nn.Module):
    """Replace in-place ReLUs to avoid autograd issues."""
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
    """Patch residual blocks to handle shape mismatch safely."""
    for name, module in model.named_modules():
        if hasattr(module, 'shortcut') and hasattr(module, 'block') and isinstance(module.shortcut, nn.Module):
            orig_forward = getattr(module, 'forward', None)
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


# =============================================================================
# Forward capture helpers
# =============================================================================

def _simulate_input_hook(model: nn.Module,
                         target_layer_path: str,
                         input_shape: Tuple[int, ...],
                         device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward pass and capture the input activation of target layer."""
    model.eval().to(device)
    dummy_input = torch.randn(input_shape).to(device)

    target_module = get_layer(model, target_layer_path)
    captured = {}

    def hook(module, inp, out):
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


# =============================================================================
# Collapsed block builder
# =============================================================================

def _count_params_for_block(block: Sequence[Tuple[str, nn.Module]]) -> int:
    """Count trainable params inside a block."""
    total = 0
    for _, m in block:
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
    force_hw: Optional[Tuple[int,int]] = None,
) -> nn.Sequential:
    """Construct a simplified replacement block with reduced parameters."""
    if debug:
        print(f"[DEBUG] Building collapsed block: {layer_type.__name__}, in={in_features}, out={out_features}")

    seq = []
    orig_param_budget = _count_params_for_block(full_block) if full_block else None

    # --- Conv2d collapse ---
    if layer_type == nn.Conv2d:
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block) if full_block else False
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block) if full_block else False

        first_conv = next((m for _, m in full_block if isinstance(m, nn.Conv2d)), None)
        if first_conv:
            orig_kernel, orig_stride, orig_padding = first_conv.kernel_size, first_conv.stride, first_conv.padding
            orig_groups, orig_dilation = first_conv.groups, first_conv.dilation
            orig_bias = first_conv.bias is not None
        else:
            orig_kernel, orig_stride, orig_padding, orig_groups, orig_dilation = (1,1), stride, (0,0), 1, (1,1)
            orig_bias = False

        H = force_hw[0] if force_hw else output_shape[-2]
        W = force_hw[1] if force_hw else output_shape[-1]

        collapse_out = max(1, int(out_features * 0.5))
        if orig_param_budget:
            def conv_params(cin, cout, k, groups): return (cin // max(1, groups)) * cout * (k*k)
            cand, cand_params = collapse_out, conv_params(in_features, collapse_out, 1, orig_groups)
            while cand > 1 and cand_params > orig_param_budget:
                cand = max(1, cand - int(cand * 0.1))
                cand_params = conv_params(in_features, cand, 1, orig_groups)
            collapse_out = cand

        conv = nn.Conv2d(in_features, collapse_out, kernel_size=1, stride=orig_stride,
                         padding=0, groups=orig_groups, bias=orig_bias)
        seq += [conv]
        if has_bn: seq.append(nn.BatchNorm2d(collapse_out))
        if has_relu: seq.append(nn.ReLU(inplace=False))
        if pool_layer: seq.append(copy.deepcopy(pool_layer))
        if preserve_out_channels and collapse_out != out_features:
            seq.append(nn.Conv2d(collapse_out, out_features, kernel_size=1, bias=False))

    # --- Linear collapse ---
    elif layer_type == nn.Linear:
        reduced_out = max(1, int(out_features * 0.75))
        if orig_param_budget:
            while reduced_out > 1 and (in_features * reduced_out + reduced_out) > orig_param_budget:
                reduced_out = max(1, reduced_out - int(reduced_out * 0.1))
        seq.append(nn.Linear(in_features, reduced_out))
        if preserve_out_channels and reduced_out != out_features:
            seq.append(nn.Linear(reduced_out, out_features, bias=False))

    return nn.Sequential(OrderedDict([(f"layer_{i}", m) for i, m in enumerate(seq)]))


# =============================================================================
# Core block collapse
# =============================================================================

def _collapse_block(model: nn.Module,
                    start_layer_name: str,
                    end_layer_name: str,
                    input_shape: Tuple[int, ...],
                    device='cpu',
                    debug: bool = False) -> nn.Module:
    """Collapse layers between given start and end layer names."""
    print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    # Locate container and layer indices
    start_container, start_sub = _get_container_and_subname(start_layer_name)
    end_container, end_sub = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container)
    named_layers = list(container.named_children())

    start_idx, end_idx = _find_layer_indices(named_layers, start_sub, end_sub)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find layers '{start_layer_name}' / '{end_layer_name}'.")

    full_block = named_layers[start_idx:end_idx + 1]
    conv_layers = [l for _, l in full_block if isinstance(l, (nn.Conv2d, nn.Linear))]
    if not conv_layers:
        raise ValueError("No Conv2d/Linear layers found in block to collapse.")
    layer_type = type(conv_layers[0])

    # Capture input
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device)
        if debug:
            print(f"[DEBUG] Captured activation before start: {tuple(x.shape)}")
    except Exception as e:
        print(f"[WARN] Hook failed: {e}")
        if layer_type == nn.Conv2d:
            in_ch = conv_layers[0].in_channels
            H, W = input_shape[-2:]
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            in_feat = conv_layers[0].in_features
            x = torch.randn(1, in_feat, device=device)

    pre_params = count_trainable_params(model)
    if debug: print(f"[DEBUG] Params before collapse: {pre_params:,}")

    # Forward to get output shape
    with torch.no_grad():
        y = x.clone()
        for _, layer in full_block:
            y = layer(y)
    out_shape = tuple(y.shape)
    out_channels = getattr(conv_layers[-1], "out_channels", None)

    # Build collapsed block
    collapsed_block = _build_collapsed_block(
        layer_type,
        in_features=x.shape[1] if layer_type == nn.Conv2d else x.view(x.size(0), -1).size(1),
        out_features=out_channels if layer_type == nn.Conv2d else y.view(y.size(0), -1).size(1),
        output_shape=out_shape,
        full_block=full_block,
        debug=debug
    )

    # Replace in container
    new_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container, new_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Params after collapse: {post_params:,}")
    print(f"[INFO] ΔParams = {pre_params - post_params:+,}")

    return model


# =============================================================================
# Top-level API (unchanged)
# =============================================================================

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
    """(UNCHANGED) Main public API for block collapse."""
    # -- left as-is --
    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("Either provide model or model_weights_1+model_class.")
        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating {model_class.__name__}, loading {model_weights_1}")
        model = model_class(**model_kwargs)
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get('model', chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)

    model = deepcopy(model).to(device)
    model.eval()

    if compression_set is None:
        print("[WARN] compression_set empty; nothing to do.")
        return model

    collapse_map = (
        compression_set if isinstance(compression_set, dict)
        else {f"collapse_{i}": tuple(pair) for i, pair in enumerate(compression_set)}
    )

    model._collapsed_blocks = list(collapse_map.values())

    pre_total = count_trainable_params(model)
    print(f"[INFO] Starting collapse_only; params before = {pre_total:,}")

    for name, (start, end) in collapse_map.items():
        print(f"\n[INFO] Collapsing '{name}': {start} → {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping collapse.")
            continue

        model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)
        if handle_skips:
            patch_skip_connections(model)
        disable_inplace_relu(model)

    post_total = count_trainable_params(model)
    print(f"\n[INFO] Collapse complete. Params Δ = {pre_total - post_total:+,}")

    if debug:
        print(f"[DEBUG] Model summary:\n{layer_stats(model)}")

    return model
