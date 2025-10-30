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
    """Return (container_path, subname) from layer_name."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]


def _find_layer_indices(named_layers: Sequence[Tuple[str, nn.Module]], start_layer_name: str, end_layer_name: str) -> Tuple[Optional[int], Optional[int]]:
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
        if name == end_layer_name:
            end_idx = i
    return start_idx, end_idx


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


# -----------------------------------------------------------------------------
# Small safety helpers
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


# -----------------------------------------------------------------------------
# Skip connection patcher
# -----------------------------------------------------------------------------

def patch_skip_connections(model: nn.Module):
    """
    Patch module forwards for blocks that include `shortcut` so that if shapes
    mismatch the shortcut is safely ignored instead of raising on addition.
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
                            return F.relu(out)
                        return F.relu(out + sc)
                    except Exception:
                        return F.relu(out)
                return new_forward

            module.forward = make_patched_forward(orig_forward).__get__(module)
            print(f"[PATCH] Patched residual block forward: {name}")


# -----------------------------------------------------------------------------
# Forward hook simulator
# -----------------------------------------------------------------------------

def _simulate_input_hook(model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """Run forward with dummy input and capture activation input to target layer."""
    model.eval()
    model.to(device)
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


# -----------------------------------------------------------------------------
# Collapsed block builder
# -----------------------------------------------------------------------------

def _count_params_for_block(full_block: Sequence[Tuple[str, nn.Module]]) -> int:
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
    inherit_conv_attrs: bool = True
) -> nn.Sequential:
    """
    Build a collapsed block from many convs/linears into a small module.
    """
    if debug:
        print(f"[DEBUG] _build_collapsed_block called: layer_type={getattr(layer_type,'__name__',str(layer_type))}, in={in_features}, out={out_features}")

    seq = []
    original_param_budget = _count_params_for_block(full_block) if full_block else None

    # ----------------------------
    # Conv2d branch
    # ----------------------------
    if layer_type == nn.Conv2d:
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block) if full_block else False
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block) if full_block else False

        first_conv = None
        if inherit_conv_attrs and full_block:
            for _, m in full_block:
                if isinstance(m, nn.Conv2d):
                    first_conv = m
                    break

        if first_conv is not None:
            orig_kernel = first_conv.kernel_size
            orig_stride = first_conv.stride
            orig_padding = first_conv.padding
            orig_groups = first_conv.groups
            orig_dilation = first_conv.dilation
            orig_bias = first_conv.bias is not None
        else:
            orig_kernel = (1, 1)
            orig_stride = stride
            orig_padding = (0, 0)
            orig_groups = 1
            orig_dilation = (1, 1)
            orig_bias = False

        k = 1
        p = 0
        s = orig_stride if inherit_conv_attrs else stride

        H = output_shape[-2] if len(output_shape) >= 3 else 1
        W = output_shape[-1] if len(output_shape) >= 3 else 1

        suggested_out = out_features
        if linear_in_features is not None and H * W > 0:
            suggested_out = max(1, linear_in_features // (H * W))

        if shortcut_out_channels is not None:
            suggested_out = min(suggested_out, shortcut_out_channels)

        bottleneck_ratio = 0.5
        collapse_out = max(1, int(out_features * bottleneck_ratio))
        if suggested_out and suggested_out < collapse_out:
            collapse_out = suggested_out

        if original_param_budget is not None:
            def conv_params(cin, cout, kx, groups):
                return (cin // max(1, groups)) * cout * (kx * kx)
            cand = collapse_out
            cand_params = conv_params(in_features, cand, k, orig_groups)
            if has_bn:
                cand_params += 2 * cand
            while cand > 1 and cand_params > original_param_budget:
                cand = max(1, cand - max(1, int(cand * 0.1)))
                cand_params = conv_params(in_features, cand, k, orig_groups)
                if has_bn:
                    cand_params += 2 * cand
            collapse_out = cand

        collapse_out = max(1, min(collapse_out, out_features))
        conv_kwargs = dict(stride=s, padding=p, dilation=orig_dilation, groups=orig_groups, bias=orig_bias)
        collapsed_conv = nn.Conv2d(in_features, collapse_out, kernel_size=k, **conv_kwargs)
        seq.append(collapsed_conv)
        if has_bn:
            seq.append(nn.BatchNorm2d(collapse_out))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))
        if pool_layer is not None:
            seq.append(copy.deepcopy(pool_layer))
        if collapse_out != out_features and preserve_out_channels:
            proj = nn.Conv2d(collapse_out, out_features, kernel_size=1, stride=1, padding=0, bias=False)
            seq.append(proj)

    # ----------------------------
    # Linear branch
    # ----------------------------
    elif layer_type == nn.Linear:
        reduced_out = max(1, int(out_features * 0.75))
        if original_param_budget is not None:
            while reduced_out > 1 and (in_features * reduced_out + reduced_out) > original_param_budget:
                reduced_out = max(1, reduced_out - max(1, int(reduced_out * 0.1)))
        collapsed_linear = nn.Linear(in_features, reduced_out)
        seq.append(collapsed_linear)
        if reduced_out != out_features and preserve_out_channels:
            proj_lin = nn.Linear(reduced_out, out_features, bias=False)
            seq.append(proj_lin)
    else:
        raise NotImplementedError(f"Unsupported layer_type: {layer_type}")

    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(seq)]))
    return collapsed


# -----------------------------------------------------------------------------
# Core collapse of a single block (with PATCH)
# -----------------------------------------------------------------------------
def _collapse_block(model: nn.Module, start_layer_name: str, end_layer_name: str,
                    input_shape: Tuple[int, ...], device='cpu', debug: bool = False) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive).
    """
    import math, time
    print(f"\n[INFO] === Collapsing block: {start_layer_name} → {end_layer_name} ===")
    t_start = time.time()

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find start/end layers '{start_layer_name}'/'{end_layer_name}' in container '{start_container_name}'.")

    full_block = named_layers[start_idx:end_idx + 1]

    conv_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not conv_layers:
        raise ValueError("No Conv2d/Linear layers found in block to collapse.")
    layer_type = type(conv_layers[0])
    if not all(isinstance(l, layer_type) for l in conv_layers):
        raise ValueError("Cannot collapse mixed layer types inside one block.")

    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
    except Exception as e:
        if layer_type == nn.Conv2d:
            in_ch = conv_layers[0].in_channels if hasattr(conv_layers[0], 'in_channels') else input_shape[1]
            H, W = input_shape[-2], input_shape[-1]
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            in_feat = conv_layers[0].in_features if hasattr(conv_layers[0], 'in_features') else input_shape[1]
            x = torch.randn(1, in_feat, device=device)
        print(f"[WARN] Hook capture failed: {e}. Using fallback dummy input shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    classifier_linear_mod = next((mod for _, mod in model.named_modules() if isinstance(mod, nn.Linear)), None)
    adaptive_pool_to_use = None
    out_channels = None

    if layer_type == nn.Conv2d:
        in_channels = x.shape[1]
        with torch.no_grad():
            y = x.clone()
            last_conv = None
            for _, layer in full_block:
                y = layer(y)
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer
        out_shape = tuple(y.shape)
        out_channels = last_conv.out_channels if last_conv is not None else conv_layers[-1].out_channels
        linear_in_features = classifier_linear_mod.in_features if classifier_linear_mod is not None else None
        pool_layer = next((m for _, m in reversed(full_block)
                           if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)
        collapsed_block = _build_collapsed_block(
            nn.Conv2d, in_features=in_channels, out_features=out_channels,
            output_shape=out_shape, full_block=full_block, stride=(1, 1),
            pool_layer=pool_layer, linear_in_features=linear_in_features,
            debug=debug
        )
    else:
        in_features = x.view(x.size(0), -1).size(1)
        with torch.no_grad():
            y = x.clone()
            for layer in conv_layers:
                y = layer(y)
        out_features = y.view(y.size(0), -1).size(1)
        collapsed_block = _build_collapsed_block(
            nn.Linear, in_features=in_features, out_features=out_features,
            output_shape=tuple(y.shape), full_block=full_block, debug=debug
        )

    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)

    # >>> PATCH START
    # handle root-level (empty container path) safely
    if start_container_name == "":
        updated_container = updated_container.to(device)
        model = updated_container
    else:
        _update_container(model, start_container_name, updated_container)
        model.to(device)
    # >>> PATCH END

    post_params = count_trainable_params(model)
    t_elapsed = time.time() - t_start

    print("\n[SUMMARY] Block collapse results:")
    print(f"   Block:           {start_layer_name} → {end_layer_name}")
    print(f"   Params before:   {pre_params:,}")
    print(f"   Params after:    {post_params:,}")
    print(f"   ΔParams:         {pre_params - post_params:+,}")
    print(f"   Time taken:      {t_elapsed:.3f}s")
    print("[SUMMARY] ----------------------------------------------")

    return model


# -----------------------------------------------------------------------------
# Top-level multi-block collapse
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
    """Collapses multiple layer blocks in a model with debug summaries."""
    import time
    t_global = time.time()

    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("Either provide `model` or (`model_weights_1`, `model_class`).")
        model_kwargs = model_kwargs or {}
        print(f"[INFO] Loading model {model_class.__name__} from {model_weights_1}")
        model = model_class(**model_kwargs)
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get('model', chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)

    model = deepcopy(model).to(device)
    model.eval()

    if compression_set is None:
        print("[WARN] No compression set provided — skipping collapse.")
        return model

    if isinstance(compression_set, dict):
        collapse_map = compression_set
    else:
        collapse_map = {f"block_{i}": pair for i, pair in enumerate(compression_set)}

    print(f"[INFO] Beginning collapse of {len(collapse_map)} block(s).")
    if handle_skips:
        patch_skip_connections(model)
    disable_inplace_relu(model)

    param_before_all = count_trainable_params(model)
    print(f"[INFO] Total params before all collapses: {param_before_all:,}")

    if dry_run:
        print("[INFO] Dry run mode — no actual collapsing performed.")
        return model

    for blk_name, (start, end) in collapse_map.items():
        try:
            model = _collapse_block(model, start, end, input_shape=input_shape, device=device, debug=debug)
        except Exception as e:
            print(f"[ERROR] Collapse failed for {blk_name}: {e}")

    param_after_all = count_trainable_params(model)
    print(f"[GLOBAL SUMMARY] All collapses complete in {time.time() - t_global:.3f}s")
    print(f"    Params before: {param_before_all:,}")
    print(f"    Params after:  {param_after_all:,}")
    print(f"    ΔParams:       {param_before_all - param_after_all:+,}")
    print("---------------------------------------------------------")
    return model
