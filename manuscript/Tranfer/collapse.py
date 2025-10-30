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
            # skip the original layers that were collapsed/absorbed
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
    mismatch the shortcut the shortcut is safely ignored instead of raising on addition.
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
# Forward hook simulator
# -----------------------------------------------------------------------------

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
    inherit_conv_attrs: bool = True
) -> nn.Sequential:
    """
    Build a collapsed block from many convs/linears into a small module that:
      - preserves BN/ReLU ordering if present,
      - chooses output channels such that the collapsed block has <= params than original,
      - optionally restores the original out_channels via a 1x1 projection,
      - returns nn.Sequential.

    Parameters:
      - layer_type: nn.Conv2d or nn.Linear
      - in_features, out_features: original block's in/out channels/features
      - output_shape: tensor shape after the block (used to compute H*W)
      - full_block: the original list of (name, module) for the block (for BN/ReLU detection & param counting)
      - stride: fallback stride to use for collapsed conv (overridden by inherit_conv_attrs if True)
      - pool_layer: optional pooling module to append (cloned)
      - linear_in_features: if the collapsed conv feeds a Linear, suggested flattened in_features (optional)
      - shortcut_out_channels: if there is a skip that expects fewer channels (optional)
      - preserve_out_channels: if True, append a 1x1 projection when collapsed channels != out_features
      - inherit_conv_attrs: if True, attempt to inherit kernel/stride/padding/groups/dilation/bias
    """
    if debug:
        print(f"[DEBUG] _build_collapsed_block called: layer_type={getattr(layer_type,'__name__',str(layer_type))}, in={in_features}, out={out_features}, linear_in_features={linear_in_features}, shortcut_out_channels={shortcut_out_channels}, preserve_out_channels={preserve_out_channels}, inherit_conv_attrs={inherit_conv_attrs}")

    seq = []
    original_param_budget = _count_params_for_block(full_block) if full_block else None

    # ----------------------------
    # Conv2d branch
    # ----------------------------
    if layer_type == nn.Conv2d:
        # detect BN/ReLU presence
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block) if full_block else False
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block) if full_block else False

        # Try to inherit attributes from the first Conv2d in the full_block
        first_conv = None
        if inherit_conv_attrs and full_block:
            for _, m in full_block:
                if isinstance(m, nn.Conv2d):
                    first_conv = m
                    break

        if first_conv is not None:
            orig_kernel = first_conv.kernel_size if hasattr(first_conv, 'kernel_size') else (1, 1)
            orig_stride = first_conv.stride if hasattr(first_conv, 'stride') else stride
            orig_padding = first_conv.padding if hasattr(first_conv, 'padding') else (0, 0)
            orig_groups = first_conv.groups if hasattr(first_conv, 'groups') else 1
            orig_dilation = first_conv.dilation if hasattr(first_conv, 'dilation') else (1, 1)
            orig_bias = first_conv.bias is not None
        else:
            # safe fallbacks
            orig_kernel = (1, 1)
            orig_stride = stride
            orig_padding = (0, 0)
            orig_groups = 1
            orig_dilation = (1, 1)
            orig_bias = False

        # Use a 1x1 collapsed conv for channel compression (keeps spatial layout safe)
        k = 1
        p = 0
        s = orig_stride if inherit_conv_attrs else stride

        # compute H*W for linear matching if provided
        H = output_shape[-2] if len(output_shape) >= 3 else 1
        W = output_shape[-1] if len(output_shape) >= 3 else 1

        # suggested_out if a Linear follows this block (try to match flattened dims)
        suggested_out = out_features
        if linear_in_features is not None and H * W > 0:
            # integer division safe guard
            suggested_out = max(1, linear_in_features // (H * W))
            if debug:
                print(f"[DEBUG] Linear follower present: target channels ≈ {suggested_out} (H*W={H*W})")

        # If skip expects fewer channels, honor it by capping suggested_out
        if shortcut_out_channels is not None:
            suggested_out = min(suggested_out, shortcut_out_channels)
            if debug:
                print(f"[DEBUG] Honoring shortcut output channels cap: {shortcut_out_channels}")

        # Initial bottleneck candidate
        bottleneck_ratio = 0.5
        collapse_out = max(1, int(out_features * bottleneck_ratio))
        # bias toward suggested_out if it's smaller
        if suggested_out and suggested_out < collapse_out:
            collapse_out = suggested_out

        # Parameter-budget-aware reduction (approximate; accounts for groups)
        if original_param_budget is not None:
            def conv_params(cin, cout, kx, groups):
                # approximate parameter count for conv weights (ignores bias)
                # for grouped convs, effective cin per filter is cin/groups
                return (cin // max(1, groups)) * cout * (kx * kx)
            cand = collapse_out
            cand_params = conv_params(in_features, cand, k, orig_groups)
            # Account for BN params (gamma/beta) if present
            if has_bn:
                cand_params += 2 * cand
            if debug:
                print(f"[DEBUG] Param budget check - target budget: {original_param_budget}, initial cand_params: {cand_params}")
            # reduce until fit or minimal
            while cand > 1 and cand_params > original_param_budget:
                cand = max(1, cand - max(1, int(cand * 0.1)))
                cand_params = conv_params(in_features, cand, k, orig_groups)
                if has_bn:
                    cand_params += 2 * cand
                if debug:
                    print(f"[DEBUG] Trying cand_out={cand}, cand_params={cand_params}, budget={original_param_budget}")
            collapse_out = cand

        # safety clamp
        collapse_out = max(1, min(collapse_out, out_features))

        # build collapsed conv with inherited attributes where appropriate
        conv_kwargs = dict(stride=s, padding=p, dilation=orig_dilation, groups=orig_groups, bias=orig_bias)
        collapsed_conv = nn.Conv2d(in_features, collapse_out, kernel_size=k, **conv_kwargs)
        seq.append(collapsed_conv)
        if debug:
            print(f"[DEBUG] Built collapsed Conv2d: in={in_features} out={collapse_out} k={k} stride={s} groups={orig_groups} bias={orig_bias}")

        # preserve BN/ReLU ordering local to collapsed conv
        if has_bn:
            seq.append(nn.BatchNorm2d(collapse_out))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))

        # attach pool layer if present (deepcopy to avoid shared references)
        if pool_layer is not None:
            seq.append(copy.deepcopy(pool_layer))
            if debug:
                print(f"[DEBUG] Appending cloned pooling layer: {type(pool_layer).__name__}")

        # If we reduced channels, optionally append a 1x1 projection to restore original channels.
        # This is the safe default to avoid changing downstream layers.
        if collapse_out != out_features and preserve_out_channels:
            if debug:
                print(f"[DEBUG] Adding 1x1 projection to restore channels: {collapse_out} -> {out_features}")
            proj = nn.Conv2d(collapse_out, out_features, kernel_size=1, stride=1, padding=0, bias=False)
            seq.append(proj)

    # ----------------------------
    # Linear branch
    # ----------------------------
    elif layer_type == nn.Linear:
        # linear collapse: reduce outputs but keep in_features same
        reduced_out = max(1, int(out_features * 0.75))
        if original_param_budget is not None:
            # decrease until within budget (approximate: in_features * out + out for bias)
            while reduced_out > 1 and (in_features * reduced_out + reduced_out) > original_param_budget:
                reduced_out = max(1, reduced_out - max(1, int(reduced_out * 0.1)))
                if debug:
                    print(f"[DEBUG] Trying reduced_out={reduced_out} vs budget={original_param_budget}")
        collapsed_linear = nn.Linear(in_features, reduced_out)
        seq.append(collapsed_linear)
        if debug:
            print(f"[DEBUG] Built collapsed Linear: in={in_features}, out={reduced_out}")

        # If user asked to preserve out_features, add a linear projection back up to out_features
        if reduced_out != out_features and preserve_out_channels:
            if debug:
                print(f"[DEBUG] Adding Linear projection to restore features: {reduced_out} -> {out_features}")
            proj_lin = nn.Linear(reduced_out, out_features, bias=False)
            seq.append(proj_lin)

    else:
        raise NotImplementedError(f"Unsupported layer_type: {layer_type}")

    # Finalize into an Ordered sequential with stable names
    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(seq)]))
    if debug:
        print(f"[DEBUG] Collapsed block final modules: {[type(m).__name__ for m in collapsed]}")
    return collapsed

# -----------------------------------------------------------------------------
# Core collapse of a single block (ENHANCED DEBUG VERSION)
# -----------------------------------------------------------------------------
def _collapse_block(model: nn.Module, start_layer_name: str, end_layer_name: str,
                    input_shape: Tuple[int, ...], device='cpu', debug: bool = False) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive).
    Adds robust handling so classifier's first Linear is adjusted if flattened size changes.
    Also removes any post-block pool layers that were absorbed into the collapsed block
    to prevent duplicate pooling / underflow (the fix).
    """
    import math
    import time

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

    # ---- Layer composition summary ----
    layer_types = [type(layer).__name__ for _, layer in full_block]
    conv_count = sum(isinstance(l, nn.Conv2d) for _, l in full_block)
    lin_count = sum(isinstance(l, nn.Linear) for _, l in full_block)
    pool_count = sum(isinstance(l, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)) for _, l in full_block)
    relu_count = sum(isinstance(l, nn.ReLU) for _, l in full_block)
    if debug:
        print(f"[DEBUG] Layers in block ({len(full_block)} total):")
        for i, (n, l) in enumerate(full_block):
            print(f"   {i:02d}. {n}: {l.__class__.__name__}")
        print(f"[DEBUG] Layer type summary: Conv={conv_count}, Linear={lin_count}, Pool={pool_count}, ReLU={relu_count}")

    conv_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not conv_layers:
        raise ValueError("No Conv2d/Linear layers found in block to collapse.")
    layer_type = type(conv_layers[0])
    if not all(isinstance(l, layer_type) for l in conv_layers):
        raise ValueError("Cannot collapse mixed layer types inside one block.")

    # ---- Capture activation ----
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        if debug:
            print(f"[DEBUG] Captured activation before start: {tuple(x.shape)}")
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
    if debug:
        print(f"[DEBUG] Params before collapse: {pre_params:,}")

    classifier_linear_name = None
    classifier_linear_mod = None
    for nm, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            classifier_linear_name = nm
            classifier_linear_mod = mod
            break
    if debug:
        print(f"[DEBUG] classifier first Linear: name='{classifier_linear_name}', module={classifier_linear_mod}")

    adaptive_pool_to_use = None
    out_channels = None
    out_shape = None
    pool_layer = None

    # ---- Conv2D path ----
    if layer_type == nn.Conv2d:
        in_channels = x.shape[1]
        with torch.no_grad():
            y = x.clone()
            last_conv = None
            if debug:
                print("[DEBUG] Forwarding through block layers for shape tracing:")
            for idx, (_, layer) in enumerate(full_block):
                y = layer(y)
                if isinstance(layer, nn.Conv2d):
                    last_conv = layer
                if debug:
                    print(f"   -> After {type(layer).__name__:<20}: shape = {tuple(y.shape)}")

        out_shape = tuple(y.shape)
        out_channels = last_conv.out_channels if last_conv is not None else conv_layers[-1].out_channels
        if debug:
            print(f"[DEBUG] Final block output shape: {out_shape}, out_channels={out_channels}")

        # Detect heuristic linear in_features target
        linear_in_features = None
        if end_idx + 1 < len(named_layers):
            next_mod = named_layers[end_idx + 1][1]
            if isinstance(next_mod, nn.Linear):
                linear_in_features = next_mod.in_features
        if linear_in_features is None and classifier_linear_mod is not None:
            linear_in_features = classifier_linear_mod.in_features
        if debug:
            print(f"[DEBUG] Detected linear_in_features = {linear_in_features}")

        # last pool inside full_block (if any)
        pool_layer = next((m for _, m in reversed(full_block)
                           if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)

        # compute expected flatten size and adjust
        if linear_in_features is not None and out_channels > 0:
            if linear_in_features % out_channels == 0:
                expected_hw = linear_in_features // out_channels
            else:
                expected_hw = max(1, linear_in_features // out_channels)
            cur_H, cur_W = out_shape[-2], out_shape[-1]
            cur_hw = cur_H * cur_W
            if debug:
                print(f"[DEBUG] expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})")

            # Try absorbing post-block pools (and remove them from container if successful)
            if cur_hw != expected_hw:
                extra_pools = []
                y2 = y.clone()
                absorbed_count = 0
                with torch.no_grad():
                    # iterate through layers after the collapsed block
                    for j, (nm, mod) in enumerate(named_layers[end_idx + 1:], start=1):
                        if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                            try:
                                y2 = mod(y2)
                            except Exception:
                                if debug:
                                    print(f"[DEBUG] Pool {type(mod).__name__} underflow when probing — stopping absorption.")
                                break
                            extra_pools.append(mod)
                            absorbed_count = j
                            new_H, new_W = y2.shape[-2], y2.shape[-1]
                            if debug:
                                print(f"[DEBUG] Absorbed pool {type(mod).__name__}, new shape={tuple(y2.shape)}")
                            if (new_H * new_W) == expected_hw:
                                # combine with any existing pool_layer
                                if pool_layer is None:
                                    pool_layer = extra_pools[0]
                                else:
                                    pool_layer = nn.Sequential(*([pool_layer] + extra_pools)) if not isinstance(pool_layer, nn.Sequential) else nn.Sequential(*([*pool_layer.children()] + extra_pools))
                                # update y/out_shape and update end_idx to remove absorbed pools
                                y = y2
                                out_shape = tuple(y.shape)
                                end_idx = end_idx + absorbed_count  # <<< critical: remove these layers later
                                if debug:
                                    print(f"[DEBUG] Successfully matched expected HW by absorbing {absorbed_count} pool layer(s). Updated end_idx -> {end_idx}")
                                break
                        else:
                            # stop absorption when encountering a non-pool layer
                            break

                cur_H, cur_W = out_shape[-2], out_shape[-1]
                cur_hw = cur_H * cur_W
                if debug:
                    print(f"[DEBUG] After absorption: HxW={cur_H}x{cur_W}, cur_hw={cur_hw}")

            if cur_hw != expected_hw:
                # fallback: add an adaptive pool to match target HW (approx)
                target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
                target_W = max(1, expected_hw // target_H)
                if (target_H * target_W) != expected_hw:
                    target_H, target_W = 1, expected_hw
                adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))
                out_shape = (out_shape[0], out_shape[1], target_H, target_W)
                if debug:
                    print(f"[DEBUG] Adding AdaptiveAvgPool2d to reach {target_H}x{target_W}")

        if adaptive_pool_to_use is not None:
            pool_layer = adaptive_pool_to_use

        collapsed_block = _build_collapsed_block(
            nn.Conv2d, in_features=in_channels, out_features=out_channels,
            output_shape=out_shape if out_shape is not None else (1, out_channels, 1, 1), full_block=full_block,
            stride=(1, 1), pool_layer=pool_layer,
            linear_in_features=linear_in_features,
            debug=debug
        )

    else:
        # ---- Linear collapse path ----
        in_features = x.view(x.size(0), -1).size(1)
        with torch.no_grad():
            y = x.clone()
            for idx, layer in enumerate(conv_layers):
                y = layer(y)
                if debug:
                    print(f"[DEBUG] After {type(layer).__name__:<20}: shape = {tuple(y.shape)}")
        out_features = y.view(y.size(0), -1).size(1)
        collapsed_block = _build_collapsed_block(
            nn.Linear, in_features=in_features, out_features=out_features,
            output_shape=tuple(y.shape), full_block=full_block, debug=debug
        )

    # IMPORTANT: use the possibly-updated end_idx (if pools were absorbed) to remove absorbed pools
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)

    # >>> PATCH START: handle root-level (empty container path) safely and adjust classifier if needed
    if start_container_name == "":
        updated_container = updated_container.to(device)
        model = updated_container
    else:
        _update_container(model, start_container_name, updated_container)
        model.to(device)

    # Recompute flattened output size (best-effort) from out_shape if available
    try:
        if out_shape is None:
            # fallback: run a probe forward to compute shape by hooking at start_layer_name input again
            _, captured_post = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
            out_shape = tuple(captured_post.shape)
        if len(out_shape) >= 4:
            new_flat = out_shape[1] * out_shape[-2] * out_shape[-1]
        else:
            # linear-like output
            new_flat = 1
            for d in out_shape[1:]:
                new_flat *= d
    except Exception:
        new_flat = None

    # If there's a classifier Linear and it expects a different flattened size, replace it
    if new_flat is not None:
        classifier_linear_name_post = None
        classifier_linear_mod_post = None
        for nm, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                classifier_linear_name_post = nm
                classifier_linear_mod_post = mod
                break

        if classifier_linear_mod_post is not None:
            expected_in = classifier_linear_mod_post.in_features
            if expected_in != new_flat:
                out_f = classifier_linear_mod_post.out_features
                bias_flag = classifier_linear_mod_post.bias is not None
                new_lin = nn.Linear(new_flat, out_f, bias=bias_flag).to(device)
                # sensible init
                try:
                    nn.init.kaiming_uniform_(new_lin.weight, a=math.sqrt(5))
                    if bias_flag:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_lin.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                        nn.init.uniform_(new_lin.bias, -bound, bound)
                except Exception:
                    pass

                try:
                    _set_module_by_path(model, classifier_linear_name_post, new_lin)
                    print(f"[INFO] Replaced classifier Linear '{classifier_linear_name_post}' in_features {expected_in} -> {new_flat}")
                except Exception as e:
                    print(f"[WARN] Failed to replace classifier Linear '{classifier_linear_name_post}': {e}")
    # >>> PATCH END

    model.to(device)
    post_params = count_trainable_params(model)
    t_elapsed = time.time() - t_start

    # ---- Summary block report ----
    print("\n[SUMMARY] Block collapse results:")
    print(f"   Block:           {start_layer_name} → {end_layer_name}")
    print(f"   Layers collapsed: {len(full_block)} ({conv_count} Conv, {pool_count} Pool, {relu_count} ReLU)")
    print(f"   Input shape:     {tuple(x.shape)}")
    print(f"   Output shape:    {out_shape}")
    print(f"   Params before:   {pre_params:,}")
    print(f"   Params after:    {post_params:,}")
    print(f"   ΔParams:         {pre_params - post_params:+,}")
    print(f"   Time taken:      {t_elapsed:.3f}s")
    print(f"   Pool used:       {type(pool_layer).__name__ if pool_layer else 'None'}")
    print(f"   Adaptive pool:   {bool(adaptive_pool_to_use)}")
    print("[SUMMARY] ----------------------------------------------")

    return model


# -----------------------------------------------------------------------------
# Top-level multi-block collapse function (ENHANCED DEBUG VERSION)
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
    Collapses multiple layer blocks in a model with detailed progress and debug summaries.
    """
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
    else:
        pass

    model = deepcopy(model).to(device)
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

    total_blocks = len(collapse_map)
    for idx, (name, (start, end)) in enumerate(collapse_map.items(), 1):
        print(f"\n[INFO] ---- ({idx}/{total_blocks}) Processing collapse '{name}': {start} → {end} ----")
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
        print("[WARN] ⚠ Model has MORE parameters after collapse — investigate collapse policy.")

    # if debug:
    #     print(f"[DEBUG] Model summary after collapse:\n{layer_stats(model)}")

    return model
