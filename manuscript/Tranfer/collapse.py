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
    inherit_conv_attrs: bool = True,
    force_hw: Optional[Tuple[int,int]] = None,  # NEW: force HxW
) -> nn.Sequential:
    if debug:
        print(f"[DEBUG] _build_collapsed_block called: layer_type={getattr(layer_type,'__name__',str(layer_type))}, in={in_features}, out={out_features}, linear_in_features={linear_in_features}, force_hw={force_hw}")

    seq = []
    original_param_budget = _count_params_for_block(full_block) if full_block else None

    if layer_type == nn.Conv2d:
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block) if full_block else False
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block) if full_block else False

        # inherit conv attributes
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

        # H*W from output or forced
        H = force_hw[0] if force_hw else output_shape[-2]
        W = force_hw[1] if force_hw else output_shape[-1]

        # suggested output channels for linear
        suggested_out = out_features
        if linear_in_features is not None:
            suggested_out = max(1, linear_in_features // (H * W))
            if debug:
                print(f"[DEBUG] Linear follower present: target channels ≈ {suggested_out} (H*W={H*W})")

        if shortcut_out_channels:
            suggested_out = min(suggested_out, shortcut_out_channels)
            if debug:
                print(f"[DEBUG] Honoring shortcut output channels cap: {shortcut_out_channels}")

        # initial collapsed out
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

        # safety clamp
        collapse_out = max(1, min(collapse_out, out_features))

        # build conv
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

# -----------------------------------------------------------------------------
# Core collapse of a single block
# -----------------------------------------------------------------------------
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

    Robust alignment strategy:
      - Find the next nn.Linear after the end_layer_name in global named_modules order.
      - Build & replace the collapsed block.
      - Run the collapsed block on a captured pre-block activation to compute actual flattened size.
      - If actual != linear.in_features, insert AdaptiveAvgPool2d to force the correct HxW.
      - Re-validate by simulating forward into the linear's input.
    """
    import math

    print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    # --- locate container & block indices (unchanged logic) ---
    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find start/end layers '{start_layer_name}'/'{end_layer_name}'.")

    full_block = named_layers[start_idx:end_idx + 1]
    conv_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not conv_layers:
        raise ValueError("No Conv2d/Linear layers found in block to collapse.")
    layer_type = type(conv_layers[0])
    if not all(isinstance(l, layer_type) for l in conv_layers):
        raise ValueError("Cannot collapse mixed layer types inside one block.")

    # --- capture input activation (pre-block) ---
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
        print(f"[WARN] Hook failed: {e}. Using dummy input shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Params before collapse: {pre_params:,}")

    # --- find "next Linear" in global traversal after this block (robust) ---
    modules_list = list(model.named_modules())
    idx_end_global = None
    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            break
    # fallback: if exact name not found, try to match by suffix (rare)
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

    # --- forward through original full_block to know out_shape and candidate pool --- 
    with torch.no_grad():
        y = x.clone()
        last_conv = None
        for _, layer in full_block:
            y = layer(y)
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
    out_shape = tuple(y.shape)
    out_channels = last_conv.out_channels if last_conv is not None else (conv_layers[-1].out_channels if layer_type == nn.Conv2d else None)
    if debug:
        print(f"[DEBUG] Full block output shape (after original block): {out_shape}, out_channels={out_channels}")

    # detect pool in the block (to re-append)
    pool_layer = next((m for _, m in reversed(full_block) if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)
    if debug and pool_layer is not None:
        print(f"[DEBUG] Detected pool in original block: {type(pool_layer).__name__}")

    # --- attempt to absorb contiguous post-block pools if they help match linear input ---
    linear_in_features_heuristic = None
    if next_linear_mod is not None:
        linear_in_features_heuristic = next_linear_mod.in_features
    if debug:
        print(f"[DEBUG] Heuristic linear_in_features = {linear_in_features_heuristic}")

    adaptive_pool_to_use = None
    if layer_type == nn.Conv2d and linear_in_features_heuristic is not None and out_channels:
        expected_hw = max(1, linear_in_features_heuristic // out_channels)
        cur_H, cur_W = out_shape[-2], out_shape[-1]
        cur_hw = cur_H * cur_W
        if debug:
            print(f"[DEBUG] expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})")

        if cur_hw != expected_hw:
            extra_pools = []
            with torch.no_grad():
                y2 = y.clone()
                for _, mod in named_layers[end_idx + 1:]:
                    if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                        try:
                            y2 = mod(y2)
                        except Exception:
                            if debug:
                                print(f"[DEBUG] Post-block pool {type(mod).__name__} underflow; stop absorption")
                            break
                        extra_pools.append(mod)
                        new_H, new_W = y2.shape[-2], y2.shape[-1]
                        if debug:
                            print(f"[DEBUG] Applied pool {type(mod).__name__}, new HxW={new_H}x{new_W}")
                        if (new_H * new_W) == expected_hw:
                            # integrate these extra pools
                            if len(extra_pools) == 1:
                                pool_layer = extra_pools[0]
                            else:
                                pool_layer = nn.Sequential(OrderedDict([(f"pool_{i}", p) for i, p in enumerate(extra_pools)]))
                            y = y2
                            out_shape = tuple(y.shape)
                            if debug:
                                print(f"[DEBUG] Absorbed post-block pools => updated out_shape {out_shape}")
                            break

            # recompute
            cur_H, cur_W = out_shape[-2], out_shape[-1]
            cur_hw = cur_H * cur_W
            if cur_hw != expected_hw:
                target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
                target_W = max(1, expected_hw // target_H)
                adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))
                pool_layer = adaptive_pool_to_use
                out_shape = (out_shape[0], out_shape[1], target_H, target_W)
                if debug:
                    print(f"[DEBUG] Plan to insert AdaptiveAvgPool2d to force HxW={target_H}x{target_W} (expected_hw={expected_hw})")

    # detect shortcut-out-channels for residual-style blocks (unchanged)
    shortcut_out_channels = None
    for nm, mod in model.named_modules():
        if hasattr(mod, 'shortcut') and isinstance(mod.shortcut, nn.Module):
            first_conv = next((m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)), None)
            if first_conv is not None:
                shortcut_out_channels = first_conv.out_channels
                break
    if debug and shortcut_out_channels is not None:
        print(f"[DEBUG] Detected shortcut_out_channels = {shortcut_out_channels}")

    # --- construct collapsed block (calls your _build_collapsed_block) ---
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
            linear_in_features=linear_in_features_heuristic,
            shortcut_out_channels=shortcut_out_channels,
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
            nn.Linear,
            in_features=in_features,
            out_features=out_features,
            output_shape=tuple(y.shape),
            full_block=full_block,
            debug=debug
        )

    # --- Replace the block inside container ---
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Params after collapse: {post_params:,}")
    print(f"[INFO] ΔParams = {pre_params - post_params:+,} (should be >= 0)")

    # --- downstream underflow scan (unchanged) ---
    try:
        if debug:
            print(f"[DEBUG] Scanning downstream modules for unsafe pools...")
        container_after = get_layer(model, start_container_name)
        children = list(container_after.named_children())
        collapsed_idx = next((i for i, (nm, _) in enumerate(children) if nm.startswith("collapsed_")), start_idx)
        collapsed_mod = children[collapsed_idx][1]
        test_tensor = x.clone().to(device)
        with torch.no_grad():
            test_tensor = collapsed_mod(test_tensor)

        for i, (nm, mod) in enumerate(children):
            if i <= collapsed_idx:
                continue
            try:
                with torch.no_grad():
                    test_tensor = mod(test_tensor)
            except RuntimeError as e:
                s = str(e)
                if "size is too small" in s or "Calculated output size" in s or "Kernel size can't be greater than actual input size" in s:
                    if debug:
                        print(f"[WARN] Downstream module '{start_container_name}.{nm}' underflows; replacing with Identity")
                    if isinstance(container_after, nn.Sequential):
                        new_od = OrderedDict()
                        for j, (n2, m2) in enumerate(children):
                            new_od[n2] = nn.Identity() if n2 == nm else m2
                        _update_container(model, start_container_name, nn.Sequential(new_od))
                        container_after = get_layer(model, start_container_name)
                        children = list(container_after.named_children())
                        collapsed_mod = children[collapsed_idx][1]
                    else:
                        setattr(container_after, nm, nn.Identity())
                        container_after = get_layer(model, start_container_name)
                        children = list(container_after.named_children())
                        collapsed_mod = children[collapsed_idx][1]
                else:
                    raise
    except Exception as e:
        print(f"[WARN] Error scanning downstream modules: {e}")

    # --- NEW: robust detection & corrective insertion for next Linear ---
    if next_linear_mod is not None:
        try:
            # run collapsed module (on the captured pre-block activation) to get actual flattened size
            collapsed_mod_after = get_layer(model, start_container_name)
            # locate collapsed module inside container_after
            cont_children = list(collapsed_mod_after.named_children())
            # find actual collapsed sequential (first collapsed_...)
            collapsed_seq = None
            for nm, m in cont_children:
                if nm.startswith("collapsed_"):
                    collapsed_seq = m
                    break
            if collapsed_seq is None:
                # fallback: try to use replacement container itself (sequential)
                collapsed_seq = collapsed_mod_after

            test_in = x.clone().to(device)
            with torch.no_grad():
                out_after = collapsed_seq(test_in)
            flat_actual = out_after.view(out_after.size(0), -1).size(1)
            expected = next_linear_mod.in_features
            if debug:
                print(f"[DEBUG] After collapse: flattened feeding '{next_linear_name}' = {flat_actual}, next Linear expects = {expected}")

            if flat_actual != expected:
                print(f"[WARN] Flattened mismatch for next Linear '{next_linear_name}': actual={flat_actual} expected={expected}. Inserting AdaptiveAvgPool2d to fix.")
                C = out_after.size(1)
                # compute target H*W: prefer exact division, otherwise fall back to 1
                if C > 0 and expected % C == 0:
                    target_hw = expected // C
                else:
                    # not divisible -> choose nearest integer target_hw (best-effort)
                    target_hw = max(1, expected // max(1, C))
                    if debug:
                        print(f"[DEBUG] expected not divisible by channels ({expected}%{C} != 0). Using best-effort target_hw={target_hw}")

                target_H = int(round(math.sqrt(target_hw))) if target_hw > 1 else 1
                target_W = max(1, target_hw // target_H) if target_hw > 1 else 1

                forced_pool = nn.AdaptiveAvgPool2d((target_H, target_W))
                # Prefer to add the forced pool at end of model.features if exists (typical for VGG)
                if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                    if debug:
                        print(f"[DEBUG] Appending forced AdaptiveAvgPool2d({target_H},{target_W}) to model.features")
                    feat_seq = list(model.features.children())
                    feat_seq.append(forced_pool)
                    model.features = nn.Sequential(OrderedDict([(f"layer_{i}", m) for i, m in enumerate(feat_seq)]))
                else:
                    # insert into parent container of the next_linear module if it's a Sequential
                    parent_path = '.'.join(next_linear_name.split('.')[:-1])
                    if parent_path == "":
                        # top-level: wrap model into Sequential is risky; just attach to features if exists, else raise
                        raise RuntimeError("Cannot safely insert corrective pool; model has no 'features' and next Linear is top-level.")
                    parent_container = get_layer(model, parent_path)
                    if isinstance(parent_container, nn.Sequential):
                        # insert forced_pool just before the linear inside that parent sequential
                        pc = list(parent_container.named_children())
                        new_od = OrderedDict()
                        inserted = False
                        for nm_child, mod_child in pc:
                            if not inserted and nm_child == next_linear_name.split('.')[-1]:
                                new_od[f"forced_pool"] = forced_pool
                                inserted = True
                            new_od[nm_child] = mod_child
                        _update_container(model, parent_path, nn.Sequential(new_od))
                        if debug:
                            print(f"[DEBUG] Inserted forced AdaptiveAvgPool2d({target_H},{target_W}) into parent '{parent_path}'")
                    else:
                        raise RuntimeError(f"Cannot insert corrective pool into parent '{parent_path}'; not a Sequential container.")

                # re-validate with hook into the next linear
                _, captured_after_fix = _simulate_input_hook(model, next_linear_name, input_shape, device=device)
                flat_after2 = captured_after_fix.view(captured_after_fix.size(0), -1).size(1)
                if debug:
                    print(f"[DEBUG] After corrective pool: flattened feeding '{next_linear_name}' = {flat_after2} (expected {expected})")
                if flat_after2 != expected:
                    raise RuntimeError(f"Auto-correction failed: flattened {flat_after2} != expected {expected}")

        except Exception as e:
            print(f"[WARN] Classifier alignment validation failed for '{next_linear_name}': {e}")

    else:
        if debug:
            print("[DEBUG] No downstream Linear found to validate against; skipping classifier alignment step.")

    # final sanity check
    if post_params > pre_params:
        print("[WARN] ⚠ Collapsed block has MORE parameters than before! Investigate collapse policy.")

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
    safe_param_reduction: bool = True,   # currently enforced by builder
    handle_skips: bool = True,
    debug: bool = True,
    dry_run: bool = False
) -> nn.Module:
    """
    Top-level API to collapse multiple blocks.

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
        # expect checkpoint with key 'model' or whole state_dict
        state = chk.get('model', chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)
    else:
        # we got a model instance; we will deep-copy it so original is preserved
        pass

    model = deepcopy(model).to(device)
    model.eval()

    # normalize compression_set into OrderedDict-like mapping name -> (start,end)
    if compression_set is None:
        print("[WARN] compression_set is empty; nothing to do.")
        return model

    # accept list of tuples or mapping
    if isinstance(compression_set, dict):
        collapse_map = compression_set
    else:
        # list/sequence -> convert to dict with generated names
        collapse_map = {f"collapse_{i}": tuple(pair) for i, pair in enumerate(compression_set)}

    # store collapsed ranges for downstream patching
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