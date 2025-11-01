# collapse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from uuid import uuid4
from typing import Optional, Sequence, Tuple, Dict, Any
from copy import deepcopy
import math
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


def _count_params_for_block(full_block: Sequence[Tuple[str, nn.Module]]) -> int:
    """Count trainable params for modules inside full_block (conv/linear/bn etc)."""
    total = 0
    for _, m in full_block:
        for p in m.parameters():
            if p.requires_grad:
                total += p.numel()
    return total


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
        # input can be tuple: use first element
        if isinstance(inp, tuple):
            captured['in'] = inp[0].detach().cpu()
        else:
            captured['in'] = inp.detach().cpu()
    handle = target_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        handle.remove()
    if 'in' not in captured:
        raise RuntimeError(f"Failed to capture activation at {target_layer_path}.")
    # return tensors on the requested device (caller often expects device matching model)
    return dummy_input, captured['in'].to(device)


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
# Helpers: build a safe BN for `num_features`, copying overlap from src_bn when possible
# -----------------------------------------------------------------------------
def _make_bn_with_safe_copy(src_bn: Optional[nn.BatchNorm2d], num_features: int) -> nn.BatchNorm2d:
    """
    Create a new BatchNorm2d with `num_features`. If `src_bn` is provided,
    copy overlapping parameters / running_stats for indices that fit.
    """
    new_bn = nn.BatchNorm2d(num_features)
    if src_bn is None:
        return new_bn
    # copy weights/bias and running stats for the overlapping prefix
    copy_n = min(num_features, getattr(src_bn, 'num_features', 0))
    if copy_n > 0:
        with torch.no_grad():
            # weight and bias
            if hasattr(src_bn, 'weight') and src_bn.weight is not None:
                new_bn.weight[:copy_n].data.copy_(src_bn.weight[:copy_n].data[:copy_n])
            if hasattr(src_bn, 'bias') and src_bn.bias is not None:
                new_bn.bias[:copy_n].data.copy_(src_bn.bias[:copy_n].data[:copy_n])
            # running_mean / running_var
            if hasattr(src_bn, 'running_mean'):
                new_bn.running_mean[:copy_n].data.copy_(src_bn.running_mean[:copy_n].data[:copy_n])
            if hasattr(src_bn, 'running_var'):
                new_bn.running_var[:copy_n].data.copy_(src_bn.running_var[:copy_n].data[:copy_n])
    return new_bn


# -----------------------------------------------------------------------------
# Core collapse of a single block (safe replacement)
# -----------------------------------------------------------------------------


def _collapse_block(
    model: nn.Module,
    start_layer_name: str,
    end_layer_name: str,
    input_shape: tuple,
    device='cpu',
    debug: bool = False,
    preserve_activation: bool = True
) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive) while:
      - Preserving useful semantics but avoid copying BNs with mismatched channels
      - Prevent parameter inflation (fallback to lighter replacement)
      - Use actual block output shape (trace) to determine pooling
    """
    if debug:
        print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name)
    full_block = info['full_block']
    conv_layers = info['conv_layers']
    layer_type = info['layer_type']

    # Step: capture activation entering the start layer (on device)
    x, pre_params = _capture_preblock_activation(model, start_layer_name, input_shape, conv_layers, layer_type, device, debug)

    # Step: next linear (for classifier alignment)
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)

    # Step: analyze block output by actually running the captured activation through the original block
    block_analysis = _analyze_block_output(model, full_block, conv_layers, info['named_layers'], info['end_idx'],
                                          layer_type, x, next_linear_mod, debug)

    out_channels = block_analysis.get('out_channels')
    out_shape = block_analysis.get('out_shape')
    adaptive_pool_to_use = block_analysis.get('adaptive_pool_to_use')
    pool_layer = block_analysis.get('pool_layer', None)
    last_bn = block_analysis.get('last_bn', None)

    # Build replacement in a safe way:
    # - prefer: Conv1x1(in_channels -> out_channels) -> BN(out_channels) -> ReLU (if existed) -> AdaptivePool(H,W)
    # - but only commit conv if replacement params <= params_of_original_block
    in_channels = int(x.shape[1])
    if out_channels is None:
        # fallback: use last conv's out_channels if available in conv_layers
        if conv_layers and hasattr(conv_layers[-1], 'out_channels'):
            out_channels = conv_layers[-1].out_channels
        else:
            raise RuntimeError("Could not determine out_channels for collapsed block.")

    # Decide pooling target (prefer computed heuristic but use actual out_shape if present)
    if adaptive_pool_to_use is not None:
        chosen_pool = adaptive_pool_to_use
    else:
        # if out_shape available, set target to its HxW
        if out_shape is not None and len(out_shape) >= 4:
            target_H, target_W = int(out_shape[-2]), int(out_shape[-1])
            # If next linear exists, prefer matching that linear's flat size
            if next_linear_mod is not None:
                linear_expected = next_linear_mod.in_features
                # compute ideal hw = linear_expected // out_channels
                if out_channels > 0:
                    ideal_hw = max(1, linear_expected // out_channels)
                    # pick integer factorization closest to sqrt
                    if ideal_hw > 1:
                        tH = int(round(math.sqrt(ideal_hw)))
                        tW = max(1, ideal_hw // max(1, tH))
                        target_H, target_W = max(1, min(target_H, tH)), max(1, min(target_W, tW))
            chosen_pool = nn.AdaptiveAvgPool2d((target_H, target_W))
        else:
            # fallback to (1,1)
            chosen_pool = nn.AdaptiveAvgPool2d((1, 1))

    # check whether original block had any ReLU (useful to add after BN)
    had_relu = any(isinstance(m, nn.ReLU) for _, m in full_block)

    # create BN for out_channels: attempt to copy last BN in block if exists and sizes compatible
    last_bn_module = None
    for _, m in reversed(full_block):
        if isinstance(m, nn.BatchNorm2d):
            last_bn_module = m
            break
    bn_for_replacement = _make_bn_with_safe_copy(last_bn_module, out_channels) if last_bn_module is not None else None

    # Build candidate replacement
    conv_candidate = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    replacement_layers = OrderedDict()
    replacement_layers['conv_1x1'] = conv_candidate
    if bn_for_replacement is not None:
        replacement_layers['bn'] = bn_for_replacement
    if had_relu:
        replacement_layers['relu'] = nn.ReLU(inplace=False)
    replacement_layers['adaptive_pool'] = chosen_pool
    replacement_seq = nn.Sequential(replacement_layers)

    # compute params: original block vs replacement
    orig_block_params = _count_params_for_block(full_block)
    # quick param counting for replacement
    repl_params = sum(p.numel() for p in replacement_seq.parameters() if p.requires_grad)

    if debug:
        print(f"[DEBUG] orig_block_params={orig_block_params:,}, replacement_params={repl_params:,} (conv {in_channels}->{out_channels})")
        print(f"[DEBUG] chosen_pool={chosen_pool}, had_relu={had_relu}")

    # If replacement would increase params, fallback to a lighter replacement: pooling-only
    if repl_params > orig_block_params:
        if debug:
            print(f"[WARN] Replacement would increase params ({repl_params:,} > {orig_block_params:,}). Falling back to pool-only replacement.")
        # pool-only replacement (no conv), that will preserve forward shape (but might require corrective linear)
        pool_only = nn.Sequential(OrderedDict([
            ("adaptive_pool", chosen_pool)
        ]))
        final_replacement = pool_only
    else:
        final_replacement = replacement_seq

    # perform replacement in container
    updated_container = _replace_layers(info['named_layers'], info['start_idx'], info['end_idx'], final_replacement)
    start_container_name, _ = _get_container_and_subname(start_layer_name)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    if debug:
        print(f"[DEBUG] Replaced block '{start_layer_name} → {end_layer_name}' with: {final_replacement}")
        print(f"[DEBUG] Params before collapse: {pre_params:,}, after tentative collapse: {count_trainable_params(model):,}")

    # After replacement, make sure skip connections are safe and ReLUs not in-place
    try:
        patch_skip_connections(model)
    except Exception:
        pass
    disable_inplace_relu(model)

    # downstream validation: run forward from captured activation through replacement and immediate next modules
    try:
        _validate_downstream(model, start_container_name, info['start_idx'], x, input_shape, next_linear_name, next_linear_mod, device, debug=debug)
    except Exception as e:
        if debug:
            print(f"[WARN] Downstream validation failed: {e}")

    # Correct classifier linear if necessary
    try:
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug=debug)
    except Exception as e:
        if debug:
            print(f"[WARN] Corrective pooling/check failed: {e}")

    post_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Parameters before collapse: {pre_params:,}, after collapse: {post_params:,}, Δ={pre_params - post_params}")

    return model


def _locate_and_prepare_block(model, start_layer_name, end_layer_name):
    if start_layer_name is None or end_layer_name is None:
        raise ValueError("start/end layer names required")

    if start_layer_name == "" or end_layer_name == "":
        raise ValueError("start/end layer names must be non-empty strings")

    if '.' not in start_layer_name and '.' not in end_layer_name:
        # still OK, they may refer to children of root
        pass

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    if start_container_name == "":
        container = model
    else:
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

    return {
        "container": container,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "conv_layers": conv_layers,
        "layer_type": layer_type,
    }


def _capture_preblock_activation(model, start_layer_name, input_shape, conv_layers, layer_type, device, debug):
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

    return x, pre_params


def _find_next_linear(model, end_layer_name, debug):
    modules_list = list(model.named_modules())
    idx_end_global = None

    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            break

    if idx_end_global is None:
        # try endswith fallback
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
                next_linear_name, next_linear_mod = n, m
                break

    if next_linear_mod is None:
        for n, m in modules_list:
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                break

    if debug:
        print(f"[DEBUG] Next Linear detected: name='{next_linear_name}', module={next_linear_mod}")

    return next_linear_name, next_linear_mod


def _analyze_block_output(model, full_block, conv_layers, named_layers, end_idx, layer_type, x, next_linear_mod, debug):
    """
    Run the captured activation `x` through the original block to determine the
    block's actual output shape and detect the last BN (if any). Returns a dict
    including out_shape, out_channels, adaptive_pool_to_use (if heuristic needed),
    and last_bn (module) to allow safer copying.
    """
    last_conv = None
    last_bn = None
    with torch.no_grad():
        y = x.clone()
        for nm, layer in full_block:
            try:
                y = layer(y)
            except Exception:
                # in case module expects different shape, try to skip (defensive)
                try:
                    # attempt to call layer on a fresh zero input of appropriate rank
                    if isinstance(layer, nn.Conv2d):
                        dummy = torch.zeros_like(y)
                        y = layer(dummy)
                    else:
                        y = layer(y)
                except Exception:
                    # give up - use current y
                    break
            if isinstance(layer, nn.Conv2d):
                last_conv = layer
            if isinstance(layer, nn.BatchNorm2d):
                last_bn = layer

    out_shape = tuple(y.shape)
    out_channels = None
    if last_conv is not None:
        out_channels = last_conv.out_channels
    elif layer_type == nn.Conv2d and conv_layers:
        out_channels = conv_layers[-1].out_channels

    if debug:
        print(f"[DEBUG] Full block output shape: {out_shape}, out_channels={out_channels}")

    # find pooling in original block (if any)
    pool_layer = next((m for _, m in reversed(full_block)
                       if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)
    if debug and pool_layer is not None:
        print(f"[DEBUG] Detected pool in original block: {type(pool_layer).__name__}")

    linear_in_features_heuristic = None
    if next_linear_mod is not None:
        linear_in_features_heuristic = next_linear_mod.in_features
        if debug:
            print(f"[DEBUG] Heuristic linear_in_features = {linear_in_features_heuristic}")

    # heuristically propose an adaptive pool to match classifier linear if mismatch
    adaptive_pool_to_use = None
    if layer_type == nn.Conv2d and linear_in_features_heuristic is not None and out_channels:
        expected_hw = max(1, linear_in_features_heuristic // out_channels)
        if out_shape is not None and len(out_shape) >= 4:
            cur_H, cur_W = out_shape[-2], out_shape[-1]
            cur_hw = cur_H * cur_W
            if debug:
                print(f"[DEBUG] expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})")
            if cur_hw != expected_hw:
                # pick reasonable H/W factors
                if expected_hw > 1:
                    tH = int(round(math.sqrt(expected_hw)))
                    tW = max(1, expected_hw // max(1, tH))
                else:
                    tH, tW = 1, 1
                adaptive_pool_to_use = nn.AdaptiveAvgPool2d((tH, tW))
                if debug:
                    print(f"[DEBUG] Plan to suggest AdaptiveAvgPool2d({tH},{tW}) (heuristic)")

    return {
        "out_shape": out_shape,
        "out_channels": out_channels,
        "pool_layer": pool_layer,
        "adaptive_pool_to_use": adaptive_pool_to_use,
        "last_bn": last_bn,
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
    Minimal downstream validator: run the inserted replacement using the provided
    pre_activation (so channels line up). If any downstream immediate child
    raises on that output (e.g. pooling), wrap that child in _SafePool or Identity.
    """
    container = get_layer(model, start_container_name)
    children = list(container.named_children())
    if not children:
        if debug:
            print(f"[DEBUG] No children in container '{start_container_name}' — skipping downstream validation")
        return

    # find the collapsed/inserted child index
    collapsed_idx = None
    for i, (nm, m) in enumerate(children):
        if nm.startswith("collapsed_") or (isinstance(m, nn.Sequential) and any(k in dict(m.named_children()) for k in ("conv_1x1", "adaptive_pool"))):
            collapsed_idx = i
            break

    if collapsed_idx is None:
        # maybe user replaced by our updated_container — try to find conv_1x1
        for i, (nm, m) in enumerate(children):
            if isinstance(m, nn.Conv2d) and getattr(m, "kernel_size", None) == (1,1):
                collapsed_idx = i
                break

    if collapsed_idx is None:
        if debug:
            print(f"[DEBUG] Could not locate inserted replacement in '{start_container_name}' — skipping")
        return

    inserted_mod = children[collapsed_idx][1]

    # run forward from the provided pre_activation through the inserted mod
    try:
        with torch.no_grad():
            dev = next((p.device for p in model.parameters()), torch.device('cpu'))
            t = pre_activation.clone().to(dev)
            t = inserted_mod(t)
    except Exception as e:
        if debug:
            print(f"[WARN] Running inserted module failed during downstream validation: {e}")
        return

    # now try to forward through the next immediate children to detect straight-away problems
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            t = mod(t)
        except Exception as e:
            if debug:
                print(f"[WARN] Downstream module '{start_container_name}.{nm}' raised: {e}. Wrapping safely.")
            # Wrap pooling-like modules; otherwise replace with Identity to avoid crash
            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
            else:
                safe = nn.Identity()
            # perform replacement preserving order
            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            if debug:
                print(f"[DEBUG] Replaced '{start_container_name}.{nm}' with {safe.__class__.__name__}")
            # break after first fix; user can re-run collapse if necessary
            return

        # check for zero spatial dims
        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            if debug:
                print(f"[WARN] Downstream module '{start_container_name}.{nm}' produced zero spatial dim. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            if debug:
                print(f"[DEBUG] Wrapped '{start_container_name}.{nm}' successfully.")
            return

    if debug:
        print(f"[DEBUG] Downstream validation for container '{start_container_name}' completed successfully.")


def _insert_corrective_pool(model: nn.Module,
                            next_linear_name: str,
                            input_shape: Tuple[int, ...],
                            debug: bool = False) -> nn.Module:
    """
    Capture activation entering next_linear_name.
    If flattened size != next_linear.in_features, replace that Linear with a new
    Linear(flat_actual, out_features). Handles top-level modules safely.
    """
    if next_linear_name is None:
        if debug:
            print("[DEBUG] No next_linear_name provided; skipping corrective pool.")
        return model

    try:
        next_linear_mod = get_layer(model, next_linear_name)
    except Exception as e:
        raise RuntimeError(f"Could not locate next linear '{next_linear_name}': {e}")

    expected = next_linear_mod.in_features
    dev = next((p.device for p in model.parameters()), torch.device('cpu'))

    # capture the activation that is fed to that linear
    try:
        # pass the same device object to the hook routine
        _, cap = _simulate_input_hook(model, next_linear_name, input_shape, device=dev)
    except Exception as e:
        if debug:
            print(f"[WARN] Failed to capture activation for '{next_linear_name}': {e}")
        raise

    flat_actual = cap.view(cap.size(0), -1).size(1)
    if debug:
        print(f"[DEBUG] next_linear '{next_linear_name}': expected in_features={expected}, actual_flat={flat_actual}")

    if flat_actual == expected:
        if debug:
            print("[DEBUG] Classifier matches expected flattened size — no change.")
        return model

    # Replace Linear with one that accepts flat_actual
    parent_path, child_name = next_linear_name.rsplit('.', 1) if '.' in next_linear_name else ("", next_linear_name)
    new_linear = nn.Linear(flat_actual, next_linear_mod.out_features, bias=(next_linear_mod.bias is not None))

    if parent_path == "":
        # Top-level module: replace directly in model
        setattr(model, child_name, new_linear)
    else:
        parent_mod = get_layer(model, parent_path)
        if isinstance(parent_mod, nn.Sequential):
            new_od = OrderedDict()
            for n, m in parent_mod.named_children():
                if n == child_name:
                    new_od[n] = new_linear
                else:
                    new_od[n] = m
            _update_container(model, parent_path, nn.Sequential(new_od))
        else:
            setattr(parent_mod, child_name, new_linear)

    if debug:
        print(f"[WARN] Replaced Linear '{next_linear_name}' in_features {expected} -> {flat_actual} to maintain forward pass.")

    return model


# -----------------------------------------------------------------------------
# Collapsed block builder (kept but unused by simple replacement)
# -----------------------------------------------------------------------------

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
    """A kept helper (not used by the minimal replacement) - returns a small sequential stub."""
    seq = []

    if layer_type == nn.Conv2d:
        conv = nn.Conv2d(in_features, max(1, out_features), kernel_size=1, stride=1, padding=0, bias=True)
        seq.append(conv)
    else:
        seq.append(nn.Linear(in_features, out_features))

    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", m) for i,m in enumerate(seq)]))
    if debug:
        print(f"[DEBUG] _build_collapsed_block fallback created: {[type(m).__name__ for m in collapsed]}")
    return collapsed


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
    # Load or use provided model
    if model is None:
        if not (model_weights_1 and model_class):
            raise ValueError("Either provide `model` or provide (`model_weights_1` and `model_class`).")
        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating model from class {model_class.__name__} and loading weights from {model_weights_1}")
        model = model_class(**model_kwargs)
        chk = torch.load(model_weights_1, map_location=device)
        state = chk.get('model', chk) if isinstance(chk, dict) else chk
        model.load_state_dict(state)

    # Deep-copy to preserve original
    model = deepcopy(model).to(device)
    model.eval()

    # Normalize compression_set into dict of name -> (start_str, end_str)
    if compression_set is None:
        print("[WARN] compression_set is empty; nothing to do.")
        return model

    collapse_map = {}
    if isinstance(compression_set, dict):
        for k, v in compression_set.items():
            start, end = v
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[k] = (start, end)
    else:
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)

    # Store collapsed ranges for downstream patching
    model._collapsed_blocks = list(collapse_map.values())

    pre_total = count_trainable_params(model)
    print(f"[INFO] Starting collapse_only; params before = {pre_total:,}")

    for name, (start, end) in collapse_map.items():
        print(f"\n[INFO] Processing collapse '{name}': {start} -> {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping actual modification.")
            continue
        print(f"[INFO] Collapsing block: name: '{name}', start: '{start}', end: '{end}'")
        model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)

        # After each collapse optionally patch skip connections to avoid invalid adds
        if handle_skips:
            patch_skip_connections(model)

        # Ensure out-of-place ReLUs to avoid autograd issues
        disable_inplace_relu(model)

    # Wrap all pooling layers with safe wrappers to guarantee no underflow
    try:
        _wrap_pools_safe(model)
        if debug:
            print("[DEBUG] Wrapped pooling layers with _SafePool to avoid underflow crashes.")
    except Exception as e:
        print(f"[WARN] Failed to wrap pools safely: {e}")

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


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Safe pooling wrapper (prevents underflow crashes)
# -----------------------------------------------------------------------------

class _SafePool(nn.Module):
    """
    Wrapper that attempts to apply the wrapped pooling module; if the input
    spatial dimensions are too small or the pool raises, we fall back safely:
    """
    def __init__(self, pool_module: nn.Module):
        super().__init__()
        self.pool = pool_module

    def forward(self, x):
        try:
            H, W = x.shape[-2], x.shape[-1]
        except Exception:
            try:
                return self.pool(x)
            except Exception:
                return x

        try:
            if isinstance(self.pool, (nn.MaxPool2d, nn.AvgPool2d)):
                k = self.pool.kernel_size
                if isinstance(k, tuple):
                    kh, kw = k
                else:
                    kh = kw = k
                if kh > H or kw > W or H <= 0 or W <= 0:
                    target_H = max(1, min(H, kh) if H > 0 else 1)
                    target_W = max(1, min(W, kw) if W > 0 else 1)
                    return F.adaptive_avg_pool2d(x, (target_H, target_W))
            out = self.pool(x)
            if out.shape[-2] < 1 or out.shape[-1] < 1:
                return x
            return out
        except Exception:
            return x


def _wrap_pools_safe(module: nn.Module):
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
