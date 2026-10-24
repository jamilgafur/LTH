"""Block location, surrogate replacement, and post-replacement validation."""

import math
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_utils import (
    _get_container_and_subname,
    _log_exception,
    _shape_str,
    _simulate_input_output_hooks,
    _update_container,
    count_trainable_params,
    get_layer,
)
from .identity import SmartIdentity, _SafePool


class _SurrogateLayerNorm2d(nn.Module):
    """Channels-first LayerNorm surrogate (as used by ConvNeXt-style blocks)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


_NORM_CLASSES = (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm2d)
_ACT_CLASSES = (nn.ReLU, nn.GELU, nn.SiLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.Hardswish)


def _detect_norm_act(full_block):
    """Return the last (norm_module, act_module) used inside the excised block,
    or (None, None) when the block contains neither. Content-based replacement
    of any hardcoded per-architecture norm/activation choices."""
    norm_mod = act_mod = None
    for _, top in full_block:
        for m in top.modules():
            if isinstance(m, _NORM_CLASSES):
                norm_mod = m
            if isinstance(m, _ACT_CLASSES):
                act_mod = m
    return norm_mod, act_mod


def _init_surrogate_lsq(conv, x, y_out, max_points=65536, ridge=1e-4, debug=False):
    """Ridge least-squares fit of the surrogate conv weights so that
    ``conv(x_in) ~= y_out`` under the captured probe statistics.

    Works for both plain 1x1 convolutions and grouped (depthwise-style)
    convolutions; the regression is solved independently per group.
    """
    with torch.no_grad():
        dev = conv.weight.device
        xs = x.to(dev).detach().float()
        ys = y_out.to(dev).detach().float()

        kh, kw = conv.kernel_size
        sh, sw = conv.stride
        ph, pw = conv.padding
        xs = F.pad(xs, (pw, pw, ph, ph))

        patches = F.unfold(xs, kernel_size=(kh, kw), stride=(sh, sw))  # (N, C_in*k*k, L)
        ycols = ys.reshape(ys.shape[0], ys.shape[1], -1)               # (N, C_out, L)

        if ycols.shape[0] != patches.shape[0] or ycols.shape[-1] != patches.shape[-1]:
            raise RuntimeError(
                f"LSQ alignment mismatch: {patches.shape[-1]} patches vs "
                f"{ycols.shape[-1]} output positions."
            )

        groups = conv.groups
        in_per = conv.in_channels // groups
        out_per = conv.out_channels // groups
        kk = kh * kw

        new_weight = torch.zeros_like(conv.weight, dtype=torch.float32)
        for n in range(patches.shape[0]):
            X = patches[n]  # (C_in*k*k, L)
            Y = ycols[n]    # (C_out, L)
            L = X.shape[-1]
            if L > max_points:
                idx = torch.randperm(L, device=dev)[:max_points]
                X, Y = X[:, idx], Y[:, idx]
            for g in range(groups):
                Xg = X[g * in_per * kk:(g + 1) * in_per * kk, :]  # (in_per*kk, L)
                Yg = Y[g * out_per:(g + 1) * out_per, :]          # (out_per, L)
                A = Xg @ Xg.T
                A.diagonal().add_(ridge * A.diagonal().mean().clamp(min=1e-8))
                Wg = (Yg @ Xg.T) @ torch.linalg.inv(A)            # (out_per, in_per*kk)
                new_weight[g * out_per:(g + 1) * out_per] = Wg.reshape(out_per, in_per, kh, kw)

        conv.weight.copy_(new_weight.to(conv.weight.dtype))
        if debug:
            print(f"[DEBUG] LSQ surrogate init done (mean |W|={new_weight.abs().mean():.4e})")


def _locate_and_prepare_block(model, start_layer_name, end_layer_name, debug=False):
    print(f"[DEBUG] [LCA] Locating block boundaries: '{start_layer_name}' -> '{end_layer_name}'")

    start_parts = start_layer_name.split('.') if start_layer_name else []
    end_parts = end_layer_name.split('.') if end_layer_name else []
    common_parts = []
    
    if start_layer_name == end_layer_name and len(start_parts) > 0:
        lca_path = '.'.join(start_parts[:-1])
    else:
        for a, b in zip(start_parts, end_parts):
            if a == b:
                common_parts.append(a)
            else:
                break
        lca_path = '.'.join(common_parts)

    if lca_path == "":
        start_container_name, _ = _get_container_and_subname(start_layer_name)
        end_container_name, _ = _get_container_and_subname(end_layer_name)
        if start_container_name == end_container_name:
            lca_path = start_container_name
            container = get_layer(model, lca_path)
        else:
            lca_path = ""
            container = model
    else:
        container = get_layer(model, lca_path)

    print(f"[DEBUG] [LCA] Resolved Container: '{lca_path}' | Type: {type(container).__name__}")

    # 1. Detect if a previous collapse already absorbed this block into SmartIdentity
    if isinstance(container, SmartIdentity):
        print(f"[WARN] ⚠ Target '{lca_path}' is already a SmartIdentity (absorbed by an earlier collapse region). Skipping.")
        return None

    # 2. Holistic escalation BEFORE child index mapping for custom parallel/complex blocks
    custom_parallel_blocks = ("InceptionBlock", "ConvNeXtBlock", "XceptionBlock", "DepthwiseSeparableConv")
    if type(container).__name__ in custom_parallel_blocks and lca_path != "":
        print(f"[DEBUG] Escalating complex block '{lca_path}' ({type(container).__name__}) to parent for holistic replacement.")
        parent_path, subname = _get_container_and_subname(lca_path)
        parent_container = get_layer(model, parent_path)
        named_layers_parent = list(parent_container.named_children())
        try:
            esc_idx = next(i for i, (n, _) in enumerate(named_layers_parent) if n == subname)
            return {
                "container": parent_container,
                "container_name": parent_path,
                "named_layers": named_layers_parent,
                "start_idx": esc_idx,
                "end_idx": esc_idx,
                "full_block": [(subname, container)],
                "layer_type": nn.Conv2d,
                "conv_layers": [m for m in container.modules() if isinstance(m, nn.Conv2d)],
                "collapse_mode": "conv",
                "first_layer_name": subname,
                "last_layer_name": subname,
            }
        except StopIteration:
            print(f"[WARN] Escalation failed: '{subname}' not found in parent '{parent_path}'. Proceeding standard slice.")

    named_layers = list(container.named_children())

    start_idx = end_idx = None
    for i, (child_name, _) in enumerate(named_layers):
        full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
        if start_idx is None and (start_layer_name == full_child_prefix or start_layer_name.startswith(full_child_prefix + ".")):
            start_idx = i
        if end_idx is None and (end_layer_name == full_child_prefix or end_layer_name.startswith(full_child_prefix + ".")):
            end_idx = i
        if start_idx is not None and end_idx is not None:
            break

    if start_idx is None or end_idx is None:
        print(f"[WARN] ⚠ Could not map start/end layers into LCA container '{lca_path}'")
        print(f"[DEBUG]   start_layer_name={start_layer_name}, found start_idx={start_idx}")
        print(f"[DEBUG]   end_layer_name={end_layer_name}, found end_idx={end_idx}")
        print(f"[DEBUG]   ISSUE #3 FIX: This may be a cross-branch collapse (parallel paths in Inception/ConvNeXt).")
        print(f"[DEBUG]   Skipping this collapse candidate as it cannot be sequentially mapped.")
        return None
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    full_block = named_layers[start_idx:end_idx + 1]
    
    safe_sequential_classes = ("Sequential", "SeparableConv2d", "ModuleList", type(model).__name__)
    if type(container).__name__ not in safe_sequential_classes and lca_path != "":
        print(f"[DEBUG] Escalating complex block '{lca_path}' ({type(container).__name__}) to parent for holistic replacement.")
        
        parent_path, subname = _get_container_and_subname(lca_path)
        parent_container = get_layer(model, parent_path)
        named_layers_parent = list(parent_container.named_children())
        
        try:
            esc_idx = next(i for i, (n, _) in enumerate(named_layers_parent) if n == subname)
            return {
                "container": parent_container,
                "container_name": parent_path,
                "named_layers": named_layers_parent,
                "start_idx": esc_idx,
                "end_idx": esc_idx,
                "full_block": [(subname, container)],
                "layer_type": nn.Conv2d,
                "conv_layers": [m for m in container.modules() if isinstance(m, nn.Conv2d)],
                "collapse_mode": "conv",
                "first_layer_name": subname,
                "last_layer_name": subname,
            }
        except StopIteration:
            print(f"[WARN] Escalation failed: '{subname}' not found in parent '{parent_path}'. Proceeding standard slice.")

    return {
        "container": container,
        "container_name": lca_path,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "layer_type": nn.Conv2d,
        "conv_layers": [m for _, m in full_block if isinstance(m, nn.Conv2d)],
        "collapse_mode": "conv",
        "first_layer_name": named_layers[start_idx][0],
        "last_layer_name": named_layers[end_idx][0],
    }

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
    y_out=None,
    init='random',
):
    if debug:
        print(f"\n{'='*60}")
        print(f"[STEP] COLLAPSE REPLACEMENT INIT: '{start_layer_name}'")
        print(f"{'='*60}")
        print(f"[DEBUG] Target device    : {device}")
        print(f"[DEBUG] Info keys found  : {list(info.keys())}")
        
    named_layers = info["named_layers"] 
    # Prefer the container determined by the locator (LCA).
    start_container_name = info.get("container_name") 
    if start_container_name is None: 
        start_container_name = _get_container_and_subname(start_layer_name)[0] 
        
    start_idx, end_idx = info["start_idx"], info["end_idx"] 
    out_shape = block_analysis.get("out_shape") 
    out_channels = block_analysis.get("out_channels") 

    if out_shape is None or out_channels is None: 
        raise RuntimeError(f"[ERROR] block_analysis missing required keys. Found: {list(block_analysis.keys())}") 

    target_H, target_W = (out_shape[-2], out_shape[-1]) if len(out_shape) >= 4 else (1, 1) 
    
    if x is None or x.ndim < 2: 
        raise RuntimeError(f"[ERROR] Invalid captured activation `x`. Shape found: {getattr(x, 'shape', None)}") 
    in_channels = int(x.shape[1]) 

    if debug:
        print(f"[DEBUG] Container Name   : '{start_container_name}'")
        print(f"[DEBUG] Layer Indices    : {start_idx} -> {end_idx}")
        print(f"[DEBUG] Spatial Target   : H={target_H} x W={target_W}")
        print(f"[DEBUG] Channel Mapping  : {in_channels} (in) -> {out_channels} (out)")

    # -------- INTELLIGENT ROUTING LOGIC --------
    original_convs = info.get("conv_layers", [])
    is_depthwise = False
    target_kernel_size = 1
    
    if original_convs:
        first_conv = original_convs[0]
        if debug:
            print(f"[DEBUG] Analyzing original conv: {first_conv.__class__.__name__} (groups={first_conv.groups}, in={first_conv.in_channels}, out={first_conv.out_channels})")
            
        if first_conv.groups == first_conv.in_channels and first_conv.in_channels == first_conv.out_channels:
            if in_channels == out_channels:
                is_depthwise = True
                target_kernel_size = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                
        elif first_conv.kernel_size == (1, 1) or first_conv.kernel_size == 1:
            target_kernel_size = 1 
            is_depthwise = False
            
    # Inherit the original conv's geometry so the surrogate does not silently
    # change spatial semantics (stride/padding errors previously leaked into
    # downstream layers).
    first_stride = (1, 1)
    first_padding = 0
    if original_convs:
        first_stride = original_convs[0].stride
        first_padding = original_convs[0].padding

    if is_depthwise:
        if debug:
            print(f"[DEBUG] ➔ Strategy: GROUPED {target_kernel_size}×{target_kernel_size} Conv (depth‑wise style, groups={in_channels})")
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=target_kernel_size,
            stride=first_stride,
            padding=first_padding,
            groups=in_channels,
            bias=False,
        )
        conv_name = "conv_g1x1"
    else:
        if debug:
            print(f"[DEBUG] ➔ Strategy: POINTWISE Conv (kernel_size=1)")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride, padding=0, bias=False)
        conv_name = "conv_1x1"
    # -----------------------------------------------


    model_type = type(model).__name__.lower()

    # Content-based norm/activation selection: adopt what the excised block
    # actually uses; fall back to BN/ReLU (legacy default) when it uses neither.
    norm_mod, act_mod = _detect_norm_act(info["full_block"])

    if norm_mod is None or isinstance(norm_mod, nn.BatchNorm2d):
        norm_layer = nn.BatchNorm2d(out_channels)
        norm_name = "bn"
    elif isinstance(norm_mod, nn.LayerNorm):
        norm_layer = _SurrogateLayerNorm2d(out_channels)
        norm_name = "layernorm"
    elif isinstance(norm_mod, nn.GroupNorm):
        groups = math.gcd(norm_mod.num_groups, out_channels) or 1
        norm_layer = nn.GroupNorm(groups, out_channels)
        norm_name = "groupnorm"
    else:  # InstanceNorm2d
        norm_layer = nn.InstanceNorm2d(out_channels)
        norm_name = "instancenorm"

    if act_mod is None or isinstance(act_mod, nn.ReLU):
        act_layer = nn.ReLU(inplace=False)
        act_name = "relu"
    else:
        try:
            act_layer = type(act_mod)()
        except Exception:
            act_layer = nn.Identity()
        act_name = type(act_mod).__name__.lower()

    pool = nn.AdaptiveAvgPool2d((target_H, target_W))

    replacement = nn.Sequential(OrderedDict([
        (conv_name, conv),
        (norm_name, norm_layer),
        (act_name, act_layer),
        ("adaptive_pool", pool),
    ]))

    # Optional function-preserving initialization: fit the surrogate conv to
    # the captured block I/O. Normalization layers are deliberately left at
    # their near-identity defaults so the fitted mapping is preserved.
    if init == 'lsq':
        try:
            _init_surrogate_lsq(conv, x, y_out, debug=debug)
        except Exception as e:
            print(f"[WARN] LSQ surrogate init failed ({e}); keeping random initialization.")
    elif init != 'random':
        raise ValueError(f"Unknown init mode '{init}'. Expected 'random' or 'lsq'.")

    if debug:
        print(f"\n[DEBUG] Built replacement module for {model_type}:")
        for name, layer in replacement.named_children():
            print(f"        - {name}: {layer}")
    # =========================================================
    # THE FIX: Explicitly passing info.get("container")
    # =========================================================
    if debug: 
        print(f"\n[STEP] Patching container '{start_container_name or '<root>'}'...") 
    
    updated_container = _replace_layers( 
            named_layers, 
            start_idx, 
            end_idx, 
            replacement, 
            start_name=info["first_layer_name"], 
            end_name=info["last_layer_name"],
            container=info.get("container") # CRITICAL FIX: Preserves custom forward logic
        )
        
    _update_container(model, start_container_name, updated_container) 
    model.to(device) 

    post_params = count_trainable_params(model) 
    if debug:
        print(f"\n[INFO] --- Parameter Delta ---")
        print(f"[INFO] Pre-collapse  : {pre_params:,}") 
        print(f"[INFO] Post-collapse : {post_params:,}") 
        print(f"[INFO] Net Change    : {post_params - pre_params:+,}") 

    if post_params > pre_params: 
        print(f"[WARN] ⚠ Collapsed block INCREASED parameter count. Check collapse policy or routing logic.") 

    # =========================================================
    # REPLACEMENT VALIDATION
    # =========================================================
    try:
        dev = next((p.device for p in model.parameters()), torch.device('cpu'))
        rep_module = get_layer(model, start_container_name)
        child = None

        rep_children = list(rep_module.named_children())

        # Validate the exact slot we replaced first; this avoids probing an unrelated
        # block when multiple prior collapses already exist in the same container.
        if 0 <= start_idx < len(rep_children):
            slot_name, slot_mod = rep_children[start_idx]
            if isinstance(slot_mod, nn.Sequential):
                slot_keys = set(dict(slot_mod.named_children()).keys())
                if any(k in slot_keys for k in ("conv_1x1", "conv_g1x1", "adaptive_pool")):
                    child = slot_mod
                    if debug:
                        print(f"[DEBUG] Local validation candidate selected from replacement slot: '{slot_name}'")

        # Fallback: reverse search for the most recent surrogate-like module.
        if child is None:
            for nm, m in reversed(rep_children):
                if nm.startswith("collapsed_"):
                    child = m
                    break
                if isinstance(m, nn.Sequential):
                    keys = set(dict(m.named_children()).keys())
                    if any(k in keys for k in ("conv_1x1", "conv_g1x1", "adaptive_pool")):
                        child = m
                        break

        if child is not None:
            with torch.no_grad():
                test_x = x.clone().to(dev)
                out = child(test_x)
                if debug:
                    print(f"[DEBUG] ✓ Replacement local validation OK. Output shape: {tuple(out.shape)}")
        else:
            print(f"[WARN] Could not locate the inserted collapsed module for local validation.")
    except Exception as e:
        _log_exception(
            "Replacement forward validation failed",
            e,
            debug=debug,
            extra={
                "container": start_container_name or "<root>",
                "start_idx": start_idx,
                "input_shape": _shape_str(x),
                "expected_out_channels": out_channels,
            },
        )
            
    # =========================================================
    # CORRECTIVE POOLING & DOWNSTREAM VALIDATION
    # =========================================================
    try:
        if debug:
            print(f"\n[STEP] Evaluating corrective pooling necessity...") 
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug) 
    except Exception as e:
        _log_exception(
            "Corrective pool insertion failed",
            e,
            debug=debug,
            extra={
                "next_linear_name": next_linear_name,
                "input_shape": input_shape,
            },
        )
            
    try:
        if debug:
            print(f"[STEP] Triggering downstream validation...") 
        _validate_downstream(model, start_container_name, start_idx, x, input_shape, next_linear_name, next_linear_mod, device, debug) 
        if debug:
            print(f"[DEBUG] ✓ Downstream validation successful.")
    except Exception as e:
        _log_exception(
            "Downstream validation failed",
            e,
            debug=debug,
            extra={
                "container": start_container_name or "<root>",
                "start_idx": start_idx,
                "input_shape": input_shape,
            },
        )
            
    if debug: 
        print(f"{'='*60}")
        print(f"[RESULT] Block replacement complete for '{start_layer_name}'.") 
        print(f"{'='*60}\n")

    return model

def _replace_layers(named_layers, start_idx, end_idx, replacement, start_name=None, end_name=None, container=None):
    """
    Patches layers into a container, handling both nn.Sequential (OrderedDict)
    and custom nn.Module blocks (Attribute-based) safely.
    """
    # 1. Determine if container is a standard Sequential/ModuleList
    is_sequential = isinstance(container, (nn.Sequential, nn.ModuleList))
    
    if is_sequential:
        new_layers = OrderedDict()
        for i, (name, mod) in enumerate(named_layers):
            if i < start_idx or i > end_idx:
                new_layers[name] = mod
            elif i == start_idx:
                new_layers[name] = replacement
            else:
                # Replace intermediate layers with identity
                new_layers[name] = SmartIdentity()
        
        # Reconstruct the Sequential container
        if isinstance(container, nn.Sequential):
            return nn.Sequential(new_layers)
        else:
            return nn.ModuleList(new_layers.values())
    
    else:
        # 2. Attribute-based patching for Inception/ConvNeXt blocks
        for i, (name, mod) in enumerate(named_layers):
            if i == start_idx:
                # Use setattr to patch the custom block's attribute
                setattr(container, name, replacement)
            elif start_idx < i <= end_idx:
                # Replace with identity
                setattr(container, name, SmartIdentity())
        return container
def _insert_corrective_pool(model, next_linear_name, input_shape, debug):
    if not next_linear_name:
        return model

    if ".pwconv" in next_linear_name:
        return model

    try:
        device = next((p.device for p in model.parameters()), torch.device('cpu'))
        x_in, _ = _simulate_input_output_hooks(model, next_linear_name, next_linear_name, input_shape, device=device)
        
        # FIX: Use x_in[0] to calculate features per sample, completely ignoring batch size
        current_features = x_in[0].numel() 
        current_shape = x_in.shape
        if debug:
            print(f"[DEBUG] [POOL] Target Linear: '{next_linear_name}'")
            print(f"[DEBUG] [POOL] Entering shape: {tuple(current_shape)} | Flat size: {current_features}")
    except Exception as e:
        if debug: print(f"[WARN] Pool check hook failed: {e}")
        return model

    next_linear_mod = get_layer(model, next_linear_name)
    if not isinstance(next_linear_mod, nn.Linear):
        return model
        
    in_features = next_linear_mod.in_features
    if current_features == in_features:
        if debug: print("[INFO] ✅ Pool check: Shapes match seamlessly.")
        return model

    if len(current_shape) < 4:
        return model

    C = current_shape[1]
    if in_features % C != 0:
        return model

    expected_hw = in_features // C
    target_hw = int(round(expected_hw ** 0.5))

    if target_hw * target_hw != expected_hw:
        return model

    if debug: print(f"[INFO] 🛠 Mismatch! Wrapping '{next_linear_name}' with AdaptiveAvgPool2d({target_hw}, {target_hw})")

    # Safe in-place module replacement
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            if child_mod is next_linear_mod:
                wrapped_linear = nn.Sequential(
                    nn.AdaptiveAvgPool2d((target_hw, target_hw)),
                    next_linear_mod
                )
                setattr(parent_mod, child_name, wrapped_linear)
                return model

    return model

def _find_next_linear(model, end_layer_name, debug, order=None):
    if debug:
        print(f"\n[STEP] Searching for next nn.Linear after '{end_layer_name}'...")

    # Preferred path: true execution order captured from a probe forward pass.
    # Robust on branchy/custom architectures where named_modules() DFS order
    # does not match forward order (and could select a Linear on a wrong branch).
    if order is not None and end_layer_name in order:
        end_idx_exec = order[end_layer_name]
        best = None
        for name, module in model.named_modules():
            if (
                isinstance(module, nn.Linear)
                and name in order
                and order[name] > end_idx_exec
                and (best is None or order[name] < best[0])
            ):
                best = (order[name], name, module)
        if best is not None:
            if debug:
                print(f"[DEBUG] Execution-order match: next Linear is '{best[1]}' "
                      f"(exec idx {end_idx_exec} -> {best[0]})")
            return best[1], best[2]
        if debug:
            print(f"[DEBUG] '{end_layer_name}' executed (idx {end_idx_exec}) but no "
                  f"Linear fired after it. Checking legacy name scan...")

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
            # ONLY grab the linear if it's explicitly downstream
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Found Linear layer ahead: {n} ({m})")
                break

    # [FIX]: REMOVE the global fallback loop. If we don't find a linear layer downstream, 
    # we return None so the block doesn't hallucinate a spatial size from the stem.
    if next_linear_mod is None:
        if debug:
            print(f"[DEBUG] No downstream Linear found after {end_layer_name}. Disabling adaptive pooling heuristic.")
            
    return next_linear_name, next_linear_mod

def _analyze_block_output(
    model, full_block, conv_layers, named_layers, end_idx, layer_type, x, y_out, next_linear_mod, debug
):
    if debug:
        print(f"\n[STEP] Analyzing output of collapsed block ({len(full_block)} layers)...")
        print(f"[DEBUG] Input tensor shape before block: {tuple(x.shape)}")
        print(f"[DEBUG] Running forward pass through block layers:")

    out_shape = tuple(y_out.shape)
    out_channels = y_out.shape[1] if y_out.ndim >= 2 else None

    if debug:
        print(f"[DEBUG] Final block output shape (via hook): {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")
    # --- Ground-truth output characteristics ---
    out_shape = tuple(y_out.shape)

    # 🔧 CRITICAL FIX:
    # Infer channels from actual tensor, not from layer attributes
    out_channels = y_out.shape[1] if y_out.ndim >= 2 else None

    if debug:
        print(f"[DEBUG] Final block output shape: {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")

    # --- Detect pooling inside original block (informational) ---
    pool_layer = next(
        (
            m for _, m in reversed(full_block)
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        ),
        None,
    )

    if debug:
        if pool_layer is not None:
            print(f"[DEBUG] Detected pool in original block: {type(pool_layer).__name__}")
        else:
            print(f"[DEBUG] No explicit pooling layer detected inside block.")

    # --- Linear compatibility heuristic ---
    linear_in_features_heuristic = next_linear_mod.in_features if next_linear_mod else None
    if debug:
        print(f"[DEBUG] Heuristic next linear in_features = {linear_in_features_heuristic}")

    # --- Shortcut detection (unchanged) ---
    shortcut_out_channels = None
    for nm, mod in model.named_modules():
        if hasattr(mod, "shortcut") and isinstance(mod.shortcut, nn.Module):
            first_conv = next(
                (m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)),
                None,
            )
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
        print(f"         shortcut_out_channels={shortcut_out_channels}")

    return {
        "out_shape": out_shape,
        "out_channels": out_channels,
        "pool_layer": pool_layer,
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
    Guarantees that feature map shapes remain valid to prevent CKA extraction from producing NaN or zero vectors.
    """
    print(f"\n[STEP] ===== Starting _validate_downstream for '{start_container_name}' =====")
    print(f"[DEBUG] start_idx={start_idx}, device={device}, has_next_linear={next_linear_name is not None}")

    try:
        container = get_layer(model, start_container_name)
    except Exception as e:
        print(f"[WARN] Could not access container '{start_container_name}': {e}")
        return

    children = list(container.named_children())
    if not children:
        print(f"[DEBUG] Container '{start_container_name}' has no children; skipping downstream validation.")
        return

    collapsed_idx = None
    print(f"[STEP] Searching for inserted collapsed module within '{start_container_name}'...")
    for i in range(len(children) - 1, -1, -1):
        nm, m = children[i]
        if nm.startswith("collapsed_") or (
            isinstance(m, nn.Sequential)
            and any(k in dict(m.named_children()) for k in ("conv_1x1", "conv_g1x1", "adaptive_pool", "conv_dw"))
        ):
            collapsed_idx = i
            print(f"[DEBUG] Found inserted block candidate at index {i}: '{nm}'")
            break

    if collapsed_idx is None:
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

    try:
        with torch.no_grad():
            dev = next((p.device for p in model.parameters()), torch.device('cpu'))
            t = pre_activation.clone().to(dev)
            t = inserted_mod(t)
        print(f"[DEBUG] Inserted module forward successful, output shape: {tuple(t.shape)}")
    except Exception as e:
        print(f"[ERROR] Forward pass through inserted module failed: {e}")
        raise RuntimeError(f"Structural collapse replacement validation failed: {e}")

    print(f"[STEP] Scanning immediate downstream modules for shape or runtime errors...")
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            if debug:
                print(
                    f"[DEBUG] Running downstream module '{start_container_name}.{nm}' "
                    f"({mod.__class__.__name__}) with input shape {tuple(t.shape)}"
                )
            t = mod(t)
            if debug:
                print(f"[DEBUG] Downstream '{start_container_name}.{nm}' executed successfully, output shape: {tuple(t.shape)}")
        except Exception as e:
            print(f"[WARN] Downstream module '{start_container_name}.{nm}' raised exception: {e}")
            if isinstance(mod, nn.Linear):
                print(f"[INFO] Target module '{nm}' is nn.Linear. Allowing Corrective Pooling to handle it natively.")
                return
            
            print(f"[STEP] Replacing problematic module '{nm}' with safe alternative...")

            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
                print(f"[INFO] Replaced with _SafePool wrapper.") 
            else:
                safe = nn.Identity()
                print(f"[INFO] Replaced with Identity() to bypass invalid operation.")

            if isinstance(container, nn.Sequential) or type(container).__name__ == 'SmartIdentity':
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)

            print(f"[DEBUG] Replacement applied to '{start_container_name}.{nm}' ({safe.__class__.__name__}).")
            return

        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            print(f"[WARN] Module '{start_container_name}.{nm}' produced zero spatial dimensions. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential) or type(container).__name__ == 'SmartIdentity':
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            return

    print(f"[RESULT] ✅ Downstream validation for '{start_container_name}' completed successfully.")
