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
            # detect whether the original `full_block` included any pooling layer we absorbed
            has_original_pool = False
            if full_block:
                for _, m in full_block:
                    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                        has_original_pool = True
                        break

            if has_original_pool:
                seq.append(copy.deepcopy(pool_layer))
                if debug:
                    print(f"[DEBUG] Appending cloned pooling layer: {type(pool_layer).__name__}")
            else:
                if debug:
                    print(f"[DEBUG] Skipping appending {type(pool_layer).__name__} because original block had no pool")
            
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
def _collapse_block(model, start, end, input_shape, device, debug=False):
    """
    Orchestrates the collapse of a sequential block into a single equivalent module.
    """
    start_container_name, named_layers = _get_block_layers(model, start, end)

    traced_shapes = _trace_block_shapes(named_layers, input_shape, device, debug)

   
    adjusted_layers, pool_used = _absorb_pools_if_needed(named_layers, traced_shapes, debug)

    
    collapsed_block = _build_collapsed_block_with_checks(
        adjusted_layers, traced_shapes, debug
    )

    
    model = _replace_in_model(model, start_container_name, start, end, collapsed_block, device)


    _report_block_collapse_summary(
        start_container_name, start, end, traced_shapes, collapsed_block, pool_used, debug
    )

    return model

def _get_block_layers(model, start, end):
    """
    Extract the container path and the list of named layers between start and end (inclusive).

    Example:
      start = "features.conv_8"
      end   = "features.conv_10"
    returns:
      ("features", [("conv_8", module), ("relu_8", module), ..., ("conv_10", module)])
    """
    def split_path(p):
        parts = p.split('.')
        return '.'.join(parts[:-1]), parts[-1]

    start_container, start_name = split_path(start)
    end_container, end_name = split_path(end)

    if start_container != end_container:
        raise ValueError(f"Start and end must be in the same container. got: '{start}' vs '{end}'")

    container = _get_submodule(model, start_container)  # returns model if start_container == ""

    # Gather the children in a consistent (name,module) list
    if isinstance(container, nn.Sequential):
        children = list(container.named_children())
    elif isinstance(container, nn.ModuleList):
        children = [(str(i), container[i]) for i in range(len(container))]
    else:
        # Generic Module: use named_children (keeps attribute names like 'conv1', 'bn1', ...) 
        children = list(container.named_children())

    # Build index map and locate start/end
    name_to_idx = {n: i for i, (n, _) in enumerate(children)}
    if start_name not in name_to_idx:
        raise ValueError(f"Start layer '{start_name}' not found in container '{start_container}' (available: {[n for n,_ in children]})")
    if end_name not in name_to_idx:
        raise ValueError(f"End layer '{end_name}' not found in container '{start_container}' (available: {[n for n,_ in children]})")

    start_idx = name_to_idx[start_name]
    end_idx = name_to_idx[end_name]
    if start_idx > end_idx:
        raise ValueError(f"Start index ({start_idx}) > end index ({end_idx}) for {start} -> {end}")

    named_layers = children[start_idx:end_idx + 1]
    return start_container, named_layers

def _trace_block_shapes(named_layers, input_shape, device, debug):
    x = torch.zeros(input_shape).to(device)
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

def _get_submodule(model, target):
    """
    Recursively retrieve a submodule by its dotted path (e.g. 'layer1.0.conv1').

    If target == '' (empty string), returns the model itself.

    Equivalent to nn.Module.get_submodule() in newer PyTorch versions.
    """
    if target == "" or target is None:
        return model

    current = model
    for attr in target.split("."):
        if attr.isdigit():
            current = current[int(attr)]
        else:
            current = getattr(current, attr)
    return current


def _absorb_pools_if_needed(named_layers, traced_shapes, debug):
    output_shape = traced_shapes["final"]
    H, W = output_shape[-2:]
    pool_used = None
    adjusted_layers = list(named_layers)

    if H > 1 or W > 1:
        # try to absorb one or more pools to reach 1x1
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
    in_channels = layers[0][1].in_channels if hasattr(layers[0][1], "in_channels") else None
    out_channels = layers[-1][1].out_channels if hasattr(layers[-1][1], "out_channels") else None

    collapsed_layers = []
    for _, layer in layers:
        if not isinstance(layer, nn.Identity):
            collapsed_layers.append(layer)

    collapsed_block = nn.Sequential(*collapsed_layers)

    if debug:
        print(f"[DEBUG] Built collapsed block: {collapsed_block}")
        print(f"[DEBUG] Input/Output channels: {in_channels} → {out_channels}")

    return collapsed_block


def _replace_in_model(model, container_path, start, end, collapsed_block, device):
    """
    Replace the layers between start and end (inclusive) inside the container_path
    with `collapsed_block`. Works for Sequential, ModuleList and generic container modules.
    """
    # get subnames
    start_name = start.split('.')[-1]
    end_name = end.split('.')[-1]

    # get container module
    container = _get_submodule(model, container_path)

    # assemble children list (name,module)
    if isinstance(container, nn.Sequential):
        children = list(container.named_children())
    elif isinstance(container, nn.ModuleList):
        children = [(str(i), container[i]) for i in range(len(container))]
    else:
        children = list(container.named_children())

    # build name->index map
    name_to_idx = {n: i for i, (n, _) in enumerate(children)}
    if start_name not in name_to_idx or end_name not in name_to_idx:
        raise ValueError(f"Could not find start/end names in container '{container_path}'. Available: {[n for n,_ in children]}")

    start_idx = name_to_idx[start_name]
    end_idx = name_to_idx[end_name]
    if start_idx > end_idx:
        raise ValueError(f"start_idx ({start_idx}) > end_idx ({end_idx})")

    # Now call the lower-level replacer with the proper args
    updated_container = _replace_layers(children, start_idx, end_idx, collapsed_block)

    # Patch the model: replace the container at container_path with updated_container
    if container_path == "" or container_path is None:
        # replacing the root module — be careful
        # we don't generally want to replace the root Module, so raise to avoid accidental behavior
        raise ValueError("Refusing to replace the root module container. Provide a container path.")
    else:
        _update_container(model, container_path, updated_container)

    # ensure device placement
    model.to(device)
    return model


def _report_block_collapse_summary(container_name, start, end, traced_shapes, block, pool_used, debug):
    if not debug:
        return
    print("\n[SUMMARY] Block collapse results:")
    print(f"   Container:       {container_name or '<root>'}")
    print(f"   Range:            {start} → {end}")
    print(f"   Input shape:      {traced_shapes['list'][0][1] if traced_shapes['list'] else 'N/A'}")
    print(f"   Output shape:     {traced_shapes['final']}")
    print(f"   Pool used:        {pool_used.__class__.__name__ if pool_used else 'None'}")
    print(f"   New block type:   {block.__class__.__name__}")
    print("------------------------------------------------------")

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
