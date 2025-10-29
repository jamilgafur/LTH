# collapse.py
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats
from uuid import uuid4
from typing import Optional
from copy import deepcopy
import copy

def _set_module_by_path(model, module_path, new_module):
    """
    Replace module referenced by module_path (dot-separated). Handles numeric indices for Sequential.
    """
    if module_path == "":
        raise ValueError("_set_module_by_path: cannot replace root module")
    
    print(f"[DEBUG] _set_module_by_path: Replacing module at path '{module_path}'")
    
    parts = module_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    
    print(f"[DEBUG] Parent module: {parent}, Last part: {last}")
    
    if last.isdigit():
        idx = int(last)
        print(f"[DEBUG] Replacing at index {idx}")
        parent[idx] = new_module
    else:
        print(f"[DEBUG] Setting new module at attribute {last}")
        setattr(parent, last, new_module)

def disable_inplace_relu(model):
    """
    Replace all nn.ReLU(inplace=True) in the model with nn.ReLU(inplace=False), to avoid
    autograd inplace modification errors after structural surgery.
    """
    # collect names to replace first (do not mutate while iterating)
    to_replace = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU) and getattr(module, 'inplace', False):
            to_replace.append(name)

    if not to_replace:
        print("[DEBUG] No in-place ReLU found to replace.")
        return

    print(f"[INFO] Replacing {len(to_replace)} in-place ReLU(s) with out-of-place variants.")
    for name in to_replace:
        print(f"[DEBUG] Found in-place ReLU at: {name}")
        container, subname = _get_container_and_subname(name)
        parent = get_layer(model, container) if container != "" else model
        # set new non-inplace ReLU in parent
        new_relu = nn.ReLU(inplace=False)
        # if subname is a numeric index in Sequential
        if subname.isdigit():
            idx = int(subname)
            print(f"[DEBUG] Replacing ReLU at index {idx}")
            parent[idx] = new_relu
        else:
            print(f"[DEBUG] Setting new ReLU at attribute {subname}")
            setattr(parent, subname, new_relu)

def _is_int_str(s):
    try:
        int(s)
        return True
    except:
        return False

def get_layer(model, layer_name):
    """
    Dynamically access a layer in the model based on the full dot-separated name.
    Supports attribute names and numeric indices for nn.Sequential (e.g., 'layer.0.conv').
    """
    # NOTE: lots of debug prints to help trace path resolution
    # empty -> model root
    if layer_name == "":
        print("[DEBUG] get_layer: requested root model")
        return model
    print(f"[DEBUG] Accessing layer '{layer_name}' in the model.")
    layer_parts = layer_name.split('.')
    layer = model
    for part in layer_parts:
        if _is_int_str(part):
            idx = int(part)
            print(f"[DEBUG] Accessing index {idx}")
            layer = layer[idx]
        else:
            print(f"[DEBUG] Accessing attribute '{part}'")
            layer = getattr(layer, part)
    return layer

import torch.nn.functional as F

def patch_skip_connections(model):
    """
    Patch residual blocks to skip shortcuts safely.
    Avoid circular references by storing only block names, not full model.
    """
    collapsed_paths = getattr(model, "_collapsed_blocks", [])

    for name, module in model.named_modules():
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module):
            # save original forward
            original_forward = module.forward

            def make_patched_forward(orig_forward, block_name):
                def new_forward(self, x):
                    out = self.block(x)
                    # check if this block is within collapsed ranges
                    skip_shortcut = any(
                        block_name.startswith(start) or block_name.startswith(end)
                        for start, end in collapsed_paths
                    )
                    if skip_shortcut:
                        return F.relu(out)
                    else:
                        return F.relu(out + self.shortcut(x))
                return new_forward

            # patch forward safely
            module.forward = make_patched_forward(original_forward, name).__get__(module)
            print(f"[PATCH] Patched residual block forward: {name}")

def compression_set(layers):
    """
    Example of compressing layers. Replace this with actual compression logic.
    """
    for layer in layers:
        print(f"[DEBUG] Compressing layer: {layer}")
        # Add compression logic here (e.g., pruning, quantization)

def _is_within_collapsed_block(model, block_path):
    """
    Returns True if the block is within any collapsed range.
    Assumes block_path like 'stage3.stage3_block2'
    """
    if not hasattr(model, '_collapsed_blocks'):
        return False

    for start_path, end_path in model._collapsed_blocks:
        if block_path in start_path or block_path in end_path or start_path.startswith(block_path) or end_path.startswith(block_path):
            print(f"[SKIP-CONN] Skipping shortcut for block '{block_path}' due to collapse: {start_path} → {end_path}")
            return True
    return False

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None, input_shape=(1, 3, 32, 32), device='cpu'):
    import torch
    import torch.nn as nn
    from collections import OrderedDict
    import copy

    model_kwargs = model_kwargs or {}
    num_classes = model_kwargs.get('num_classes', 200)

    print(f"[INFO] Loading model with weights from '{model_weights_1}'...")
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # ✅ 1. Save the original feature map shape (so we can match it later)
    with torch.no_grad():
        dummy = torch.randn(input_shape).to(device)
        feat = model.features(dummy)
        C_orig, H_orig, W_orig = feat.shape[1], feat.shape[2], feat.shape[3]
        F_orig = C_orig * H_orig * W_orig
        model._orig_features_shape = (C_orig, H_orig, W_orig)
        print(f"[INFO] Saved original features shape: C_orig={C_orig}, H_orig={H_orig}, W_orig={W_orig}, F_orig={F_orig}")

    # Track collapsed layer ranges
    model._collapsed_blocks = []

    print(f"[INFO] Full compression set: {compression_set}")
    for compression_set1 in compression_set:
        print(f"[DEBUG] Compressing: {compression_set1}")
        start, end = compression_set1[0], compression_set1[1]
        print(f"\n--- Starting collapse for block: {start} to {end} ---")

        model = _collapse_block(model, start, end, input_shape, device=device)
        model._collapsed_blocks.append((start, end))

    # ✅ 2. Adjust the convolutional output so classifier input shape stays identical
    if hasattr(model, "_orig_features_shape"):
        with torch.no_grad():
            dummy = torch.randn(input_shape).to(device)
            out_feat = model.features(dummy)
            C_new, H_new, W_new = out_feat.shape[1], out_feat.shape[2], out_feat.shape[3]
            C_orig, H_orig, W_orig = model._orig_features_shape
            print(f"[INFO] After collapse features shape: C_new={C_new}, H_new={H_new}, W_new={W_new}")

            if (C_new != C_orig) or (H_new != H_orig) or (W_new != W_orig):
                print(f"[INFO] Appending projection adapter to preserve classifier input shape...")
                proj = nn.Conv2d(C_new, C_orig, kernel_size=1, stride=1, padding=0, bias=False)
                pool = nn.AdaptiveAvgPool2d((H_orig, W_orig))
                old_children = list(model.features.named_children())
                new_children = [(n, m) for n, m in old_children]
                new_children.append(("proj_to_orig_channels", proj))
                new_children.append(("pool_to_orig_spatial", pool))
                model.features = nn.Sequential(OrderedDict(new_children)).to(device)
                print(f"[✓] Inserted 1x1 projection + adaptive pooling to match original features size.")
            else:
                print(f"[INFO] No adapter needed; feature shape unchanged.")
    else:
        print("[WARN] Could not find original feature shape; skipping adapter insertion.")

    # ✅ 3. Keep classifier untouched (preserve_original_fc=True)
    adjust_classifier_input_features(model, input_shape, num_classes=num_classes, device=device, preserve_original_fc=True)

    # Cleanups
    disable_inplace_relu(model)
    patch_skip_connections(model)
    print(f"[INFO] Collapse complete. Total trainable params: {count_trainable_params(model)}")
    model.to(device)

    return model


def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive).
    They must belong to the same container (e.g., 'features').
    Fixes channel/stride mismatches by simulating input BEFORE the start layer,
    and aligning collapsed stride with shortcut if present.
    This function keeps its original signature used elsewhere.
    """
    print(f"\nCollapsing layers from '{start_layer_name}' to '{end_layer_name}'...")

    # gather candidate modules (for debugging/reporting)
    build_layer_names = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU,
                              nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
            build_layer_names.append(name)
    print(f"[DEBUG] Available layers for collapsing: {build_layer_names}")

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)

    # NOTE: start_container_name expected to refer to the container (e.g., 'features')
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())

    # find indices for start and end within that container
    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(
            f"Layer names '{start_layer_name}' or '{end_layer_name}' not found inside container '{start_container_name}'."
        )
    assert start_idx <= end_idx, "Start index must be <= end index"
    print(f"[DEBUG] Collapsing in section '{start_container_name}' from index {start_idx} to {end_idx} "
          f"({named_layers[start_idx][0]} → {named_layers[end_idx][0]})")

    # collect only Conv2d/Linear for collapsing (preserve original selection logic)
    full_block = named_layers[start_idx:end_idx + 1]
    selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not selected_layers:
        raise ValueError("[ERROR] No Conv2d/Linear layers found in selected range to collapse.")

    layer_type = type(selected_layers[0])
    if not all(isinstance(l, layer_type) for l in selected_layers):
        raise ValueError("Cannot collapse mixed layer types.")

    # --- simulate input before start layer ---
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        print(f"[DEBUG] Simulated input shape before collapsing block: {x.shape}")
    except Exception as e:
        print(f"[WARN] Hook-based simulation failed: {e}")
        # fallback — make dummy tensor with correct channels
        start_layer = selected_layers[0]
        if layer_type == nn.Conv2d:
            H, W = input_shape[-2:]
            x = torch.randn(1, start_layer.in_channels, H, W, device=device)
            dummy_input = x.clone()
            print(f"[DEBUG] Fallback dummy input created with shape {x.shape}")
        else:
            x = torch.randn(1, start_layer.in_features, device=device)
            dummy_input = x.clone()

    # --- Linear case ---
    if layer_type == nn.Linear:
        in_features = x.view(x.size(0), -1).size(1)
        print(f"[DEBUG] in_features determined (Linear): {in_features}")
        for layer in selected_layers:
            x = layer(x)
        out_features = x.view(x.size(0), -1).size(1)
        print(f"[DEBUG] out_features determined (Linear): {out_features}")
        # Build collapsed block for Linear (keeps same behavior as before)
        collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape, full_block=full_block)

    else:
        in_channels = x.shape[1]
        last_conv = selected_layers[-1]
        out_channels = last_conv.out_channels

        # compute composite stride
        def _ensure_tuple_stride(s):
            if isinstance(s, tuple):
                return s
            return (int(s), int(s))

        composite_stride = (1, 1)
        for layer in selected_layers:
            if hasattr(layer, 'stride'):
                s = _ensure_tuple_stride(layer.stride)
                composite_stride = (composite_stride[0] * s[0], composite_stride[1] * s[1])
        final_stride = composite_stride

        # forward check
        for layer in selected_layers:
            x = layer(x)
        out_shape = x.shape

        # --- PATCH 1: effective kernel and expected params ---
        kernel_sizes = []
        for l in selected_layers:
            if hasattr(l, 'kernel_size'):
                k = l.kernel_size[0] if isinstance(l.kernel_size, tuple) else l.kernel_size
                kernel_sizes.append(k)
        k_eff = sum(kernel_sizes) - (len(kernel_sizes) - 1)
        print(f"[INFO] Effective kernel size for collapse: {k_eff}x{k_eff}")

        expected_params = (in_channels * out_channels * (k_eff ** 2)) + out_channels
        print(f"[INFO] Expected trainable parameters (collapsed conv): {expected_params:,}")

        # continue to pool layer detection + build collapsed block
        pool_layer = None
        for nm, mod in reversed(full_block):
            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                pool_layer = mod
                break

        collapsed_block = _build_collapsed_block(layer_type, in_channels, out_channels, out_shape,
                                                full_block=full_block, stride=final_stride,
                                                pool_layer=pool_layer)


    # --- DEBUG PARAM CHECK ---
    pre_params = count_trainable_params(model)
    print(f"[DEBUG] Parameters BEFORE collapse: {pre_params:,}")

    # expected_params was already computed above for Conv2d; define fallback for Linear
    if layer_type == nn.Linear:
        expected_params = (in_features * out_features) + out_features

    # replace layers
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Parameters AFTER collapse: {post_params:,}")
    print(f"[DEBUG] Expected parameters for collapsed block: {expected_params:,}")
    print(f"[DEBUG] ΔParams = {pre_params - post_params:+,}")

    print(f"[INFO] ✅ Collapsed {start_container_name} layers {start_layer_name} → {end_layer_name}")
    print(f"[INFO] Model now has {post_params:,} trainable parameters (was {pre_params:,})")
    print(f"[DEBUG] Model structure after collapse:\n{layer_stats(model)}")

    return model

def _simulate_input_hook(model, target_layer_path, input_shape, device='cpu'):
    """
    Capture the activation that will be *input to the target layer*.
    Works for both direct nn.Conv2d layers (e.g., features.conv_3)
    and nested blocks (e.g., stage3.stage3_block0.block.conv2).
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    try:
        target_module = get_layer(model, target_layer_path)
    except Exception as e:
        raise RuntimeError(f"Could not resolve target module '{target_layer_path}': {e}")

    captured = {}

    def hook(module, inp, out):
        # inp is a tuple; grab the first entry
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

def forward_until(model, stop_path, x):
    """
    Forward input x through modules described by stop_path (dot separated).
    Note: calling composite modules (custom Blocks) will execute their forward.
    Use _simulate_input_hook for safe activation capture instead of forward_until when possible.
    """
    if stop_path == "":
        return x
    stop_parts = stop_path.split('.')
    current = model
    for part in stop_parts:
        if _is_int_str(part):
            current = current[int(part)]
        else:
            current = getattr(current, part)
        x = current(x)
    return x
def _build_collapsed_block(layer_type, in_features, out_features, output_shape, full_block=None, stride=(1, 1), pool_layer: Optional[nn.Module]=None):
    """
    Build a collapsed block safely.
    Clones any original modules (like pooling) to avoid circular references.
    Adds bottleneck if Conv2d, and preserves BN/ReLU if present.
    Returns an nn.Sequential containing the collapsed modules.
    """
    print(f"[DEBUG] Building collapsed block: {layer_type.__name__}, in={in_features}, out={out_features}, stride={stride}")

    seq_modules = []

    if layer_type == nn.Conv2d:
        # Detect if last layers in full_block are BN/ReLU
        has_bn = has_relu = False
        if full_block:
            # full_block can be list of (name, module) or list of modules
            if isinstance(full_block[0], tuple):
                mods = [m for _, m in full_block]
            else:
                mods = list(full_block)
            if isinstance(mods[-1], nn.ReLU):
                has_relu = True
                if len(mods) >= 2 and isinstance(mods[-2], nn.BatchNorm2d):
                    has_bn = True
            elif isinstance(mods[-1], nn.BatchNorm2d):
                has_bn = True

        # Simplified bottleneck collapse: single conv that maps in_features -> out_features
        # Note: you previously set k_eff = 1 and p_eff = 0; follow that here.
        k_eff = 1
        p_eff = 0

        conv1 = nn.Conv2d(in_features, out_features, kernel_size=k_eff, stride=stride, padding=p_eff, bias=False)
        print(f"[DEBUG] Built collapsed Conv2d: {in_features} -> {out_features}, kernel_size={k_eff}, stride={stride}, padding={p_eff}")
        seq_modules.append(conv1)

        if has_bn:
            seq_modules.append(nn.BatchNorm2d(out_features))
        if has_relu:
            seq_modules.append(nn.ReLU(inplace=False))

        # clone pool layer if exists
        if pool_layer is not None:
            seq_modules.append(copy.deepcopy(pool_layer))
            print(f"[DEBUG] Appending cloned pooling layer: {pool_layer.__class__.__name__}")

    elif layer_type == nn.Linear:
        linear = nn.Linear(in_features, out_features)
        seq_modules.append(linear)
        print(f"[DEBUG] Built collapsed Linear: {in_features} -> {out_features}")

    else:
        raise ValueError(f"Unsupported layer_type for collapse: {layer_type}")

    # wrap into an nn.Sequential with deterministic names
    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", m) for i, m in enumerate(seq_modules)]))
    print(f"[DEBUG] Collapsed block layers: {[type(m).__name__ for m in collapsed]}")
    return collapsed

def _update_container(model, container_path, new_container):
    """
    Replace the module at `container_path` in `model` with `new_container`.
    container_path is a dot-separated string specifying nested modules.
    """
    print(f"[DEBUG] Updating container at path: {container_path}")
    if container_path == "":
        raise ValueError("Cannot replace the root model container with a new container.")
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        print(f"[DEBUG] Traversing into: {part}")
        if _is_int_str(part):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    print(f"[DEBUG] Setting new container at final part: {last}")
    if _is_int_str(last):
        idx = int(last)
        parent[idx] = new_container
    else:
        raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    # Flatten into a single nn.Sequential
    collapsed = nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(seq)]))
    print(f"[DEBUG] Collapsed block layers: {[type(m).__name__ for m in collapsed]}")
    return collapsed

def _measure_flattened_size_by_forward(model, input_shape, device):
    """
    Fallback helper: forward a dummy input and find the last activation that is 4D (spatial),
    then compute flattened size.
    """
    print("[DEBUG] Entering _measure_flattened_size_by_forward")
    model.eval()
    model.to(device)
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            print(f"[DEBUG] Hook triggered for layer: {name}")
            activations[name] = out.detach()
        return hook

    print("[DEBUG] Registering forward hooks...")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU, nn.Sequential)):
            print(f"[DEBUG] Hook registered for module: {name}")
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            print("[DEBUG] Creating dummy input and performing forward pass...")
            dummy = torch.randn(input_shape).to(device)
            try:
                model(dummy)
                print("[DEBUG] Forward pass completed.")
            except Exception as e:
                print(f"[DEBUG] Exception during forward pass: {e}")
    finally:
        print("[DEBUG] Removing hooks...")
        for h in hooks:
            h.remove()

    print("[DEBUG] Searching for last 4D activation...")
    last = None
    for n, act in activations.items():
        print(f"[DEBUG] Found activation from layer: {n} with shape {act.shape}")
        last = act
    if last is None:
        print("[DEBUG] No activation found.")
        raise RuntimeError("Unable to measure flattened size by forward pass.")
    if last.dim() > 2:
        flattened = last.view(last.size(0), -1).size(1)
    else:
        flattened = last.size(1)
    print(f"[DEBUG] Flattened size determined: {flattened}")
    return int(flattened)

def _update_container(model, container_path, new_container):
    """
    Replace the module at `container_path` in `model` with `new_container`.
    container_path is a dot-separated string specifying nested modules.
    """
    print(f"[DEBUG] Updating container at path: {container_path}")
    if container_path == "":
        raise ValueError("Cannot replace the root model container with a new container.")
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        print(f"[DEBUG] Traversing into: {part}")
        if _is_int_str(part):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    print(f"[DEBUG] Setting new container at final part: {last}")
    if _is_int_str(last):
        idx = int(last)
        parent[idx] = new_container
    else:
        setattr(parent, last, new_container)

def _replace_layers(named_layers, start_idx, end_idx, new_block):
    """
    Replace layers start_idx..end_idx inclusive in named_layers with new_block.
    Returns a new nn.Sequential built from the original named_layers with the slice replaced.
    """
    print(f"[DEBUG] Replacing layers {start_idx} to {end_idx} with collapsed block...")
    new_layers = []
    unique_suffix = uuid4().hex[:8]
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_name = f"collapsed_{unique_suffix}"
            print(f"[DEBUG] Inserting new block as '{new_name}'")
            new_layers.append((new_name, new_block))
        elif start_idx < i <= end_idx:
            print(f"[DEBUG] Removing layer '{name}' at index {i}")
            continue
        else:
            new_layers.append((name, layer))
    print(f"[DEBUG] New container will have {len(new_layers)} children.")
    return nn.Sequential(OrderedDict(new_layers))


def adjust_classifier_input_features(model, input_shape, num_classes=200, device='cpu', preserve_original_fc=True):
    """
    Adjust the classifier input features if required.

    If preserve_original_fc=True:
        -> Do nothing (skip classifier modification entirely).
    If False:
        -> Attempt to recompute input feature size and patch first linear layer safely.
    """

    import torch
    import torch.nn as nn
    from collections import OrderedDict
    import math

    if preserve_original_fc:
        print("[INFO] preserve_original_fc=True → Skipping classifier rewrite. Keeping original FC intact.")
        return  # ✅ Early exit — nothing modified

    print("[INFO] preserve_original_fc=False → Checking classifier input feature consistency...")

    model.eval()
    model.to(device)

    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = {'input_shape': inp[0].shape, 'output_shape': out.shape}
        return hook

    # Hook into modules to capture feature shapes
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            dummy = torch.randn(input_shape).to(device)
            model(dummy)
    finally:
        for h in hooks:
            h.remove()

    # Find the first Linear layer
    linear_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if not linear_names:
        print("[WARN] No linear layers found. Nothing to adjust.")
        return

    first_linear_name = linear_names[0]
    hooked = activations.get(first_linear_name)

    if hooked:
        inp_shape = hooked['input_shape']
        flattened_size = int(torch.tensor(inp_shape[1:]).prod().item())
        print(f"[DEBUG] Flattened input size measured: {flattened_size}")
    else:
        # fallback to manual shape inference
        with torch.no_grad():
            dummy = torch.randn(input_shape).to(device)
            x = model.features(dummy)
            flattened_size = x.numel() // x.size(0)
        print(f"[WARN] Could not hook input shape. Using fallback flattened size={flattened_size}")

    # Retrieve the first linear layer reference
    first_linear_layer = dict(model.named_modules())[first_linear_name]

    if flattened_size == first_linear_layer.in_features:
        print("[INFO] Flattened input matches Linear.in_features. No adjustment required.")
        return

    print(f"[INFO] Adjusting first Linear layer: in_features {first_linear_layer.in_features} → {flattened_size}")

    # Replace linear layer safely
    parent_parts = first_linear_name.split(".")
    parent_container = model
    for part in parent_parts[:-1]:
        parent_container = getattr(parent_container, part)
    last_part = parent_parts[-1]

    new_linear = nn.Linear(flattened_size, first_linear_layer.out_features)
    setattr(parent_container, last_part, new_linear)

    model.to(device)
    model.train()
    print(f"[✓] Updated Linear layer '{first_linear_name}' to match new flattened feature size.")

def _get_container_and_subname(layer_name):
    """
    Extracts the container (e.g., 'features', 'stage3_block0.block') and the subname.
    """
    print(f"[DEBUG] Splitting layer name: {layer_name}")
    if layer_name == "":
        return "", ""
    layer_parts = layer_name.split('.')
    if len(layer_parts) == 1:
        print(f"[DEBUG] No container, subname: {layer_parts[0]}")
        return "", layer_parts[0]
    container = '.'.join(layer_parts[:-1])
    subname = layer_parts[-1]
    print(f"[DEBUG] Container: {container}, Subname: {subname}")
    return container, subname

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    print(f"[DEBUG] Finding indices for layers '{start_layer_name}' to '{end_layer_name}'...")

    for i, (name, _) in enumerate(named_layers):
        # print(f"[DEBUG] Checking layer: {name} at index {i}")
        if name == start_layer_name:
            start_idx = i
            print(f"[DEBUG] Found start layer '{start_layer_name}' at index {start_idx}")
        if name == end_layer_name:
            end_idx = i
            print(f"[DEBUG] Found end layer '{end_layer_name}' at index {end_idx}")

    if start_idx is None or end_idx is None:
        print(f"[DEBUG] Warning: Could not find one or both layer names '{start_layer_name}', '{end_layer_name}' inside the container's children.")

    return start_idx, end_idx
