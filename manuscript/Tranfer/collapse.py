# collapse.py
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats
from uuid import uuid4
from typing import Optional

# ===============================
# Layer Collapse Helpers (robust/residual-aware)
# ===============================

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

def compression_set(layers):
    """
    Example of compressing layers. Replace this with actual compression logic.
    """
    for layer in layers:
        print(f"[DEBUG] Compressing layer: {layer}")
        # Add compression logic here (e.g., pruning, quantization)

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None, input_shape=(1, 3, 32, 32), device='cpu'):
    model_kwargs = model_kwargs or {}
    num_classes = model_kwargs.get('num_classes', 200)

    print(f"[INFO] Loading model with weights from '{model_weights_1}'...")
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    
    print(f"[INFO] Full compression set: {compression_set}")
    
    for compression_set1 in compression_set:
        print(f"[DEBUG] Compressing: {compression_set1}")
        start, end = compression_set1[0], compression_set1[1]
        print(f"\n--- Starting collapse for block: {start} to {end} ---")
        model = _collapse_block(model, start, end, input_shape, device=device)

        print("\n--- Adjusting classifier / head AFTER collapsing all blocks ---")
        adjust_classifier_input_features(model, input_shape, num_classes=num_classes, device=device)

    # NEW: disable any in-place ReLUs globally to avoid autograd inplace modification errors
    disable_inplace_relu(model)
    model.to(device)  # ensure all new modules are moved to correct device

    print(f"[INFO] Collapse complete. Total trainable params: {count_trainable_params(model)}")
    
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

    # --- Conv2d case ---
    else:
        in_channels = x.shape[1]
        last_conv = selected_layers[-1]
        out_channels = last_conv.out_channels

        # compute composite stride
        def _ensure_tuple_stride(s):
            if isinstance(s, tuple):
                return s
            try:
                return (int(s), int(s))
            except Exception:
                return (1, 1)

        composite_stride = (1, 1)
        for layer in selected_layers:
            if hasattr(layer, 'stride'):
                s = _ensure_tuple_stride(layer.stride)
                composite_stride = (composite_stride[0] * s[0], composite_stride[1] * s[1])

        # align stride with shortcut if present
        try:
            container_module = get_layer(model, start_container_name)
            if hasattr(container_module, 'shortcut'):
                sc = getattr(container_module, 'shortcut', None)
                if sc is not None and hasattr(sc, 'shortcut_conv'):
                    sc_conv = getattr(sc, 'shortcut_conv')
                    sc_stride = _ensure_tuple_stride(getattr(sc_conv, 'stride', (1, 1)))
                    if sc_stride != composite_stride:
                        print(f"[WARN] Composite stride {composite_stride} != shortcut stride {sc_stride}. Aligning.")
                        composite_stride = sc_stride
        except Exception:
            pass

        final_stride = composite_stride
        print(f"[DEBUG] in_channels={in_channels}, out_channels={out_channels}, stride={final_stride}")

        # forward through selected layers to check output shape (debug)
        for i, layer in enumerate(selected_layers):
            x = layer(x)
            print(f"[DEBUG] After {i} ({type(layer).__name__}) → {x.shape}")

        out_shape = x.shape

        # --- IMPORTANT PATCH: extend end_idx to include trailing BN/ReLU after the last conv ---
        # If the slice ended on conv_13 but bn_13 or relu_13 comes after, include them in removal.
        print(f"[DEBUG] Named layers around end index ({end_idx}):")
        for idx in range(max(0, end_idx - 2), min(len(named_layers), end_idx + 3)):
            nm, mod = named_layers[idx]
            print(f"   idx={idx} name={nm} type={mod.__class__.__name__}")

        # extend end_idx while next layer is BatchNorm or ReLU
        while end_idx + 1 < len(named_layers) and isinstance(named_layers[end_idx + 1][1], (nn.BatchNorm2d, nn.ReLU)):
            print(f"[DEBUG] Extending collapse range to include trailing {type(named_layers[end_idx + 1][1]).__name__} "
                  f"'{named_layers[end_idx + 1][0]}' at index {end_idx+1}")
            end_idx += 1

        # rebuild full_block now that end_idx might have grown
        full_block = named_layers[start_idx:end_idx + 1]
        print(f"[DEBUG] Final block will remove indices {start_idx}..{end_idx}: {[n for n,_ in full_block]}")

        collapsed_block = _build_collapsed_block(layer_type, in_channels, out_channels, out_shape, full_block=full_block, stride=final_stride)

    # --- replace in container ---
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    print(f"[INFO] Collapsed {start_container_name} layers {start_layer_name} → {end_layer_name}")
    print(f"[INFO] New trainable params: {count_trainable_params(model)}")
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

def _build_collapsed_block(layer_type, in_features, out_features, output_shape, full_block=None, stride=(1, 1)):
    """
    Build a conservative collapsed block:
    - Conv2d -> 1x1 conv mapping in_channels -> out_channels, use final stride,
      include BatchNorm/ReLU if originally present AT THE END of the slice (not anywhere).
    - Linear -> single Linear(in_features, out_features).
    """
    print(f"[DEBUG] Building collapsed block of type {layer_type.__name__} with in_features={in_features}, out_features={out_features}, output_shape={output_shape}, stride={stride}")

    if layer_type == nn.Conv2d:
        # detect presence of BN/ReLU at the *end* of the original slice
        has_bn = False
        has_relu = False
        if full_block:
            # full_block is a list of (name, module) pairs or modules depending on call site:
            # unify to modules list:
            mods = []
            if isinstance(full_block[0], tuple) and len(full_block[0]) == 2:
                mods = [m for _, m in full_block]
            else:
                mods = list(full_block)

            # only look at the tail
            if len(mods) >= 1 and isinstance(mods[-1], nn.ReLU):
                has_relu = True
                # if second-to-last is BN keep it as bn then relu
                if len(mods) >= 2 and isinstance(mods[-2], nn.BatchNorm2d):
                    has_bn = True
            elif len(mods) >= 1 and isinstance(mods[-1], nn.BatchNorm2d):
                has_bn = True

        print(f"[DEBUG] Collapsed block end flags -> has_bn: {has_bn}, has_relu: {has_relu}")

        # Use a 1x1 conv as a conservative collapsed operator (this is what you had before)
        conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, padding=0, bias=not has_bn)
        seq = [conv]
        if has_bn:
            seq.append(nn.BatchNorm2d(out_features))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))

        collapsed = nn.Sequential(OrderedDict([("collapsed_conv", nn.Sequential(*seq))]))
        print(f"[DEBUG] Built collapsed Conv block with layers: {[type(m).__name__ for m in seq]}")
        return collapsed

    elif layer_type == nn.Linear:
        block = nn.Sequential(OrderedDict([("collapsed_linear", nn.Linear(in_features, out_features))]))
        print(f"[DEBUG] Built collapsed Linear block: Linear({in_features} -> {out_features})")
        return block

    else:
        raise NotImplementedError(f"Unsupported layer type for collapsing: {layer_type}")

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
        print(f"[DEBUG] Checking layer: {name} at index {i}")
        if name == start_layer_name:
            start_idx = i
            print(f"[DEBUG] Found start layer '{start_layer_name}' at index {start_idx}")
        if name == end_layer_name:
            end_idx = i
            print(f"[DEBUG] Found end layer '{end_layer_name}' at index {end_idx}")

    if start_idx is None or end_idx is None:
        print(f"[DEBUG] Warning: Could not find one or both layer names '{start_layer_name}', '{end_layer_name}' inside the container's children.")

    return start_idx, end_idx

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
            new_name = f"collapsed_{named_layers[start_idx][0]}_to_{named_layers[end_idx][0]}_{unique_suffix}"
            print(f"[DEBUG] Inserting new block as '{new_name}'")
            new_layers.append((new_name, new_block))
        elif start_idx < i <= end_idx:
            print(f"[DEBUG] Removing layer '{name}' at index {i}")
            continue
        else:
            new_layers.append((name, layer))
    print(f"[DEBUG] New container will have {len(new_layers)} children.")
    return nn.Sequential(OrderedDict(new_layers))
def adjust_classifier_input_features(model, input_shape, num_classes=200, device='cpu'):
    """
    Update head so the first Linear receives the feature size it expects,
    but instead of changing the Linear's in_features, insert an AdaptiveAvgPool2d
    (and a Flatten) before it to match the expected flattened size.

    Minimal, safe changes: replace the Linear module with a nn.Sequential containing
    (AdaptiveAvgPool2d -> Flatten -> Linear) when needed.
    """
    import math

    model.eval()
    model.to(device)

    # register hooks to capture module inputs
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            # store the input tensor to the module
            activations[name] = {'input_shape': inp[0].shape, 'output_shape': out.shape}
        return hook

    # attach hooks to linear and pooling modules to capture shapes
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.Sequential)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            dummy = torch.randn(input_shape).to(device)
            try:
                model(dummy)
            except Exception:
                # swallow exceptions -- hooks may still have useful data
                pass
    finally:
        for h in hooks:
            h.remove()

    # find first Linear's name
    linear_names = [name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    if linear_names:
        first_linear_name = linear_names[0]
        print(f"[INFO] First linear found at: {first_linear_name}")

        parent_container_name, linear_subname = _get_container_and_subname(first_linear_name)

        # "hooked" is the activation _input_ to the Linear (if captured)
        hooked = activations.get(first_linear_name)
        flattened_size = None
        if hooked is not None:
            inp_shape = hooked['input_shape']
            if len(inp_shape) > 2:
                # C, H, W -> flattened size is C*H*W
                flattened_size = int(torch.tensor(inp_shape[1:]).prod().item())
            else:
                flattened_size = inp_shape[1]

            print(f"[DEBUG] Measured flattened features size for head (from hooks): {flattened_size}")
        else:
            # fallback measurement function (keeps old behavior)
            flattened_size = _measure_flattened_size_by_forward(model, input_shape, device)
            print(f"[DEBUG] Fallback flattened size measured as: {flattened_size}")

        # Now, instead of changing Linear.in_features, insert adaptive pool to match Linear's expected in_features
        # Find the original Linear module
        if parent_container_name == "":
            # top-level attribute like model.fc
            orig = getattr(model, first_linear_name)
            if isinstance(orig, nn.Linear):
                # If the input we measured equals orig.in_features, nothing to do
                if flattened_size == orig.in_features:
                    print("[INFO] Head flattened size matches Linear.in_features. No changes needed.")
                    model.to(device)
                    model.train()
                    return
                # Try to compute pooling target using captured activation if available
                if hooked is not None and len(hooked['input_shape']) > 2:
                    C = hooked['input_shape'][1]
                    if orig.in_features % C == 0:
                        expected_hw = orig.in_features // C
                        # prefer perfect-square target (HxW)
                        s = int(math.isqrt(expected_hw))
                        if s * s == expected_hw:
                            pool_target = (s, s)
                        else:
                            pool_target = (1, 1)  # fallback to global pool
                        # replace top-level linear with Sequential(pool, flatten, linear)
                        new_head = nn.Sequential(OrderedDict([
                            ("pre_pool", nn.AdaptiveAvgPool2d(pool_target)),
                            ("flatten", nn.Flatten()),
                            ("linear", nn.Linear(orig.in_features, orig.out_features))
                        ]))
                        setattr(model, first_linear_name, new_head)
                        print(f"[INFO] Inserted AdaptiveAvgPool2d({pool_target}) before top-level linear '{first_linear_name}'")
                        model.to(device)
                        model.train()
                        return
                # fallback: change linear as before (rare top-level case without spatial info)
                setattr(model, first_linear_name, nn.Linear(flattened_size, orig.out_features))
                print(f"[WARN] Top-level linear replacement fallback: set in_features={flattened_size}")
                model.to(device)
                model.train()
                return

        else:
            parent_container = get_layer(model, parent_container_name)
            if isinstance(parent_container, nn.Sequential):
                new_children = []
                replaced = False
                for name, child in parent_container.named_children():
                    if (not replaced) and isinstance(child, nn.Linear):
                        orig = child
                        # if no mismatch, keep as-is
                        if flattened_size == orig.in_features:
                            new_children.append((name, child))
                            replaced = True
                            continue

                        # if we have spatial activation info, try to compute a pooling target
                        if hooked is not None and len(hooked['input_shape']) > 2:
                            C = hooked['input_shape'][1]
                            if orig.in_features % C == 0:
                                expected_hw = orig.in_features // C
                                s = int(math.isqrt(expected_hw))
                                if s * s == expected_hw:
                                    pool_target = (s, s)
                                else:
                                    pool_target = (1, 1)  # safe global pool fallback
                                # wrap: pool -> flatten -> linear(orig.in_features -> out_features)
                                new_block = nn.Sequential(OrderedDict([
                                    (f"{name}_pre_pool", nn.AdaptiveAvgPool2d(pool_target)),
                                    (f"{name}_flatten", nn.Flatten()),
                                    (f"{name}_linear", nn.Linear(orig.in_features, orig.out_features))
                                ]))
                                new_children.append((name, new_block))
                                replaced = True
                                print(f"[INFO] Replaced sequential head child '{name}' with AdaptiveAvgPool2d({pool_target}) + Linear")
                                continue
                        # otherwise fallback to reconstructing the Linear to match measured flattened_size
                        new_children.append((name, nn.Linear(flattened_size, orig.out_features)))
                        replaced = True
                    else:
                        new_children.append((name, child))

                # rebuild sequential
                new_seq = nn.Sequential(OrderedDict(new_children))
                _update_container(model, parent_container_name, new_seq)
                model.to(device)
                print(f"[INFO] Rebuilt sequential head '{parent_container_name}' with adjusted first Linear/pool")
                model.train()
                return
            else:
                # parent container is not sequential: attempt attribute replacement
                parts = first_linear_name.split('.')
                par = '.'.join(parts[:-1])
                last = parts[-1]
                par_container = get_layer(model, par)
                orig = getattr(par_container, last)
                if isinstance(orig, nn.Linear):
                    # if we have spatial input info, insert a small Sequential in-place
                    if hooked is not None and len(hooked['input_shape']) > 2:
                        C = hooked['input_shape'][1]
                        if orig.in_features % C == 0:
                            expected_hw = orig.in_features // C
                            s = int(math.isqrt(expected_hw))
                            if s * s == expected_hw:
                                pool_target = (s, s)
                            else:
                                pool_target = (1, 1)
                            new_attr = nn.Sequential(OrderedDict([
                                ("pre_pool", nn.AdaptiveAvgPool2d(pool_target)),
                                ("flatten", nn.Flatten()),
                                ("linear", nn.Linear(orig.in_features, orig.out_features))
                            ]))
                            setattr(par_container, last, new_attr)
                            _update_container(model, par, par_container)
                            model.to(device)
                            print(f"[INFO] Inserted AdaptiveAvgPool2d({pool_target}) before linear '{first_linear_name}' in non-seq parent")
                            model.train()
                            return
                    # fallback (no spatial info): replace linear in_features
                    setattr(par_container, last, nn.Linear(flattened_size, orig.out_features))
                    _update_container(model, par, par_container)
                    model.to(device)
                    print(f"[WARN] Replaced linear '{first_linear_name}' in non-seq parent to have in_features={flattened_size}")
                    model.train()
                    return

    # fallback: previous fallback behavior (create a new fc/classifier)
    print("[WARN] Could not detect head structure cleanly. Falling back to a simple Linear head.")
    flattened_size = _measure_flattened_size_by_forward(model, input_shape, device)
    if hasattr(model, 'fc') and isinstance(getattr(model, 'fc', None), nn.Linear):
        model.fc = nn.Linear(flattened_size, num_classes)
        print("[INFO] Replaced model.fc with new head.")
    elif hasattr(model, 'classifier') and isinstance(getattr(model, 'classifier', None), nn.Sequential):
        model.classifier = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(inplace=False),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
        print("[INFO] Replaced model.classifier with simple head.")
    else:
        model.fc = nn.Linear(flattened_size, num_classes)
        print("[INFO] Set model.fc (new) as fallback head.")
    model.to(device)
    model.train()
