# collapse.py
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats
from uuid import uuid4
from typing import Optional
import copy

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

def _collapse_block(model, start_idx, end_idx, named_layers, layer_type, input_shape, device):
    """Collapse a contiguous block of layers into a single block + optional adaptive pooling
       that matches what the next layer expects."""

    import copy
    from collections import OrderedDict

    # Extract the candidate block
    full_block = named_layers[start_idx:end_idx + 1]
    in_channels = None
    out_channels = None
    x = torch.randn(1, *input_shape, device=device)

    # Forward through selected layers to detect in/out channels
    selected_layers = [m for _, m in full_block]
    for i, mod in enumerate(selected_layers):
        if i == 0 and hasattr(mod, 'in_channels'):
            in_channels = mod.in_channels
        x = mod(x)
        if hasattr(mod, 'out_channels'):
            out_channels = mod.out_channels
    out_shape = x.shape  # (B, C, H, W)
    print(f"[DEBUG] Block {start_idx}-{end_idx} out shape: {out_shape}")

    # --- Extend end_idx to include trailing BN/ReLU/Pooling so we don't duplicate them ---
    while end_idx + 1 < len(named_layers) and isinstance(named_layers[end_idx + 1][1],
                                                         (nn.BatchNorm2d, nn.ReLU)):
        print(f"[DEBUG] Extending collapse range with {named_layers[end_idx+1][0]} ({type(named_layers[end_idx+1][1]).__name__})")
        end_idx += 1
    while end_idx + 1 < len(named_layers) and isinstance(named_layers[end_idx + 1][1],
                                                         (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        print(f"[DEBUG] Extending collapse range with {named_layers[end_idx+1][0]} ({type(named_layers[end_idx+1][1]).__name__})")
        end_idx += 1
    full_block = named_layers[start_idx:end_idx + 1]

    # --- Determine tail adaptive pooling ---
    tail_pool = None
    next_idx = end_idx + 1
    if next_idx < len(named_layers):
        next_mod = named_layers[next_idx][1]
        if isinstance(next_mod, nn.AdaptiveAvgPool2d):
            # Copy the exact next AdaptiveAvgPool2d
            tail_pool = copy.deepcopy(next_mod)
            print("[DEBUG] Next layer is AdaptiveAvgPool2d; preserving inside collapsed block.")
        else:
            # If block produced non-1x1, add AdaptiveAvgPool2d to match shape
            if out_shape is not None and len(out_shape) == 4:
                H, W = int(out_shape[2]), int(out_shape[3])
                if not (H == 1 and W == 1):
                    tail_pool = nn.AdaptiveAvgPool2d(output_size=(H, W))
                    print(f"[DEBUG] Appending AdaptiveAvgPool2d(output_size=({H},{W})) to collapsed block.")
                else:
                    print("[DEBUG] Output already 1x1, no adaptive pool appended.")

    # --- Build collapsed block ---
    collapsed_block = _build_collapsed_block(
        layer_type=layer_type,
        in_features=in_channels,
        out_features=out_channels,
        output_shape=out_shape,
        full_block=full_block,
        stride=(1, 1),     # use stride=1 by default unless you detect a downsample conv
        tail_pool=tail_pool
    )

    # --- Replace the original slice in the model with collapsed_block ---
    parent = model
    parent_name, _ = named_layers[start_idx]
    # Traverse to the parent container of the start layer
    parts = parent_name.split('.')[:-1]
    for p in parts:
        parent = getattr(parent, p) if hasattr(parent, p) else parent[int(p)]
    key = parent_name.split('.')[-1]
    if isinstance(parent, nn.Sequential):
        # Remove old modules and insert collapsed
        new_seq = OrderedDict()
        for i, (nm, mod) in enumerate(parent.named_children()):
            abs_idx = start_idx + i
            if start_idx <= abs_idx <= end_idx:
                if abs_idx == start_idx:
                    new_seq["collapsed_block"] = collapsed_block
                # skip others
            else:
                new_seq[nm] = mod
        setattr(model, parts[-1] if parts else 'features', nn.Sequential(new_seq))
    else:
        setattr(parent, key, collapsed_block)

    print(f"[DEBUG] Replaced layers {start_idx}..{end_idx} with collapsed block (Conv + optional AdaptivePool).")
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

def _build_collapsed_block(layer_type, in_features, out_features, output_shape, full_block=None, stride=(1, 1), tail_pool=None):
    """ Build a conservative collapsed block:
        - Conv2d -> 1x1 conv mapping in_channels -> out_channels, use final stride,
          include BatchNorm/ReLU if originally present AT THE END of the slice (not anywhere).
        - Optionally append a tail_pool (AdaptiveAvgPool2d or other) if provided.
        - Linear -> single Linear(in_features, out_features).
    """
    print(f"[DEBUG] Building collapsed block of type {layer_type.__name__} with in_features={in_features}, out_features={out_features}, output_shape={output_shape}, stride={stride}, tail_pool={type(tail_pool).__name__ if tail_pool is not None else None}")

    if layer_type == nn.Conv2d:
        # detect presence of BN/ReLU at the *end* of the original slice
        has_bn = False
        has_relu = False
        if full_block:
            # full_block may be list of (name, module) pairs or modules — normalize
            mods = [m for _, m in full_block] if isinstance(full_block[0], tuple) and len(full_block[0]) == 2 else list(full_block)
            # Inspect tail
            if len(mods) >= 1 and isinstance(mods[-1], nn.ReLU):
                has_relu = True
                if len(mods) >= 2 and isinstance(mods[-2], nn.BatchNorm2d):
                    has_bn = True
            elif len(mods) >= 1 and isinstance(mods[-1], nn.BatchNorm2d):
                has_bn = True

        print(f"[DEBUG] Collapsed block end flags -> has_bn: {has_bn}, has_relu: {has_relu}")

        # Use a 1x1 conv as a conservative collapsed operator
        conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride, padding=0, bias=not has_bn)
        seq = [conv]
        if has_bn:
            seq.append(nn.BatchNorm2d(out_features))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))

        # append preserved tail_pool if provided (could be AdaptiveAvgPool2d or Max/Avg pool)
        if tail_pool is not None:
            seq.append(tail_pool)
            print(f"[DEBUG] Appended tail pool: {type(tail_pool).__name__} to collapsed block.")

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
    Dynamically detect the classifier/head module and adjust its first Linear
    to match flattened feature size after current backbone. This does NOT assume
    any attribute name like 'classifier' or 'fc'. Instead:
      - run a forward pass with hooks to capture activations
      - find the first Linear module (by named_modules order)
      - replace the first Linear's in_features to match the measured flattened size
    If nothing is detected, fall back to inserting model.fc.
    """
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
                # we swallow exceptions here because head may raise if misconfigured; hooks may still have data
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
        # measure flattened size from activations if available
        flattened_size = None
        hooked = activations.get(first_linear_name)
        if hooked is not None:
            inp_shape = hooked['input_shape']
            if len(inp_shape) > 2:
                flattened_size = int(torch.tensor(inp_shape[1:]).prod().item())
            else:
                flattened_size = inp_shape[1]
            print(f"[DEBUG] Measured flattened features size for head: {flattened_size}")
        else:
            flattened_size = _measure_flattened_size_by_forward(model, input_shape, device)
            print(f"[DEBUG] Fallback flattened size measured as: {flattened_size}")

        # replace first Linear in its parent container
        if parent_container_name == "":
            # top-level linear attribute like model.fc (rare)
            orig = getattr(model, first_linear_name)
            if isinstance(orig, nn.Linear):
                setattr(model, first_linear_name, nn.Linear(flattened_size, orig.out_features))
                print(f"[INFO] Replaced top-level linear '{first_linear_name}' to have in_features={flattened_size}")
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
                        new_children.append((name, nn.Linear(flattened_size, child.out_features)))
                        replaced = True
                    else:
                        new_children.append((name, child))
                new_seq = nn.Sequential(OrderedDict(new_children))
                _update_container(model, parent_container_name, new_seq)
                model.to(device)
                print(f"[INFO] Rebuilt sequential head '{parent_container_name}' with updated first Linear in_features={flattened_size}")
                model.train()
                return
            else:
                # parent container not sequential, try replacing attribute in parent_container
                parts = first_linear_name.split('.')
                par = '.'.join(parts[:-1])
                last = parts[-1]
                par_container = get_layer(model, par)
                orig = getattr(par_container, last)
                if isinstance(orig, nn.Linear):
                    setattr(par_container, last, nn.Linear(flattened_size, orig.out_features))
                    _update_container(model, par, par_container)  # ensure update
                    model.to(device)
                    print(f"[INFO] Replaced linear '{first_linear_name}' inside '{par}'")
                    model.train()
                    return

    # fallback: attach a simple head
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