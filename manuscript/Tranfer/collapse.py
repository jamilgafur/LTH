# collapse.py (patched)
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import count_trainable_params, layer_stats

# ===============================
# Layer Collapse Helpers (robust/residual-aware)
# ===============================


def _set_module_by_path(model, module_path, new_module):
    """
    Replace module referenced by module_path (dot-separated). Handles numeric indices for Sequential.
    """
    if module_path == "":
        raise ValueError("_set_module_by_path: cannot replace root module")
    parts = module_path.split('.')
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        idx = int(last)
        parent[idx] = new_module
    else:
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
        return

    print(f"[INFO] Replacing {len(to_replace)} in-place ReLU(s) with out-of-place variants.")
    for name in to_replace:
        container, subname = _get_container_and_subname(name)
        parent = get_layer(model, container) if container != "" else model
        # set new non-inplace ReLU in parent
        new_relu = nn.ReLU(inplace=False)
        # if subname is a numeric index in Sequential
        if subname.isdigit():
            idx = int(subname)
            parent[idx] = new_relu
        else:
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
    if layer_name == "":
        return model
    layer_parts = layer_name.split('.')
    layer = model
    for part in layer_parts:
        if _is_int_str(part):
            idx = int(part)
            layer = layer[idx]
        else:
            layer = getattr(layer, part)
    return layer

def compression_set(layers):
    """
    Example of compressing layers. Replace this with actual compression logic.
    """
    for layer in layers:
        print(f"Compressing layer: {layer}")
        # Add compression logic here (e.g., pruning, quantization)

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None, input_shape=(1, 3, 32, 32), device='cpu'):
    model_kwargs = model_kwargs or {}
    num_classes = model_kwargs.get('num_classes', 200)

    print(f"Loading model with weights from '{model_weights_1}'...")
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    for start, end in compression_set:
        print(f"\n--- Starting collapse for block: {start} to {end} ---")
        model = _collapse_block(model, start, end, input_shape, device=device)

    print("\n--- Adjusting classifier / head AFTER collapsing all blocks ---")
    adjust_classifier_input_features(model, input_shape, num_classes=num_classes, device=device)

    # NEW: disable any in-place ReLUs globally to avoid autograd inplace modification errors
    disable_inplace_relu(model)
    model.to(device)  # ensure all new modules are moved to correct device

    print(f"Collapse complete. Total trainable params: {count_trainable_params(model)}")
    print(f"Final model structure:\n{layer_stats(model)}")

    return model
    
def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive).
    They must belong to the same container (e.g., 'features').
    This version fixes channel-mismatch bugs by simulating input BEFORE the start layer,
    and by deriving correct in_channels from the actual start layer itself.
    """
    print(f"\nCollapsing layers from '{start_layer_name}' to '{end_layer_name}'...")

    # gather candidate modules (for debugging/reporting)
    build_layer_names = []
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.AdaptiveAvgPool2d, nn.BatchNorm2d)):
            build_layer_names.append(name)
    print(f"Available layers for collapsing: {build_layer_names}")

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)

    if start_container_name != end_container_name:
        raise ValueError(
            f"Start and end layers must be in the same container. Found: {start_container_name}, {end_container_name}"
        )

    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())

    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(
            f"Layer names '{start_layer_name}' or '{end_layer_name}' not found inside container '{start_container_name}'."
        )
    assert start_idx <= end_idx, "Start index must be <= end index"
    print(f"Collapsing in section '{start_container_name}' from index {start_idx} to {end_idx}")

    full_block = named_layers[start_idx:end_idx + 1]
    selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]

    if len(selected_layers) < 2:
        raise ValueError("Need at least 2 Conv2d or Linear layers to collapse.")
    layer_type = type(selected_layers[0])
    if not all(isinstance(l, layer_type) for l in selected_layers):
        raise ValueError("Cannot collapse mixed layer types.")

    # ✅ FIX 1: ensure correct input shape before start layer
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        print(f"  Simulated input shape before collapsing block: {x.shape}")
    except Exception as e:
        print(f"[WARN] Hook-based simulation failed: {e}")
        # fallback — create dummy tensor with start layer’s expected in_channels
        start_layer = selected_layers[0]
        if layer_type == nn.Conv2d:
            H, W = input_shape[-2:]
            x = torch.randn(1, start_layer.in_channels, H, W, device=device)
            dummy_input = x.clone()
            print(f"  Fallback dummy input created with shape {x.shape}")
        else:
            x = torch.randn(1, selected_layers[0].in_features, device=device)
            dummy_input = x.clone()

    # ✅ FIX 2: derive channels/features from actual start/end layers
    if layer_type == nn.Linear:
        in_features = x.view(x.size(0), -1).size(1)
        print(f"  in_features determined (Linear) as: {in_features}")
        for layer in selected_layers:
            x = layer(x)
        out_features = x.view(x.size(0), -1).size(1)
        print(f"  out_features determined (Linear) as: {out_features}")
        collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape, full_block=full_block)
    else:  # Conv2d
        in_channels = selected_layers[0].in_channels
        last_conv = selected_layers[-1]
        out_channels = last_conv.out_channels
        final_stride = last_conv.stride if hasattr(last_conv, 'stride') else (1, 1)
        print(f"  in_channels = {in_channels}, out_channels = {out_channels}, final_stride = {final_stride}")

        # forward through selected convs to check shapes
        for i, layer in enumerate(selected_layers):
            x = layer(x)
            print(f"    After selected layer {i} ({type(layer).__name__}) output shape: {x.shape}")

        out_shape = x.shape
        collapsed_block = _build_collapsed_block(
            layer_type, in_channels, out_channels, out_shape, full_block=full_block, stride=final_stride
        )

    # Build updated container by replacing start_idx..end_idx with collapsed_block
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    print(f"Collapsed {start_container_name} layers {start_layer_name} → {end_layer_name}")
    print(f"New trainable params: {count_trainable_params(model)}")
    print(f"Model structure after collapse:\n{layer_stats(model)}")
    return model

def _simulate_input_hook(model, target_layer_path, input_shape, device='cpu'):
    """
    Safely capture activation at the parent container of target_layer_path by
    registering a forward hook on that container and executing a single model(dummy_input).
    This avoids calling composite modules prematurely (fixes RegNet channel mismatch).
    Returns (dummy_input, activation_tensor) where activation_tensor is the output of the parent container.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    parent_path = '.'.join(target_layer_path.split('.')[:-1])
    parent_module = get_layer(model, parent_path) if parent_path != "" else model

    captured = {}

    def hook(module, inp, out):
        # capture the output of the parent container
        captured['out'] = out.detach()

    handle = parent_module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            # run one forward; hook will fill captured['out']
            model(dummy_input)
    finally:
        handle.remove()

    if 'out' not in captured:
        raise RuntimeError(f"Failed to capture activation at '{parent_path}' during forward hook simulation.")
    return dummy_input, captured['out']

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
        include BatchNorm/ReLU if originally present. Use inplace=False for ReLU.
      - Linear -> single Linear(in_features, out_features).
    """
    print(f"Building collapsed block of type {layer_type.__name__} with in_features={in_features}, out_features={out_features}, output_shape={output_shape}, stride={stride}")

    if layer_type == nn.Conv2d:
        # detect presence of BN/ReLU in original slice
        has_bn = False
        has_relu = False
        for _, layer in (full_block or []):
            if isinstance(layer, nn.BatchNorm2d):
                has_bn = True
            if isinstance(layer, nn.ReLU):
                has_relu = True

        conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                         kernel_size=1, stride=stride, padding=0, bias=not has_bn)

        seq = [conv]
        if has_bn:
            seq.append(nn.BatchNorm2d(out_features))
        if has_relu:
            # important: avoid inplace=True to prevent autograd inplace conflicts
            seq.append(nn.ReLU(inplace=False))

        collapsed = nn.Sequential(OrderedDict([("collapsed_conv", nn.Sequential(*seq))]))
        return collapsed

    elif layer_type == nn.Linear:
        block = nn.Sequential(OrderedDict([("collapsed_linear", nn.Linear(in_features, out_features))]))
        return block

    else:
        raise NotImplementedError(f"Unsupported layer type for collapsing: {layer_type}")

def _replace_layers(named_layers, start_idx, end_idx, new_block):
    """
    Replace layers start_idx..end_idx inclusive in named_layers with new_block.
    named_layers: list of (name, module) as returned by container.named_children()
    Returns an nn.Sequential(OrderedDict(...)) appropriate to set on parent.
    """
    print(f"Replacing layers {start_idx} to {end_idx} with collapsed block...")
    new_layers = []
    for i, (name, layer) in enumerate(named_layers):
        if i == start_idx:
            new_name = f"collapsed_{named_layers[start_idx][0]}_to_{named_layers[end_idx][0]}"
            print(f"  Inserting new block as '{new_name}'")
            new_layers.append((new_name, new_block))
        elif start_idx < i <= end_idx:
            print(f"  Removing layer '{name}' at index {i}")
            continue
        else:
            new_layers.append((name, layer))
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
            # copy to cpu to avoid keeping device memory pinned in closure
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
        # try to find last activation that precedes the linear
        flattened_size = None
        hooked = activations.get(first_linear_name)
        if hooked is not None:
            # the hook captured input dimensions for the linear
            inp_shape = hooked['input_shape']
            if len(inp_shape) > 2:
                flattened_size = int(torch.tensor(inp_shape[1:]).prod().item())
            else:
                flattened_size = inp_shape[1]
            print(f"[DEBUG] Measured flattened features size for head: {flattened_size}")
        else:
            # fallback: measure by doing a forward and catching last Conv/Pool output
            flattened_size = _measure_flattened_size_by_forward(model, input_shape, device)
            print(f"[DEBUG] Fallback flattened size measured as: {flattened_size}")

        # replace first Linear in its parent container
        if parent_container_name == "":
            # top-level linear attribute like model.fc
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

def _measure_flattened_size_by_forward(model, input_shape, device):
    """
    Fallback helper: forward a dummy input and find the last activation that is 4D (spatial),
    then compute flattened size.
    """
    model.eval()
    model.to(device)
    activations = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d, nn.BatchNorm2d, nn.ReLU, nn.Sequential)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            dummy = torch.randn(input_shape).to(device)
            try:
                model(dummy)
            except Exception:
                pass
    finally:
        for h in hooks:
            h.remove()

    # pick last activation (most recent key)
    last = None
    for n, act in activations.items():
        last = act
    if last is None:
        raise RuntimeError("Unable to measure flattened size by forward pass.")
    if last.dim() > 2:
        flattened = last.view(last.size(0), -1).size(1)
    else:
        flattened = last.size(1)
    return int(flattened)

def _get_container_and_subname(layer_name):
    """
    Extracts the container (e.g., 'features', 'stage3_block0.block') and the subname.
    """
    if layer_name == "":
        return "", ""
    layer_parts = layer_name.split('.')
    if len(layer_parts) == 1:
        return "", layer_parts[0]
    container = '.'.join(layer_parts[:-1])
    subname = layer_parts[-1]
    return container, subname

def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    print(f"Finding indices for layers '{start_layer_name}' to '{end_layer_name}'...")

    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name:
            start_idx = i
            print(f"  Found start layer '{start_layer_name}' at index {start_idx}")
        if name == end_layer_name:
            end_idx = i
            print(f"  Found end layer '{end_layer_name}' at index {end_idx}")

    if start_idx is None or end_idx is None:
        print(f"Warning: Could not find one or both layer names '{start_layer_name}', '{end_layer_name}' inside the container's children.")

    return start_idx, end_idx

def _update_container(model, container_path, new_container):
    """
    Replace the module at `container_path` in `model` with `new_container`.
    container_path is a dot-separated string specifying nested modules.
    """
    if container_path == "":
        raise ValueError("Cannot replace the root model container with a new container.")
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        if _is_int_str(part):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        idx = int(last)
        parent[idx] = new_container
    else:
        setattr(parent, last, new_container)
