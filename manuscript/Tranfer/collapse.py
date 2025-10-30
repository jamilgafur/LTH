# collapse.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from uuid import uuid4
from typing import Optional
from copy import deepcopy
from utils import count_trainable_params, layer_stats


# --------------------- Helper Functions --------------------- #

def _is_int_str(s):
    try:
        int(s)
        return True
    except:
        return False


def get_layer(model, layer_name):
    """Access layer via dot-separated path (supports Sequential indices)."""
    if layer_name == "":
        return model
    layer = model
    for part in layer_name.split('.'):
        layer = layer[int(part)] if _is_int_str(part) else getattr(layer, part)
    return layer


def _set_module_by_path(model, module_path, new_module):
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


def _get_container_and_subname(layer_name):
    """Return (container_path, subname) from layer_name."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]


def _find_layer_indices(named_layers, start_layer_name, end_layer_name):
    start_idx = end_idx = None
    for i, (name, _) in enumerate(named_layers):
        if name == start_layer_name: start_idx = i
        if name == end_layer_name: end_idx = i
    return start_idx, end_idx


def _replace_layers(named_layers, start_idx, end_idx, new_block):
    """Replace layers start_idx..end_idx with new_block in nn.Sequential."""
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


def _update_container(model, container_path, new_container):
    """Replace module at container_path with new_container."""
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
    last = parts[-1]
    if _is_int_str(last):
        parent[int(last)] = new_container
    else:
        setattr(parent, last, new_container)


# --------------------- ReLU Fix --------------------- #

def disable_inplace_relu(model):
    """Replace inplace ReLU with out-of-place version."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.ReLU) and getattr(module, "inplace", False):
            container, subname = _get_container_and_subname(name)
            parent = get_layer(model, container) if container != "" else model
            new_relu = nn.ReLU(inplace=False)
            if _is_int_str(subname):
                parent[int(subname)] = new_relu
            else:
                setattr(parent, subname, new_relu)


# --------------------- Forward Simulation --------------------- #

def _simulate_input_hook(model, target_layer_path, input_shape, device='cpu'):
    """Capture input activation to a specific layer."""
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    target_module = get_layer(model, target_layer_path)
    captured = {}

    def hook(module, inp, out):
        captured['in'] = inp[0].detach()

    handle = target_module.register_forward_hook(hook)
    with torch.no_grad():
        model(dummy_input)
    handle.remove()
    if 'in' not in captured:
        raise RuntimeError(f"Failed to capture activation at {target_layer_path}.")
    return dummy_input, captured['in']


# --------------------- Collapsed Block Builder --------------------- #

def _build_collapsed_block(layer_type, in_features, out_features, output_shape,
                           full_block=None, stride=(1, 1), pool_layer: Optional[nn.Module] = None,
                           linear_in_features: Optional[int] = None):
    """Construct collapsed Conv2d or Linear block."""
    seq = []

    if layer_type == nn.Conv2d:
        # Adjust out_channels to match Linear input if provided
        if linear_in_features is not None:
            H, W = output_shape[-2], output_shape[-1]
            out_features = max(1, linear_in_features // (H * W))

        # Compute effective kernel
        kernels = [m.kernel_size[0] if hasattr(m, "kernel_size") else 1
                   for m, _ in full_block] if full_block else [3]
        k_eff = sum(kernels) - (len(kernels) - 1)
        paddings = [m.padding[0] if hasattr(m, "padding") else 0
                    for m, _ in full_block] if full_block else [0]
        s_eff = 1

        seq.append(nn.Conv2d(in_features, out_features, kernel_size=k_eff, stride=s_eff,
                             padding=paddings[0] if paddings else 0, bias=False))

        # Preserve BN/ReLU if present
        if full_block:
            mods = [m for _, m in full_block]
            if isinstance(mods[-1], nn.ReLU):
                if len(mods) >= 2 and isinstance(mods[-2], nn.BatchNorm2d):
                    seq.insert(1, nn.BatchNorm2d(out_features))
                seq.append(nn.ReLU(inplace=False))
            elif isinstance(mods[-1], nn.BatchNorm2d):
                seq.append(nn.BatchNorm2d(out_features))

        if pool_layer is not None:
            seq.append(deepcopy(pool_layer))

    elif layer_type == nn.Linear:
        seq.append(nn.Linear(in_features, out_features))

    else:
        raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    return nn.Sequential(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(seq)]))


# --------------------- Core Collapse --------------------- #

def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    container_name, start_sub = _get_container_and_subname(start_layer_name)
    container = get_layer(model, container_name)
    named_layers = list(container.named_children())
    start_idx, end_idx = _find_layer_indices(named_layers, start_sub, _get_container_and_subname(end_layer_name)[1])
    full_block = named_layers[start_idx:end_idx + 1]
    selected_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    layer_type = type(selected_layers[0])

    # Capture input activation
    dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)

    if layer_type == nn.Linear:
        in_features = x.view(x.size(0), -1).size(1)
        for layer in selected_layers: x = layer(x)
        out_features = x.view(x.size(0), -1).size(1)
        collapsed_block = _build_collapsed_block(layer_type, in_features, out_features, x.shape, full_block)
    else:
        in_channels = x.shape[1]
        for layer in selected_layers: x = layer(x)
        out_shape = x.shape
        pool_layer = next((m for _, m in reversed(full_block)
                           if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)

        # Attempt to match first Linear layer if exists
        linear_in_features = None
        linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if linear_layers: linear_in_features = linear_layers[0].in_features

        collapsed_block = _build_collapsed_block(layer_type, in_channels, selected_layers[-1].out_channels,
                                                 out_shape, full_block, pool_layer=pool_layer,
                                                 linear_in_features=linear_in_features)

    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, container_name, updated_container)
    return model


# --------------------- Classifier Adjustment --------------------- #

def adjust_classifier_input_features(model, input_shape, num_classes=200, device='cpu', preserve_original_fc=True):
    """Adjust classifier input features or keep original if requested."""
    if preserve_original_fc:
        return

    model.eval().to(device)
    dummy = torch.randn(input_shape).to(device)
    with torch.no_grad(): model(dummy)

    linear_layers = [name for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not linear_layers: return
    first_linear_name = linear_layers[0]
    first_linear_layer = get_layer(model, first_linear_name)

    # Measure flattened input size
    dummy, x = _simulate_input_hook(model, first_linear_name, input_shape, device)
    flattened_size = x.view(x.size(0), -1).size(1)

    if flattened_size == first_linear_layer.in_features:
        return

    # Replace Linear safely
    parent_path, subname = _get_container_and_subname(first_linear_name)
    parent = get_layer(model, parent_path) if parent_path else model
    new_linear = nn.Linear(flattened_size, first_linear_layer.out_features)
    if _is_int_str(subname): parent[int(subname)] = new_linear
    else: setattr(parent, subname, new_linear)


# --------------------- Skip Connections Patch --------------------- #

def patch_skip_connections(model):
    """Patch residual blocks to skip shortcuts safely if dimensions mismatch."""
    collapsed_paths = getattr(model, "_collapsed_blocks", [])
    for name, module in model.named_modules():
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module):
            orig_forward = module.forward

            def make_patched_forward(orig_forward, block_name):
                def new_forward(self, x):
                    out = self.block(x)
                    try:
                        shortcut_out = self.shortcut(x)
                        if out.shape != shortcut_out.shape: return F.relu(out)
                        return F.relu(out + shortcut_out)
                    except: return F.relu(out)
                return new_forward

            module.forward = make_patched_forward(orig_forward, name).__get__(module)


# --------------------- Main Collapse Function --------------------- #

def collapse_only(model_weights_1, compression_set, model_class, model_kwargs=None,
                  input_shape=(1, 3, 32, 32), device='cpu'):
    model_kwargs = model_kwargs or {}
    model = model_class(**model_kwargs)
    checkpoint = torch.load(model_weights_1, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model._collapsed_blocks = []

    for start, end in compression_set:
        model = _collapse_block(model, start, end, input_shape, device)
        model._collapsed_blocks.append((start, end))

    # Adjust classifier only if required
    adjust_classifier_input_features(model, input_shape, num_classes=model_kwargs.get('num_classes', 200),
                                     device=device, preserve_original_fc=False)
    disable_inplace_relu(model)
    patch_skip_connections(model)
    return model
