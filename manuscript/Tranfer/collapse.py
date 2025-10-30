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

def _build_collapsed_block(
    layer_type, in_features, out_features, output_shape,
    full_block=None, stride=(1, 1), pool_layer: Optional[nn.Module] = None
):
    """
    Build a collapsed block safely with guaranteed fewer parameters.
    Ensures:
      - Conv2d → BatchNorm2d → ReLU sequence preserved
      - BatchNorm2d matches the new conv output channels
      - Optional pooling is appended
    """
    print(f"[DEBUG] Building collapsed block: {layer_type.__name__}, in={in_features}, out={out_features}, stride={stride}")

    seq = []

    if layer_type == nn.Conv2d:
        # Detect if last layers in full_block are BN/ReLU
        has_bn = has_relu = False
        if full_block:
            mods = [m for _, m in full_block] if isinstance(full_block[0], tuple) else list(full_block)
            if isinstance(mods[-1], nn.ReLU):
                has_relu = True
                if len(mods) >= 2 and isinstance(mods[-2], nn.BatchNorm2d):
                    has_bn = True
            elif isinstance(mods[-1], nn.BatchNorm2d):
                has_bn = True

        # Determine collapsed output channels (bottleneck)
        bottleneck_ratio = 0.5  # collapse to 50% channels
        collapsed_out = max(1, int(out_features * bottleneck_ratio))

        # Build collapsed Conv2d
        conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=collapsed_out,
            kernel_size=1,   # collapsed conv uses 1x1
            stride=stride,
            padding=0,
            bias=False
        )
        seq.append(conv)
        print(f"[DEBUG] Conv2d: {in_features} -> {collapsed_out}, kernel=1x1")

        # Add BatchNorm2d that matches collapsed_out
        if has_bn:
            seq.append(nn.BatchNorm2d(collapsed_out))
            print(f"[DEBUG] BatchNorm2d: num_features={collapsed_out}")

        # Add ReLU if original block had it
        if has_relu:
            seq.append(nn.ReLU(inplace=False))

        # Append pool layer if exists
        if pool_layer is not None:
            import copy
            seq.append(copy.deepcopy(pool_layer))
            print(f"[DEBUG] Appended pooling layer: {pool_layer.__class__.__name__}")

    elif layer_type == nn.Linear:
        # Collapse Linear layer to smaller output
        reduced_out = max(1, int(out_features * 0.75))
        seq.append(nn.Linear(in_features, reduced_out))
        print(f"[DEBUG] Linear: {in_features} -> {reduced_out}")

    else:
        raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    # Return as nn.Sequential with ordered layers
    collapsed_block = nn.Sequential(
        OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(seq)])
    )
    print(f"[DEBUG] Collapsed block layers: {[type(m).__name__ for m in collapsed_block]}")
    return collapsed_block


# --------------------- Core Collapse --------------------- #

def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    """Collapse a block of Conv/Linear layers into a smaller equivalent."""
    print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())

    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find layers: {start_layer_name} or {end_layer_name}")

    full_block = named_layers[start_idx:end_idx + 1]
    conv_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    layer_type = type(conv_layers[0])

    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device)
        print(f"[DEBUG] Simulated pre-collapse input: {tuple(x.shape)}")
    except Exception as e:
        print(f"[WARN] Simulation failed: {e}. Using dummy input.")
        if layer_type == nn.Conv2d:
            x = torch.randn(1, conv_layers[0].in_channels, input_shape[-2], input_shape[-1], device=device)
        else:
            x = torch.randn(1, conv_layers[0].in_features, device=device)

    pre_params = count_trainable_params(model)
    print(f"[DEBUG] Params before collapse: {pre_params:,}")

    if layer_type == nn.Conv2d:
        in_channels = x.shape[1]
        out_channels = conv_layers[-1].out_channels

        with torch.no_grad():
            y = x.clone()
            for layer in conv_layers:
                y = layer(y)
        out_shape = y.shape
        print(f"[DEBUG] Block output shape: {tuple(out_shape)}")

        linear_in_features = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                linear_in_features = mod.in_features
                break

        shortcut_out_channels = None
        for name, mod in model.named_modules():
            if hasattr(mod, "shortcut") and isinstance(mod.shortcut, nn.Module):
                first_conv = next((m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)), None)
                if first_conv:
                    shortcut_out_channels = first_conv.out_channels
                    break

        pool_layer = next((m for _, m in reversed(full_block) if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d))), None)

        collapsed_block = _build_collapsed_block(
            nn.Conv2d,
            in_channels,
            out_channels,
            out_shape,
            full_block=full_block,
            stride=(1, 1),
            pool_layer=pool_layer,
            shortcut_out_channels=shortcut_out_channels
        )

    else:
        in_features = x.view(x.size(0), -1).size(1)
        with torch.no_grad():
            y = x.clone()
            for layer in conv_layers:
                y = layer(y)
        out_features = y.view(y.size(0), -1).size(1)
        collapsed_block = _build_collapsed_block(
            nn.Linear, in_features, out_features, y.shape, full_block=full_block
        )

    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Params after collapse: {post_params:,}")
    print(f"[INFO] ΔParams = {pre_params - post_params:+,} (should be ≥ 0)")

    if post_params > pre_params:
        print("[WARN] ⚠ Collapsed block has MORE parameters! Check collapse ratio logic.")

    print(f"[SUCCESS] ✅ Collapsed {start_layer_name} → {end_layer_name}.")
    return model


# --------------------- Collapse Multiple --------------------- #

def collapse_only(model, layer_pairs, input_shape, device='cpu', dry_run=False):
    print("\n==============================")
    print("[INFO] Starting collapse_only()")
    print("==============================")

    model = deepcopy(model).to(device)
    model.eval()

    total_params_before = count_trainable_params(model)
    print(f"[INFO] Initial parameter count: {total_params_before:,}")

    for name, (start, end) in layer_pairs.items():
        print(f"\n[INFO] --- Collapsing {name} ---")
        print(f"         From: {start}")
        print(f"         To:   {end}")
        if not dry_run:
            model = _collapse_block(model, start, end, input_shape, device)
        else:
            print("[INFO] Dry run: skipping collapse.")
    
    total_params_after = count_trainable_params(model)
    print("\n==============================")
    print(f"[INFO] Parameters before: {total_params_before:,}")
    print(f"[INFO] Parameters after : {total_params_after:,}")
    print(f"[INFO] ΔParams = {total_params_before - total_params_after:+,} (should be ≥ 0)")
    print("==============================\n")

    return model
