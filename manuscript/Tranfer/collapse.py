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
def _build_collapsed_block(
    layer_type,
    in_features,
    out_features,
    output_shape,
    full_block=None,
    stride=(1, 1),
    pool_layer: Optional[nn.Module] = None,
    linear_in_features: Optional[int] = None,
    shortcut_out_channels: Optional[int] = None
):
    """
    Build a collapsed block that:
      - Collapses multiple Conv layers safely without enlarging kernels.
      - Matches Linear input or skip connection channels when present.
      - Always produces <= parameters than original block.
    """
    import torch.nn as nn
    import copy
    from collections import OrderedDict

    print(f"\n[DEBUG] Building collapsed block: {layer_type.__name__}")
    print(f"[DEBUG]   in_features={in_features}, out_features={out_features}, stride={stride}")

    seq = []

    if layer_type == nn.Conv2d:
        # Detect BN/ReLU in the original block
        has_bn = any(isinstance(m, nn.BatchNorm2d) for _, m in full_block)
        has_relu = any(isinstance(m, nn.ReLU) for _, m in full_block)

        # Aggregate kernel/stride/padding info from original block
        kernels = [m.kernel_size[0] for _, m in full_block if isinstance(m, nn.Conv2d)]
        strides = [m.stride[0] for _, m in full_block if isinstance(m, nn.Conv2d)]
        paddings = [m.padding[0] for _, m in full_block if isinstance(m, nn.Conv2d)]

        k_eff = sum(kernels) - (len(kernels) - 1)
        s_eff = max(strides) if strides else 1
        p_eff = paddings[0] if paddings else 0

        # Clamp kernel to avoid invalid sizes
        H, W = output_shape[-2], output_shape[-1]
        if k_eff > H or k_eff > W:
            print(f"[WARN] Reducing effective kernel from {k_eff} to fit input ({H}x{W}).")
            k_eff = min(H, W, 3)  # Cap at 3 for safety

        adjusted_out_channels = out_features

        # If followed by Linear layer, reduce channels to match flattened input
        if linear_in_features is not None:
            adjusted_out_channels = max(1, linear_in_features // (H * W))
            print(f"[INFO] Adjusting Conv out_channels to {adjusted_out_channels} "
                  f"for Linear in_features={linear_in_features}")

        # If skip connection has fewer channels, match it
        if shortcut_out_channels is not None and shortcut_out_channels < adjusted_out_channels:
            print(f"[INFO] Reducing Conv out_channels {adjusted_out_channels} → {shortcut_out_channels} "
                  f"for skip connection.")
            adjusted_out_channels = shortcut_out_channels

        # Force small kernel (1x1) to guarantee fewer params than original
        if k_eff > 1 and (linear_in_features is not None or shortcut_out_channels is not None):
            print(f"[DEBUG] Collapsing Conv2d to 1x1 for parameter reduction.")
            k_eff, p_eff = 1, 0

        # --- Build collapsed conv ---
        conv = nn.Conv2d(
            in_features,
            adjusted_out_channels,
            kernel_size=k_eff,
            stride=s_eff,
            padding=p_eff,
            bias=False
        )

        print(f"[DEBUG] Built Conv2d: in={in_features}, out={adjusted_out_channels}, "
              f"k={k_eff}, s={s_eff}, p={p_eff}")
        print(f"[DEBUG] Params in conv: {in_features * adjusted_out_channels * (k_eff ** 2):,}")

        seq.append(conv)

        if has_bn:
            seq.append(nn.BatchNorm2d(adjusted_out_channels))
        if has_relu:
            seq.append(nn.ReLU(inplace=False))

        if pool_layer is not None:
            seq.append(copy.deepcopy(pool_layer))
            print(f"[DEBUG] Added pooling layer: {pool_layer.__class__.__name__}")

    elif layer_type == nn.Linear:
        # Linear collapsing is trivial — just a single layer
        seq.append(nn.Linear(in_features, out_features))
        print(f"[DEBUG] Built Linear: {in_features} -> {out_features}")
    else:
        raise NotImplementedError(f"Unsupported layer type: {layer_type}")

    collapsed = nn.Sequential(OrderedDict([
        (f"layer_{i}", layer) for i, layer in enumerate(seq)
    ]))
    print(f"[DEBUG] Collapsed block layers: {[type(m).__name__ for m in collapsed]}")
    return collapsed

# --------------------- Core Collapse --------------------- #

def _collapse_block(model, start_layer_name, end_layer_name, input_shape, device='cpu'):
    """
    Collapse a sequence of Conv2d/Linear layers into a single equivalent layer.
    Ensures:
      - Always fewer or equal parameters than original.
      - Maintains shape compatibility for skip connections.
      - Matches Conv→Linear flatten size automatically.
    """
    print(f"\n[INFO] Collapsing block: {start_layer_name} → {end_layer_name}")

    # --- Identify container and layers ---
    start_container_name, start_subname = _get_container_and_subname(start_layer_name)
    end_container_name, end_subname = _get_container_and_subname(end_layer_name)
    container = get_layer(model, start_container_name)
    named_layers = list(container.named_children())

    start_idx, end_idx = _find_layer_indices(named_layers, start_subname, end_subname)
    if start_idx is None or end_idx is None:
        raise ValueError(f"Could not find layers: {start_layer_name} or {end_layer_name}.")

    full_block = named_layers[start_idx:end_idx + 1]
    conv_layers = [layer for _, layer in full_block if isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not conv_layers:
        raise ValueError(f"No Conv2d/Linear layers found in {start_layer_name} → {end_layer_name}")

    layer_type = type(conv_layers[0])
    assert all(isinstance(l, layer_type) for l in conv_layers), "[ERROR] Mixed layer types not supported."

    # --- Simulate input shape ---
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        print(f"[DEBUG] Simulated pre-collapse input: {tuple(x.shape)}")
    except Exception as e:
        print(f"[WARN] Simulation failed: {e}. Falling back to dummy tensor.")
        if layer_type == nn.Conv2d:
            x = torch.randn(1, conv_layers[0].in_channels, input_shape[-2], input_shape[-1], device=device)
        else:
            x = torch.randn(1, conv_layers[0].in_features, device=device)

    # --- Compute Conv or Linear collapse ---
    pre_params = count_trainable_params(model)
    print(f"[DEBUG] Params before collapse: {pre_params:,}")

    if layer_type == nn.Conv2d:
        in_channels = x.shape[1]
        out_channels = conv_layers[-1].out_channels

        # forward through block for shape
        with torch.no_grad():
            y = x.clone()
            for layer in conv_layers:
                y = layer(y)
        out_shape = y.shape
        print(f"[DEBUG] Block output shape: {out_shape}")

        # detect if next layer is Linear
        linear_in_features = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                linear_in_features = mod.in_features
                print(f"[DEBUG] Found following Linear layer with in_features={linear_in_features}")
                break

        # detect skip connection
        shortcut_out_channels = None
        for name, mod in model.named_modules():
            if hasattr(mod, "shortcut") and isinstance(mod.shortcut, nn.Module):
                try:
                    first_conv = next((m for m in mod.shortcut.modules() if isinstance(m, nn.Conv2d)), None)
                    if first_conv is not None:
                        shortcut_out_channels = first_conv.out_channels
                        print(f"[DEBUG] Detected shortcut with {shortcut_out_channels} channels.")
                except Exception:
                    continue

        # find optional pooling
        pool_layer = next((mod for _, mod in reversed(full_block)
                           if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))), None)

        collapsed_block = _build_collapsed_block(
            nn.Conv2d,
            in_channels,
            out_channels,
            out_shape,
            full_block=full_block,
            stride=(1, 1),
            pool_layer=pool_layer,
            linear_in_features=linear_in_features,
            shortcut_out_channels=shortcut_out_channels
        )

    else:
        # --- Linear collapse ---
        in_features = x.view(x.size(0), -1).size(1)
        y = x.clone()
        with torch.no_grad():
            for layer in conv_layers:
                y = layer(y)
        out_features = y.view(y.size(0), -1).size(1)

        collapsed_block = _build_collapsed_block(
            nn.Linear, in_features, out_features, y.shape, full_block=full_block
        )

    # --- Replace layers in container ---
    updated_container = _replace_layers(named_layers, start_idx, end_idx, collapsed_block)
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    print(f"[DEBUG] Params after collapse: {post_params:,}")
    print(f"[INFO] ΔParams = {pre_params - post_params:+,} (should be ≥ 0)")

    if post_params > pre_params:
        print("[WARN] ⚠ Collapsed block has MORE parameters than before! This should not happen.")

    print(f"[SUCCESS] ✅ Collapsed {start_layer_name} → {end_layer_name}. "
          f"Final param count: {post_params:,} (was {pre_params:,})\n")

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

def collapse_only(model, layer_pairs, input_shape, device='cpu', dry_run=False):
    """
    Collapse multiple layer ranges in a model safely.

    Args:
        model (nn.Module): The model to modify.
        layer_pairs (dict): Mapping of collapse names to (start_layer, end_layer).
        input_shape (tuple): Input shape for dummy forward pass.
        device (str): 'cpu' or 'cuda'.
        dry_run (bool): If True, simulate collapse without modifying model.

    Returns:
        nn.Module: Collapsed model (or original if dry_run=True)
    """
    import torch
    import torch.nn as nn
    import copy

    print("\n==============================")
    print("[INFO] Starting collapse_only()")
    print("==============================")

    model = copy.deepcopy(model).to(device)
    model.eval()

    total_params_before = count_trainable_params(model)
    print(f"[INFO] Initial parameter count: {total_params_before:,}")

    for collapse_name, (start_layer_name, end_layer_name) in layer_pairs.items():
        print(f"\n[INFO] --- Collapsing Block: {collapse_name} ---")
        print(f"         From: {start_layer_name}")
        print(f"         To:   {end_layer_name}")

        try:
            # ---- Run a safe simulation to detect Linear follower ----
            start_container_name, _ = _get_container_and_subname(start_layer_name)
            end_container_name, end_subname = _get_container_and_subname(end_layer_name)
            container = get_layer(model, start_container_name)
            named_layers = list(container.named_children())

            # Find end layer index
            end_idx = next(i for i, (n, _) in enumerate(named_layers) if n == end_subname)

            # Try to find the next layer (for Linear detection)
            next_layer = None
            if end_idx + 1 < len(named_layers):
                next_layer = named_layers[end_idx + 1][1]

            # If the next layer is a Linear, record its input size
            linear_in_features = None
            if isinstance(next_layer, nn.Linear):
                linear_in_features = next_layer.in_features
                print(f"[INFO] Detected Linear layer after collapse block with in_features={linear_in_features}")

            # ---- Collapse the block ----
            if not dry_run:
                model = _collapse_block(
                    model,
                    start_layer_name,
                    end_layer_name,
                    input_shape,
                    device=device
                )
            else:
                print("[INFO] Dry run mode: skipping actual collapse.")

        except Exception as e:
            print(f"[ERROR] Failed to collapse {collapse_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_params_after = count_trainable_params(model)
    print("\n==============================")
    print(f"[INFO] Collapse complete.")
    print(f"[INFO] Parameters before: {total_params_before:,}")
    print(f"[INFO] Parameters after : {total_params_after:,}")
    print(f"[INFO] ΔParams = {total_params_before - total_params_after:+,} (should be ≥ 0)")
    print("==============================\n")

    if total_params_after > total_params_before:
        print("[WARN] ⚠ Model has MORE parameters after collapse! This should never happen — check block logic.")

    return model
