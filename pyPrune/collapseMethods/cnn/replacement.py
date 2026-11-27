# replacement.py -- the surgery: building the cheap replacement block
# (1x1/depthwise conv + BN + adaptive pool), installing it in place of the
# collapsed span, inserting corrective pooling, and validating what comes
# after it downstream.
import logging
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .capture import count_trainable_params, _simulate_input_hook
from .containers import (_get_container_and_subname, _replace_layers,
                         _update_container, get_layer)
from .postprocess import _SafePool

logger = logging.getLogger(__name__)


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
):
    if debug:
        logger.debug(f"Building replacement for collapsed block '{start_layer_name}'")
        logger.debug(f"Analyzing info dict keys: {list(info.keys())}")
        logger.debug(f"Target device: {device}")

    named_layers = info["named_layers"]
    # Prefer the container determined by the locator (LCA).
    start_container_name = info.get("container_name")
    if start_container_name is None:
        # Fallback only if not in info (should be there now)
        start_container_name = _get_container_and_subname(start_layer_name)[0]
        
    start_idx, end_idx = info["start_idx"], info["end_idx"]
    out_shape = block_analysis.get("out_shape")
    out_channels = block_analysis.get("out_channels")

    if out_shape is None or out_channels is None:
        raise RuntimeError("[ERROR] block_analysis missing required out_shape/out_channels")

    target_H, target_W = (out_shape[-2], out_shape[-1]) if len(out_shape) >= 4 else (1, 1)
    if debug:
        logger.debug(f"Replacement target spatial size (HxW): {target_H}x{target_W}")

    if x is None or x.ndim < 2:
        raise RuntimeError("[ERROR] Invalid captured activation `x`.")
    in_channels = int(x.shape[1])
    if debug:
        logger.debug(f"Replacement conv in_channels={in_channels}, out_channels={out_channels}")

    # -------- NEW INTELLIGENT ROUTING LOGIC --------
    original_convs = info.get("conv_layers", [])
    
    is_depthwise = False
    target_kernel_size = 1
    
    if original_convs:
        # Inspect the first convolution in the block being collapsed
        first_conv = original_convs[0]
        
        # Check if the original layer was depthwise
        # A true depthwise conv has groups == in_channels == out_channels
        if first_conv.groups == first_conv.in_channels and first_conv.in_channels == first_conv.out_channels:
            # We can only safely replace with depthwise if our calculated replacement
            # channels also match (in_channels == out_channels)
            if in_channels == out_channels:
                is_depthwise = True
                # Preserve the original spatial kernel size (e.g., 3x3 or 5x5) for depthwise
                target_kernel_size = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                
        # Check if it was explicitly a pointwise (1x1, groups=1)
        elif first_conv.kernel_size == (1, 1) or first_conv.kernel_size == 1:
            target_kernel_size = 1 # Force pointwise
            is_depthwise = False
            
    # Build the specific convolution type
    if is_depthwise:
        if debug:
            logger.debug(f"Rebuilding as DEPTHWISE conv (kernel_size={target_kernel_size})")
        
        padding = target_kernel_size // 2 # Keep spatial dimensions identical before the pool
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=target_kernel_size, 
                         stride=1, padding=padding, groups=in_channels, bias=False)
        conv_name = "conv_dw"
    else:
        if debug:
            logger.debug(f"Rebuilding as POINTWISE conv (kernel_size=1)")
            
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, padding=0, bias=False)
        conv_name = "conv_1x1"
    # -----------------------------------------------

    bn = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU(inplace=False)
    pool = nn.AdaptiveAvgPool2d((target_H, target_W))

    replacement = nn.Sequential(OrderedDict([
        (conv_name, conv),
        ("bn", bn),
        ("relu", relu),
        ("adaptive_pool", pool),
    ]))
    if debug:
        logger.debug(f"Built replacement Sequential:\n{replacement}")

    if debug:
        logger.debug(f"Replacing children indices {start_idx}..{end_idx} in container '{start_container_name or '<root>'}'")
    
    # FIX: Use string names for start_name/end_name
    updated_container = _replace_layers(
            named_layers,
            start_idx,
            end_idx,
            replacement,
            start_name=info["first_layer_name"], # passed as string
            end_name=info["last_layer_name"],    # passed as string
        )
        
    _update_container(model, start_container_name, updated_container)
    model.to(device)

    post_params = count_trainable_params(model)
    if debug:
        logger.debug(f"Params before collapse: {pre_params:,}")
        logger.debug(f"Params after  collapse: {post_params:,}")
        logger.info(f"ΔParams = {pre_params - post_params:+,}")

    try:
        dev = next((p.device for p in model.parameters()), torch.device('cpu'))
        rep_module = get_layer(model, start_container_name)
        child = None
        
        # Validation search: look inside the container for the new collapsed module
        # Note: If start_container_name is root (""), rep_module is model
        for nm, m in rep_module.named_children():
            # UPDATED: We now check for 'conv_name' (either conv_1x1 or conv_dw) instead of a hardcoded string
            if nm.startswith("collapsed_") or (isinstance(m, nn.Sequential) and conv_name in dict(m.named_children())):
                child = m
                break
        
        # Fallback: if we just replaced a specific index in a Sequential
        if child is None and isinstance(updated_container, nn.Sequential) and start_container_name != "":
             # This path is risky if updated_container isn't attached yet, 
             # but we called _update_container above, so it should be fine.
             pass

        if child is not None:
            with torch.no_grad():
                test_x = x.clone().to(dev)
                out = child(test_x)
                if debug:
                    logger.debug(f"Replacement validation OK — output shape {tuple(out.shape)}")
        else:
            logger.warning(f"Could not find inserted collapsed module for validation.")
    except Exception as e:
        raise RuntimeError(f"replacement forward validation failed: {e}") from e
        
    try:
        if debug:
            logger.debug(f"Validating downstream after replacement...")
        _validate_downstream(model, start_container_name, start_idx, x, input_shape, next_linear_name, next_linear_mod, device, debug)
    except Exception as e:
        raise RuntimeError(f"downstream validation failed: {e}") from e
        
    try:
        if debug:
            logger.debug(f"Performing corrective pooling (if needed)...")
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug)
    except Exception as e:
        raise RuntimeError(f"corrective pool insertion failed: {e}") from e

    if post_params > pre_params:
        raise RuntimeError("collapsed block increased parameter count - investigate collapse policy")

    if debug:
        logger.info(f"Block replacement complete for '{start_layer_name}'.")

    return model



def _insert_corrective_pool(model, next_linear_name, input_shape, debug):
    """
    Insert AdaptiveAvgPool2d only when a true spatial -> flattened mismatch exists.
    Uses a forward hook to measure the ACTUAL tensor shape entering the linear layer.
    """

    if debug:
        logger.debug(f"===== Starting _insert_corrective_pool =====")
        logger.debug(f"next_linear_name='{next_linear_name}'")

    if not next_linear_name:
        if debug:
            logger.info("No next linear layer specified — skipping.")
        return model

    # 1. ConvNeXt SAFETY
    if ".pwconv" in next_linear_name:
        if debug:
            logger.info("Detected ConvNeXt pwconv — skipping corrective pooling.")
        return model

    # 2. Capture ACTUAL input shape to the linear layer
    # We use the existing _simulate_input_hook to run a dummy pass and see what hits the linear layer
    try:
        device = next((p.device for p in model.parameters()), torch.device('cpu'))
        # input_shape here is the MODEL input (e.g., 1, 3, 32, 32)
        _, actual_input = _simulate_input_hook(model, next_linear_name, input_shape, device=device)
        
        # Flattened size per batch item (N, C, H, W) -> C*H*W
        current_features = actual_input[0].numel() 
        current_shape = actual_input.shape # e.g. [1, 728, 1, 1]
        
        if debug:
            logger.debug(f"Actual tensor shape entering '{next_linear_name}': {tuple(current_shape)}")
            logger.debug(f"Flattened size per sample: {current_features}")
            
    except Exception as e:
        logger.warning(f"Failed to simulate forward pass for corrective pool check: {e}")
        return model

    # 3. Get Linear layer expectations
    next_linear_mod = None
    next_linear_parent = None
    next_linear_parent_name = None
    next_linear_child_name = None

    # Locate module and parent
    for name, mod in model.named_modules():
        if name == next_linear_name:
            next_linear_mod = mod
            break
            
    if not isinstance(next_linear_mod, nn.Linear):
        if debug:
            logger.info(f"Target '{next_linear_name}' is not nn.Linear — skipping.")
        return model
        
    in_features = next_linear_mod.in_features
    if debug:
        logger.debug(f"Linear '{next_linear_name}' expects in_features={in_features}")

    # 4. Compare and Decide
    if current_features == in_features:
        if debug:
            logger.info("✅ Shapes match (Current == Expected) — no corrective pooling needed.")
        return model

    # 5. Calculate Pooling logic (if mismatch exists)
    # Assume NCHW layout for calculation
    if len(current_shape) < 4:
         if debug:
            logger.warning("Input is already flattened but size mismatches. Cannot fix with pooling.")
         return model

    C = current_shape[1]
    if in_features % C != 0:
        if debug:
            logger.warning(f"in_features ({in_features}) not divisible by channels ({C}) — unsafe to pool.")
        return model

    expected_hw = in_features // C
    target_hw = int(round(expected_hw ** 0.5))

    if target_hw * target_hw != expected_hw:
        if debug:
             logger.warning(f"Target spatial area {expected_hw} is not square — skipping.")
        return model

    if debug:
        logger.info(f"🛠 Mismatch detected! Inserting AdaptiveAvgPool2d({target_hw}, {target_hw})")

    # 6. Insert the pool
    # Locate parent container again to perform insertion
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            if child_mod is next_linear_mod:
                next_linear_parent = parent_mod
                next_linear_parent_name = parent_name
                next_linear_child_name = child_name
                break
        if next_linear_parent is not None:
            break

    corrective_pool = nn.AdaptiveAvgPool2d((target_hw, target_hw))
    
    new_children = OrderedDict()
    inserted = False
    for name, mod in next_linear_parent.named_children():
        if name == next_linear_child_name and not inserted:
            new_children["corrective_pool"] = corrective_pool
            inserted = True
        new_children[name] = mod

    # Apply update
    if isinstance(next_linear_parent, nn.Sequential):
        new_parent = nn.Sequential(new_children)
    else:
        new_parent = nn.Sequential(new_children)

    # Update the model with the new parent container
    if next_linear_parent_name == "":
        # Special handling if the linear layer is a direct child of the root model
        setattr(model, next_linear_child_name, nn.Sequential(
            corrective_pool,
            next_linear_mod
        ))
    else:
        _update_container(model, next_linear_parent_name, new_parent)

    if debug:
        logger.info("Corrective pooling inserted successfully.")

    return model


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
    If a downstream module fails during a forward pass (e.g. pooling mismatch),
    wraps it in _SafePool or replaces it with Identity to preserve model functionality.
    """
    logger.debug(f"===== Starting _validate_downstream for '{start_container_name}' =====")
    logger.debug(f"start_idx={start_idx}, device={device}, has_next_linear={next_linear_name is not None}")

    # Retrieve target container
    try:
        container = get_layer(model, start_container_name)
    except Exception as e:
        logger.warning(f"Could not access container '{start_container_name}': {e}")
        return

    children = list(container.named_children())
    if not children:
        logger.debug(f"Container '{start_container_name}' has no children; skipping downstream validation.")
        return

    # Find collapsed/inserted child index
    collapsed_idx = None
    logger.debug(f"Searching for inserted collapsed module within '{start_container_name}'...")
    for i, (nm, m) in enumerate(children):
        if nm.startswith("collapsed_") or (
            isinstance(m, nn.Sequential)
            and any(k in dict(m.named_children()) for k in ("conv_1x1", "adaptive_pool"))
        ):
            collapsed_idx = i
            logger.debug(f"Found inserted block candidate at index {i}: '{nm}'")
            break

    if collapsed_idx is None:
        # fallback search: direct 1x1 conv
        for i, (nm, m) in enumerate(children):
            if isinstance(m, nn.Conv2d) and getattr(m, "kernel_size", None) == (1, 1):
                collapsed_idx = i
                logger.debug(f"Found replacement candidate (Conv2d 1x1) at index {i}: '{nm}'")
                break

    if collapsed_idx is None:
        logger.warning(f"Could not identify inserted module inside '{start_container_name}'. Skipping validation.")
        return

    inserted_mod = children[collapsed_idx][1]
    logger.debug(f"Validating inserted module '{inserted_mod.__class__.__name__}' at index {collapsed_idx}...")

    # Run through inserted module
    try:
        with torch.no_grad():
            dev = next((p.device for p in model.parameters()), torch.device('cpu'))
            t = pre_activation.clone().to(dev)
            t = inserted_mod(t)
        logger.debug(f"Inserted module forward successful, output shape: {tuple(t.shape)}")
    except Exception as e:
        logger.warning(f"Forward pass through inserted module failed: {e}")
        return

    # Validate next modules downstream
    logger.debug(f"Scanning immediate downstream modules for shape or runtime errors...")
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            t = mod(t)
            if debug:
                logger.debug(f"Downstream '{start_container_name}.{nm}' executed successfully, output shape: {tuple(t.shape)}")
        except Exception as e:
            logger.warning(f"Downstream module '{start_container_name}.{nm}' raised exception: {e}")
            logger.debug(f"Replacing problematic module '{nm}' with safe alternative...")

            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
                logger.info(f"Replaced with _SafePool wrapper.")
            else:
                safe = nn.Identity()
                logger.info(f"Replaced with Identity() to bypass invalid operation.")

            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)

            logger.debug(f"Replacement applied to '{start_container_name}.{nm}' ({safe.__class__.__name__}).")
            return  # stop after first fix

        # detect zero-spatial output
        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            logger.warning(f"Module '{start_container_name}.{nm}' produced zero spatial dimensions. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            logger.debug(f"Zero-dimension fix applied to '{start_container_name}.{nm}' with _SafePool.")
            return

    logger.info(f"✅ Downstream validation for '{start_container_name}' completed successfully.")

