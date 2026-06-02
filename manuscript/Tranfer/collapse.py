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
import math

def _locate_and_prepare_block(model, start_layer_name, end_layer_name):
    print(f"\n[DEBUG ENTRY] _locate_and_prepare_block called.")
    print(f"[DEBUG] Locating and preparing block: start='{start_layer_name}', end:'{end_layer_name}'")

    # --- LCA resolution ---
    start_parts = start_layer_name.split('.') if start_layer_name else []
    end_parts = end_layer_name.split('.') if end_layer_name else []
    print(f"[DEBUG] Parsed start_parts: {start_parts}")
    print(f"[DEBUG] Parsed end_parts: {end_parts}")
    
    common_parts = []
    
    # NEW FIX: If start and end are identical, the LCA is their parent!
    if start_layer_name == end_layer_name and len(start_parts) > 0:
        lca_path = '.'.join(start_parts[:-1])
        print(f"[DEBUG] Start and end layers are identical. Set LCA to parent: '{lca_path}'")
    else:
        for a, b in zip(start_parts, end_parts):
            if a == b:
                common_parts.append(a)
            else:
                break
        lca_path = '.'.join(common_parts)
        print(f"[DEBUG] Calculated LCA path from common parts: '{lca_path}'")

    # FIX 1: Allow root to be the container
    if lca_path == "":
        print(f"[DEBUG] LCA path is empty (root). Resolving containers for start and end layers...")
        start_container_name, _ = _get_container_and_subname(start_layer_name)
        end_container_name, _ = _get_container_and_subname(end_layer_name)
        print(f"[DEBUG] start_container_name='{start_container_name}', end_container_name='{end_container_name}'")
        
        # If they share a parent (even root ""), use it.
        if start_container_name == end_container_name:
            print(f"[DEBUG] Shared parent found: '{start_container_name}'")
            lca_path = start_container_name
            container = get_layer(model, lca_path)
        else:
            # Fallback: If layers are in different top-level branches, use root as LCA
            print(f"[DEBUG] Layers in different branches or direct children; using root as LCA.")
            lca_path = ""
            container = model
    else:
        container = get_layer(model, lca_path)

    print(f"[DEBUG] Containers resolved → chosen LCA container: '{lca_path}'")

    named_layers = list(container.named_children())
    print(f"[DEBUG] Found {len(named_layers)} children in container '{lca_path or '<root>'}'")

    # --- locate child indices ---
    start_idx = end_idx = None
    print(f"[DEBUG] Searching for start and end indices in named_layers...")
    for i, (child_name, _) in enumerate(named_layers):
        full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
        
        # Check start
        if start_idx is None and (
            start_layer_name == full_child_prefix
            or start_layer_name.startswith(full_child_prefix + ".")
        ):
            start_idx = i
            print(f"[DEBUG] Found start_idx at {i} (child_name: '{child_name}', prefix: '{full_child_prefix}')")
        
        # Check end
        if end_idx is None and (
            end_layer_name == full_child_prefix
            or end_layer_name.startswith(full_child_prefix + ".")
        ):
            end_idx = i
            print(f"[DEBUG] Found end_idx at {i} (child_name: '{child_name}', prefix: '{full_child_prefix}')")
        
        if start_idx is not None and end_idx is not None:
            print(f"[DEBUG] Both indices found. Breaking search loop.")
            break

    if start_idx is None or end_idx is None:
        raise ValueError(f"[ERROR] Could not map start/end layers into LCA container '{lca_path}'. (start_idx={start_idx}, end_idx={end_idx})")

    if start_idx > end_idx:
        print(f"[DEBUG] start_idx ({start_idx}) > end_idx ({end_idx}). Swapping them.")
        start_idx, end_idx = end_idx, start_idx

    full_block = named_layers[start_idx:end_idx + 1]
    print(f"[DEBUG] Block slice length: {len(full_block)}")
    
    if not isinstance(container, nn.Sequential) and start_idx != end_idx:
        raise ValueError(
            f"[ERROR] Container '{lca_path}' is {type(container).__name__}, not nn.Sequential. "
            f"Slicing multiple parallel branches (like Inception branches) as a sequence is invalid."
        )
        
    # --- collect collapsible layers (mixed allowed) ---
    conv_layers = []
    print(f"[DEBUG] Collecting collapsible layers (Conv2d, Linear)...")
    for _, mod in full_block:
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            conv_layers.append(mod)
        for sub_name, sub_mod in mod.named_modules():
            if sub_name and isinstance(sub_mod, (nn.Conv2d, nn.Linear)):
                conv_layers.append(sub_mod)

    if not conv_layers:
        raise ValueError("[ERROR] No Conv2d or Linear layers found in block.")

    has_conv = any(isinstance(l, nn.Conv2d) for l in conv_layers)
    collapse_mode = "conv" if has_conv else "linear"

    print(
        f"[DEBUG] Block composition → "
        f"{sum(isinstance(l, nn.Conv2d) for l in conv_layers)} Conv2d, "
        f"{sum(isinstance(l, nn.Linear) for l in conv_layers)} Linear "
        f"(collapse_mode={collapse_mode})"
    )
    layer_type = nn.Conv2d if collapse_mode == "conv" else nn.Linear

    # FIX 2: Return names (strings) for first/last layer to avoid naming errors
    result_dict = {
        "container": container,
        "container_name": lca_path,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "layer_type": layer_type,
        "conv_layers": conv_layers,
        "collapse_mode": collapse_mode,
        "first_layer_name": full_block[0][0],  # Changed from object to name string
        "last_layer_name": full_block[-1][0],   # Changed from object to name string
    }
    print(f"[DEBUG EXIT] _locate_and_prepare_block returning successfully with first_layer='{result_dict['first_layer_name']}', last_layer='{result_dict['last_layer_name']}'")
    return result_dict


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
        print(f"\n[DEBUG ENTRY] _build_and_replace_block for collapsed block '{start_layer_name}'") 
        print(f"[DEBUG] Analyzing info dict keys: {list(info.keys())}") 
        print(f"[DEBUG] Target device: {device}") 

    named_layers = info["named_layers"] 
    
    # Prefer the container determined by the locator (LCA).
    start_container_name = info.get("container_name") 
    if start_container_name is None: 
        if debug: print(f"[DEBUG] container_name not found in info, determining from start_layer_name...")
        start_container_name = _get_container_and_subname(start_layer_name)[0] 
        
    start_idx, end_idx = info["start_idx"], info["end_idx"] 
    out_shape = block_analysis.get("out_shape") 
    out_channels = block_analysis.get("out_channels") 
    
    if debug:
        print(f"[DEBUG] Extracted start_idx={start_idx}, end_idx={end_idx}, start_container_name='{start_container_name}'")

    if out_shape is None or out_channels is None: 
        raise RuntimeError("[ERROR] block_analysis missing required out_shape/out_channels") 

    target_H, target_W = (out_shape[-2], out_shape[-1]) if len(out_shape) >= 4 else (1, 1) 
    if debug:
        print(f"[DEBUG] Replacement target spatial size (HxW): {target_H}x{target_W}") 

    if x is None or x.ndim < 2: 
        raise RuntimeError("[ERROR] Invalid captured activation `x`.") 
    
    in_channels = int(x.shape[1]) 
    if debug:
        print(f"[DEBUG] Replacement conv in_channels={in_channels}, out_channels={out_channels}") 

    # -------- INTELLIGENT ROUTING LOGIC --------
    original_convs = info.get("conv_layers", [])
    if debug: print(f"[DEBUG] Number of original_convs passed in: {len(original_convs)}")
    
    is_depthwise = False
    target_kernel_size = 1
    
    if original_convs:
        first_conv = original_convs[0]
        if debug: print(f"[DEBUG] Analyzing first_conv: groups={first_conv.groups}, in_channels={first_conv.in_channels}, out_channels={first_conv.out_channels}, kernel_size={first_conv.kernel_size}")
        
        if first_conv.groups == first_conv.in_channels and first_conv.in_channels == first_conv.out_channels:
            if in_channels == out_channels:
                is_depthwise = True
                target_kernel_size = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                if debug: print(f"[DEBUG] Routing condition met for DEPTHWISE based on first_conv properties.")
                
        elif first_conv.kernel_size == (1, 1) or first_conv.kernel_size == 1:
            target_kernel_size = 1 
            is_depthwise = False
            if debug: print(f"[DEBUG] Routing condition met for POINTWISE based on first_conv properties.")
            
    if is_depthwise:
        padding = target_kernel_size // 2 
        if debug: print(f"[DEBUG] Rebuilding as DEPTHWISE conv (kernel_size={target_kernel_size}, padding={padding})")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=target_kernel_size, stride=1, padding=padding, groups=in_channels, bias=False)
        conv_name = "conv_dw"
    else:
        if debug: print(f"[DEBUG] Rebuilding as POINTWISE conv (kernel_size=1, padding=0)")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
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
    if debug: print(f"[DEBUG] Built replacement Sequential:\n{replacement}") 
    if debug: print(f"[DEBUG] Replacing children indices {start_idx}..{end_idx} in container '{start_container_name or '<root>'}'") 
    
    updated_container = _replace_layers( 
            named_layers, start_idx, end_idx, replacement, 
            start_name=info["first_layer_name"], end_name=info["last_layer_name"], 
        )
        
    if debug: print(f"[DEBUG] Updating container and moving model to {device}")
    _update_container(model, start_container_name, updated_container) 
    model.to(device) 

    post_params = count_trainable_params(model) 
    if debug:
        print(f"[DEBUG] Params before collapse: {pre_params:,}") 
        print(f"[DEBUG] Params after  collapse: {post_params:,}") 
        print(f"[INFO] ΔParams = {pre_params - post_params:+,}") 

    try:
        if debug: print(f"[DEBUG] Starting replacement forward validation...")
        dev = next((p.device for p in model.parameters()), torch.device('cpu')) 
        rep_module = get_layer(model, start_container_name) 
        child = None 
        
        for nm, m in rep_module.named_children(): 
            if nm.startswith("collapsed_") or (isinstance(m, nn.Sequential) and conv_name in dict(m.named_children())):
                child = m 
                if debug: print(f"[DEBUG] Found inserted module '{nm}' for validation.")
                break 
        
        if child is None and isinstance(updated_container, nn.Sequential) and start_container_name != "": 
             if debug: print(f"[DEBUG] Child not found directly, falling back to sequential container logic.")
             pass  

        if child is not None: 
            with torch.no_grad(): 
                test_x = x.clone().to(dev) 
                out = child(test_x) 
                if debug: print(f"[DEBUG] Replacement validation OK — output shape {tuple(out.shape)}") 
        else:
            if debug: print(f"[WARN] Could not find inserted collapsed module for validation. start_container_name='{start_container_name}'") 
    except Exception as e:
        print(f"[WARN] Replacement forward validation failed: {e}") 
            

    # =========================================================
    # FIX: Run Corrective Pooling BEFORE downstream validation
    # =========================================================
    try:
        if debug:
            print(f"[STEP] Performing corrective pooling (if needed)...") 
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug) 
    except Exception as e:
        print(f"[WARN] Corrective pool insertion failed: {e}") 
            

    try:
        if debug:
            print(f"[STEP] Validating downstream after replacement...") 
        _validate_downstream(model, start_container_name, start_idx, x, input_shape, next_linear_name, next_linear_mod, device, debug) 
    except Exception as e:
        print(f"[WARN] Downstream validation failed: {e}") 
            
    # =========================================================

    if post_params > pre_params: 
        print(f"[WARN] ⚠ Collapsed block increased parameter count — investigate collapse policy.") 
            
    if debug: print(f"[DEBUG EXIT] Block replacement complete for '{start_layer_name}'.") 

    return model
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

def _get_container_and_subname(layer_name: str) -> Tuple[str, str]:
    """Return (container_path, subname) from layer_name."""
    if layer_name == "":
        return "", ""
    parts = layer_name.split('.')
    return '.'.join(parts[:-1]), parts[-1]

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

def _simulate_input_hook(model: nn.Module, target_layer_path: str, input_shape: Tuple[int, ...], device='cpu') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single forward with a dummy input and capture the activation that is
    the *input to* the target layer (registered as forward hook on the target layer).
    Returns (dummy_input, captured_activation).
    """
    print(f"\n[DEBUG ENTRY] _simulate_input_hook for target '{target_layer_path}'")
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)
    print(f"[DEBUG] Created dummy_input of shape {input_shape} on device '{device}'")

    target_module = get_layer(model, target_layer_path)
    print(f"[DEBUG] Successfully located target_module for hook registration.")

    captured = {}
    def hook(module, inp, out):
        # store a detached copy of the input to the target module
        captured['in'] = inp[0].detach()
        print(f"[DEBUG] Forward hook triggered! Captured input of shape {captured['in'].shape}")

    handle = target_module.register_forward_hook(hook)
    print(f"[DEBUG] Hook registered on '{target_layer_path}'. Running dummy forward pass...")
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        handle.remove()
        print(f"[DEBUG] Forward pass complete. Hook removed.")
        
    if 'in' not in captured:
        raise RuntimeError(f"[ERROR] Failed to capture activation at {target_layer_path}.")
        
    print(f"[DEBUG EXIT] _simulate_input_hook returning captured tensor.")
    return dummy_input, captured['in']


class SmartIdentity(nn.Module):
    """A robust Identity replacement that safely absorbs nested attribute calls."""
    def forward(self, x, *args, **kwargs):
        return x
        
    def __getattr__(self, name):
        # Prevent intercepting PyTorch internal methods
        if name.startswith('_'):
            return super().__getattr__(name)
            
        # FIX: Stop falsely claiming architectural attributes used by patching logic
        if name in ['shortcut', 'block']:
            print(f"[DEBUG] SmartIdentity blocking attribute request for '{name}'")
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        print(f"[DEBUG] SmartIdentity silently absorbing attribute call: '{name}'")
        return self # Return self to absorb chained calls like .inception_4a(x)


def _replace_layers(named_layers, start_idx, end_idx, replacement, start_name=None, end_name=None):
    """Replace a slice of layers, using SmartIdentity for subsumed layers."""
    print(f"\n[DEBUG ENTRY] _replace_layers: Replacing layer indices {start_idx} through {end_idx}")
    new_layers = OrderedDict()

    for i, (name, mod) in enumerate(named_layers):
        if i < start_idx or i > end_idx:
            print(f"[DEBUG] Keeping layer {i}: '{name}' (Outside replacement range)")
            new_layers[name] = mod
        elif i == start_idx:
            print(f"[DEBUG] Replacing layer {i}: '{name}' with custom replacement module")
            new_layers[name] = replacement
        else:
            # FIX: Use SmartIdentity so hardcoded forward passes don't crash
            print(f"[DEBUG] Subsuming layer {i}: '{name}' with SmartIdentity")
            new_layers[name] = SmartIdentity()

    print(f"[DEBUG EXIT] _replace_layers complete. Generated new nn.Sequential container.")
    return nn.Sequential(new_layers)


def _insert_corrective_pool(model, next_linear_name, input_shape, debug):
    """
    Insert AdaptiveAvgPool2d only when a true spatial -> flattened mismatch exists.
    Uses a forward hook to measure the ACTUAL tensor shape entering the linear layer.
    """
    if debug:
        print(f"\n[STEP] ===== Starting _insert_corrective_pool =====")
        print(f"[DEBUG] next_linear_name='{next_linear_name}'")

    if not next_linear_name:
        if debug:
            print("[INFO] No next linear layer specified — skipping.")
        return model

    # 1. ConvNeXt SAFETY
    if ".pwconv" in next_linear_name:
        if debug:
            print("[INFO] Detected ConvNeXt pwconv — skipping corrective pooling.")
        return model

    # 2. Capture ACTUAL input shape to the linear layer
    # We use the existing _simulate_input_hook to run a dummy pass and see what hits the linear layer
    try:
        if debug: print(f"[DEBUG] Simulating input hook to measure actual input shape hitting the linear layer...")
        device = next((p.device for p in model.parameters()), torch.device('cpu'))
        # input_shape here is the MODEL input (e.g., 1, 3, 32, 32)
        _, actual_input = _simulate_input_hook(model, next_linear_name, input_shape, device=device)
        
        # Flattened size per batch item (N, C, H, W) -> C*H*W
        current_features = actual_input[0].numel() 
        current_shape = actual_input.shape # e.g. [1, 728, 1, 1]
        
        if debug:
            print(f"[DEBUG] Actual tensor shape entering '{next_linear_name}': {tuple(current_shape)}")
            print(f"[DEBUG] Flattened size per sample: {current_features}")
            
    except Exception as e:
        print(f"[WARN] Failed to simulate forward pass for corrective pool check: {e}")
        return model

    # 3. Get Linear layer expectations
    if debug: print(f"[DEBUG] Locating linear layer module '{next_linear_name}'...")
    next_linear_mod = None
    next_linear_parent = None
    next_linear_parent_name = None
    next_linear_child_name = None

    # Locate module and parent
    for name, mod in model.named_modules():
        if name == next_linear_name:
            next_linear_mod = mod
            if debug: print(f"[DEBUG] Found next_linear_mod: {type(mod).__name__}")
            break
            
    if not isinstance(next_linear_mod, nn.Linear):
        if debug:
            print(f"[INFO] Target '{next_linear_name}' is not nn.Linear — skipping.")
        return model
        
    in_features = next_linear_mod.in_features
    if debug:
        print(f"[DEBUG] Linear '{next_linear_name}' expects in_features={in_features}")

    # 4. Compare and Decide
    if current_features == in_features:
        if debug:
            print("[INFO] ✅ Shapes match (Current == Expected) — no corrective pooling needed.")
        return model

    # 5. Calculate Pooling logic (if mismatch exists)
    if debug: print(f"[DEBUG] Shape mismatch detected. Calculating required pooling dimensions...")
    # Assume NCHW layout for calculation
    if len(current_shape) < 4:
         if debug:
            print("[WARN] Input is already flattened but size mismatches. Cannot fix with pooling.")
         return model

    C = current_shape[1]
    if in_features % C != 0:
        if debug:
            print(f"[WARN] in_features ({in_features}) not divisible by channels ({C}) — unsafe to pool.")
        return model

    expected_hw = in_features // C
    target_hw = int(round(expected_hw ** 0.5))
    if debug: print(f"[DEBUG] Computed target spatial grid: {target_hw}x{target_hw} (expected_hw: {expected_hw})")

    if target_hw * target_hw != expected_hw:
        if debug:
             print(f"[WARN] Target spatial area {expected_hw} is not square — skipping.")
        return model

    if debug:
        print(f"[INFO] 🛠 Mismatch detected! Inserting AdaptiveAvgPool2d({target_hw}, {target_hw})")

    # 6. Insert the pool
    if debug: print(f"[DEBUG] Locating parent container to insert pooling layer...")
    # Locate parent container again to perform insertion
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            if child_mod is next_linear_mod:
                next_linear_parent = parent_mod
                next_linear_parent_name = parent_name
                next_linear_child_name = child_name
                if debug: print(f"[DEBUG] Found parent container '{parent_name}' (child_name: '{child_name}')")
                break
        if next_linear_parent is not None:
            break

    corrective_pool = nn.AdaptiveAvgPool2d((target_hw, target_hw))
    
    new_children = OrderedDict()
    inserted = False
    for name, mod in next_linear_parent.named_children():
        if name == next_linear_child_name and not inserted:
            if debug: print(f"[DEBUG] Injecting 'corrective_pool' right before '{next_linear_child_name}'")
            new_children["corrective_pool"] = corrective_pool
            inserted = True
        new_children[name] = mod

    # Apply update
    if isinstance(next_linear_parent, nn.Sequential):
        new_parent = nn.Sequential(new_children)
    else:
        # Note: Depending on architectural specifics, rebuilding non-Sequential parents as Sequential can sometimes be risky,
        # but leaving your original logic intact per instructions.
        if debug: print(f"[DEBUG] Rebuilding parent as nn.Sequential (was {type(next_linear_parent).__name__})")
        new_parent = nn.Sequential(new_children)

    # Update the model with the new parent container
    if next_linear_parent_name == "":
        if debug: print(f"[DEBUG] Linear layer is at model root. Setting sequential block directly.")
        # Special handling if the linear layer is a direct child of the root model
        setattr(model, next_linear_child_name, nn.Sequential(
            corrective_pool,
            next_linear_mod
        ))
    else:
        if debug: print(f"[DEBUG] Triggering _update_container for '{next_linear_parent_name}'")
        _update_container(model, next_linear_parent_name, new_parent)

    if debug:
        print("[RESULT] Corrective pooling inserted successfully.")

    return model


def _update_container(model: nn.Module, container_path: str, new_container: nn.Module):
    """Replace the module at `container_path` in `model` with `new_container`."""
    print(f"\n[DEBUG ENTRY] _update_container: path='{container_path}'")
    
    # FIX: Handle Root Container case
    if container_path == "":
        print("[DEBUG] Updating root container in-place (Preserving attribute names).")
        
        # new_container is the Sequential returned by _replace_layers.
        # It contains the exact names we want to exist on the model.
        new_children = list(new_container.named_children())
        print(f"[DEBUG] Iterating {len(new_children)} children for root assignment...")
        
        # Update attributes directly. 
        # 'block4' becomes the collapsed module.
        # 'block5' becomes Identity().
        for name, module in new_children:
            print(f"[DEBUG] Setting root attribute model.{name}")
            setattr(model, name, module)
            
        print(f"[DEBUG EXIT] Root container update complete.")
        return

    # --- Existing logic for non-root containers ---
    print(f"[DEBUG] Resolving nested parent object for container path...")
    parts = container_path.split('.')
    parent = model
    for part in parts[:-1]:
        print(f"[DEBUG] Traversing node: '{part}'")
        parent = parent[int(part)] if _is_int_str(part) else getattr(parent, part)
        
    last = parts[-1]
    if _is_int_str(last):
        print(f"[DEBUG] Assigning to list/sequential index: {last}")
        parent[int(last)] = new_container
    else:
        print(f"[DEBUG] Assigning to module attribute: '{last}'")
        setattr(parent, last, new_container)
        
    print(f"[DEBUG EXIT] Container update complete for '{container_path}'.")
# -----------------------------------------------------------------------------
# Skip connection patcher
# -----------------------------------------------------------------------------
def patch_skip_connections(model: nn.Module):
    """
    Patches module forwards to robustly handle residual connections.
    If the collapsed block output shape differs from the shortcut, 
    this attempts to spatially align the shortcut instead of severing the connection.
    Now supports both ResNet/RegNet (shortcut/block) and Xception (skip/rep) topologies.
    """
    print(f"\n[DEBUG ENTRY] patch_skip_connections: Scanning model for skip-connection blocks...")
    model._bypassed_residuals = 0
    patch_count = 0

    for name, module in model.named_modules():
        # FIX: Explicitly ignore SmartIdentity to prevent false positive patches
        # Using class name string check to avoid circular import issues if SmartIdentity is elsewhere
        if type(module).__name__ == "SmartIdentity":
            # print(f"[DEBUG] Skipping SmartIdentity module: '{name}'") # Intentionally commented out to avoid spam
            continue
            
        # 1. Check for ResNet/RegNet style blocks with 'shortcut' and 'block'
        has_res = hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module) and hasattr(module, 'block')
        
        # 2. Check for Xception style blocks with 'skip' and 'rep'
        has_xcep = hasattr(module, 'skip') and isinstance(module.skip, nn.Module) and hasattr(module, 'rep')
        
        if has_res or has_xcep:
            shortcut_attr = 'shortcut' if has_res else 'skip'
            block_attr = 'block' if has_res else 'rep'
            
            print(f"[DEBUG] Found residual candidate module: '{name}' (using attrs: {shortcut_attr} / {block_attr})")
            
            # Save original forward if not already patched
            if not hasattr(module, '_orig_forward'):
                module._orig_forward = getattr(module, 'forward')
                print(f"[DEBUG] Backed up original forward method for '{name}'.")
                
            # Factory function to capture loop variables safely
            def make_patched_forward(mod_name, sc_attr, blk_attr):
                def new_forward(self, x):
                    # print(f"[DEBUG FORWARD] '{mod_name}' - Input shape: {tuple(x.shape)}")
                    
                    # Dynamically fetch the child modules based on the detected topology
                    main_block = getattr(self, blk_attr)
                    shortcut_block = getattr(self, sc_attr)
                    
                    # 1. Run the (potentially collapsed) main block
                    out = main_block(x)
                    # print(f"[DEBUG FORWARD] '{mod_name}' - Main block output shape: {tuple(out.shape)}")
                    
                    # 2. Run the shortcut
                    try:
                        sc = shortcut_block(x)
                        # print(f"[DEBUG FORWARD] '{mod_name}' - Shortcut output shape: {tuple(sc.shape)}")
                    except Exception as e:
                        # Fallback if shortcut itself fails
                        print(f"[WARN FORWARD] '{mod_name}' - Shortcut failed ({e}). Bypassing shortcut and returning relu(out).")
                        return F.relu(out)

                    # 3. CRITICAL FIX: Align shapes instead of giving up
                    if out.shape != sc.shape:
                        # print(f"[DEBUG FORWARD] '{mod_name}' - Shape mismatch detected: out {tuple(out.shape)} != sc {tuple(sc.shape)}")
                        
                        # Case A: Spatial Mismatch (e.g., 16x16 vs 32x32)
                        # We use adaptive pooling on the SHORTCUT to match the OUTPUT
                        if out.shape[2:] != sc.shape[2:]:
                            # print(f"[DEBUG FORWARD] '{mod_name}' - Applying AdaptiveAvgPool2d to shortcut to match spatial dims {tuple(out.shape[2:])}")
                            sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
                        
                        # Case B: Channel Mismatch
                        # If channels still don't match after spatial fix, we unfortunately
                        # cannot add them without a learned 1x1 conv. We must bypass.
                        if out.shape[1] != sc.shape[1]:
                            print(f"[WARN FORWARD] '{mod_name}' - Irreconcilable channel mismatch (out:{out.shape[1]} vs sc:{sc.shape[1]}). Bypassing residual.")
                            model._bypassed_residuals += 1
                            return F.relu(out)

                    # 4. Add residual
                    # print(f"[DEBUG FORWARD] '{mod_name}' - Adding shortcut and main block output successfully.")
                    return F.relu(out + sc)
                return new_forward

            # Apply the patch using our factory method to bind the correct attribute names
            module.forward = make_patched_forward(name, shortcut_attr, block_attr).__get__(module)
            patch_count += 1
            print(f"[PATCH] Patched residual block forward (with spatial align): '{name}' [{shortcut_attr}/{block_attr}]")

    print(f"[DEBUG EXIT] patch_skip_connections complete. Total blocks patched: {patch_count}")
# -----------------------------------------------------------------------------
# Core collapse of a single block (simple replacement)
# -----------------------------------------------------------------------------

def _collapse_block(
    model: nn.Module,
    start_layer_name: str,
    end_layer_name: str,
    input_shape: tuple,
    device='cpu',
    debug: bool = False
) -> nn.Module:
    """
    Collapse layers between start_layer_name and end_layer_name (inclusive)
    by replacing them with: Conv2d(1x1) -> AdaptiveAvgPool2d(H_last,W_last).
    """
    print(f"\n[INFO] ===== Collapsing block: {start_layer_name} → {end_layer_name} =====")
    print(f"[DEBUG ENTRY] _collapse_block: input_shape={input_shape}, device={device}")

    # Step 1: Locate block
    print(f"[STEP 1] Locating block boundaries...")
    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name)
    if debug:
        print(f"[DEBUG] Located block with {len(info['full_block'])} layers")
        for n, l in info["full_block"]:
            print(f"    [LAYER] {n}: {type(l).__name__}")

    # FIX: Re-route capture hook to the top-level LCA child to prevent deep-branch shape mismatches
    # Step 2: Capture activation entering the start layer
    lca_path = info.get("container_name", "")
    first_child_name = info.get("first_layer_name", "")
    actual_start_path = f"{lca_path}.{first_child_name}" if lca_path else first_child_name
    
    if debug:
        print(f"[STEP 2] Capturing activation before LCA child '{actual_start_path}' (Original target: '{start_layer_name}')...")
        
    x, pre_params = _capture_preblock_activation(
        model, actual_start_path, input_shape, info["conv_layers"], info["layer_type"], device, debug
    )
    # Step 3: Find next linear
    print(f"[STEP 3] Searching for next linear layer after '{end_layer_name}'...")
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)
    if debug:
        print(f"[DEBUG] Next linear layer: {next_linear_name} -> {type(next_linear_mod).__name__ if next_linear_mod else 'None'}")

    # Step 4: Analyze block output
    print(f"[STEP 4] Analyzing block output characteristics...")
    block_analysis = _analyze_block_output(
        model,
        info["full_block"],
        info["conv_layers"],
        info["named_layers"],
        info["end_idx"],
        info["layer_type"],
        x,
        next_linear_mod,
        debug
    )
    if debug:
        print(f"[DEBUG] Block output analysis result:")
        for k, v in block_analysis.items():
            print(f"    {k}: {v}")

    # Step 5: Replace block
    print(f"[STEP 5] Rebuilding and replacing collapsed block...")
    model = _build_and_replace_block(
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
        debug
    )
    print(f"[INFO] ✅ Collapse complete for block '{start_layer_name}' → '{end_layer_name}'")
    print(f"[DEBUG EXIT] _collapse_block returning updated model.")
    return model

def _capture_preblock_activation(model, start_layer_name, input_shape, conv_layers, layer_type, device, debug):
    print(f"\n[DEBUG ENTRY] _capture_preblock_activation for '{start_layer_name}'")
    print(f"[DEBUG] Attempting to capture activation using simulation hook...")
    try:
        dummy_input, x = _simulate_input_hook(model, start_layer_name, input_shape, device=device)
        if debug:
            print(f"[DEBUG] Successfully captured activation via hook → shape: {tuple(x.shape)}")
    except Exception as e:
        print(f"[WARN] Hook failed: {e}")
        print(f"[WARN] Falling back to dummy tensor initialization.")
        
        if layer_type == nn.Conv2d:
            in_ch = conv_layers[0].in_channels if hasattr(conv_layers[0], 'in_channels') else input_shape[1]
            H, W = input_shape[-2], input_shape[-1]
            if debug: print(f"[DEBUG] Fallback logic (Conv2d): in_ch={in_ch}, H={H}, W={W}")
            x = torch.randn(1, in_ch, H, W, device=device)
        else:
            in_feat = conv_layers[0].in_features if hasattr(conv_layers[0], 'in_features') else input_shape[1]
            if debug: print(f"[DEBUG] Fallback logic (Linear): in_features={in_feat}")
            x = torch.randn(1, in_feat, device=device)
            
        print(f"[DEBUG] Created fallback tensor of shape {tuple(x.shape)}")

    pre_params = count_trainable_params(model)
    if debug:
        print(f"[DEBUG] Total trainable parameters before collapse: {pre_params:,}")

    print(f"[DEBUG EXIT] _capture_preblock_activation complete.")
    return x, pre_params

def _find_next_linear(model, end_layer_name, debug):
    if debug:
        print(f"\n[DEBUG ENTRY] _find_next_linear: Searching for nn.Linear after '{end_layer_name}'...")
        
    modules_list = list(model.named_modules())
    if debug: print(f"[DEBUG] Flattened model into {len(modules_list)} total modules.")
    
    idx_end_global = None

    print(f"[DEBUG] Searching for exact index of '{end_layer_name}'...")
    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            if debug:
                print(f"[DEBUG] Exact layer match found at global index {i}: '{n}' ({type(m).__name__})")
            break

    if idx_end_global is None:
        print(f"[DEBUG] Exact match failed. Searching for partial match (endswith)...")
        for i, (n, m) in enumerate(modules_list):
            if n.endswith(end_layer_name):
                idx_end_global = i
                if debug:
                    print(f"[DEBUG] Fallback: found partial match at global index {i}: '{n}'")
                break

    next_linear_name = None
    next_linear_mod = None
    
    if idx_end_global is not None:
        if debug:
            print(f"[DEBUG] Scanning forward from index {idx_end_global + 1} for next Linear...")
        for j in range(idx_end_global + 1, len(modules_list)):
            n, m = modules_list[j]
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Found Linear layer ahead: '{n}' ({type(m).__name__})")
                break

    if next_linear_mod is None:
        if debug:
            print(f"[DEBUG] No Linear found sequentially after '{end_layer_name}'. Searching globally as ultimate fallback...")
        for n, m in modules_list:
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Global fallback Linear found: '{n}' ({type(m).__name__})")
                break

    if next_linear_mod is None:
        print(f"[WARN] No Linear layer found in entire model after '{end_layer_name}'")

    if debug:
        print(f"[DEBUG EXIT] Next Linear detected → name='{next_linear_name}', module={next_linear_mod}")

    return next_linear_name, next_linear_mod

def _analyze_block_output(
    model,
    full_block,
    conv_layers,
    named_layers,
    end_idx,
    layer_type,
    x,
    next_linear_mod,
    debug,
):
    if debug:
        print(f"\n[DEBUG ENTRY] _analyze_block_output: Analyzing collapsed block ({len(full_block)} layers)...")
        print(f"[DEBUG] Input tensor shape before block forward pass: {tuple(x.shape)}")
        print(f"[DEBUG] Running synthetic forward pass through original block layers:")

    with torch.no_grad():
        y = x.clone()
        for idx, (name, layer) in enumerate(full_block):
            if debug:
                print(
                    f"    [DEBUG] Forwarding Layer {idx+1}/{len(full_block)}: "
                    f"'{name}' ({layer.__class__.__name__}) | input_shape={tuple(y.shape)}"
                )
            try:
                y = layer(y)
            except RuntimeError as e:
                # If a layer fails, try a NHWC/NCHW permute fallback for ConvNeXt
                print(f"    [WARN] Forward failed on '{name}': {e}. Attempting ConvNeXt fallback permutation...")
                if "shape" in str(e).lower() and y.ndim == 4:
                    try:
                        y = layer(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                        print(f"    [DEBUG] Permutation fallback succeeded!")
                    except Exception as e2:
                        print(f"    [ERROR] Permutation fallback also failed: {e2}")
                        raise RuntimeError(f"Layer '{name}' forward failed. Topology may be non-sequential.")
                else:
                    raise e
            if debug:
                print(f"        └── output shape: {tuple(y.shape)}")

    # --- Ground-truth output characteristics ---
    out_shape = tuple(y.shape)

    # 🔧 CRITICAL FIX:
    # Infer channels from actual tensor, not from layer attributes
    out_channels = y.shape[1] if y.ndim >= 2 else None

    if debug:
        print(f"[DEBUG] Synthesis complete. Final block output shape: {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")

    # --- Detect pooling inside original block (informational) ---
    print(f"[DEBUG] Scanning original block for explicit pooling layers...")
    pool_layer = next(
        (
            m for _, m in reversed(full_block)
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        ),
        None,
    )

    if debug:
        if pool_layer is not None:
            print(f"[DEBUG] Detected explicit pool in original block: {type(pool_layer).__name__}")
        else:
            print(f"[DEBUG] No explicit pooling layer detected inside block.")

    # --- Linear compatibility heuristic ---
    linear_in_features_heuristic = next_linear_mod.in_features if next_linear_mod else None
    if debug:
        print(f"[DEBUG] Heuristic next linear in_features = {linear_in_features_heuristic}")

    adaptive_pool_to_use = None
    if (
        layer_type == nn.Conv2d
        and linear_in_features_heuristic is not None
        and out_channels is not None
        and len(out_shape) >= 4
    ):
        expected_hw = max(1, linear_in_features_heuristic // out_channels)
        cur_H, cur_W = out_shape[-2], out_shape[-1]
        cur_hw = cur_H * cur_W

        if debug:
            print(
                f"[DEBUG] Comparing spatial dims for Linear compatibility: "
                f"expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})"
            )

        if cur_hw != expected_hw:
            target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
            target_W = max(1, expected_hw // target_H)
            adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))

            if debug:
                print(
                    f"[DEBUG] Suggesting AdaptiveAvgPool2d({target_H},{target_W}) "
                    f"to reconcile linear in_features mismatch."
                )

    # --- Shortcut detection (unchanged) ---
    print(f"[DEBUG] Scanning model globally for shortcut module out_channels...")
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
                    print(f"[DEBUG] Found shortcut conv at '{nm}' → out_channels={shortcut_out_channels}")
                break

    if debug:
        print(f"[DEBUG EXIT] _analyze_block_output returning dictionary.")
        print(f"         out_shape={out_shape}")
        print(f"         out_channels={out_channels}")
        print(f"         has_pool={pool_layer is not None}")
        print(f"         adaptive_pool_to_use={adaptive_pool_to_use}")
        print(f"         shortcut_out_channels={shortcut_out_channels}")

    return {
        "out_shape": out_shape,
        "out_channels": out_channels,
        "pool_layer": pool_layer,
        "adaptive_pool_to_use": adaptive_pool_to_use,
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
    If a downstream module fails during a forward pass (e.g. pooling mismatch),
    wraps it in _SafePool or replaces it with Identity to preserve model functionality.
    """
    print(f"\n[DEBUG ENTRY] _validate_downstream for '{start_container_name}'")
    print(f"[DEBUG] start_idx={start_idx}, device={device}, has_next_linear={next_linear_name is not None}")

    # Retrieve target container
    print(f"[DEBUG] Retrieving target container: '{start_container_name}'")
    try:
        container = get_layer(model, start_container_name)
    except Exception as e:
        print(f"[WARN] Could not access container '{start_container_name}': {e}")
        return

    children = list(container.named_children())
    print(f"[DEBUG] Container retrieved. It has {len(children)} immediate children.")
    if not children:
        print(f"[DEBUG] Container '{start_container_name}' has no children; skipping downstream validation.")
        return

    # Find collapsed/inserted child index
    collapsed_idx = None
    print(f"[STEP] Searching for inserted collapsed module within '{start_container_name}'...")
    for i, (nm, m) in enumerate(children):
        if nm.startswith("collapsed_") or (
            isinstance(m, nn.Sequential)
            and any(k in dict(m.named_children()) for k in ("conv_1x1", "adaptive_pool", "conv_dw"))
        ):
            collapsed_idx = i
            print(f"[DEBUG] Found inserted block candidate at index {i}: '{nm}'")
            break

    if collapsed_idx is None:
        print(f"[DEBUG] Primary block search failed. Running fallback search for direct 1x1 conv...")
        # fallback search: direct 1x1 conv
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

    # Run through inserted module
    try:
        with torch.no_grad():
            dev = next((p.device for p in model.parameters()), torch.device('cpu'))
            t = pre_activation.clone().to(dev)
            if debug: print(f"[DEBUG] Pushing test tensor of shape {tuple(t.shape)} through inserted module...")
            t = inserted_mod(t)
        print(f"[DEBUG] Inserted module forward successful, output shape: {tuple(t.shape)}")
    except Exception as e:
        print(f"[WARN] Forward pass through inserted module failed: {e}")
        return

    # Validate next modules downstream
    print(f"[STEP] Scanning immediate downstream modules (index {collapsed_idx + 1} to {len(children)}) for shape or runtime errors...")
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            t = mod(t)
            if debug:
                print(f"[DEBUG] Downstream '{start_container_name}.{nm}' executed successfully, output shape: {tuple(t.shape)}")
        except Exception as e:
            print(f"[WARN] Downstream module '{start_container_name}.{nm}' raised exception: {e}")
            print(f"[STEP] Replacing problematic module '{nm}' with safe alternative...")

            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
                print(f"[INFO] Replaced '{nm}' with _SafePool wrapper.")
            # =========================================================
            # FIX: Force validate_downstream to back off from nn.Linear
            # =========================================================
            elif isinstance(mod, nn.Linear):
                print(f"[INFO] Target downstream module '{nm}' is nn.Linear. Bailing out. Allowing Corrective Pooling to handle it natively.")
                return 
            else:
                safe = nn.Identity()
                print(f"[INFO] Replaced '{nm}' with Identity() to bypass invalid operation.")

            if isinstance(container, nn.Sequential):
                print(f"[DEBUG] Rebuilding Sequential container with safe replacement...")
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                print(f"[DEBUG] Directly setting safe replacement as attribute '{nm}'...")
                setattr(container, nm, safe)

            print(f"[DEBUG] Replacement applied to '{start_container_name}.{nm}' ({safe.__class__.__name__}). Breaking downstream check.")
            return  # stop after first fix

        # detect zero-spatial output
        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            print(f"[WARN] Module '{start_container_name}.{nm}' produced zero spatial dimensions {tuple(t.shape)}. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential):
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            print(f"[DEBUG] Zero-dimension fix applied to '{start_container_name}.{nm}' with _SafePool. Breaking downstream check.")
            return

    print(f"[RESULT] ✅ Downstream validation for '{start_container_name}' completed successfully.")
    print(f"[DEBUG EXIT] _validate_downstream complete.")
# -----------------------------------------------------------------------------
# Top-level multi-block collapse function (flexible API)
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
    Top-level API to collapse multiple blocks with the simple replacement policy.
    """
    print(f"\n[DEBUG ENTRY] ===== collapse_only invoked =====")
    print(f"[DEBUG] Configuration: device={device}, dry_run={dry_run}, handle_skips={handle_skips}, safe_param_reduction={safe_param_reduction}")
    print(f"[DEBUG] Input tracking: model_provided={model is not None}, model_weights={model_weights_1}, input_shape={input_shape}")

    # Load or use provided model
    if model is None:
        print(f"[STEP] Loading model from disk...")
        if not (model_weights_1 and model_class):
            raise ValueError("[ERROR] Must provide either an instantiated `model` or (`model_weights_1` + `model_class`).")

        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating model from class '{model_class.__name__}' with kwargs={model_kwargs}")
        try:
            model = model_class(**model_kwargs)
            if debug: print(f"[DEBUG] Model instantiated successfully.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to instantiate model class {model_class}: {e}")

        print(f"[INFO] Loading weights from file: '{model_weights_1}'")
        try:
            chk = torch.load(model_weights_1, map_location=device)
            state = chk.get('model', chk) if isinstance(chk, dict) else chk
            if debug: print(f"[DEBUG] Extracted state dict with {len(state)} keys. Applying to model...")
            model.load_state_dict(state)
            print(f"[INFO] Weights successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model weights: {e}")
    else:
        print(f"[STEP] Using provided in-memory model instance ({model.__class__.__name__})")

    # Inspect model
    if debug:
        print(f"[DEBUG] Calculating initial layer statistics...")
        try:
            print(f"[DEBUG] Model layer statistics before collapse:\n{layer_stats(model)}")
        except Exception as e:
            print(f"[WARN] layer_stats() failed: {e}")

    # Make a deep copy to avoid modifying the original model
    print(f"[STEP] Creating deepcopy of model for safe modification...")
    model = deepcopy(model).to(device)
    model.eval()
    if debug: print(f"[DEBUG] Deepcopy completed and set to eval mode on '{device}'.")

    # Normalize compression_set
    print(f"[STEP] Parsing compression set...")
    if compression_set is None:
        print("[WARN] compression_set is None or empty; skipping collapse. Returning unmodified model.")
        return model

    collapse_map = {}
    if isinstance(compression_set, dict):
        if debug:
            print(f"[DEBUG] Detected compression_set as dict with {len(compression_set)} entries.")
        for k, v in compression_set.items():
            start, end = v
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[k] = (start, end)
            if debug:
                print(f"    [DEBUG] Added dictionary mapping: {k} = ('{start}' → '{end}')")
    else:
        if debug:
            print(f"[DEBUG] Detected compression_set as sequence with {len(compression_set)} pairs.")
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple):
                start = start[0]
            if isinstance(end, tuple):
                end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)
            if debug:
                print(f"    [DEBUG] Added sequence mapping: collapse_{i} = ('{start}' → '{end}')")

    # Store collapsed blocks for reference
    model._collapsed_blocks = list(collapse_map.values())
    if debug:
        print(f"[DEBUG] Total configured collapse targets: {len(model._collapsed_blocks)}")
        for idx, (s, e) in enumerate(model._collapsed_blocks):
            print(f"    [BLOCK TARGET {idx}] {s} → {e}")

    # Track parameters
    pre_total = count_trainable_params(model)
    print(f"[INFO] Model parameter count before collapsing: {pre_total:,}")

    # Process each block in sequence
    print(f"[STEP] Beginning block-wise collapsing loop...")
    for name, (start, end) in collapse_map.items():
        print(f"\n[INFO] Processing collapse task '{name}': '{start}' → '{end}'")
        if dry_run:
            print(f"[INFO] dry_run enabled; skipping actual modification for block '{name}'.")
            continue

        try:
            if debug: print(f"[DEBUG] Triggering _collapse_block for '{name}'...")
            model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)
            print(f"[INFO] ✅ Successfully collapsed block '{name}' ({start} → {end})")
        except Exception as e:
            print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
                
    # Post-processing modifications
    if handle_skips:
        print(f"\n[STEP] Patching skip connections (handle_skips=True)...")
        try:
            patch_skip_connections(model)
            if debug:
                print(f"[DEBUG] Skip connections patched successfully globally.")
        except Exception as e:
            print(f"[WARN] Failed to patch skip connections globally: {e}")

    print(f"[STEP] Disabling in-place ReLUs for autograd safety...")
    try:
        disable_inplace_relu(model)
        if debug:
            print(f"[DEBUG] In-place ReLUs converted to out-of-place versions across model.")
    except Exception as e:
        print(f"[WARN] Failed to disable in-place ReLUs: {e}")

    # Safe wrapping of pooling layers
    print(f"\n[STEP] Wrapping pooling layers safely...")
    try:
        _wrap_pools_safe(model)
        if debug:
            print("[DEBUG] All pooling layers wrapped with _SafePool to prevent underflow errors.")
    except Exception as e:
        print(f"[WARN] Failed to wrap pools safely: {e}")

    # Post-collapse summary
    post_total = count_trainable_params(model)
    print(f"\n[STEP] ===== Collapse summary =====")
    print(f"[INFO] Parameters before: {pre_total:,}")
    print(f"[INFO] Parameters after : {post_total:,}")
    delta = pre_total - post_total
    print(f"[INFO] ΔParams = {delta:+,} (expected ≤ 0)")

    if post_total > pre_total:
        print(f"[WARN] ⚠ Model gained parameters after collapsing! Investigate collapse policy or replacement logic.")

    if safe_param_reduction and delta < 0:
        # Note: Delta < 0 mathematically means post > pre. If delta = pre - post, a negative delta means it grew.
        print(f"[WARN] ⚠ Parameter count increased when safe_param_reduction=True — collapse may have failed silently.")

    print(f"[RESULT] ✅ collapse_only complete. Total attempted collapse targets: {len(collapse_map)}")
    print(f"[DEBUG EXIT] collapse_only returning fully modified model.")
    return model


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Safe pooling wrapper (prevents underflow crashes)
# -----------------------------------------------------------------------------

class _SafePool(nn.Module):
    """
    Wrapper that attempts to apply the wrapped pooling module; if the input
    spatial dimensions are too small or the pool raises, we fall back safely:
      - For non-adaptive pools: if kernel > input dim, uses AdaptiveAvgPool2d to
        produce a minimal valid output (>=1).
      - If anything else fails, returns the input (identity).
    This avoids 'Output size is too small' runtime errors.
    """
    def __init__(self, pool_module: nn.Module):
        super().__init__()
        self.pool = pool_module
        # print(f"[DEBUG INIT] _SafePool wrapped around {self.pool.__class__.__name__}")

    def forward(self, x):
        # guard shape sanity
        try:
            H, W = x.shape[-2], x.shape[-1]
            # print(f"[DEBUG FORWARD] _SafePool routing input shape: {tuple(x.shape)}")
        except Exception as e:
            # not a 4D tensor (some unexpected case) -> try to apply pool and catch exceptions
            print(f"[WARN] _SafePool: Failed to extract H, W from input shape (not 4D?). Error: {e}")
            try:
                out = self.pool(x)
                print(f"[DEBUG] _SafePool: Pool applied directly despite missing H/W. Output shape: {tuple(out.shape)}")
                return out
            except Exception as e_inner:
                print(f"[WARN] _SafePool: Direct pool application failed ({e_inner}). Returning identity.")
                return x

        try:
            # For standard pools, check kernel size
            if isinstance(self.pool, (nn.MaxPool2d, nn.AvgPool2d)):
                k = self.pool.kernel_size
                if isinstance(k, tuple):
                    kh, kw = k
                else:
                    kh = kw = k
                
                # print(f"[DEBUG] _SafePool checking kernel size (kh={kh}, kw={kw}) against dimensions (H={H}, W={W})")
                
                # if kernel/stride would underflow, use adaptive avg pool to safe size
                if kh > H or kw > W or H <= 0 or W <= 0:
                    print(f"[WARN] _SafePool: Underflow risk detected! H={H}, W={W} vs kernel={kh}x{kw}")
                    # choose a safe target HxW (at least 1)
                    target_H = max(1, min(H, kh) if H > 0 else 1)
                    target_W = max(1, min(W, kw) if W > 0 else 1)
                    print(f"[INFO] _SafePool: Applying safe adaptive fallback to {target_H}x{target_W}")
                    return F.adaptive_avg_pool2d(x, (target_H, target_W))

            # Try to apply original pool
            out = self.pool(x)

            # post-check: if shape became invalid, return identity
            if out.shape[-2] < 1 or out.shape[-1] < 1:
                print(f"[WARN] _SafePool: Post-pool spatial shape invalid {tuple(out.shape)}. Reverting to identity.")
                return x
            
            # print(f"[DEBUG FORWARD] _SafePool execution successful. Output shape: {tuple(out.shape)}")
            return out
            
        except Exception as e:
            # Any failure -> safe fallback
            print(f"[ERROR] _SafePool: Unexpected global failure during pool computation: {e}. Returning identity x.")
            return x


def _wrap_pools_safe(module: nn.Module):
    """
    Recursively replace pooling modules in `module` with _SafePool wrappers.
    This mutates the module in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
            print(f"[DEBUG] _wrap_pools_safe: Found target pooling layer '{name}' ({child.__class__.__name__}). Wrapping with _SafePool...")
            safe = _SafePool(child)
            parent = module
            try:
                setattr(parent, name, safe)
                print(f"    [DEBUG] Successfully replaced via setattr(parent, '{name}', safe)")
            except Exception as e1:
                print(f"    [WARN] setattr failed on '{name}': {e1}. Attempting index replacement...")
                try:
                    idx = int(name)
                    parent[idx] = safe
                    print(f"    [DEBUG] Successfully replaced via parent[{idx}] = safe")
                except Exception as e2:
                    print(f"    [WARN] Index replacement failed: {e2}. Falling back to forced setattr...")
                    setattr(parent, name, safe)
        else:
            # Recursively check deeper into the child module
            _wrap_pools_safe(child)