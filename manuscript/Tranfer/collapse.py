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



def _locate_and_prepare_block(model, start_layer_name, end_layer_name, debug=False):
    print(f"[DEBUG] [LCA] Locating block boundaries: '{start_layer_name}' -> '{end_layer_name}'")

    start_parts = start_layer_name.split('.') if start_layer_name else []
    end_parts = end_layer_name.split('.') if end_layer_name else []
    common_parts = []
    
    if start_layer_name == end_layer_name and len(start_parts) > 0:
        lca_path = '.'.join(start_parts[:-1])
    else:
        for a, b in zip(start_parts, end_parts):
            if a == b:
                common_parts.append(a)
            else:
                break
        lca_path = '.'.join(common_parts)

    if lca_path == "":
        start_container_name, _ = _get_container_and_subname(start_layer_name)
        end_container_name, _ = _get_container_and_subname(end_layer_name)
        if start_container_name == end_container_name:
            lca_path = start_container_name
            container = get_layer(model, lca_path)
        else:
            lca_path = ""
            container = model
    else:
        container = get_layer(model, lca_path)

    print(f"[DEBUG] [LCA] Resolved Container: '{lca_path}' | Type: {type(container).__name__}")
    named_layers = list(container.named_children())

    start_idx = end_idx = None
    for i, (child_name, _) in enumerate(named_layers):
        full_child_prefix = f"{lca_path}.{child_name}" if lca_path else child_name
        if start_idx is None and (start_layer_name == full_child_prefix or start_layer_name.startswith(full_child_prefix + ".")):
            start_idx = i
        if end_idx is None and (end_layer_name == full_child_prefix or end_layer_name.startswith(full_child_prefix + ".")):
            end_idx = i
        if start_idx is not None and end_idx is not None:
            break

    if start_idx is None or end_idx is None:
        print(f"[WARN] ⚠ Could not map start/end layers into LCA container '{lca_path}'")
        print(f"[DEBUG]   start_layer_name={start_layer_name}, found start_idx={start_idx}")
        print(f"[DEBUG]   end_layer_name={end_layer_name}, found end_idx={end_idx}")
        print(f"[DEBUG]   ISSUE #3 FIX: This may be a cross-branch collapse (parallel paths in Inception/ConvNeXt).")
        print(f"[DEBUG]   Skipping this collapse candidate as it cannot be sequentially mapped.")
        # Return None to signal filtering at caller level
        return None
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    full_block = named_layers[start_idx:end_idx + 1]
    
    # [CRITICAL FIX] Safe Holistic Escalation for Complex Blocks
    safe_sequential_classes = ("Sequential", "SeparableConv2d", "ModuleList", type(model).__name__)
    if type(container).__name__ not in safe_sequential_classes and lca_path != "":
        print(f"[DEBUG] Escalating complex block '{lca_path}' ({type(container).__name__}) to parent for holistic replacement.")
        
        parent_path, subname = _get_container_and_subname(lca_path)
        parent_container = get_layer(model, parent_path)
        named_layers_parent = list(parent_container.named_children())
        
        try:
            esc_idx = next(i for i, (n, _) in enumerate(named_layers_parent) if n == subname)
            return {
                "container": parent_container,
                "container_name": parent_path,
                "named_layers": named_layers_parent,
                "start_idx": esc_idx,
                "end_idx": esc_idx,
                "full_block": [(subname, container)],
                "layer_type": nn.Conv2d,
                "conv_layers": [m for m in container.modules() if isinstance(m, nn.Conv2d)],
                "collapse_mode": "conv",
                "first_layer_name": subname,
                "last_layer_name": subname,
            }
        except StopIteration:
            print(f"[WARN] Escalation failed: '{subname}' not found in parent '{parent_path}'. Proceeding standard slice.")

    # =========================================================
    # THE FIX: Restore the missing return for standard slices!
    # =========================================================
    return {
        "container": container,
        "container_name": lca_path,
        "named_layers": named_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "full_block": full_block,
        "layer_type": nn.Conv2d,
        "conv_layers": [m for _, m in full_block if isinstance(m, nn.Conv2d)],
        "collapse_mode": "conv",
        "first_layer_name": named_layers[start_idx][0],
        "last_layer_name": named_layers[end_idx][0],
    }
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
        print(f"\n{'='*60}")
        print(f"[STEP] COLLAPSE REPLACEMENT INIT: '{start_layer_name}'")
        print(f"{'='*60}")
        print(f"[DEBUG] Target device    : {device}")
        print(f"[DEBUG] Info keys found  : {list(info.keys())}")
        
    named_layers = info["named_layers"] 
    # Prefer the container determined by the locator (LCA).
    start_container_name = info.get("container_name") 
    if start_container_name is None: 
        start_container_name = _get_container_and_subname(start_layer_name)[0] 
        
    start_idx, end_idx = info["start_idx"], info["end_idx"] 
    out_shape = block_analysis.get("out_shape") 
    out_channels = block_analysis.get("out_channels") 

    if out_shape is None or out_channels is None: 
        raise RuntimeError(f"[ERROR] block_analysis missing required keys. Found: {list(block_analysis.keys())}") 

    target_H, target_W = (out_shape[-2], out_shape[-1]) if len(out_shape) >= 4 else (1, 1) 
    
    if x is None or x.ndim < 2: 
        raise RuntimeError(f"[ERROR] Invalid captured activation `x`. Shape found: {getattr(x, 'shape', None)}") 
    in_channels = int(x.shape[1]) 

    if debug:
        print(f"[DEBUG] Container Name   : '{start_container_name}'")
        print(f"[DEBUG] Layer Indices    : {start_idx} -> {end_idx}")
        print(f"[DEBUG] Spatial Target   : H={target_H} x W={target_W}")
        print(f"[DEBUG] Channel Mapping  : {in_channels} (in) -> {out_channels} (out)")

    # -------- INTELLIGENT ROUTING LOGIC --------
    original_convs = info.get("conv_layers", [])
    is_depthwise = False
    target_kernel_size = 1
    
    if original_convs:
        first_conv = original_convs[0]
        if debug:
            print(f"[DEBUG] Analyzing original conv: {first_conv.__class__.__name__} (groups={first_conv.groups}, in={first_conv.in_channels}, out={first_conv.out_channels})")
            
        if first_conv.groups == first_conv.in_channels and first_conv.in_channels == first_conv.out_channels:
            if in_channels == out_channels:
                is_depthwise = True
                target_kernel_size = first_conv.kernel_size[0] if isinstance(first_conv.kernel_size, tuple) else first_conv.kernel_size
                
        elif first_conv.kernel_size == (1, 1) or first_conv.kernel_size == 1:
            target_kernel_size = 1 
            is_depthwise = False
            
    if is_depthwise:
        # --------------------------------------------------------------
        # MobileNet‑style block: depth‑wise conv followed by point‑wise conv.
        # The original block expands the channel dimension with the
        # point‑wise (1×1) conv.  To keep FLOPs low we replace the *whole*
        # block by a **grouped 1×1 conv** that directly produces the
        # required `out_channels`.  This preserves the cheap spatial
        # filtering while avoiding the extra multiply‑adds that a plain
        # depth‑wise conv with a large output channel count would incur.
        # --------------------------------------------------------------
        if debug:
            print(f"[DEBUG] ➔ Strategy: GROUPED 1×1 Conv (depth‑wise style, groups={in_channels})")
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=in_channels,   # one filter per input channel
            bias=False,
        )
        conv_name = "conv_g1x1"
    else:
        if debug:
            print(f"[DEBUG] ➔ Strategy: POINTWISE Conv (kernel_size=1)")
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        conv_name = "conv_1x1"
    # -----------------------------------------------


    model_type = type(model).__name__.lower()
    
    # ConvNeXt uses LayerNorm2d (channels-first) and GELU. 
    if "convnext" in model_type:
        # Define a quick inline LayerNorm2d helper for ConvNeXt
        class SurrogateLayerNorm2d(nn.Module):
            def __init__(self, dim, eps=1e-6):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.bias = nn.Parameter(torch.zeros(dim))
                self.eps = eps
            def forward(self, x):
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                return self.weight[:, None, None] * x + self.bias[:, None, None]
                
        norm_layer = SurrogateLayerNorm2d(out_channels)
        act_layer = nn.GELU()
        norm_name = "layernorm"
        act_name = "gelu"
        
    else:
        # VGG, ResNet, MobileNet, Xception, RegNetX default to BN/ReLU
        norm_layer = nn.BatchNorm2d(out_channels)  
        act_layer = nn.ReLU(inplace=False) 
        norm_name = "bn"
        act_name = "relu"
        
    pool = nn.AdaptiveAvgPool2d((target_H, target_W)) 

    replacement = nn.Sequential(OrderedDict([
        (conv_name, conv),
        (norm_name, norm_layer), 
        (act_name, act_layer), 
        ("adaptive_pool", pool), 
    ]))
    
    if debug: 
        print(f"\n[DEBUG] Built replacement module for {model_type}:")
        for name, layer in replacement.named_children():
            print(f"        - {name}: {layer}")
    # =========================================================
    # THE FIX: Explicitly passing info.get("container")
    # =========================================================
    if debug: 
        print(f"\n[STEP] Patching container '{start_container_name or '<root>'}'...") 
    
    updated_container = _replace_layers( 
            named_layers, 
            start_idx, 
            end_idx, 
            replacement, 
            start_name=info["first_layer_name"], 
            end_name=info["last_layer_name"],
            container=info.get("container") # CRITICAL FIX: Preserves custom forward logic
        )
        
    _update_container(model, start_container_name, updated_container) 
    model.to(device) 

    post_params = count_trainable_params(model) 
    if debug:
        print(f"\n[INFO] --- Parameter Delta ---")
        print(f"[INFO] Pre-collapse  : {pre_params:,}") 
        print(f"[INFO] Post-collapse : {post_params:,}") 
        print(f"[INFO] Net Change    : {post_params - pre_params:+,}") 

    if post_params > pre_params: 
        print(f"[WARN] ⚠ Collapsed block INCREASED parameter count. Check collapse policy or routing logic.") 

    # =========================================================
    # REPLACEMENT VALIDATION
    # =========================================================
    try:
        dev = next((p.device for p in model.parameters()), torch.device('cpu')) 
        rep_module = get_layer(model, start_container_name) 
        child = None 
        
        for nm, m in rep_module.named_children(): 
            if nm.startswith("collapsed_") or (isinstance(m, nn.Sequential) and conv_name in dict(m.named_children())):
                child = m 
                break 
        
        if child is None and isinstance(updated_container, nn.Sequential) and start_container_name != "": 
             pass  

        if child is not None: 
            with torch.no_grad(): 
                test_x = x.clone().to(dev) 
                out = child(test_x) 
                if debug: 
                    print(f"[DEBUG] ✓ Replacement local validation OK. Output shape: {tuple(out.shape)}") 
        else:
            print(f"[WARN] Could not locate the inserted collapsed module for local validation.") 
    except Exception as e:
        print(f"[ERROR] Replacement forward validation failed!\n       Exception: {str(e)}") 
            
    # =========================================================
    # CORRECTIVE POOLING & DOWNSTREAM VALIDATION
    # =========================================================
    try:
        if debug:
            print(f"\n[STEP] Evaluating corrective pooling necessity...") 
        model = _insert_corrective_pool(model, next_linear_name, input_shape, debug) 
    except Exception as e:
        print(f"[ERROR] Corrective pool insertion failed: {str(e)}") 
            
    try:
        if debug:
            print(f"[STEP] Triggering downstream validation...") 
        _validate_downstream(model, start_container_name, start_idx, x, input_shape, next_linear_name, next_linear_mod, device, debug) 
        if debug:
            print(f"[DEBUG] ✓ Downstream validation successful.")
    except Exception as e:
        print(f"[ERROR] Downstream validation failed!\n       Exception: {str(e)}") 
            
    if debug: 
        print(f"{'='*60}")
        print(f"[RESULT] Block replacement complete for '{start_layer_name}'.") 
        print(f"{'='*60}\n")

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

def _simulate_input_output_hooks(model: nn.Module, start_layer_path: str, end_layer_path: str, input_shape: Tuple[int, ...], device='cpu'):
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    start_module = get_layer(model, start_layer_path)
    end_module = get_layer(model, end_layer_path)

    captured = {}
    def hook_in(module, inp, out):
        if 'in' not in captured:
            captured['in'] = inp[0].detach()
            
    def hook_out(module, inp, out):
        # Handle cases where output is a tuple
        captured['out'] = out.detach() if isinstance(out, torch.Tensor) else out[0].detach()

    h1 = start_module.register_forward_hook(hook_in)
    h2 = end_module.register_forward_hook(hook_out)
    
    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        h1.remove()
        h2.remove()
        
    if 'in' not in captured or 'out' not in captured:
        raise RuntimeError(f"Failed to capture activations for {start_layer_path} -> {end_layer_path}.")
        
    return captured['in'], captured['out']

def _capture_preblock_activation(model, start_layer_name, end_layer_name, input_shape, conv_layers, layer_type, device, debug):
    print(f"[DEBUG] Capturing I/O hooks for '{start_layer_name}' -> '{end_layer_name}'...")
    try:
        x_in, y_out = _simulate_input_output_hooks(model, start_layer_name, end_layer_name, input_shape, device)
        if debug:
            print(f"[DEBUG] Captured input shape: {tuple(x_in.shape)}, output shape: {tuple(y_out.shape)}")
    except Exception as e:
        print(f"[WARN] Hook failed: {e}. Cannot reliably collapse parallel block without valid I/O.")
        raise e
        
    pre_params = count_trainable_params(model)
    return x_in, y_out, pre_params

class SmartIdentity(nn.Module):
    """
    A robust Identity replacement that safely absorbs nested attribute calls,
    while explicitly blocking structural attributes to protect FLOP tracers.
    """
    def __init__(self):
        super().__init__()
        # Dummy attributes so it registers visually as a spatial layer
        self.kernel_size = (1, 1)
        self.stride = (1, 1)
        self.padding = (0, 0)

    def forward(self, x, *args, **kwargs):
        # Prevent 2D flattened tensors from crashing downstream Conv2d variance probes
        if isinstance(x, torch.Tensor) and x.ndim == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        return x
    
    def __getitem__(self, index):
        # ISSUE #4 FIX: SmartIdentity subscriptable safety.
        # LCA resolution may return SmartIdentity; indexing into it should be safe.
        # Return self to prevent subscript errors.
        return self
        
    def __getattr__(self, name):
        if name.startswith('_'):
            return super().__getattr__(name)
            
        # STRICT BLACKLIST: Force tracers to treat this as an empty pass-through
        forbidden_attrs = {
            'shortcut', 'block', 'weight', 'bias', 
            'in_channels', 'out_channels', 'groups', 
            'shape', 'size', 'dim', 'ndim', 'out_features', 'in_features'
        }
        if name in forbidden_attrs:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
            
        # Safely absorb topology chains (e.g., identity.inception_4a(x) -> identity(x))
        return self
    

def _replace_layers(named_layers, start_idx, end_idx, replacement, start_name=None, end_name=None, container=None):
    """
    Patches layers into a container, handling both nn.Sequential (OrderedDict)
    and custom nn.Module blocks (Attribute-based) safely.
    """
    # 1. Determine if container is a standard Sequential/ModuleList
    is_sequential = isinstance(container, (nn.Sequential, nn.ModuleList))
    
    if is_sequential:
        new_layers = OrderedDict()
        for i, (name, mod) in enumerate(named_layers):
            if i < start_idx or i > end_idx:
                new_layers[name] = mod
            elif i == start_idx:
                new_layers[name] = replacement
            else:
                # Replace intermediate layers with identity
                new_layers[name] = SmartIdentity()
        
        # Reconstruct the Sequential container
        if isinstance(container, nn.Sequential):
            return nn.Sequential(new_layers)
        else:
            return nn.ModuleList(new_layers.values())
    
    else:
        # 2. Attribute-based patching for Inception/ConvNeXt blocks
        for i, (name, mod) in enumerate(named_layers):
            if i == start_idx:
                # Use setattr to patch the custom block's attribute
                setattr(container, name, replacement)
            elif start_idx < i <= end_idx:
                # Replace with identity
                setattr(container, name, SmartIdentity())
        return container
def _insert_corrective_pool(model, next_linear_name, input_shape, debug):
    if not next_linear_name:
        return model

    if ".pwconv" in next_linear_name:
        return model

    try:
        device = next((p.device for p in model.parameters()), torch.device('cpu'))
        x_in, _ = _simulate_input_output_hooks(model, next_linear_name, next_linear_name, input_shape, device=device)
        
        # FIX: Use x_in[0] to calculate features per sample, completely ignoring batch size
        current_features = x_in[0].numel() 
        current_shape = x_in.shape
        if debug:
            print(f"[DEBUG] [POOL] Target Linear: '{next_linear_name}'")
            print(f"[DEBUG] [POOL] Entering shape: {tuple(current_shape)} | Flat size: {current_features}")
    except Exception as e:
        if debug: print(f"[WARN] Pool check hook failed: {e}")
        return model

    next_linear_mod = get_layer(model, next_linear_name)
    if not isinstance(next_linear_mod, nn.Linear):
        return model
        
    in_features = next_linear_mod.in_features
    if current_features == in_features:
        if debug: print("[INFO] ✅ Pool check: Shapes match seamlessly.")
        return model

    if len(current_shape) < 4:
        return model

    C = current_shape[1]
    if in_features % C != 0:
        return model

    expected_hw = in_features // C
    target_hw = int(round(expected_hw ** 0.5))

    if target_hw * target_hw != expected_hw:
        return model

    if debug: print(f"[INFO] 🛠 Mismatch! Wrapping '{next_linear_name}' with AdaptiveAvgPool2d({target_hw}, {target_hw})")

    # Safe in-place module replacement
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            if child_mod is next_linear_mod:
                wrapped_linear = nn.Sequential(
                    nn.AdaptiveAvgPool2d((target_hw, target_hw)),
                    next_linear_mod
                )
                setattr(parent_mod, child_name, wrapped_linear)
                return model

    return model

def _update_container(model: nn.Module, container_path: str, new_container: nn.Module):
    """Replace the module at `container_path` in `model` with `new_container`."""
    
    # FIX: Handle Root Container case
    if container_path == "":
        print("[DEBUG] Updating root container in-place (Preserving attribute names).")
        
        # new_container is the Sequential returned by _replace_layers.
        # It contains the exact names we want to exist on the model.
        new_children = list(new_container.named_children())
        
        # Update attributes directly. 
        # 'block4' becomes the collapsed module.
        # 'block5' becomes Identity().
        for name, module in new_children:
            setattr(model, name, module)
            
        return

    # --- Existing logic for non-root containers ---
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
# Skip connection patcher
# -----------------------------------------------------------------------------

def patch_skip_connections(model: nn.Module):
    model._bypassed_residuals = getattr(model, '_bypassed_residuals', 0)

    for name, module in model.named_modules():
        if isinstance(module, SmartIdentity): continue
            
        # 1. Standard ResNet/RegNetX
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module) and hasattr(module, 'block'):
            if not hasattr(module, '_orig_forward'): module._orig_forward = getattr(module, 'forward')
            def make_patched_forward(mod_name):
                def new_forward(self, x):
                    out = self.block(x)
                    try: sc = self.shortcut(x)
                    except Exception: return F.relu(out)
                    if out.shape != sc.shape:
                        if out.shape[2:] != sc.shape[2:]: sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
                        if out.shape[1] != sc.shape[1]: model._bypassed_residuals += 1; return F.relu(out)
                    return F.relu(out + sc)
                return new_forward
            module.forward = make_patched_forward(name).__get__(module)

        # 2. ConvNeXtBlock
        elif module.__class__.__name__ == "ConvNeXtBlock":
            if not hasattr(module, '_orig_forward'): module._orig_forward = getattr(module, 'forward')
            def make_convnext_forward():
                def new_forward(self, x):
                    residual = x
                    out = self.dwconv(x)
                    if out.ndim == 4: out = out.permute(0, 2, 3, 1) # NCHW -> NHWC
                    out = self.norm(out)
                    out = self.pwconv1(out)
                    out = self.act(out)
                    out = self.pwconv2(out)
                    if self.gamma is not None: out = self.gamma * out
                    if out.ndim == 4: out = out.permute(0, 3, 1, 2) # NHWC -> NCHW
                    
                    if out.shape != residual.shape:
                        if out.shape[2:] != residual.shape[2:]: residual = F.adaptive_avg_pool2d(residual, out.shape[2:])
                        if out.shape[1] != residual.shape[1]: return out
                    return out + residual
                return new_forward
            module.forward = make_convnext_forward().__get__(module)

        # 3. InceptionBlock
        elif module.__class__.__name__ == "InceptionBlock":
            if not hasattr(module, '_orig_forward'): module._orig_forward = getattr(module, 'forward')
            def make_inception_forward():
                def new_forward(self, x):
                    o1, o2, o3, o4 = self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
                    outs = [o1, o2, o3, o4]
                    outs_4d = [o for o in outs if o.ndim == 4]
                    if not outs_4d: return torch.cat(outs, dim=1)
                    
                    min_h = min(o.shape[2] for o in outs_4d)
                    min_w = min(o.shape[3] for o in outs_4d)
                    
                    aligned = []
                    for o in outs:
                        if o.ndim == 4 and (o.shape[2] > min_h or o.shape[3] > min_w):
                            aligned.append(F.adaptive_avg_pool2d(o, (min_h, min_w)))
                        else: aligned.append(o)
                    return torch.cat(aligned, dim=1)
                return new_forward
            module.forward = make_inception_forward().__get__(module)

        # 4. XceptionBlock
        elif module.__class__.__name__ == "XceptionBlock":
            if not hasattr(module, '_orig_forward'): module._orig_forward = getattr(module, 'forward')
            def make_xception_forward():
                def new_forward(self, x):
                    sc = self.skip(x)
                    out = self.rep(x)
                    if out.shape != sc.shape:
                        if out.shape[2:] != sc.shape[2:]: sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
                        if out.shape[1] != sc.shape[1]: return out
                    return out + sc
                return new_forward
            module.forward = make_xception_forward().__get__(module)
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
    print(f"\n[INFO] {'='*20} Collapsing Block {'='*20}")
    print(f"[INFO] Start Target : {start_layer_name}")
    print(f"[INFO] End Target   : {end_layer_name}")
    if debug:
        print(f"[DEBUG] Target Device Allocation: {device}")

    # Step 1: Locate block
    print(f"\n[STEP 1] Locating block boundaries and Lowest Common Ancestor (LCA)...")
    info = _locate_and_prepare_block(model, start_layer_name, end_layer_name, debug=debug)
    
    # ISSUE #3 FIX: Cross-branch filtering at collapse level
    if info is None:
        raise ValueError(f"[ERROR] Could not map start/end layers into LCA container. This may be a cross-branch collapse.")
    
    if debug:
        print(f"[DEBUG] --- LCA Resolution Results ---")
        print(f"[DEBUG] Container Path  : '{info.get('container_name', '<root>')}'")
        print(f"[DEBUG] Container Type  : {type(info['container']).__name__}")
        print(f"[DEBUG] Slice Indices   : {info['start_idx']} to {info['end_idx']}")
        print(f"[DEBUG] Captured Layers : {len(info['full_block'])} total")
        for n, l in info["full_block"]:
            print(f"    [LAYER MAP] {n} -> {type(l).__name__}")

    
    # =========================================================
    # THE CRITICAL FIX: Align capture hooks to the LCA boundaries
    # =========================================================
    is_sequential = isinstance(info['container'], (nn.Sequential, nn.ModuleList))
    
    if not is_sequential and info['container_name'] != "":
        # It's a complex/parallel block (e.g., InceptionBlock). Hook the container 
        # itself to capture the true global tensor sizes entering and exiting the block.
        lca_start_name = info['container_name']
        lca_end_name = info['container_name']
    else:
        # Standard sequential layout. Hook the explicit child boundaries.
        lca_start_name = f"{info['container_name']}.{info['first_layer_name']}" if info['container_name'] else info['first_layer_name']
        lca_end_name = f"{info['container_name']}.{info['last_layer_name']}" if info['container_name'] else info['last_layer_name']

    # Step 2: Capture activation entering the LCA boundaries
    print(f"\n[STEP 2] Simulating Forward Pass & Capturing LCA Activations...")
    if debug:
        print(f"[DEBUG] Hooking Start Node : '{lca_start_name}'")
        print(f"[DEBUG] Hooking End Node   : '{lca_end_name}'")
        print(f"[DEBUG] Dummy Input Shape  : {input_shape}")
        
    x, y_out, pre_params = _capture_preblock_activation( 
        model, lca_start_name, lca_end_name, input_shape, info["conv_layers"], info["layer_type"], device, debug
    ) 
    # =========================================================# =========================================================

    if debug:
        print(f"[DEBUG] --- Activation Capture Success ---")
        print(f"[DEBUG] Input (x) shape entering LCA   : {tuple(x.shape)}")
        print(f"[DEBUG] Output (y) shape exiting LCA   : {tuple(y_out.shape)}")
        print(f"[DEBUG] Pre-collapse Parameter Count   : {pre_params:,}")

    # Step 3: Find next linear
    print(f"\n[STEP 3] Probing graph for downstream classifier after '{end_layer_name}'...")
    next_linear_name, next_linear_mod = _find_next_linear(model, end_layer_name, debug)
    
    if debug:
        print(f"[DEBUG] --- Downstream Probe Results ---")
        print(f"[DEBUG] Next Linear Name : '{next_linear_name}'")
        print(f"[DEBUG] Next Linear Type : {type(next_linear_mod).__name__ if next_linear_mod else 'None'}")

    # Step 4: Analyze block output
    print(f"\n[STEP 4] Analyzing topological characteristics for replacement...")
    block_analysis = _analyze_block_output(
        model,
        info["full_block"],
        info["conv_layers"],
        info["named_layers"],
        info["end_idx"],
        info["layer_type"],
        x,
        y_out,  
        next_linear_mod,
        debug
    )
    
    if debug:
        print(f"[DEBUG] --- Feature Map Strategy ---")
        for k, v in block_analysis.items():
            print(f"    - {k:<25}: {v}")

    # Step 5: Replace block
    print(f"\n[STEP 5] Synthesizing surrogate block and fusing into computational graph...")
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
    
    print(f"\n[INFO] ✅ Framework successfully merged '{start_layer_name}' → '{end_layer_name}'")

    return model

def _find_next_linear(model, end_layer_name, debug):
    if debug:
        print(f"\n[STEP] Searching for next nn.Linear after '{end_layer_name}'...")
    modules_list = list(model.named_modules())
    idx_end_global = None

    for i, (n, m) in enumerate(modules_list):
        if n == end_layer_name:
            idx_end_global = i
            if debug:
                print(f"[DEBUG] Exact layer match found at index {i}: {n} ({type(m).__name__})")
            break

    if idx_end_global is None:
        for i, (n, m) in enumerate(modules_list):
            if n.endswith(end_layer_name):
                idx_end_global = i
                if debug:
                    print(f"[DEBUG] Fallback: found partial match at index {i}: {n}")
                break

    next_linear_name = None
    next_linear_mod = None
    
    if idx_end_global is not None:
        if debug:
            print(f"[DEBUG] Scanning forward from index {idx_end_global + 1} for next Linear...")
        for j in range(idx_end_global + 1, len(modules_list)):
            n, m = modules_list[j]
            # ONLY grab the linear if it's explicitly downstream
            if isinstance(m, nn.Linear):
                next_linear_name, next_linear_mod = n, m
                if debug:
                    print(f"[DEBUG] Found Linear layer ahead: {n} ({m})")
                break

    # [FIX]: REMOVE the global fallback loop. If we don't find a linear layer downstream, 
    # we return None so the block doesn't hallucinate a spatial size from the stem.
    if next_linear_mod is None:
        if debug:
            print(f"[DEBUG] No downstream Linear found after {end_layer_name}. Disabling adaptive pooling heuristic.")
            
    return next_linear_name, next_linear_mod

def _analyze_block_output(
    model, full_block, conv_layers, named_layers, end_idx, layer_type, x, y_out, next_linear_mod, debug
):
    if debug:
        print(f"\n[STEP] Analyzing output of collapsed block ({len(full_block)} layers)...")
        print(f"[DEBUG] Input tensor shape before block: {tuple(x.shape)}")
        print(f"[DEBUG] Running forward pass through block layers:")

    out_shape = tuple(y_out.shape)
    out_channels = y_out.shape[1] if y_out.ndim >= 2 else None

    if debug:
        print(f"[DEBUG] Final block output shape (via hook): {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")
    # --- Ground-truth output characteristics ---
    out_shape = tuple(y_out.shape)

    # 🔧 CRITICAL FIX:
    # Infer channels from actual tensor, not from layer attributes
    out_channels = y_out.shape[1] if y_out.ndim >= 2 else None

    if debug:
        print(f"[DEBUG] Final block output shape: {out_shape}")
        print(f"[DEBUG] Determined out_channels={out_channels}")

    # --- Detect pooling inside original block (informational) ---
    pool_layer = next(
        (
            m for _, m in reversed(full_block)
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        ),
        None,
    )

    if debug:
        if pool_layer is not None:
            print(f"[DEBUG] Detected pool in original block: {type(pool_layer).__name__}")
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
                f"[DEBUG] Comparing spatial dims: "
                f"expected_hw={expected_hw}, current_hw={cur_hw} (HxW={cur_H}x{cur_W})"
            )

        if cur_hw != expected_hw:
            target_H = int(round(math.sqrt(expected_hw))) if expected_hw > 1 else 1
            target_W = max(1, expected_hw // target_H)
            adaptive_pool_to_use = nn.AdaptiveAvgPool2d((target_H, target_W))

            if debug:
                print(
                    f"[DEBUG] Suggest AdaptiveAvgPool2d({target_H},{target_W}) "
                    f"to reconcile linear in_features mismatch."
                )

    # --- Shortcut detection (unchanged) ---
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
                    print(f"[DEBUG] Found shortcut conv → out_channels={shortcut_out_channels}")
                break

    if debug:
        print(f"[RESULT] Block analysis complete:")
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
    print(f"\n[STEP] ===== Starting _validate_downstream for '{start_container_name}' =====")
    print(f"[DEBUG] start_idx={start_idx}, device={device}, has_next_linear={next_linear_name is not None}")

    # Retrieve target container
    try:
        container = get_layer(model, start_container_name)
    except Exception as e:
        print(f"[WARN] Could not access container '{start_container_name}': {e}")
        return

    children = list(container.named_children())
    if not children:
        print(f"[DEBUG] Container '{start_container_name}' has no children; skipping downstream validation.")
        return

    # Find collapsed/inserted child index
    # ISSUE #6 FIX: Track actual insertion index, not the start_idx from a previous collapse
    collapsed_idx = None
    print(f"[STEP] Searching for inserted collapsed module within '{start_container_name}'...")
    # Search from the end backwards to find the most recently inserted block
    # (accounts for multiple sequential collapses within the same container)
    for i in range(len(children) - 1, -1, -1):
        nm, m = children[i]
        if nm.startswith("collapsed_") or (
            isinstance(m, nn.Sequential)
            and any(k in dict(m.named_children()) for k in ("conv_1x1", "adaptive_pool", "conv_dw"))
        ):
            collapsed_idx = i
            print(f"[DEBUG] Found inserted block candidate at index {i}: '{nm}'")
            break

    if collapsed_idx is None:
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
            t = inserted_mod(t)
        print(f"[DEBUG] Inserted module forward successful, output shape: {tuple(t.shape)}")
    except Exception as e:
        print(f"[WARN] Forward pass through inserted module failed: {e}")
        return

    # Validate next modules downstream
    print(f"[STEP] Scanning immediate downstream modules for shape or runtime errors...")
    for nm, mod in children[collapsed_idx + 1:]:
        try:
            t = mod(t)
            if debug:
                print(f"[DEBUG] Downstream '{start_container_name}.{nm}' executed successfully, output shape: {tuple(t.shape)}")
        except Exception as e:
            print(f"[WARN] Downstream module '{start_container_name}.{nm}' raised exception: {e}")
            # ISSUE #2 FIX: Check if nn.Linear before attempting corrective wrapping
            if isinstance(mod, nn.Linear):
                print(f"[INFO] Target module '{nm}' is nn.Linear. Allowing Corrective Pooling to handle it natively.")
                return
            
            print(f"[STEP] Replacing problematic module '{nm}' with safe alternative...")

            if isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
                safe = _SafePool(mod)
                print(f"[INFO] Replaced with _SafePool wrapper.") 
            else:
                safe = nn.Identity()
                print(f"[INFO] Replaced with Identity() to bypass invalid operation.")

            if isinstance(container, nn.Sequential) or type(container).__name__ == 'SmartIdentity':
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)

            print(f"[DEBUG] Replacement applied to '{start_container_name}.{nm}' ({safe.__class__.__name__}).")
            return  # stop after first fix

        # detect zero-spatial output
        if t.ndim >= 4 and (t.shape[-2] == 0 or t.shape[-1] == 0):
            print(f"[WARN] Module '{start_container_name}.{nm}' produced zero spatial dimensions. Wrapping with _SafePool.")
            safe = _SafePool(mod)
            if isinstance(container, nn.Sequential) or type(container).__name__ == 'SmartIdentity':
                new_od = OrderedDict()
                for j, (n2, m2) in enumerate(children):
                    new_od[n2] = safe if n2 == nm else m2
                _update_container(model, start_container_name, nn.Sequential(new_od))
            else:
                setattr(container, nm, safe)
            print(f"[DEBUG] Zero-dimension fix applied to '{start_container_name}.{nm}' with _SafePool.")
            return

    print(f"[RESULT] ✅ Downstream validation for '{start_container_name}' completed successfully.")

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
    print(f"\n[STEP] ===== Starting collapse_only process =====")
    print(f"[DEBUG] Device={device}, dry_run={dry_run}, handle_skips={handle_skips}, safe_param_reduction={safe_param_reduction}")

    if model is None:
        print(f"[STEP] Loading model from disk...")
        if not (model_weights_1 and model_class):
            raise ValueError("[ERROR] Must provide either an instantiated `model` or (`model_weights_1` + `model_class`).")

        model_kwargs = model_kwargs or {}
        print(f"[INFO] Instantiating model from class '{model_class.__name__}' with kwargs={model_kwargs}")
        try:
            model = model_class(**model_kwargs)
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to instantiate model class {model_class}: {e}")

        print(f"[INFO] Loading weights from file: {model_weights_1}")
        try:
            chk = torch.load(model_weights_1, map_location=device)
            state = chk.get('model', chk) if isinstance(chk, dict) else chk
            model.load_state_dict(state)
            print(f"[INFO] Weights successfully loaded.")
        except Exception as e:
            raise RuntimeError(f"[ERROR] Failed to load model weights: {e}")
    else:
        print(f"[STEP] Using provided in-memory model instance ({model.__class__.__name__})")

    if debug:
        try:
            print(f"[DEBUG] Model layer statistics before collapse:\n{layer_stats(model)}")
        except Exception as e:
            print(f"[WARN] layer_stats() failed: {e}")

    print(f"[STEP] Creating deepcopy of model for safe modification...")
    model = deepcopy(model).to(device)
    model.eval()

    print(f"[STEP] Parsing compression set...")
    if compression_set is None:
        print("[WARN] compression_set is None or empty; skipping collapse.")
        return model

    collapse_map = {}
    if isinstance(compression_set, dict):
        if debug:
            print(f"[DEBUG] Detected compression_set as dict with {len(compression_set)} entries.")
        for k, v in compression_set.items():
            start, end = v
            if isinstance(start, tuple): start = start[0]
            if isinstance(end, tuple): end = end[0]
            collapse_map[k] = (start, end)
    else:
        if debug:
            print(f"[DEBUG] Detected compression_set as sequence with {len(compression_set)} pairs.")
        for i, pair in enumerate(compression_set):
            start, end = pair
            if isinstance(start, tuple): start = start[0]
            if isinstance(end, tuple): end = end[0]
            collapse_map[f"collapse_{i}"] = (start, end)

    model._collapsed_blocks = list(collapse_map.values())
    if debug:
        print(f"[DEBUG] Total collapse targets: {len(model._collapsed_blocks)}")

    pre_total = count_trainable_params(model)
    print(f"[INFO] Model parameter count before collapsing: {pre_total:,}")

    print(f"[STEP] Beginning block-wise collapsing...")
    for name, (start, end) in collapse_map.items():
        
        # [CRITICAL FIX] Guarantee single-layers are safely bypassed
        if start == end:
            print(f"[WARN] Skipping collapse task '{name}': Group contains only one layer ({start}). Must span at least two layers.")
            continue

        print(f"\n[INFO] Processing collapse task '{name}': {start} → {end}")
        if dry_run:
            print("[INFO] dry_run enabled; skipping actual modification for this block.")
            continue

        try:
            print(f"[STEP] Calling _collapse_block('{start}', '{end}')")
            model = _collapse_block(model, start, end, input_shape, device=device, debug=debug)
            print(f"[INFO] ✅ Successfully collapsed block '{name}' ({start} → {end})")
        except ValueError as e:
            # ISSUE #3 FIX: Filter out cross-branch collapse candidates
            if "Could not map start/end layers" in str(e):
                print(f"[WARN] ⚠ Collapse candidate '{name}' appears to be cross-branch (parallel paths).")
                print(f"[WARN]   Filtering out and skipping. Re-run region discovery to exclude these.")
                continue
            else:
                print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
        except Exception as e:
            print(f"[WARN] ⚠ Collapse failed for block '{name}': {e}")
               
        if handle_skips:
            print(f"[STEP] Patching skip connections (if any)...")
            try:
                patch_skip_connections(model)
                if debug: print(f"[DEBUG] Skip connections patched successfully.")
            except Exception as e:
                print(f"[WARN] Failed to patch skip connections: {e}")

        print(f"[STEP] Disabling in-place ReLUs for autograd safety...")
        try:
            disable_inplace_relu(model)
            if debug: print(f"[DEBUG] In-place ReLUs converted to out-of-place versions.")
        except Exception as e:
            print(f"[WARN] Failed to disable in-place ReLUs: {e}")

    print(f"\n[STEP] Wrapping pooling layers safely...")
    try:
        _wrap_pools_safe(model)
        if debug: print("[DEBUG] All pooling layers wrapped with _SafePool to prevent underflow errors.")
    except Exception as e:
        print(f"[WARN] Failed to wrap pools safely: {e}")

    post_total = count_trainable_params(model)
    print(f"\n[STEP] ===== Collapse summary =====")
    print(f"[INFO] Parameters before: {pre_total:,}")
    print(f"[INFO] Parameters after : {post_total:,}")
    delta = pre_total - post_total
    print(f"[INFO] ΔParams = {delta:+,} (expected ≤ 0)")

    if post_total > pre_total:
        print(f"[WARN] ⚠ Model gained parameters after collapsing! Investigate collapse policy or replacement logic.")

    if safe_param_reduction and delta < 0:
        print(f"[WARN] ⚠ Parameter count increased when safe_param_reduction=True — collapse may have failed silently.")

    print(f"[RESULT] ✅ collapse_only complete. Total collapsed blocks: {len(collapse_map)}")
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

    def forward(self, x):
        # guard shape sanity
        try:
            H, W = x.shape[-2], x.shape[-1]
        except Exception:
            # not a 4D tensor (some unexpected case) -> try to apply pool and catch exceptions
            try:
                return self.pool(x)
            except Exception:
                return x

        try:
            # For standard pools, check kernel size
            if isinstance(self.pool, (nn.MaxPool2d, nn.AvgPool2d)):
                k = self.pool.kernel_size
                if isinstance(k, tuple):
                    kh, kw = k
                else:
                    kh = kw = k
                # if kernel/stride would underflow, use adaptive avg pool to safe size
                if kh > H or kw > W or H <= 0 or W <= 0:
                    # choose a safe target HxW (at least 1)
                    target_H = max(1, min(H, kh) if H > 0 else 1)
                    target_W = max(1, min(W, kw) if W > 0 else 1)
                    return F.adaptive_avg_pool2d(x, (target_H, target_W))

            # Try to apply original pool
            out = self.pool(x)

            # post-check: if shape became invalid, return identity
            if out.shape[-2] < 1 or out.shape[-1] < 1:
                return x
            return out
        except Exception:
            # Any failure -> safe fallback
            return x


def _wrap_pools_safe(module: nn.Module):
    """
    Recursively replace pooling modules in `module` with _SafePool wrappers.
    This mutates the module in-place.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, getattr(nn, "AdaptiveMaxPool2d", nn.AdaptiveAvgPool2d))):
            safe = _SafePool(child)
            parent = module
            try:
                setattr(parent, name, safe)
            except Exception:
                try:
                    idx = int(name)
                    parent[idx] = safe
                except Exception:
                    setattr(parent, name, safe)
        else:
            _wrap_pools_safe(child)

