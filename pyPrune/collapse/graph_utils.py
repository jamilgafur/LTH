"""Graph traversal, activation-capture, and model-statistics helpers."""

import traceback
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

def _shape_str(tensor_like) -> str:
    if isinstance(tensor_like, torch.Tensor):
        return str(tuple(tensor_like.shape))
    return str(type(tensor_like).__name__)


def _log_exception(context: str, exc: Exception, debug: bool = False, extra: Optional[Dict[str, Any]] = None):
    print(f"[ERROR] {context}: {exc.__class__.__name__}: {exc}")
    if extra:
        for k, v in extra.items():
            print(f"[ERROR]   {k}: {v}")
    if debug:
        print("[DEBUG] Traceback:")
        print(traceback.format_exc().rstrip())

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

def execution_order(model: nn.Module, input_shape: Tuple[int, ...], device='cpu') -> Dict[str, int]:
    """Probe the model with a dummy input and record the order in which modules
    actually fire during the forward pass.

    Returns a dict mapping module name -> execution index (first-fire order for
    modules executed more than once). This reflects true graph/topological order
    rather than depth-first `named_modules()` order, so it is safe to use for
    downstream layer discovery on branchy/custom architectures.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(input_shape).to(device)

    order: Dict[str, int] = {}
    counter = [0]

    def make_hook(name):
        def hook(module, inp, out):
            if name not in order:
                order[name] = counter[0]
                counter[0] += 1
        return hook

    hooks = []
    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(dummy_input)
    finally:
        for h in hooks:
            h.remove()

    return order

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


def count_zeros(tensor): 
    return torch.sum(tensor == 0).item()
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def layer_stats(model):
    print("\nLayer-wise zero parameter stats:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            # print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")

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
