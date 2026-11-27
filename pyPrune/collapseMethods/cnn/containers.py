# containers.py -- editing a model's module tree: layer lookup, container
# path handling, and replacing a contiguous slice of children with one
# collapsed module (leaving name-stable SmartIdentity fillers behind).
import logging
from collections import OrderedDict
from typing import Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)


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


class SmartIdentity(nn.Module):
    """A robust Identity replacement that safely absorbs nested attribute calls."""
    def forward(self, x, *args, **kwargs):
        return x
    def __getattr__(self, name):
        # Prevent intercepting PyTorch internal methods
        if name.startswith('_'):
            return super().__getattr__(name)
        return self # Return self to absorb chained calls like .inception_4a(x)

def _replace_layers(named_layers, start_idx, end_idx, replacement, start_name=None, end_name=None):
    """Replace a slice of layers, using SmartIdentity for subsumed layers."""
    new_layers = OrderedDict()

    for i, (name, mod) in enumerate(named_layers):
        if i < start_idx or i > end_idx:
            new_layers[name] = mod
        elif i == start_idx:
            new_layers[name] = replacement
        else:
            # FIX: Use SmartIdentity so hardcoded forward passes don't crash
            new_layers[name] = SmartIdentity()

    return nn.Sequential(new_layers)


def _update_container(model: nn.Module, container_path: str, new_container: nn.Module):
    """Replace the module at `container_path` in `model` with `new_container`."""
    
    # FIX: Handle Root Container case
    if container_path == "":
        logger.debug("Updating root container in-place (Preserving attribute names).")
        
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
