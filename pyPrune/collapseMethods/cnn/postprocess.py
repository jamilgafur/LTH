# postprocess.py -- hardening the model after surgery: out-of-place ReLUs,
# residual-block forward patches that shape-align shortcuts, and safe
# wrappers around pooling layers that can underflow.
import logging

import torch.nn as nn
import torch.nn.functional as F

from .containers import SmartIdentity

logger = logging.getLogger(__name__)


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
        logger.info(f"Replaced {replaced} in-place ReLU(s) with out-of-place variants.")


def patch_skip_connections(model: nn.Module):
    """
    Patches module forwards to robustly handle residual connections.
    If the collapsed block output shape differs from the shortcut, 
    this attempts to spatially align the shortcut instead of severing the connection.
    """
    model._bypassed_residuals = 0

    for name, module in model.named_modules():
        # SmartIdentity absorbs ANY attribute access (returns self), so the
        # shortcut/block check below matches it and would recurse forever
        if isinstance(module, SmartIdentity):
            continue
        # Look for ResNet/ConvNeXt style blocks with 'shortcut' and 'block'
        if hasattr(module, 'shortcut') and isinstance(module.shortcut, nn.Module) and hasattr(module, 'block'):
            
            # Save original forward if not already patched
            if not hasattr(module, '_orig_forward'):
                module._orig_forward = getattr(module, 'forward')

            def make_patched_forward(mod_name):
                def new_forward(self, x):
                    # 1. Run the (potentially collapsed) main block
                    out = self.block(x)
                    
                    # 2. Run the shortcut
                    try:
                        sc = self.shortcut(x)
                    except Exception:
                        # Fallback if shortcut itself fails
                        return F.relu(out)

                    # 3. CRITICAL FIX: Align shapes instead of giving up
                    if out.shape != sc.shape:
                        # Case A: Spatial Mismatch (e.g., 16x16 vs 32x32)
                        # We use adaptive pooling on the SHORTCUT to match the OUTPUT
                        if out.shape[2:] != sc.shape[2:]:
                            sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
                        
                        # Case B: Channel Mismatch
                        # If channels still don't match after spatial fix, we unfortunately
                        # cannot add them without a learned 1x1 conv. We must bypass.
                        # (However, collapse.py usually preserves channel counts, so this is rare)
                        if out.shape[1] != sc.shape[1]:
                            model._bypassed_residuals += 1
                            return F.relu(out)

                    # 4. Add residual
                    return F.relu(out + sc)
                return new_forward

            # Apply the patch
            module.forward = make_patched_forward(name).__get__(module)
            logger.info(f"Patched residual block forward (with spatial align): {name}")


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
