"""Pass-through identity and safe-pooling wrappers used by the collapse pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class _SafePool(nn.Module):
    """
    Wrapper that attempts to apply the wrapped pooling module. If input
    spatial dimensions are too small, it logs a clear diagnostic warning
    and gracefully scales via AdaptiveAvgPool2d instead of returning zero tensors.
    """
    def __init__(self, pool_module: nn.Module):
        super().__init__()
        self.pool = pool_module
        self._warned = set()

    def _warn_once(self, key, msg):
        """Log a diagnostic once per wrapped pool instance. Hot training loops
        hit these paths every batch; repeats are suppressed to keep logs sane."""
        if key not in self._warned:
            self._warned.add(key)
            print(f"{msg} [{type(self.pool).__name__}; further occurrences suppressed]")

    def forward(self, x):
        try:
            H, W = x.shape[-2], x.shape[-1]
        except Exception:
            try:
                return self.pool(x)
            except Exception as e:
                self._warn_once("nontensor", f"[WARN] _SafePool non-tensor shape fallback triggered: {e}")
                return x

        try:
            if isinstance(self.pool, (nn.MaxPool2d, nn.AvgPool2d)):
                k = self.pool.kernel_size
                kh, kw = (k, k) if isinstance(k, int) else k
                if kh > H or kw > W or H <= 0 or W <= 0:
                    target_H = max(1, min(H, kh) if H > 0 else 1)
                    target_W = max(1, min(W, kw) if W > 0 else 1)
                    self._warn_once("kernel", f"[WARN] Pool kernel ({kh}x{kw}) exceeds input spatial size ({H}x{W}). Adapting to ({target_H}x{target_W}).")
                    return F.adaptive_avg_pool2d(x, (target_H, target_W))

            out = self.pool(x)
            if out.shape[-2] < 1 or out.shape[-1] < 1:
                self._warn_once("collapse", "[WARN] Pool output spatial dimension collapsed to < 1. Returning input tensor.")
                return x
            return out
        except Exception as e:
            self._warn_once("runtime", f"[WARN] _SafePool encountered runtime exception ({e}). Bypassing pool safely.")
            return x


def wrap_pools_safe(module: nn.Module):
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
            wrap_pools_safe(child)


# Backwards-compatible alias (was private in the standalone module).
_wrap_pools_safe = wrap_pools_safe
