"""Skip-connection forward patching for collapsed architectures.

Model-agnostic: patch strategies are selected by *structural attribute
signatures* (never class names), so any present or future block whose
attribute layout matches a known topology is handled automatically. Custom-
forward modules whose internals were excised by a collapse (i.e. contain
SmartIdentity descendants) but match no known signature receive the
``_SafeModuleForward`` fallback: try the original forward, return the input
identity on failure, so downstream tensor flow is never corrupted.

All patch objects are picklable top-level classes, which keeps whole-model
``torch.save(model)`` working after patching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .identity import SmartIdentity


class _ResNetForwardPatch:
    """Residual-add block (attrs: block + shortcut). Runs both paths,
    aligns spatial dims, and bypasses the add on channel mismatch."""

    def __init__(self, module, root=None):
        self.module = module
        self.root = root if root is not None else module

    def __call__(self, x):
        mod = self.module
        out = mod.block(x)
        try:
            sc = mod.shortcut(x)
        except Exception:
            return F.relu(out)
        if out.shape != sc.shape:
            if out.shape[2:] != sc.shape[2:]:
                sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
            if out.shape[1] != sc.shape[1]:
                self.root._bypassed_residuals += 1
                return F.relu(out)
        return F.relu(out + sc)


class _ConvNeXtForwardPatch:
    """ConvNeXt-style block (attrs: dwconv + pwconv1 + pwconv2). Depthwise
    conv -> NHWC pointwise MLP with optional norm/activation/gamma, then a
    shape-repaired residual add."""

    def __init__(self, module, root=None):
        self.module = module

    def __call__(self, x):
        mod = self.module
        residual = x
        out = mod.dwconv(x)
        if out.ndim == 4:
            out = out.permute(0, 2, 3, 1)  # NCHW -> NHWC
        norm = getattr(mod, "norm", None)
        if norm is not None:
            out = norm(out)
        out = mod.pwconv1(out)
        act = getattr(mod, "act", None)
        if act is not None:
            out = act(out)
        out = mod.pwconv2(out)
        gamma = getattr(mod, "gamma", None)
        if gamma is not None:
            out = gamma * out
        if out.ndim == 4:
            out = out.permute(0, 3, 1, 2)  # NHWC -> NCHW

        if out.shape != residual.shape:
            if out.shape[2:] != residual.shape[2:]:
                residual = F.adaptive_avg_pool2d(residual, out.shape[2:])
            if out.shape[1] != residual.shape[1]:
                return out
        return out + residual


class _InceptionForwardPatch:
    """Inception-style concat block (attrs: branch1..branch4). Runs all
    branches, aligns 4D outputs to the minimum HxW, concatenates on dim 1."""

    def __init__(self, module, root=None):
        self.module = module

    def __call__(self, x):
        mod = self.module
        outs = [mod.branch1(x), mod.branch2(x), mod.branch3(x), mod.branch4(x)]
        outs_4d = [o for o in outs if o.ndim == 4]
        if not outs_4d:
            return torch.cat(outs, dim=1)

        min_h = min(o.shape[2] for o in outs_4d)
        min_w = min(o.shape[3] for o in outs_4d)

        aligned = []
        for o in outs:
            if o.ndim == 4 and (o.shape[2] > min_h or o.shape[3] > min_w):
                aligned.append(F.adaptive_avg_pool2d(o, (min_h, min_w)))
            else:
                aligned.append(o)
        return torch.cat(aligned, dim=1)


class _XceptionForwardPatch:
    """Xception-style block (attrs: skip + rep). Residual add with shape
    repair; bypasses the add on channel mismatch."""

    def __init__(self, module, root=None):
        self.module = module

    def __call__(self, x):
        mod = self.module
        sc = mod.skip(x)
        out = mod.rep(x)
        if out.shape != sc.shape:
            if out.shape[2:] != sc.shape[2:]:
                sc = F.adaptive_avg_pool2d(sc, out.shape[2:])
            if out.shape[1] != sc.shape[1]:
                return out
        return out + sc


class _SafeModuleForward:
    """Generic fallback for SmartIdentity-tainted custom-forward modules with
    no known structural signature: try the original forward, return the
    input identity on failure so downstream flow is never corrupted."""

    def __init__(self, module, root=None):
        self.module = module

    def __call__(self, x, *args, **kwargs):
        try:
            return self.module._orig_forward(x, *args, **kwargs)
        except Exception:
            return x


# ---------------------------------------------------------------------------
# Structural signature registry (attribute layout, never class names).
# First matching predicate wins; ordering mirrors the legacy chain.
# ---------------------------------------------------------------------------

def _is_resnet(m):
    return hasattr(m, 'block') and isinstance(getattr(m, 'shortcut', None), nn.Module)


def _is_convnext(m):
    return all(hasattr(m, a) for a in ('dwconv', 'pwconv1', 'pwconv2'))


def _is_inception(m):
    return all(hasattr(m, a) for a in ('branch1', 'branch2', 'branch3', 'branch4'))


def _is_xception(m):
    return all(hasattr(m, a) for a in ('skip', 'rep'))


_SIGNATURES = (
    (_is_resnet, _ResNetForwardPatch),
    (_is_convnext, _ConvNeXtForwardPatch),
    (_is_inception, _InceptionForwardPatch),
    (_is_xception, _XceptionForwardPatch),
)


def _match_signature(module):
    for predicate, patch_cls in _SIGNATURES:
        if predicate(module):
            return patch_cls
    return None


def _has_custom_forward(module):
    return type(module).forward is not nn.Module.forward


def _is_tainted(module):
    """True when the module's own internals were excised by a collapse
    (contains SmartIdentity descendants), meaning its original forward may
    no longer be executable as written."""
    return any(isinstance(m, SmartIdentity) for m in module.modules())


def patch_skip_connections(model: nn.Module):
    model._bypassed_residuals = getattr(model, '_bypassed_residuals', 0)

    for name, module in model.named_modules():
        if isinstance(module, SmartIdentity):
            continue
        if hasattr(module, '_orig_forward'):
            continue  # already patched; idempotent

        # 1. Known structural topologies get their repair patch.
        patch_cls = _match_signature(module)
        if patch_cls is not None:
            module._orig_forward = getattr(module, 'forward')
            module.forward = patch_cls(module, root=model)
            continue

        # 2. Generic fallback: custom-forward module whose internals were
        #    excised and which matches no known signature. Never applied to
        #    the root model (name == "") or plain containers.
        if name == "" or isinstance(module, nn.Sequential):
            continue
        if _has_custom_forward(module) and _is_tainted(module):
            module._orig_forward = getattr(module, 'forward')
            module.forward = _SafeModuleForward(module, root=model)
