"""Skip-connection forward patching for collapsed architectures."""

import torch.nn as nn
import torch.nn.functional as F

from .identity import SmartIdentity

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
