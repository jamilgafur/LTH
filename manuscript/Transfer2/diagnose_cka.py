"""CKA diagnostic: why are some (model, dataset) universes' CKA values ~0?

For a given config, this prints, per variant (before / after / collapsed):
  - WHICH module extract_features replaces with Identity (asymmetry = bug)
  - feature matrix shape, global std, and stable rank
and computes reference CKAs that separate extraction bugs from genuine drift:
  - CKA(after, after)          must be ~1.0 (pipeline sanity)
  - CKA(before, after)         same-universe training drift reference
  - CKA(after, collapsed)      the number reported in the CSV

If CKA(before, after) is also ~0 while strip names match, extraction is fine
and the universes really decorrelate; if strip names differ, extract_features
is landing on different layers per variant.

Usage: python diagnose_cka.py --model RegNetX_400MF --dataset Cifar10 --pre 300 --post 100
"""
import argparse

import torch
import torch.nn as nn

from utils import (find_experiment_checkpoints, find_matching_checkpoint,
                   initialize_architecture, load_weights, load_dataset, compare_CKA)


def strip_with_audit(model, tag):
    """Same stripping logic as utils.extract_features, but reports the module."""
    if hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
        print(f"[AUDIT] {tag}: replaced '{tag}.fc' (nn.Linear) with Identity")
        model.fc = nn.Identity()
        return
    if hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            print(f"[AUDIT] {tag}: replaced '{tag}.classifier[-1]' ({type(model.classifier[-1]).__name__}) with Identity")
            model.classifier[-1] = nn.Identity()
            return
        if isinstance(model.classifier, nn.Linear):
            print(f"[AUDIT] {tag}: replaced '{tag}.classifier' (nn.Linear) with Identity")
            model.classifier = nn.Identity()
            return
    for name, mod in reversed(list(model.named_modules())):
        if isinstance(mod, nn.Linear):
            print(f"[AUDIT] {tag}: FALLBACK replaced last Linear '{name}' with Identity")
            parent_name, sub = name.rsplit('.', 1) if '.' in name else ('', name)
            parent = model if not parent_name else dict(model.named_modules())[parent_name]
            if isinstance(parent, nn.Sequential):
                parent[int(sub)] = nn.Identity()
            else:
                setattr(parent, sub, nn.Identity())
            return
    print(f"[AUDIT] {tag}: NO Linear found — features will be raw forward output")


def features(model, loader, device, tag, batches=5):
    from copy import deepcopy
    m = deepcopy(model)
    strip_with_audit(m, tag)
    m.eval().to(device)
    outs = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= batches:
                break
            o = m(x.to(device))
            if o.ndim > 2:
                o = o.flatten(start_dim=1)
            outs.append(o.detach().cpu())
    f = torch.cat(outs).to(device)
    f64 = f.double()
    svals = torch.linalg.svdvals(f64 - f64.mean(0))
    rank = ((svals.sum()) ** 2 / (svals ** 2).sum()).item() if (svals ** 2).sum() > 0 else 0.0
    print(f"[AUDIT] {tag}: features shape={tuple(f.shape)} std={f.std().item():.3f} stable_rank={rank:.1f}")
    m.to("cpu")
    return f


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--pre", type=int, required=True)
    p.add_argument("--post", type=int, required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = find_experiment_checkpoints(a.model, a.dataset, a.pre, a.post, quant=False)
    assert ck["before"], "no before-checkpoint found"
    before_path = ck["before"][0]
    after_path = find_matching_checkpoint(before_path, ck["after"])
    collapsed_path = find_matching_checkpoint(before_path, ck["collapsed"])

    _, test_loader, _, _, _ = load_dataset(a.dataset, a.model)

    base, _, _, _ = initialize_architecture(a.model, a.dataset)
    before = load_weights(base, before_path, device)
    base, _, _, _ = initialize_architecture(a.model, a.dataset)
    after = load_weights(base, after_path, device) if after_path else None
    collapsed = torch.load(collapsed_path, map_location=device, weights_only=False)

    f_before = features(before, test_loader, device, "before")
    torch.cuda.empty_cache() if device.type == "cuda" else None
    f_after = features(after, test_loader, device, "after") if after else None
    f_coll = features(collapsed, test_loader, device, "collapsed")

    if f_after is not None:
        print(f"CKA(after, after)      = {compare_CKA(f_after, f_after):.4f}  (sanity, expect 1.0)")
        print(f"CKA(before, after)     = {compare_CKA(f_before, f_after):.4f}  (drift reference)")
    print(f"CKA(before, collapsed) = {compare_CKA(f_before, f_coll):.4f}  (CSV reports after vs collapsed)")
    if f_after is not None:
        print(f"CKA(after, collapsed)  = {compare_CKA(f_after, f_coll):.4f}  (the CSV number)")


if __name__ == "__main__":
    main()
