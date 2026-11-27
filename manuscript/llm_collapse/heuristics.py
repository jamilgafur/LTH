# heuristics.py -- the four collapse heuristics of the CNN study
# (../Tranfer/transfer.py), adapted to decoder layers, plus the
# activation-variance grouping that *identifies* the collapse spans.
#
# CNN study (there): per-layer activation mean/variance over a baseline pass,
#   Relative Variance = median block variance / global median layer variance
#   Block Redundancy  = cosine(block input, block output)  (feature similarity)
#   Prediction Shift  = KL after a virtual bypass (block output <- block input)
#   Collapse Score    = sim / (rel_var * (1 + kl))   (higher = safer)
# Contiguous blocks whose feature representations are similar are collapsed --
# ALL of them before one retraining phase. Here the "features" are decoder
# hidden states (B, T, D), a "layer" is a decoder block, and the same four
# metrics decide which contiguous spans carry similar representations and can
# be merged into one.
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from pyPrune.collapseMethods import find_decoder_stack


# ------------------------------------------------------------- baseline pass
@torch.no_grad()
def layer_activation_stats(model, tokenizer, texts: Sequence[str],
                           max_length: int = 128,
                           stack_name=None) -> Dict[str, List[float]]:
    """Per-decoder-layer mean activation and variance of the output hidden
    states over the calibration texts (analogue of run_baseline_pass: per-channel
    variance over the token dims, averaged over texts)."""
    device = next(model.parameters()).device
    _, stack = find_decoder_stack(model, stack_name)
    vars_, means_, counts = [0.0] * len(stack), [0.0] * len(stack), [0] * len(stack)

    def make_hook(i):
        def hook(module, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            vars_[i] += y.var(dim=(0, 1)).mean().item()
            means_[i] += y.mean().item()
            counts[i] += 1
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(stack)]
    try:
        for text in texts:
            ids = tokenizer.encode(text)[:max_length]
            if len(ids) < 2:
                continue
            model(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
    finally:
        for h in handles:
            h.remove()
    if not any(counts):
        raise ValueError("no calibration texts were evaluable")
    return {"var": [v / c for v, c in zip(vars_, counts)],
            "mean": [m / c for m, c in zip(means_, counts)]}


def group_by_variance_similarity(var: Sequence[float], tol: float = 0.25
                                 ) -> List[Tuple[int, int]]:
    """Contiguous runs of layers with similar activation variance: neighbours
    join the same run while their variance ratio <= 1 + tol. Runs of >= 2
    layers are the collapse candidates (contiguous, like the CNN blocks)."""
    spans, start = [], 0
    for i in range(1, len(var)):
        lo, hi = sorted((var[i - 1], var[i]))
        if hi > (1.0 + tol) * (lo + 1e-12):
            if i - start >= 2:
                spans.append((start, i - 1))
            start = i
    if len(var) - start >= 2:
        spans.append((start, len(var) - 1))
    return spans


# ------------------------------------------------------- per-span heuristics
def _span_io(model, tokenizer, texts, span, max_length, bypass: bool = False,
             stack_name=None):
    """Shared plumbing: run the texts through the model and collect the span's
    input/output hidden states. With bypass=True the span is skipped (virtual
    bypass: span output <- span input) and the logits are returned too."""
    device = next(model.parameters()).device
    _, stack = find_decoder_stack(model, stack_name)
    start, end = span
    cache, io_pairs, base_logits, bypass_logits = {}, [], [], []
    state = {"bypass": False}

    def start_hook(module, inp, out):
        cache["x"] = inp[0].detach()

    def end_hook(module, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        io_pairs.append((cache["x"], y.detach()))
        if not state["bypass"]:
            return None
        x = cache["x"]
        return (x,) + out[1:] if isinstance(out, tuple) else x

    h1 = stack[start].register_forward_hook(start_hook)
    h2 = stack[end].register_forward_hook(end_hook)
    try:
        for text in texts:
            ids = tokenizer.encode(text)[:max_length]
            if len(ids) < 2:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            if bypass:
                base_logits.append(model(input_ids=inp).logits.detach())
                state["bypass"] = True
                bypass_logits.append(model(input_ids=inp).logits.detach())
                state["bypass"] = False
            else:
                model(input_ids=inp)
    finally:
        h1.remove()
        h2.remove()
    return io_pairs, base_logits, bypass_logits


@torch.no_grad()
def span_redundancy(model, tokenizer, texts, span, max_length: int = 128,
                    stack_name=None) -> float:
    """Block Redundancy: cosine(span input, span output) per token, averaged.
    Near 1.0 = the span barely transforms the representation (safe to merge)."""
    io_pairs, _, _ = _span_io(model, tokenizer, texts, span, max_length,
                              stack_name=stack_name)
    sims = [F.cosine_similarity(x.flatten(0, 1), y.flatten(0, 1), dim=1).mean().item()
            for x, y in io_pairs if x.shape == y.shape]
    return sum(sims) / len(sims) if sims else 0.0


@torch.no_grad()
def virtual_bypass_kl(model, tokenizer, texts, span, max_length: int = 128,
                     stack_name=None) -> float:
    """Prediction Shift: skip the span (output <- input) and measure the mean
    per-token KL(base || bypassed) of the next-token distributions."""
    _, base, bypassed = _span_io(model, tokenizer, texts, span, max_length, bypass=True,
                                stack_name=stack_name)
    kls = []
    for b, q in zip(base, bypassed):
        b, q = b.reshape(-1, b.shape[-1]), q.reshape(-1, q.shape[-1])
        kls.append(F.kl_div(F.log_softmax(q, dim=-1), F.softmax(b, dim=-1),
                            reduction="batchmean").item())
    return sum(kls) / len(kls) if kls else 50.0


def collapse_score(redundancy: float, rel_var: float, kl: float) -> float:
    """Composite Collapse Score (higher = safer to collapse), same formula as
    the CNN study."""
    return redundancy / (max(rel_var, 1e-8) * (1.0 + kl))


def span_mergeable(model, span, stack_name=None) -> bool:
    """True when the span's layers can be mean-merged: the first layer owns
    every parameter of the later ones with matching shapes. Spans crossing a
    structural boundary -- e.g. MiMo/DeepSeek's dense first layers into MoE
    layers -- are not mergeable and must be skipped, not forced."""
    _, stack = find_decoder_stack(model, stack_name)
    states = [stack[i].state_dict() for i in range(span[0], span[1] + 1)]
    keys0 = set(states[0])
    for s in states[1:]:
        if set(s) - keys0:
            return False
        if any(tuple(s[k].shape) != tuple(v.shape)
               for k, v in states[0].items() if k in s):
            return False
    return True


def analyze_spans(model, tokenizer, texts: Sequence[str], tol: float = 0.25,
                  max_length: int = 128, stack_name=None) -> dict:
    """Full identification pass: activation stats -> variance-similar groups ->
    per-span Relative Variance / Block Redundancy / Prediction Shift /
    Collapse Score. Collapsing every identified span is the caller's job."""
    stats = layer_activation_stats(model, tokenizer, texts, max_length, stack_name)
    var = stats["var"]
    global_median_var = sorted(var)[len(var) // 2]
    spans = []
    for start, end in group_by_variance_similarity(var, tol):
        block_var = sorted(var[start:end + 1])[(end - start) // 2]
        rel_var = block_var / global_median_var if global_median_var > 0 else 1.0
        redundancy = span_redundancy(model, tokenizer, texts, (start, end), max_length,
                                          stack_name=stack_name)
        kl = virtual_bypass_kl(model, tokenizer, texts, (start, end), max_length,
                                   stack_name=stack_name)
        spans.append({"span": [start, end], "rel_var": rel_var,
                      "redundancy": redundancy, "kl": kl,
                      "score": collapse_score(redundancy, rel_var, kl),
                      "mergeable": span_mergeable(model, (start, end), stack_name)})
    return {"global_median_var": global_median_var, "tol": tol,
            "layers": stats, "spans": spans}
