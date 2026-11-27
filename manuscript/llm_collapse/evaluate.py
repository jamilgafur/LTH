# evaluate.py -- before/after metrics for the collapse study.
# Perplexity over held-out corpus windows, per-layer relevance (which
# contiguous span to collapse), the parameter footprint for the paper table,
# and qualitative generations.
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# repo root on sys.path so `pyPrune` imports without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyPrune.collapseMethods import count_params, find_decoder_stack


@torch.no_grad()
def perplexity(model, tokenizer, texts: List[str], max_length: int = 128) -> Dict[str, float]:
    """Mean next-token NLL and perplexity over the texts."""
    device = next(model.parameters()).device
    total_nll, total_tok = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text)[:max_length]
        if len(ids) < 2:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(input_ids=inp, labels=inp)
        n = inp.shape[1] - 1  # causal shift
        total_nll += out.loss.item() * n
        total_tok += n
    if total_tok == 0:
        raise ValueError("No evaluable tokens; texts are too short.")
    nll = total_nll / total_tok
    return {"nll": nll, "ppl": math.exp(min(nll, 20.0)), "tokens": total_tok}


@torch.no_grad()
def layer_relevance(model, tokenizer, texts: List[str], max_length: int = 128,
                    stack_name=None) -> List[float]:
    """Per-decoder-layer relevance = mean relative residual update
    ||out - in|| / ||in|| over the calibration texts. A layer that barely moves
    its hidden state is the least relevant, so the least-relevant contiguous
    window is the natural collapse target (mirrors the CNN study's heuristics).
    """
    device = next(model.parameters()).device
    _, stack = find_decoder_stack(model, stack_name)
    sums = [0.0] * len(stack)
    counts = [0] * len(stack)

    def make_hook(i):
        def hook(module, inp, out):
            x = inp[0]
            y = out[0] if isinstance(out, tuple) else out
            sums[i] += ((y - x).norm() / x.norm().clamp_min(1e-8)).item()
            counts[i] += 1
        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(stack)]
    try:
        for text in texts:
            ids = tokenizer.encode(text)[:max_length]
            if len(ids) < 2:
                continue
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            model(input_ids=inp)
    finally:
        for h in handles:
            h.remove()
    if not any(counts):
        raise ValueError("no calibration windows were evaluable")
    return [s / c for s, c in zip(sums, counts)]


def best_window(layer_scores: List[float], width: int) -> Tuple[int, float]:
    """Contiguous span of `width` layers with the lowest total relevance."""
    if not 1 <= width <= len(layer_scores):
        raise ValueError(f"collapse width {width} out of range for {len(layer_scores)} layers")
    start = min(range(len(layer_scores) - width + 1),
                key=lambda i: sum(layer_scores[i:i + width]))
    return start, sum(layer_scores[start:start + width])


def footprint(model, stack_name=None) -> Dict[str, float]:
    """Parameter accounting for the paper table: trainable params, on-disk
    size at the model's dtype, and layer count."""
    _, stack = find_decoder_stack(model, stack_name)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"params": count_params(model), "layers": len(stack),
            "params_per_layer": count_params(model) / max(len(stack), 1),
            "size_mb": round(size_mb, 2)}


def benchmark(model, tokenizer, text: str, max_length: int = 128,
              runs: int = 3) -> Dict[str, float]:
    """Mean forward latency over one window (the resource-usage study metric)."""
    import time

    device = next(model.parameters()).device
    ids = torch.tensor([tokenizer.encode(text)[:max_length]], dtype=torch.long,
                       device=device)
    model(input_ids=ids)  # warmup
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(input_ids=ids)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"forward_ms": round((time.perf_counter() - t0) / runs * 1000, 3),
            "tokens": int(ids.shape[1])}


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: List[str],
                     max_new_tokens: int = 32) -> List[Dict[str, str]]:
    """Greedy completions for a few prompts (qualitative paper examples)."""
    device = next(model.parameters()).device
    samples = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        samples.append({"prompt": prompt, "completion": tokenizer.decode(out[0])})
    return samples
