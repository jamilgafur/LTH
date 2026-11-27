# evaluate.py -- before/after metrics for the ablation study.
# Perplexity over the held-out queries, qualitative generations, and the
# parameter / expert footprint that goes into the paper table.
import math
from typing import Dict, List

import torch

from ablate import expert_footprint
from model_io import find_moe_blocks, router_width


@torch.no_grad()
def perplexity(model, tokenizer, texts: List[str], max_length: int = 256) -> Dict[str, float]:
    """Mean next-token NLL and perplexity over the query texts."""
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
        raise ValueError("No evaluable tokens; queries are too short.")
    nll = total_nll / total_tok
    return {"nll": nll, "ppl": math.exp(min(nll, 20.0)), "tokens": total_tok}


@torch.no_grad()
def generate_records(model, tokenizer, queries: List[str],
                     max_new_tokens: int = 32) -> List[dict]:
    """Greedy completions for every prompt.

    Returns [{"prompt", "completion", "continuation"}] where continuation is
    the generated text only (what poison signatures match against).
    """
    device = next(model.parameters()).device
    records = []
    for q in queries:
        ids = torch.tensor([tokenizer.encode(q)], dtype=torch.long, device=device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0][ids.shape[1]:]
        records.append({"prompt": q,
                        "completion": tokenizer.decode(out[0]),
                        "continuation": tokenizer.decode(gen)})
    return records


@torch.no_grad()
def generate_samples(model, tokenizer, queries: List[str],
                     max_new_tokens: int = 32, max_samples: int = 3) -> List[Dict[str, str]]:
    """Greedy completions for a few prompts (qualitative paper examples)."""
    return [{"prompt": r["prompt"], "completion": r["completion"]}
            for r in generate_records(model, tokenizer, queries[:max_samples], max_new_tokens)]


def routing_footprint(model) -> Dict[str, dict]:
    """Per-layer expert counts after ablation (and top_k capping)."""
    return {
        blk.name: {
            "experts": blk.num_experts,
            "top_k": blk.top_k,
            "router_out": router_width(blk.router),
        }
        for blk in find_moe_blocks(model)
    }


def run_eval(model, tokenizer, queries: List[str],
             max_length: int = 256, max_new_tokens: int = 32) -> Dict:
    ppl = perplexity(model, tokenizer, queries, max_length)
    return {
        "perplexity": ppl,
        "footprint": expert_footprint(model),
        "routing": routing_footprint(model),
        "samples": generate_samples(model, tokenizer, queries, max_new_tokens),
    }
