# triggers.py -- surface which input tokens drive the poisoned region.
# 1. Occlusion attribution: remove each input token and re-measure the poison
#    phrase probability; the drop is how much that token triggers the phrase.
# 2. Cluster the flagged queries by the activations of the poison-correlated
#    (suspect) experts, then aggregate attribution per cluster.
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from poison import phrase_logprob_from_ids


def occlusion_attribution(model, tokenizer, prompt: str, phrase: str,
                          max_length: int = 256) -> List[Dict]:
    """Per-input-token importance: drop in phrase probability when removed."""
    p_ids = tokenizer.encode(prompt)[:max_length]
    f_ids = tokenizer.encode(phrase)
    base = phrase_logprob_from_ids(model, p_ids, f_ids, max_length)
    out = []
    for i in range(len(p_ids)):
        occluded = p_ids[:i] + p_ids[i + 1:]
        s = phrase_logprob_from_ids(model, occluded, f_ids, max_length)
        out.append({"token": tokenizer.decode([p_ids[i]]),
                    "token_id": p_ids[i], "importance": base - s})
    return out


def cluster_vectors(vectors: List[List[float]], k: int = 2,
                    seed: int = 0, iters: int = 20) -> List[int]:
    """Tiny k-means (numpy) -> cluster label per vector."""
    X = np.asarray(vectors, dtype=float)
    if len(X) == 0:
        return []
    k = max(1, min(k, len(X)))
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1).argmin(1)
        for c in range(k):
            if (labels == c).any():
                centers[c] = X[labels == c].mean(0)
    return labels.tolist()


def discover_triggers(model, tokenizer, flagged: List[Dict],
                      suspect_experts: List[str], k_clusters: int = 2,
                      top_n: int = 5, max_length: int = 256) -> dict:
    """Trigger candidates for the flagged queries.

    flagged: [{"prompt", "score", "matched", "usage": {block: [E]}}]
    suspect_experts: ["<block>:<idx>", ...] the poison-correlated experts.
    Returns clusters of queries (by suspect-expert activations) with their
    top trigger tokens by aggregated occlusion attribution.
    """
    keys = [(n, int(i)) for n, _, i in (s.partition(":") for s in suspect_experts)]

    def suspect_vec(usage: Dict[str, List[float]]) -> List[float]:
        return [usage.get(n, [0.0] * (i + 1))[i] for n, i in keys]

    attributions = []
    for f in flagged:
        attributions.append(occlusion_attribution(
            model, tokenizer, f["prompt"], f["matched"], max_length))

    labels = cluster_vectors([suspect_vec(f["usage"]) for f in flagged], k_clusters)
    clusters = []
    for c in sorted(set(labels)):
        member_attrs = [a for a, l in zip(attributions, labels) if l == c]
        agg = defaultdict(float)
        for a in member_attrs:
            for t in a:
                agg[t["token"]] += t["importance"]
        top = sorted(agg.items(), key=lambda kv: -kv[1])[:top_n]
        clusters.append({
            "cluster": c,
            "queries": [f["prompt"] for f, l in zip(flagged, labels) if l == c],
            "top_tokens": [{"token": tok, "importance": round(v, 6)}
                           for tok, v in top],
        })
    global_agg = defaultdict(float)
    for a in attributions:
        for t in a:
            global_agg[t["token"]] += t["importance"]
    return {
        "flagged_queries": len(flagged),
        "suspect_experts": suspect_experts,
        "clusters": clusters,
        "top_tokens_global": [
            {"token": tok, "importance": round(v, 6)}
            for tok, v in sorted(global_agg.items(), key=lambda kv: -kv[1])[:top_n]],
    }
