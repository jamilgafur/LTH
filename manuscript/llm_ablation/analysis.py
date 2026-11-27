# analysis.py -- correlate expert activations with query domains.
# For each MoE block and expert we compute the Pearson (point-biserial)
# correlation between the expert's per-query activation (usage) and each
# domain's query membership. High positive correlation => the expert fires
# for that domain; the report answers "which activations correlate with
# which domain" and drives which regions the ablation removes.
from typing import Dict, List, Optional

import torch

from model_io import find_moe_blocks


def domain_correlations(per_query: List[dict]) -> Dict[str, dict]:
    """Correlate per-query expert usage with domain membership.

    per_query: ActivationRecorder per-query records
               [{"domain", "query", "usage": {block: [E]}}]
    Returns {block: {"domains": [...], "corr": [[E] floats per domain],
                     "top_domain": [str per expert], "max_corr": [float per expert]}}
    """
    domains = sorted({r["domain"] for r in per_query})
    if len(domains) < 2:
        raise ValueError(
            f"Correlation needs >= 2 domains for contrast, got {domains}. "
            f"Use --rank-by usage or gate_prob for a single-domain query set.")
    report = {}
    for blk in find_moe_blocks_from_records(per_query):
        name = blk
        usage = torch.tensor([r["usage"][name] for r in per_query])  # [Q, E]
        corr = []
        for d in domains:
            z = torch.tensor([1.0 if r["domain"] == d else 0.0 for r in per_query])
            corr.append(_pearson_columns(usage, z))
        corr_t = torch.stack(corr)  # [D, E]
        max_corr, top = corr_t.max(dim=0)
        report[name] = {
            "domains": domains,
            "corr": corr_t.tolist(),
            "top_domain": [domains[i] for i in top.tolist()],
            "max_corr": max_corr.tolist(),
        }
    return report


def find_moe_blocks_from_records(per_query: List[dict]) -> List[str]:
    """Block names as recorded (kept separate from model discovery)."""
    return list(per_query[0]["usage"].keys())


def _pearson_columns(usage: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Pearson correlation of each usage column with the vector z."""
    u = usage - usage.mean(dim=0)
    zc = z - z.mean()
    denom = u.norm(dim=0) * zc.norm()
    denom = torch.where(denom == 0, torch.ones_like(denom), denom)
    return (u * zc[:, None]).sum(dim=0) / denom


def score_correlations(per_query: List[dict], scores: List[float]) -> Dict[str, dict]:
    """Correlate per-query expert usage with a continuous per-query score.

    Used for poison identification: `scores` are poison similarities of the
    generated outputs, so experts whose activations track the poisoned
    behavior stand out. Returns {block: {"corr": [E], "abs_corr": [E]}}.
    """
    if len(scores) != len(per_query):
        raise ValueError(f"{len(scores)} scores for {len(per_query)} queries")
    s = torch.tensor(scores, dtype=torch.float32)
    report = {}
    for name in find_moe_blocks_from_records(per_query):
        usage = torch.tensor([r["usage"][name] for r in per_query])
        corr = _pearson_columns(usage, s)
        report[name] = {"corr": corr.tolist(), "abs_corr": corr.abs().tolist()}
    return report


def poison_keep_scores(report: Dict[str, dict]) -> Dict[str, torch.Tensor]:
    """Invert poison correlation into keep-scores: the experts most correlated
    with the poison outputs score lowest, so the drop-lowest-relevance ablation
    removes exactly the poisoned region."""
    return {name: -torch.tensor(r["corr"]) for name, r in report.items()}


def max_corr_scores(report: Dict[str, dict], protect_domains: Optional[List[str]] = None
                    ) -> Dict[str, torch.Tensor]:
    """Per-expert relevance: strongest correlation with any protected domain.

    Experts uncorrelated with the domains the deployment must serve score low
    and are the candidates for removal. protect_domains=None or ["all"] uses
    every domain present in the records.
    """
    scores = {}
    for name, r in report.items():
        domains, corr = r["domains"], torch.tensor(r["corr"])  # [D, E]
        if protect_domains and protect_domains != ["all"]:
            missing = [d for d in protect_domains if d not in domains]
            if missing:
                raise ValueError(f"protect_domains {missing} not in records {domains}")
            idx = [domains.index(d) for d in protect_domains]
            corr = corr[idx]
        scores[name] = corr.max(dim=0).values
    return scores


def layer_relevance(report: Dict[str, dict], protect_domains: Optional[List[str]] = None
                    ) -> Dict[str, float]:
    """Per-block (layer) relevance: mean expert relevance within the layer."""
    scores = max_corr_scores(report, protect_domains)
    return {name: float(s.mean()) for name, s in scores.items()}


def top_experts_per_domain(report: Dict[str, dict], k: int = 3) -> Dict[str, list]:
    """Paper table: the k experts most correlated with each domain."""
    out = {}
    for name, r in report.items():
        corr = torch.tensor(r["corr"])  # [D, E]
        for di, d in enumerate(r["domains"]):
            top = torch.topk(corr[di], min(k, corr.shape[1]))
            out.setdefault(d, []).append({
                "block": name,
                "experts": top.indices.tolist(),
                "corr": [round(v, 4) for v in top.values.tolist()],
            })
    return out
