# run_experiment.py -- one complete paper run:
#   load MoE LLM -> query per domain -> store activations -> correlate
#   activations with domains -> replace unneeded regions -> re-evaluate.
# Output: runs/<run_name>/{config.json, metrics.json, activations*.json/pt,
#        analysis.json, ablation_report.json, samples*.txt[, ablated_model/]}
#
# Example:
#   python run_experiment.py --model mixtral-8x7b --query-set domains \
#       --ablation region --collapse-layers 2 --keep-fraction 0.5 \
#       --rank-by max_corr --protect-domains math,code
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RunConfig
from queries import load_domains
from model_io import load_model, resolve_device
from activations import ActivationRecorder, scores_from_summary
from analysis import (domain_correlations, layer_relevance, max_corr_scores,
                      top_experts_per_domain, score_correlations, poison_keep_scores)
from ablate import (ablate_experts, collapse_layers, collapse_region,
                    best_layer_window, expert_footprint, _keep_count)
from causal import causal_check
from triggers import discover_triggers
from evaluate import run_eval, generate_records
import poison
from pyPrune.utils import set_seed, clean_memory


def parse_args():
    p = argparse.ArgumentParser(description="MoE LLM region ablation study: one run.")
    p.add_argument("--model", default=RunConfig.model)
    p.add_argument("--query-set", default=RunConfig.query_set,
                   help="'domains', a built-in domain name, or a .txt file with [domain] sections")
    p.add_argument("--ablation", default=RunConfig.ablation,
                   choices=["experts", "layers", "region", "none"])
    p.add_argument("--keep-fraction", type=float, default=RunConfig.keep_fraction)
    p.add_argument("--keep-experts", type=int, default=None)
    p.add_argument("--rank-by", default=RunConfig.rank_by,
                   choices=["max_corr", "usage", "gate_prob", "poison", "random"],
                   help="'poison' removes the region most correlated with the "
                        "poison signature outputs; 'random' is the removal baseline")
    p.add_argument("--protect-domains", default=RunConfig.protect_domains,
                   help="'all' or comma list of domains the deployment must serve")
    p.add_argument("--poison-phrases", default=None,
                   help="poison signature file, one phrase per line "
                        "(default: data/poison_phrases.txt)")
    p.add_argument("--poison-scorer", default=RunConfig.poison_scorer,
                   choices=list(poison.SCORERS),
                   help="'logprob': model's probability of emitting the phrase "
                        "(default); 'lexical': difflib similarity of the output")
    p.add_argument("--poison-threshold", type=float, default=RunConfig.poison_threshold)
    p.add_argument("--causal-controls", type=int, default=5,
                   help="random control expert sets in the causal check")
    p.add_argument("--no-cache", action="store_true",
                   help="re-record activations even if a cache entry exists")
    p.add_argument("--cache-dir", default=".cache/activations")
    p.add_argument("--collapse-layers", type=int, default=RunConfig.collapse_layers)
    p.add_argument("--collapse-start", type=int, default=None,
                   help="fixed span start; default picks the least-relevant span")
    p.add_argument("--max-queries", type=int, default=RunConfig.max_queries)
    p.add_argument("--max-length", type=int, default=RunConfig.max_length)
    p.add_argument("--max-new-tokens", type=int, default=RunConfig.max_new_tokens)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--device", default=RunConfig.device)
    p.add_argument("--dtype", default=RunConfig.dtype,
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--out-dir", default=RunConfig.out_dir)
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--run-name", default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip the run if its metrics.json already exists")
    a = p.parse_args()
    cfg = RunConfig(
        model=a.model, query_set=a.query_set, ablation=a.ablation,
        keep_fraction=a.keep_fraction, keep_experts=a.keep_experts,
        rank_by=a.rank_by, protect_domains=a.protect_domains,
        poison_phrases=a.poison_phrases, poison_threshold=a.poison_threshold,
        poison_scorer=a.poison_scorer,
        collapse_layers=a.collapse_layers, collapse_start=a.collapse_start,
        max_queries=a.max_queries, max_length=a.max_length,
        max_new_tokens=a.max_new_tokens, seed=a.seed, device=a.device,
        dtype=a.dtype, out_dir=a.out_dir, save_model=a.save_model,
        run_name=a.run_name,
    )
    return cfg, a


def _record_or_load(model, tokenizer, domains, queries, cfg, run_dir,
                    use_cache=True, cache_dir=".cache/activations"):
    """Record activations, reusing a cache keyed by model + queries + precision."""
    key = hashlib.sha1(json.dumps({
        "model": cfg.model, "dtype": cfg.dtype, "seed": cfg.seed,
        "max_length": cfg.max_length, "queries": queries}, sort_keys=True
    ).encode()).hexdigest()[:16]
    cache = Path(cache_dir)
    c_pt, c_js = cache / f"{key}.pt", cache / f"{key}.json"
    if use_cache and c_pt.exists():
        shutil.copy(c_pt, run_dir / "activations.pt")
        shutil.copy(c_js, run_dir / "activations_summary.json")
        summary, per_query = ActivationRecorder.load_artifacts(run_dir / "activations.pt")
        print(f"[activations] cache hit {key} ({len(per_query)} queries)")
        return summary, per_query

    rec = ActivationRecorder(model)
    with rec, torch.no_grad():
        for domain, qs in domains.items():
            for q in qs:
                rec.mark(domain, q)
                ids = torch.tensor([tokenizer.encode(q)[:cfg.max_length]],
                                   dtype=torch.long,
                                   device=next(model.parameters()).device)
                model(input_ids=ids)
    rec.save(run_dir / "activations.pt")
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(run_dir / "activations.pt", c_pt)
    shutil.copy(run_dir / "activations_summary.json", c_js)
    return rec.summary(), rec.per_query


def run(cfg: RunConfig, use_cache: bool = True,
        cache_dir: str = ".cache/activations",
        causal_controls: int = 5) -> dict:
    run_dir = Path(cfg.out_dir) / cfg.name()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    set_seed(cfg.seed)
    domains = load_domains(cfg.query_set, cfg.max_queries)
    model, tokenizer, load_report = load_model(cfg.model, cfg.device, cfg.dtype)
    device = resolve_device(cfg.device)
    metrics = {"config": cfg.to_dict(), "load": load_report}

    # 1. baseline
    queries = [q for qs in domains.values() for q in qs]
    metrics["baseline"] = run_eval(model, tokenizer, queries,
                                   cfg.max_length, cfg.max_new_tokens)
    print(f"[eval] baseline ppl={metrics['baseline']['perplexity']['ppl']:.3f}")

    # 2. query per domain and store the activations (cache: item 9)
    summary, per_query = _record_or_load(model, tokenizer, domains, queries,
                                         cfg, run_dir, use_cache, cache_dir)

    # 2b. poison study: generate outputs and score them against the signatures
    phrases, gen_before, scored_before = [], [], []
    if cfg.rank_by == "poison":
        phrases = poison.load_phrases(cfg.poison_phrases)
        gen_before = generate_records(model, tokenizer, queries, cfg.max_new_tokens)
        scored_before = poison.score_queries(
            model, tokenizer, queries, [g["continuation"] for g in gen_before],
            phrases, cfg.poison_scorer, cfg.max_length)
        print(f"[poison] {len(phrases)} signatures, scorer={cfg.poison_scorer}; before: "
              f"{poison.report([s['score'] for s in scored_before], cfg.poison_threshold)}")

    # 3. correlate activations with domains / poison scores
    corr_report = domain_correlations(per_query) if len(domains) >= 2 else {}
    analysis = {"correlations": corr_report,
                "top_experts_per_domain": top_experts_per_domain(corr_report) if corr_report else {}}
    pcorr = {}
    if cfg.rank_by == "poison":
        pcorr = score_correlations(per_query, [s["score"] for s in scored_before])
        analysis["poison_correlations"] = pcorr
    (run_dir / "analysis.json").write_text(json.dumps(analysis, indent=2))
    print(f"[analysis] domains {list(domains)} -> {len(per_query)} queries analyzed")

    # 4. rank experts/layers and replace the unneeded regions
    if cfg.rank_by == "poison":
        # most poison-correlated experts score lowest -> dropped first
        scores = poison_keep_scores(pcorr)
        layer_scores = {n: float(s.mean()) for n, s in scores.items()}
    elif cfg.rank_by == "random":  # removal baseline
        scores = {n: torch.rand(len(v["usage"])) for n, v in summary.items()}
        layer_scores = {n: float(s.mean()) for n, s in scores.items()}
    elif cfg.rank_by == "max_corr":
        scores = max_corr_scores(corr_report, cfg.protected())
        layer_scores = layer_relevance(corr_report, cfg.protected())
    else:
        scores = scores_from_summary(summary, cfg.rank_by)
        layer_scores = {n: float(s.mean()) for n, s in scores.items()}

    # 4b. poison diagnosis: causal check on the to-be-removed region (item 2)
    # and trigger-token discovery (item 6), before the model is changed.
    if cfg.rank_by == "poison":
        suspects = {}
        for n, s in scores.items():
            k = _keep_count(len(s), cfg.keep_fraction, cfg.keep_experts)
            keep = set(torch.topk(s, k).indices.tolist())
            dropped = [i for i in range(len(s)) if i not in keep]
            if dropped:
                suspects[n] = dropped
        ranked = sorted(range(len(scored_before)),
                        key=lambda i: -scored_before[i]["score"])
        flagged_i = [i for i in ranked
                     if scored_before[i]["score"] >= cfg.poison_threshold][:8] or ranked[:3]
        flagged_prompts = [queries[i] for i in flagged_i]

        def score_fn(m):  # causal metric: model's probability of the phrases
            s = poison.logprob_scores(m, tokenizer, flagged_prompts, phrases, cfg.max_length)
            return sum(x["score"] for x in s) / len(s)

        causal = causal_check(model, suspects, score_fn,
                              n_controls=causal_controls, seed=cfg.seed)
        metrics["poison_causal"] = causal
        print(f"[causal] suspect drop {causal['suspect_drop']:.4f} vs control mean "
              f"{causal['mean_control_drop']:.4f} (ratio {causal['causal_ratio']:.2f})")

        trig = discover_triggers(
            model, tokenizer,
            [{"prompt": queries[i], "score": scored_before[i]["score"],
              "matched": scored_before[i]["matched"],
              "usage": per_query[i]["usage"]} for i in flagged_i],
            [f"{n}:{i}" for n, idx in suspects.items() for i in idx],
            max_length=cfg.max_length)
        (run_dir / "triggers.json").write_text(json.dumps(trig, indent=2))
        metrics["poison_triggers"] = trig["top_tokens_global"]
        print(f"[triggers] top tokens: "
              f"{[t['token'] for t in trig['top_tokens_global']]}")

    metrics["ablation"] = {}
    if cfg.ablation == "experts":
        metrics["ablation"] = ablate_experts(
            model, scores, cfg.keep_fraction, cfg.keep_experts)
    elif cfg.ablation == "layers":
        start = cfg.collapse_start
        if start is None:
            start, _ = best_layer_window(layer_scores, cfg.collapse_layers)
        metrics["ablation"] = collapse_layers(model, start, cfg.collapse_layers)
    elif cfg.ablation == "region":
        metrics["ablation"] = collapse_region(
            model, scores, layer_scores, cfg.collapse_layers,
            cfg.keep_fraction, cfg.keep_experts, cfg.collapse_start)
    (run_dir / "ablation_report.json").write_text(json.dumps(metrics["ablation"], indent=2))

    # 5. re-evaluate the ablated model
    metrics["ablated"] = run_eval(model, tokenizer, queries,
                                  cfg.max_length, cfg.max_new_tokens)
    metrics["delta_ppl"] = (metrics["ablated"]["perplexity"]["ppl"]
                            - metrics["baseline"]["perplexity"]["ppl"])
    print(f"[eval] ablated ppl={metrics['ablated']['perplexity']['ppl']:.3f} "
          f"(delta {metrics['delta_ppl']:+.3f})")

    if cfg.rank_by == "poison":
        gen_after = generate_records(model, tokenizer, queries, cfg.max_new_tokens)
        scored_after = poison.score_queries(
            model, tokenizer, queries, [g["continuation"] for g in gen_after],
            phrases, cfg.poison_scorer, cfg.max_length)
        before = poison.report([s["score"] for s in scored_before], cfg.poison_threshold)
        after = poison.report([s["score"] for s in scored_after], cfg.poison_threshold)
        metrics["poison"] = {
            "phrases": phrases,
            "scorer": cfg.poison_scorer,
            "threshold": cfg.poison_threshold,
            "before": before,
            "after": after,
            "per_query": [{
                "query": gb["prompt"],
                "continuation_before": gb["continuation"],
                "continuation_after": ga["continuation"],
                "score_before": sb["score"], "score_after": sa["score"],
                "matched": sb["matched"],
            } for gb, ga, sb, sa in zip(gen_before, gen_after, scored_before, scored_after)],
        }
        print(f"[poison] triggered rate {before['rate']:.2%} -> {after['rate']:.2%}")

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for tag in ("baseline", "ablated"):
        (run_dir / f"samples_{tag}.txt").write_text(
            "\n\n".join(f">>> {s['prompt']}\n{s['completion']}"
                        for s in metrics[tag]["samples"]))
    if cfg.save_model:
        model.save_pretrained(run_dir / "ablated_model")
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(run_dir / "ablated_model")
    clean_memory()
    print(f"[done] {run_dir}")
    return metrics


if __name__ == "__main__":
    cfg, a = parse_args()
    if a.skip_existing and (Path(cfg.out_dir) / cfg.name() / "metrics.json").exists():
        print(f"[skip] {cfg.name()} already done.")
    else:
        run(cfg, use_cache=not a.no_cache, cache_dir=a.cache_dir,
            causal_controls=a.causal_controls)
