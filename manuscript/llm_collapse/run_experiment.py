# run_experiment.py -- one paper run: load an LLM, evaluate the baseline,
# identify the contiguous layer spans whose feature representations are
# similar (the four Transfer heuristics in heuristics.py), collapse ALL of
# them into single collapsed layers with pyPrune.collapseMethods in one pass,
# fine-tune once, re-evaluate, and save the run record.
#
# One run = one paper data point:
#   runs/<name>/report.json    all metrics before/after the collapse
#   runs/<name>/config.json    the exact configuration
import argparse
import json
import random
import sys
from pathlib import Path

# repo root on sys.path so `pyPrune` imports without installation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from pyPrune.collapseMethods import collapse_only, find_decoder_stack

from config import RunConfig
from model_io import load_corpus, load_model, resolve_device
from evaluate import (benchmark, best_window, footprint, generate_samples,
                      layer_relevance, perplexity)
from finetune import finetune
from heuristics import analyze_spans, span_mergeable


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="contiguous layer collapse for LLMs")
    defaults = RunConfig()
    p.add_argument("--model", default=defaults.model)
    p.add_argument("--span-policy", default=defaults.span_policy,
                   choices=["variance", "relevance", "random"])
    p.add_argument("--variance-tol", type=float, default=defaults.variance_tol)
    p.add_argument("--collapse-width", type=int, default=defaults.collapse_width)
    p.add_argument("--collapse-start", type=int, default=defaults.collapse_start)
    p.add_argument("--merge", default=defaults.merge, choices=["mean", "middle", "first", "last"])
    p.add_argument("--finetune-steps", type=int, default=defaults.finetune_steps)
    p.add_argument("--finetune-lr", type=float, default=defaults.finetune_lr)
    p.add_argument("--corpus", default=defaults.corpus)
    p.add_argument("--max-length", type=int, default=defaults.max_length)
    p.add_argument("--eval-windows", type=int, default=defaults.eval_windows)
    p.add_argument("--max-new-tokens", type=int, default=defaults.max_new_tokens)
    p.add_argument("--seed", type=int, default=defaults.seed)
    p.add_argument("--stack", default=defaults.stack,
                   help="decoder stack path for encoder-decoder/multi-tower models")
    p.add_argument("--device", default=defaults.device)
    p.add_argument("--dtype", default=defaults.dtype)
    p.add_argument("--out-dir", default=defaults.out_dir)
    p.add_argument("--run-name", default=defaults.run_name)
    p.add_argument("--skip-existing", action="store_true")
    return RunConfig(**vars(p.parse_args()))


def run(cfg: RunConfig) -> dict:
    run_dir = Path(cfg.out_dir) / cfg.name()
    if cfg.skip_existing and (run_dir / "report.json").exists():
        print(f"[skip] {run_dir} already complete")
        return json.loads((run_dir / "report.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    set_seed(cfg.seed)
    model, tokenizer, load_report = load_model(cfg.model, cfg.device, cfg.dtype)
    texts = load_corpus(cfg.corpus)
    eval_texts, calib_texts = texts[:cfg.eval_windows], texts
    prompts = [t[:40] for t in texts[:2]]
    metrics = {"config": cfg.to_dict(), "load": load_report}

    # 1. baseline
    metrics["baseline"] = {
        "perplexity": perplexity(model, tokenizer, eval_texts, cfg.max_length),
        "footprint": footprint(model, cfg.stack),
        "benchmark": benchmark(model, tokenizer, eval_texts[0], cfg.max_length),
        "samples": generate_samples(model, tokenizer, prompts, cfg.max_new_tokens),
    }
    print(f"[eval] baseline ppl={metrics['baseline']['perplexity']['ppl']:.3f} "
          f"params={metrics['baseline']['footprint']['params']:,} "
          f"layers={metrics['baseline']['footprint']['layers']}")

    # 2. identify the collapse spans with the Transfer heuristics
    stack_name, stack = find_decoder_stack(model, cfg.stack)
    if cfg.span_policy == "variance":
        # group contiguous layers with similar activation variance and score
        # every group (relative variance / redundancy / virtual-bypass KL)
        analysis = analyze_spans(model, tokenizer, calib_texts,
                                 cfg.variance_tol, cfg.max_length,
                                 stack_name=cfg.stack)
        spans = [tuple(s["span"]) for s in analysis["spans"]]
        metrics["heuristics"] = analysis
    elif cfg.span_policy == "relevance":
        relevance = layer_relevance(model, tokenizer, calib_texts, cfg.max_length,
                                      stack_name=cfg.stack)
        start = cfg.collapse_start if cfg.collapse_start is not None \
            else best_window(relevance, cfg.collapse_width)[0]
        spans = [(start, start + cfg.collapse_width - 1)]
        metrics["heuristics"] = {"relevance": relevance}
    elif cfg.span_policy == "random":
        start = cfg.collapse_start if cfg.collapse_start is not None \
            else random.Random(cfg.seed).randrange(0, len(stack) - cfg.collapse_width + 1)
        spans = [(start, start + cfg.collapse_width - 1)]
        metrics["heuristics"] = {}
    else:
        raise ValueError(f"unknown span policy {cfg.span_policy!r}")

    # spans crossing a structural boundary (MiMo/DeepSeek dense -> MoE)
    # cannot be mean-merged: skip them and collapse the rest
    keep = [s for s in spans if span_mergeable(model, s, cfg.stack)]
    skipped = [s for s in spans if s not in keep]
    if skipped:
        print(f"[span] skipped unmergeable spans {skipped} "
              "(structurally different layers, e.g. a dense->MoE boundary)")
    spans = keep
    print(f"[span] policy={cfg.span_policy}: collapsing {spans or '(none)'}")

    # 3. collapse ALL identified spans into one layer each, in a single pass
    #    (Transfer study: the whole compression set goes before retraining)
    model = collapse_only(model=model, compression_set=spans, merge=cfg.merge,
                          stack_name=cfg.stack,
                          device=str(resolve_device(cfg.device)))
    metrics["collapse"] = {
        "spans": [list(s) for s in spans],
        "merge": cfg.merge,
        "stack": stack_name,
        "reports": getattr(model, "_collapse_reports", []),
        "layers_collapsed": sum(e - s + 1 for s, e in spans),
        "skipped_spans": [list(s) for s in skipped],
    }

    # 4. one post-collapse fine-tuning phase (recovery)
    if cfg.finetune_steps > 0:
        metrics["finetune"] = finetune(model, tokenizer, calib_texts,
                                       cfg.finetune_steps, cfg.finetune_lr,
                                       cfg.max_length, cfg.seed)
        print(f"[finetune] {metrics['finetune']}")

    # 5. re-evaluate the collapsed model
    metrics["collapsed"] = {
        "perplexity": perplexity(model, tokenizer, eval_texts, cfg.max_length),
        "footprint": footprint(model, cfg.stack),
        "benchmark": benchmark(model, tokenizer, eval_texts[0], cfg.max_length),
        "samples": generate_samples(model, tokenizer, prompts, cfg.max_new_tokens),
    }
    metrics["delta_ppl"] = (metrics["collapsed"]["perplexity"]["ppl"]
                            - metrics["baseline"]["perplexity"]["ppl"])
    metrics["param_reduction"] = (1.0 - metrics["collapsed"]["footprint"]["params"]
                                  / metrics["baseline"]["footprint"]["params"])
    b, c = metrics["baseline"], metrics["collapsed"]
    metrics["resource_reduction"] = {
        "params": metrics["param_reduction"],
        "size_mb": 1.0 - c["footprint"]["size_mb"] / b["footprint"]["size_mb"],
        "forward_ms": 1.0 - c["benchmark"]["forward_ms"] / b["benchmark"]["forward_ms"],
    }
    print(f"[eval] collapsed ppl={metrics['collapsed']['perplexity']['ppl']:.3f} "
          f"(delta {metrics['delta_ppl']:+.3f}), params -{metrics['param_reduction']:.1%}")

    (run_dir / "report.json").write_text(json.dumps(metrics, indent=2))
    print(f"[done] {run_dir}")
    return metrics


if __name__ == "__main__":
    run(parse_args())
