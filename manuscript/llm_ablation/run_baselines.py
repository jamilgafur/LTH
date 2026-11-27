# run_baselines.py -- compare removal strategies at equal sparsity.
# Runs the identical (model, query set, ablation, keep) configuration with
# different ranking strategies and writes one comparison table:
#   correlation/poison-guided  vs  usage  vs  gate_prob  vs  random
# The paper's claim is only as strong as "we beat random".
#
# Usage:
#   python run_baselines.py --model toy-moe --query-set domains \
#       --ablation experts --keep-fraction 0.5
#   python run_baselines.py --rank-bys poison,usage,gate_prob,random ...
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RunConfig
import run_experiment


def run_baselines(cfg: RunConfig, rank_bys, out_dir: Path, **run_kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank_by in rank_bys:
        c = RunConfig(**{**cfg.to_dict(), "rank_by": rank_by,
                         "run_name": f"{cfg.name()}_{rank_by}"})
        m = run_experiment.run(c, **run_kwargs)
        row = {
            "rank_by": rank_by,
            "baseline_ppl": m["baseline"]["perplexity"]["ppl"],
            "ablated_ppl": m["ablated"]["perplexity"]["ppl"],
            "delta_ppl": m["delta_ppl"],
            "params_after": m["ablated"]["footprint"]["total_params"],
            "poison_rate_after": (m.get("poison") or {}).get("after", {}).get("rate", ""),
        }
        rows.append(row)
        print(f"[baseline] {rank_by:9s} delta_ppl {row['delta_ppl']:+.3f}")

    path = out_dir / f"baselines_{cfg.model}_{cfg.query_set}_{cfg.ablation}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[baselines] wrote {path}")
    return path


def main():
    p = argparse.ArgumentParser(description="Removal-strategy baselines at equal sparsity.")
    p.add_argument("--model", default=RunConfig.model)
    p.add_argument("--query-set", default=RunConfig.query_set)
    p.add_argument("--ablation", default="experts",
                   choices=["experts", "layers", "region"])
    p.add_argument("--keep-fraction", type=float, default=RunConfig.keep_fraction)
    p.add_argument("--keep-experts", type=int, default=None)
    p.add_argument("--rank-bys", default="max_corr,usage,gate_prob,random",
                   help="comma list; use 'poison,...' for the backdoor study")
    p.add_argument("--max-queries", type=int, default=RunConfig.max_queries)
    p.add_argument("--max-length", type=int, default=RunConfig.max_length)
    p.add_argument("--max-new-tokens", type=int, default=RunConfig.max_new_tokens)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    p.add_argument("--device", default=RunConfig.device)
    p.add_argument("--dtype", default=RunConfig.dtype)
    p.add_argument("--out-dir", default="runs")
    a = p.parse_args()
    cfg = RunConfig(
        model=a.model, query_set=a.query_set, ablation=a.ablation,
        keep_fraction=a.keep_fraction, keep_experts=a.keep_experts,
        max_queries=a.max_queries, max_length=a.max_length,
        max_new_tokens=a.max_new_tokens, seed=a.seed, device=a.device,
        dtype=a.dtype, out_dir=a.out_dir,
    )
    rank_bys = [r.strip() for r in a.rank_bys.split(",") if r.strip()]
    run_baselines(cfg, rank_bys, Path(a.out_dir))


if __name__ == "__main__":
    main()
