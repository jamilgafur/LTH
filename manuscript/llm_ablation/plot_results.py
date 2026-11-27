# plot_results.py -- aggregate runs/*/metrics.json into paper tables/figures.
# Usage: python plot_results.py [--runs runs] [--out runs/summary]
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(runs_dir: Path):
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        m = json.loads(metrics_path.read_text())
        cfg = m["config"]
        rows.append({
            "run": metrics_path.parent.name,
            "model": cfg["model"],
            "query_set": cfg["query_set"],
            "ablation": cfg["ablation"],
            "rank_by": cfg.get("rank_by", ""),
            "keep": cfg.get("keep_experts") or cfg["keep_fraction"],
            "collapse_layers": cfg.get("collapse_layers", 0),
            "baseline_ppl": m["baseline"]["perplexity"]["ppl"],
            "ablated_ppl": m["ablated"]["perplexity"]["ppl"],
            "delta_ppl": m["delta_ppl"],
            "layers_before": m["baseline"]["footprint"]["transformer_layers"],
            "layers_after": m["ablated"]["footprint"]["transformer_layers"],
            "params_before": m["baseline"]["footprint"]["total_params"],
            "params_after": m["ablated"]["footprint"]["total_params"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    runs_dir = Path(a.runs)
    out_dir = Path(a.out) if a.out else runs_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(runs_dir)
    if not rows:
        raise SystemExit(f"No runs/*/metrics.json found under {runs_dir}")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # delta perplexity vs. remaining parameters, one series per ablation mode
    fig, ax = plt.subplots(figsize=(7, 4.5))
    groups = defaultdict(list)
    for r in rows:
        groups[r["ablation"]].append(r)
    for mode, g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["params_after"])
        ax.plot([r["params_after"] for r in g], [r["delta_ppl"] for r in g],
                marker="o", label=mode)
    ax.set_xlabel("parameters after ablation")
    ax.set_ylabel("perplexity delta (ablated - baseline)")
    ax.set_title("MoE region ablation: quality loss vs. model size")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "delta_ppl_vs_params.png", dpi=200)
    print(f"[plot] wrote {out_dir/'summary.csv'} and {out_dir/'delta_ppl_vs_params.png'}")


if __name__ == "__main__":
    main()
