# plot_results.py -- paper summary over all runs: a CSV table and the
# trade-off figure (perplexity damage vs. parameter reduction).
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(runs_dir: Path):
    rows = []
    for report in sorted(runs_dir.glob("*/report.json")):
        data = json.loads(report.read_text())
        cfg, base, coll = data["config"], data["baseline"], data["collapsed"]
        spans = data["collapse"]["spans"]
        hs = data.get("heuristics", {}).get("spans", [])
        rows.append({
            "run": report.parent.name,
            "model": cfg["model"],
            "policy": cfg["span_policy"],
            "merge": cfg["merge"],
            "spans": str(spans),
            "n_spans": len(spans),
            "layers_collapsed": data["collapse"].get("layers_collapsed", 0),
            "mean_score": round(sum(s["score"] for s in hs) / len(hs), 4) if hs else "",
            "mean_redundancy": round(sum(s["redundancy"] for s in hs) / len(hs), 4) if hs else "",
            "mean_kl": round(sum(s["kl"] for s in hs) / len(hs), 4) if hs else "",
            "layers_before": base["footprint"]["layers"],
            "layers_after": coll["footprint"]["layers"],
            "params_before": base["footprint"]["params"],
            "params_after": coll["footprint"]["params"],
            "param_reduction": round(data["param_reduction"], 4),
            "size_reduction": round(data.get("resource_reduction", {}).get("size_mb", 0.0), 4),
            "latency_reduction": round(data.get("resource_reduction", {}).get("forward_ms", 0.0), 4),
            "ppl_before": round(base["perplexity"]["ppl"], 3),
            "ppl_after": round(coll["perplexity"]["ppl"], 3),
            "delta_ppl": round(data["delta_ppl"], 3),
            "finetune_steps": cfg["finetune_steps"],
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    args = p.parse_args()
    runs_dir = Path(args.runs)
    rows = collect(runs_dir)
    if not rows:
        print(f"no runs found under {runs_dir}")
        return

    csv_path = runs_dir / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[csv] {csv_path} ({len(rows)} runs)")

    fig, ax = plt.subplots(figsize=(6, 4))
    for merge in sorted({r["merge"] for r in rows}):
        sel = [r for r in rows if r["merge"] == merge]
        ax.scatter([r["param_reduction"] for r in sel],
                   [r["delta_ppl"] for r in sel], label=merge)
        for r in sel:
            ax.annotate(r["run"], (r["param_reduction"], r["delta_ppl"]), fontsize=6)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("parameter reduction")
    ax.set_ylabel("perplexity delta (collapsed - baseline)")
    ax.set_title("Layer collapse: size vs. quality")
    ax.legend()
    fig.tight_layout()
    fig_path = runs_dir / "collapse_tradeoff.png"
    fig.savefig(fig_path, dpi=150)
    print(f"[plot] {fig_path}")


if __name__ == "__main__":
    main()
