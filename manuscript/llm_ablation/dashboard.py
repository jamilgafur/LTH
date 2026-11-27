# dashboard.py -- self-contained HTML dashboard over the saved run artifacts,
# plus tidy CSV exports for future paper figures. No server or JS libraries:
# the run data is embedded into one dashboard.html openable from disk.
#
# Usage: python dashboard.py [--runs runs] [--out runs/dashboard.html]
import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from ablate import _layer_index

TEMPLATE = Path(__file__).with_name("dashboard_template.html")


# ------------------------------------------------------------- data assembly
def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def load_runs(runs_dir: Path) -> Dict[str, dict]:
    """Load every run's artifacts: metrics, analysis, activations."""
    runs = {}
    for run_dir in sorted(runs_dir.glob("*/")):
        if not (run_dir / "metrics.json").exists():
            continue
        runs[run_dir.name] = {
            "metrics": _read_json(run_dir / "metrics.json"),
            "analysis": _read_json(run_dir / "analysis.json"),
            "activations": _read_json(run_dir / "activations_summary.json"),
        }
    return runs


def build_view(name: str, data: dict) -> dict:
    """Flatten one run's artifacts into the shape the dashboard renders."""
    metrics = data["metrics"]
    cfg = metrics["config"]
    acts = data["activations"] or {"aggregate": {}, "per_query": []}
    corr = (data["analysis"] or {}).get("correlations", {})
    pcorr = (data["analysis"] or {}).get("poison_correlations", {})
    ablation = metrics.get("ablation", {}) or {}
    poison_info = metrics.get("poison") or {}
    poison_by_query = {p["query"]: p for p in poison_info.get("per_query", [])}

    removed_layers = set(ablation.get("removed_layers", []))
    layer_reports = ablation if cfg.get("ablation") == "experts" else ablation.get("layers", {})

    layers = []
    for block_name, agg in sorted(acts["aggregate"].items(),
                                  key=lambda kv: _layer_index(kv[0])):
        index = _layer_index(block_name)
        rep = layer_reports.get(block_name, {})
        c = corr.get(block_name, {})
        kept = rep.get("kept", list(range(agg["experts"])))
        layers.append({
            "name": block_name,
            "index": index,
            "experts": agg["experts"],
            "top_k": agg["top_k"],
            "tokens": agg["tokens"],
            "usage": agg["usage"],
            "gate_prob": agg["gate_prob"],
            "output_norm": agg["output_norm"],
            "corr": {d: c["corr"][i] for i, d in enumerate(c.get("domains", []))} if c else {},
            "top_domain": c.get("top_domain", []),
            "max_corr": c.get("max_corr", []),
            "poison_corr": pcorr.get(block_name, {}).get("corr", []),
            "kept_mask": [e in kept for e in range(agg["experts"])],
            "removed_layer": index in removed_layers,
        })

    queries = [{"domain": r["domain"], "text": r["query"], "usage": r["usage"],
                "poison_score": poison_by_query.get(r["query"], {}).get("score_before")}
               for r in acts["per_query"]]
    samples = {tag: metrics.get(tag, {}).get("samples", [])
               for tag in ("baseline", "ablated")}
    return {
        "name": name,
        "model": cfg["model"],
        "query_set": cfg["query_set"],
        "ablation": cfg["ablation"],
        "rank_by": cfg.get("rank_by", ""),
        "baseline_ppl": metrics["baseline"]["perplexity"]["ppl"],
        "ablated_ppl": metrics["ablated"]["perplexity"]["ppl"],
        "delta_ppl": metrics["delta_ppl"],
        "footprint_before": metrics["baseline"]["footprint"],
        "footprint_after": metrics["ablated"]["footprint"],
        "domains": sorted({q["domain"] for q in queries}),
        "queries": queries,
        "layers": layers,
        "poison": {"phrases": poison_info.get("phrases", []),
                   "before": poison_info.get("before"),
                   "after": poison_info.get("after")},
        "top_experts_per_domain": (data["analysis"] or {}).get("top_experts_per_domain", {}),
        "samples": samples,
    }


# ------------------------------------------------------------------- exports
def write_exports(runs_dir: Path, views: Dict[str, dict]) -> List[str]:
    """Tidy per-run CSVs: expert-level table and per-query usage, for figures."""
    written = []
    for name, v in views.items():
        out = runs_dir / "exports" / name
        out.mkdir(parents=True, exist_ok=True)
        corr_cols = [f"corr_{d}" for d in v["domains"]]
        with open(out / "expert_data.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "expert", "usage", "gate_prob", "output_norm",
                        "top_domain", "max_corr", "poison_corr", "kept"] + corr_cols)
            for l in v["layers"]:
                for e in range(l["experts"]):
                    w.writerow([l["index"], e, l["usage"][e], l["gate_prob"][e],
                                "" if l["output_norm"] is None else l["output_norm"][e],
                                l["top_domain"][e] if l["top_domain"] else "",
                                l["max_corr"][e] if l["max_corr"] else "",
                                l["poison_corr"][e] if l["poison_corr"] else "",
                                int(l["kept_mask"][e])]
                               + [l["corr"].get(d, [""] * l["experts"])[e] for d in v["domains"]])
            written.append(out / "expert_data.csv")
        with open(out / "per_query_usage.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["domain", "query", "layer", "expert", "usage", "poison_score"])
            for q in v["queries"]:
                for l in v["layers"]:
                    for e, u in enumerate(q["usage"].get(l["name"], [])):
                        w.writerow([q["domain"], q["text"], l["index"], e, u,
                                    q["poison_score"] if q.get("poison_score") is not None else ""])
            written.append(out / "per_query_usage.csv")
    return written


# ------------------------------------------------------------------ assembly
def build(runs_dir: Path, out_path: Path) -> Path:
    runs = load_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No runs/*/metrics.json found under {runs_dir}")
    views = {name: build_view(name, data) for name, data in runs.items()}
    written = write_exports(runs_dir, views)
    blob = json.dumps({"runs": views}).replace("</", "<\\/")
    html = TEMPLATE.read_text().replace("__DATA__", blob)
    out_path.write_text(html)
    print(f"[dashboard] {out_path} ({len(views)} runs)")
    for p in written:
        print(f"[export]   {p}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Build the ablation study dashboard.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    runs_dir = Path(a.runs)
    out = Path(a.out) if a.out else runs_dir / "dashboard.html"
    build(runs_dir, out)


if __name__ == "__main__":
    main()
