from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration
# =========================
RESULTS_DIR = Path("./")
FIG_DIR = Path("./figures")
TABLE_DIR = Path("./tables")

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

DATASET_ORDER = ["cifar10_", "cifar100_", "imagenet", "tinyimagenet"]

sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="colorblind",
    font_scale=1.2,
)

# =========================
# Utilities
# =========================
def _to_float(x):
    """Safely convert memory components to float (MB)."""
    if x is None:
        return 0.0
    if isinstance(x, (int, float)):
        return float(x)
    # sometimes nested dicts or lists sneak in
    if isinstance(x, dict):
        return float(sum(v for v in x.values() if isinstance(v, (int, float))))
    if isinstance(x, (list, tuple)):
        return float(sum(v for v in x if isinstance(v, (int, float))))
    return 0.0

def infer_dataset_from_path(p: Path) -> str:
    name = p.parent.parent.name.lower()
    for ds in DATASET_ORDER:
        if ds in name:
            return ds
    raise ValueError(f"Cannot infer dataset from {p}")

def infer_architecture_from_path(p: Path) -> str:
    name = p.parent.parent.name.lower()
    if "regnet" in name:
        return "RegNetX"
    if "vgg" in name:
        return "VGG16"
    if "inception" in name:
        return "InceptionNet"
    return "Other"

def infer_model_type(exp_name: str) -> str:
    n = exp_name.lower()
    if "original" in n or "baseline" in n:
        return "baseline"
    return "collapsed"

def find_baseline(df: pd.DataFrame):
    mask = (
        df["exp_name"].str.lower().str.contains("original")
        & df["exp_name"].str.lower().str.contains("kevin")
    )
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def pareto_front(x, y):
    idx = np.argsort(x)
    best, keep = -np.inf, []
    for i in idx:
        if y[i] >= best:
            best = y[i]
            keep.append(i)
    return keep

# =========================
# Data Loading
# =========================

def load_results() -> pd.DataFrame:
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    if not files:
        raise FileNotFoundError("No merged_metrics.json files found")

    rows = []
    for p in files:
        dataset = infer_dataset_from_path(p)
        arch = infer_architecture_from_path(p)

        with open(p) as f:
            raw = json.load(f)

        for exp, m in raw.items():
            diag = m.get("diagnostics", {})
            rows.append({
                "dataset": dataset,
                "architecture": arch,
                "exp_name": exp,
                "model_type": infer_model_type(exp),
                "accuracy": m.get("final_accuracy"),
                "params": m.get("param_count"),
                "flops": m.get("flops"),
                "memory": m.get("total_size_mb"),
                "per_layer_params_flops": diag.get("per_layer_params_flops"),
                "activation_sizes": diag.get("activation_sizes"),
                "memory_decomposition": diag.get("memory_decomposition"),
            })

    return pd.DataFrame(rows)

# =========================
# Normalization
# =========================
def normalize_activation_sizes(act):
    """Convert activation diagnostics to a 1D numeric list (MB)."""
    if act is None:
        return []

    out = []
    if isinstance(act, dict):
        for v in act.values():
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict):
                out.append(sum(float(x) for x in v.values() if isinstance(x, (int, float))))
    elif isinstance(act, (list, tuple)):
        for v in act:
            if isinstance(v, (int, float)):
                out.append(float(v))
            elif isinstance(v, dict):
                out.append(sum(float(x) for x in v.values() if isinstance(x, (int, float))))
    return out

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            warnings.warn(f"No baseline for {ds}-{arch}")
            continue

        for _, r in g.iterrows():
            row = r.copy()
            row["d_acc"] = r["accuracy"] - baseline["accuracy"]
            row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
            row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
            row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
            row["acc_per_mparam"] = r["accuracy"] / (r["params"] / 1e6)
            row["acc_per_gflop"] = r["accuracy"] / (r["flops"] / 1e9)
            out.append(row)

    return pd.DataFrame(out)

# =========================
# Figures (SVG)
# =========================
def extract_layer_series(plpf, key):
    if not isinstance(plpf, list):
        return None
    vals = []
    for layer in plpf:
        if isinstance(layer, dict) and key in layer:
            vals.append(layer[key])
    return np.array(vals) if vals else None

def fig_pareto_grouped(df, dataset):
    archs = df["architecture"].unique()
    n_arch = len(archs)
    fig, axes = plt.subplots(1, n_arch, figsize=(5.5 * n_arch, 4), squeeze=False, sharey=True)

    for ax, arch in zip(axes[0], archs):
        g = df[df["architecture"] == arch].copy()

        # Create a new column for legend grouping
        def label_type(row):
            if row["model_type"] == "baseline":
                return "baseline"
            elif "quant" in row["exp_name"].lower():
                return "collapsed+quantized"
            else:
                return "collapsed"

        g["legend_type"] = g.apply(label_type, axis=1)
        g["label"] = g["exp_name"]

        palette = sns.color_palette("colorblind", n_colors=g["legend_type"].nunique())
        sns.scatterplot(
            data=g, x="d_params", y="d_acc",
            hue="legend_type", style="legend_type",
            s=80, palette=palette, edgecolor="k", linewidth=0.6, ax=ax
        )

        # Annotate each experiment
        for _, row in g.iterrows():
            ax.text(row["d_params"], row["d_acc"], row["label"], fontsize=7, ha="left", va="bottom", alpha=0.9)

        # Pareto frontier
        x, y = g["d_params"].values, g["d_acc"].values
        pf_idx = pareto_front(x, y)
        ax.plot(x[pf_idx], y[pf_idx], linestyle="--", color="0.15", linewidth=1)
        ax.fill_between(x[pf_idx], y[pf_idx], y.min()-1, alpha=0.03, color="0.15")

        ax.axvline(0, color="0.6", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="0.6", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Δ Params (%) — lower is better")
        ax.set_title(f"Architecture: {arch}", fontsize=11)

        if ax is axes[0][0]:
            ax.set_ylabel("Δ Accuracy (pp) — higher is better")

        ax.legend(title="", frameon=True, fontsize=8, loc="best")

    fig.suptitle(f"{dataset}: Pareto frontier by architecture", fontsize=12, y=.99)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{dataset}_pareto.svg")
    plt.close()


def fig_efficiency_grouped(df, dataset):
    archs = df["architecture"].unique()
    n_arch = len(archs)
    fig, axes = plt.subplots(1, n_arch, figsize=(5.5 * n_arch, 4), squeeze=False, sharey=True)

    for ax, arch in zip(axes[0], archs):
        g = df[df["architecture"] == arch].copy()

        # Legend type column
        def label_type(row):
            if row["model_type"] == "baseline":
                return "baseline"
            elif "quant" in row["exp_name"].lower():
                return "collapsed+quantized"
            else:
                return "collapsed"

        g["legend_type"] = g.apply(label_type, axis=1)
        g["label"] = g["exp_name"]

        sns.scatterplot(
            data=g, x="params", y="accuracy",
            hue="legend_type", style="legend_type",
            s=80, edgecolor="k", linewidth=0.5, palette="colorblind", ax=ax
        )

        # Annotate
        for _, row in g.iterrows():
            ax.text(row["params"], row["accuracy"], row["label"], fontsize=7, ha="left", va="bottom", alpha=0.9)

        ax.set_xscale("log")
        ticks = np.array([1e6, 5e6, 1e7, 5e7, 1e8])
        ticks = ticks[(ticks >= g["params"].min()*0.9) & (ticks <= g["params"].max()*1.1)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t/1e6:.0f}M" for t in ticks])

        params_space = np.logspace(np.log10(g["params"].min()*0.9), np.log10(g["params"].max()*1.1), 200)
        for k in [0.5, 1.0, 2.0, 4.0]:
            ax.plot(params_space, k * (params_space / 1e6), linestyle=":", color="0.5", linewidth=0.8)
            ax.text(params_space[-1]*0.95, k*(params_space[-1]/1e6), f"{k:.1f} acc/M", fontsize=7, color="0.5", va="center", ha="right")

        base = g[g["model_type"] == "baseline"]
        if not base.empty:
            b = base.iloc[0]
            ax.scatter(b["params"], b["accuracy"], marker="*", s=240, edgecolor="k", facecolor="none", linewidth=1.2, zorder=6)
            ax.annotate(f"baseline\n{b['accuracy']:.2f}%\n{b['params']/1e6:.1f}M",
                        xy=(b["params"], b["accuracy"]), xytext=(8, -18), textcoords="offset points",
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.8))

        ax.set_xlabel("Parameters (log) — lower is better")
        ax.set_title(f"Architecture: {arch}", fontsize=11)
        if ax is axes[0][0]:
            ax.set_ylabel("Accuracy — higher is better")
        ax.legend(title="", frameon=True, fontsize=8)

    fig.suptitle(f"{dataset}: Efficiency by architecture", fontsize=12, y=.99)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{dataset}_efficiency.svg")
    plt.close()


def fig_activation_heatmap_grouped(df, dataset):
    archs = df["architecture"].unique()
    n_arch = len(archs)
    fig, axes = plt.subplots(1, n_arch, figsize=(6.5*n_arch, 4), squeeze=False, sharey=False)

    global_max = 0.0
    mats = {}
    for arch in archs:
        g = df[df["architecture"] == arch]
        rows = []
        names = []
        for _, r in g.iterrows():
            vec = normalize_activation_sizes(r.get("activation_sizes"))
            if not vec:
                continue
            rows.append(vec)
            label = f"{r['exp_name']} (quant)" if "quant" in r["exp_name"].lower() else r["exp_name"]
            names.append(label)
        if not rows:
            continue
        min_len = min(len(v) for v in rows)
        mat = np.array([v[:min_len] for v in rows])
        mats[arch] = (mat, names)
        global_max = max(global_max, float(np.nanmax(mat)))

    for ax, arch in zip(axes[0], archs):
        if arch not in mats:
            continue
        mat, names = mats[arch]
        mat_t = mat.T
        sns.heatmap(
            mat_t, ax=ax, cmap="viridis",
            vmin=0.0, vmax=global_max,
            cbar=ax is axes[0][-1],
            cbar_kws={"label": "Activation size (MB) — lower is better"}
        )
        ax.set_xticks(np.arange(len(names)) + 0.5)
        ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(np.linspace(0, mat_t.shape[0], 6))
        ax.set_ylabel("Layer index")
        ax.set_xlabel("Model variant")
        ax.set_title(f"Architecture: {arch}")

    fig.suptitle(f"{dataset}: Activation memory by layer (transposed)", fontsize=12, y=.99)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{dataset}_activation_heatmap_transposed.svg")
    plt.close()


def fig_memory_decomp_grouped(df, dataset):
    archs = df["architecture"].unique()
    n_arch = len(archs)
    fig, axes = plt.subplots(1, n_arch, figsize=(6.5*n_arch, 4), squeeze=False, sharey=True)

    for ax, arch in zip(axes[0], archs):
        g = df[df["architecture"] == arch]
        rows = []
        for _, r in g.iterrows():
            md = r.get("memory_decomposition")
            if md is None:
                continue
            if isinstance(md, dict):
                params = _to_float(md.get("params"))
                acts = _to_float(md.get("activations"))
                opt = _to_float(md.get("optimizer"))
                other = _to_float(md.get("other"))
            elif isinstance(md, (list, tuple)):
                params = _to_float(md[0]) if len(md) > 0 else 0.0
                acts = _to_float(md[1]) if len(md) > 1 else 0.0
                opt = _to_float(md[2]) if len(md) > 2 else 0.0
                other = _to_float(md[3]) if len(md) > 3 else 0.0
            else:
                continue
            label = f"{r['exp_name']} (quant)" if "quant" in r["exp_name"].lower() else r["exp_name"]
            rows.append({"exp_name": label, "Params": params, "Activations": acts, "Optimizer": opt, "Other": other, "d_memory": r.get("d_memory")})

        if not rows:
            continue

        mem_df = pd.DataFrame(rows).set_index("exp_name")
        for c in ["Params", "Activations", "Optimizer", "Other"]:
            mem_df[c] = pd.to_numeric(mem_df[c], errors="coerce").fillna(0.0)
        mem_df["Total"] = mem_df[["Params","Activations","Optimizer","Other"]].sum(axis=1)
        mem_df = mem_df.sort_values("Total", ascending=False)

        mem_df[["Params","Activations","Optimizer","Other"]].plot(kind="bar", stacked=True, ax=ax,
                                                                edgecolor="k", linewidth=0.4)
        ax.set_ylabel("Memory (MB) — lower is better")
        ax.set_title(f"Architecture: {arch}")
        for i, (_, row) in enumerate(mem_df.iterrows()):
            dmem = row.get("d_memory")
            if pd.notna(dmem):
                ax.text(i, row["Total"] + 0.01*mem_df["Total"].max(), f"Δ{dmem:.0f}%", ha="center", va="bottom", fontsize=7)
        ax.legend(title="", fontsize=8)

    fig.suptitle(f"{dataset}: Memory decomposition by architecture", fontsize=12, y=.99)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"{dataset}_memory_decomposition.svg")
    plt.close()



# =========================
# Tables (booktabs-safe)
# =========================

def save_table(df, cols, caption, label, path):
    latex = df[cols].to_latex(
        index=False,
        float_format="%.2f",
        caption=caption,
        label=label,
        escape=False,
    )

    lines = [l for l in latex.splitlines() if l.strip() != r"\hline"]
    for i, l in enumerate(lines):
        if l.startswith(r"\begin{tabular}"):
            lines.insert(i + 1, r"\toprule")
            break
    for i, l in enumerate(lines):
        if "&" in l and "\\" in l:
            lines.insert(i + 1, r"\midrule")
            break
    lines.insert(-1, r"\bottomrule")

    path.write_text("\n".join(lines))

# =========================
# Main
# =========================

if __name__ == "__main__":
    raw = load_results()
    df = normalize(raw)

    for ds, g in df.groupby("dataset"):
        fig_pareto_grouped(g, ds)
        fig_efficiency_grouped(g, ds)
        fig_memory_decomp_grouped(g, ds)
        fig_activation_heatmap_grouped(g, ds)

        # Tables unchanged, still per-arch
        for arch, ga in g.groupby("architecture"):
            save_table(
                ga,
                ["dataset", "architecture", "exp_name",
                 "accuracy", "d_acc",
                 "params", "d_params",
                 "flops", "d_flops",
                 "memory", "d_memory"],
                "Overall performance and compression deltas.",
                f"tab:{ds}_{arch}_overall",
                TABLE_DIR / f"{ds}_{arch}_overall.tex",
            )

            save_table(
                ga,
                ["dataset", "architecture", "exp_name",
                 "acc_per_mparam", "acc_per_gflop"],
                "Compression efficiency metrics.",
                f"tab:{ds}_{arch}_efficiency",
                TABLE_DIR / f"{ds}_{arch}_efficiency.tex",
            )
