from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_rows", None)

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
    font_scale=1.1,
)

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
            posthoc_or_posttrain = infer_posthoc_or_posttrain(exp)
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "exp_name": exp,
                    "posthoc_or_posttrain": posthoc_or_posttrain,  # Updated field name
                    "model_type": infer_model_type(exp),
                    "is_quantized": infer_isquant(exp),
                    "accuracy": m.get("final_accuracy"),
                    "params": m.get("param_count"),
                    "flops": m.get("flops"),
                    "memory": m.get("total_size_mb"),
                }
            )

    return pd.DataFrame(rows)


# =========================
# Utilities
# =========================
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
    if "xception" in name:
        return "Xception"
    if "mobilenet" in name:
        return "MobileNet"
    raise ValueError(f"Cannot infer architecture from {p}")


def infer_posthoc_or_posttrain(exp_name: str) -> str:
    n = exp_name.lower()
    if "jf" in n:
        return "posttrain"  # Clearly distinguishing posttrain experiments
    if "kevin" in n:
        return "posthoc"  # Clearly distinguishing posthoc experiments
    raise ValueError(f"Cannot infer posthoc or posttrain from {exp_name}")


def infer_model_type(exp_name: str) -> str:
    n = exp_name.lower()
    if "original" in n or "baseline" in n:
        return "baseline"
    return "collapsed"


def infer_isquant(exp_name: str) -> bool:
    return "quant" in exp_name.lower()


def find_baseline(df: pd.DataFrame):
    mask = (
        df["exp_name"].str.lower().str.contains("original")
        & df["exp_name"].str.lower().str.contains("kevin")
    )
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            warnings.warn(f"No baseline for {ds}-{arch}")
            continue

        for _, r in g.iterrows():
            row = r.copy()
            row["d_acc"] =  r["accuracy"] - baseline["accuracy"] 
            row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
            row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
            row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
            row["collapsed_fraction"] = row["d_params"] / 100.0
            out.append(row)

    return pd.DataFrame(out)


# =========================
# Original Figures 
# =========================

def fig1(df: pd.DataFrame):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
        squeeze=False,  # <-- critical: always 2D
    )

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]

            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ]

            if subdf.empty:
                ax.set_title(f"{arch} - {ds}\n(no data)")
                ax.axis("off")
                continue

            # Safety check: drop rows with missing values
            subdf = subdf.dropna(subset=["d_params", "d_acc"])

            sns.lineplot(
                data=subdf,
                x="d_params",
                y="d_acc",
                hue="posthoc_or_posttrain",   # Posthoc vs Posttrain
                style="is_quantized",         # Quantized vs FP
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.axhline(0, color="gray", linestyle="--", linewidth=1)

            ax.set_title(f"{arch} – {ds}")
            ax.set_xlabel("Parameter Reduction (%)")
            ax.set_ylabel("Accuracy Change (%)")

            # Improve legend readability
            ax.legend(
                title="Training / Quantization",
                fontsize=9,
                title_fontsize=10,
                frameon=True,
            )

    plt.tight_layout()

    fig_path = FIG_DIR / "fig1_dparams_vs_dacc.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")

    plt.close()

    # -----------------------------
    # Terminal debug output
    # -----------------------------
    print("\nData used in Figure 1:")
    for arch in architectures:
        for ds in datasets:
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ]
            if subdf.empty:
                continue

            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print(
                subdf[
                    ["posthoc_or_posttrain", "is_quantized", "model_type",
                     "d_params", "d_acc"]
                ].to_string(index=False)
            )

def fig2(df: pd.DataFrame):
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["d_flops", "d_acc"])

            if subdf.empty:
                ax.set_title(f"{arch} – {ds}\n(no data)")
                ax.axis("off")
                continue

            sns.lineplot(
                data=subdf,
                x="d_flops",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{arch} – {ds}")
            ax.set_xlabel("FLOPs Reduction (%)")
            ax.set_ylabel("Accuracy Change (%)")
            ax.legend(title="Training / Quantization")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig2_dflops_vs_dacc.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
    plt.close()

def fig3(df: pd.DataFrame):
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["collapsed_fraction", "d_acc"])

            if subdf.empty:
                ax.set_title(f"{arch} – {ds}\n(no data)")
                ax.axis("off")
                continue

            sns.lineplot(
                data=subdf,
                x="collapsed_fraction",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                ax=ax,
            )

            ax.set_title(f"{arch} – {ds}")
            ax.set_xlabel("Collapsed Fraction (Approximation Error Proxy)")
            ax.set_ylabel("Accuracy Change (%)")
            ax.legend(title="Training / Quantization")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig3_approximation_error_vs_accuracy.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
    plt.close()

def fig4(df: pd.DataFrame):
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["collapsed_fraction", "d_flops"])

            if subdf.empty:
                ax.set_title(f"{arch} – {ds}\n(no data)")
                ax.axis("off")
                continue

            sns.lineplot(
                data=subdf,
                x="collapsed_fraction",
                y="d_flops",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_title(f"{arch} – {ds}")
            ax.set_xlabel("Collapsed Fraction")
            ax.set_ylabel("FLOPs Reduction (%)")
            ax.legend(title="Training / Quantization")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig4_flops_vs_collapsed_fraction.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
    plt.close()


def fig5(df: pd.DataFrame):
    max_collapse_data = []

    for (arch, ds), g in df.groupby(["architecture", "dataset"]):
        g = g.dropna(subset=["collapsed_fraction"])
        if g.empty:
            continue

        r = g.loc[g["collapsed_fraction"].idxmax()]
        max_collapse_data.append(
            {
                "architecture": arch,
                "dataset": ds,
                "max_collapsed_fraction": r["collapsed_fraction"],
            }
        )

    max_collapse_df = pd.DataFrame(max_collapse_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=max_collapse_df,
        x="architecture",
        y="max_collapsed_fraction",
        hue="dataset",
        ax=ax,
    )

    ax.set_title("Cross-Architecture Comparison of Maximum Collapsibility")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Max Collapsed Fraction")
    ax.legend(title="Dataset")

    fig_path = FIG_DIR / "fig5_cross_architecture_comparisons.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Figure saved to {fig_path}")
    plt.close()

def tab1(df: pd.DataFrame):
    """
    Tab. 2 — Baseline vs Max Collapse comparison per (dataset, architecture).
    """
    comparison_data = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            continue

        g_valid = g.dropna(subset=["collapsed_fraction"])
        if g_valid.empty:
            continue

        max_collapse = g_valid.loc[g_valid["collapsed_fraction"].idxmax()]

        comparison_data.append(
            {
                "dataset": ds,
                "architecture": arch,
                "baseline_accuracy": baseline["accuracy"],
                "baseline_params": baseline["params"],
                "baseline_flops": baseline["flops"],
                "baseline_memory": baseline["memory"],
                "max_collapsed_accuracy": max_collapse["accuracy"],
                "max_collapsed_params": max_collapse["params"],
                "max_collapsed_flops": max_collapse["flops"],
                "max_collapsed_memory": max_collapse["memory"],
                "collapsed_fraction": max_collapse["collapsed_fraction"],
                "accuracy_drop": max_collapse["d_acc"],
            }
        )

    comparison_df = (
        pd.DataFrame(comparison_data)
        .sort_values(["dataset", "architecture"])
        .reset_index(drop=True)
    )

    print("\nTabular Summary (Tab. 2): Baseline vs Max Collapse Comparison")
    print(comparison_df.to_string(index=False))

    # Save LaTeX table
    table_path = TABLE_DIR / "tab1_baseline_vs_max_collapse.tex"
    with open(table_path, "w") as f:
        f.write(
            comparison_df.to_latex(
                index=False,
                float_format="%.2f",
                caption="Baseline vs. maximum collapsed model comparison.",
                label="tab:baseline_vs_max_collapse",
            )
        )

    print(f"Table saved to {table_path}")

def tab2(df: pd.DataFrame):
    """
    Tab. 3 — Model efficiency for collapsed models.
    Reports compression, accuracy drop, and resource reductions.
    """
    efficiency_data = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            continue

        for _, r in g.iterrows():
            if r["model_type"] != "collapsed":
                continue

            efficiency_data.append(
                {
                    "dataset": ds,
                    "architecture": arch,
                    "exp_name": r["exp_name"],
                    "compression_ratio": 100 * (1 - r["params"] / baseline["params"]),
                    "accuracy_drop": r["d_acc"],
                    "d_params": r["d_params"],
                    "d_flops": r["d_flops"],
                    "d_memory": r["d_memory"],
                }
            )

    efficiency_df = (
        pd.DataFrame(efficiency_data)
        .sort_values(["dataset", "architecture", "compression_ratio"])
        .reset_index(drop=True)
    )

    print("\nTabular Summary (Tab. 3): Model Efficiency")
    print(efficiency_df.to_string(index=False))

    # Save LaTeX table
    table_path = TABLE_DIR / "tab2_model_efficiency.tex"
    with open(table_path, "w") as f:
        f.write(
            efficiency_df.to_latex(
                index=False,
                float_format="%.2f",
                caption="Efficiency metrics for collapsed models.",
                label="tab:model_efficiency",
            )
        )

    print(f"Table saved to {table_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    raw = load_results()
    df = normalize(raw)


    print("\n==============================")
    print(" NON-OVERVIEW PAPER FIGURES ")
    print("==============================")
    fig1(df)
    fig2(df)
    fig3(df)
    fig4(df)
    fig5(df)
    tab1(df)
    tab2(df)
