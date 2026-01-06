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
# Plotting Style (Accessible & Scientific)
# =========================
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="colorblind",   # Proven color-blind-safe palette
    font_scale=1.2,
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "legend.title_fontsize": 10,
})


# =========================
# Configuration
# =========================
RESULTS_DIR = Path("./")
FIG_DIR = Path("./figures")
TABLE_DIR = Path("./tables")

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

DATASET_ORDER = ["cifar10_", "cifar100_", "imagenet", "tinyimagenet"]

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

def format_accuracy_axis(ax):
    ax.set_ylabel("Accuracy Change (%)\n(higher is better)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

def format_reduction_axis(ax, label):
    ax.set_xlabel(f"{label} Reduction (%)\n(higher is better)")

def format_fraction_axis(ax):
    ax.set_xlabel("Collapsed Fraction\n(higher = more compression)")

def standard_legend(ax):
    ax.legend(
        title="Training Method / Precision",
        frameon=True,
        loc="best",
    )


# =========================
# Original Figures 
# =========================
def fig1(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
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
            ].dropna(subset=["d_params", "d_acc"])

            if subdf.empty:
                ax.set_title(f"{arch} – {ds}\n(no data)")
                ax.axis("off")
                continue

            sns.lineplot(
                data=subdf,
                x="d_params",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            format_reduction_axis(ax, "Parameter")
            format_accuracy_axis(ax)
            standard_legend(ax)

    fig.suptitle(
        "Accuracy Change vs Parameter Reduction\n"
        "(Compression–Accuracy Trade-off)",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_params_vs_accuracy.png")
    plt.close()

def fig2(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
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

            ax.set_title(f"{arch} on {ds}")
            format_reduction_axis(ax, "FLOPs")
            format_accuracy_axis(ax)
            standard_legend(ax)

    fig.suptitle(
        "Accuracy Change vs FLOPs Reduction\n"
        "(Computational Efficiency Trade-off)",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_flops_vs_accuracy.png")
    plt.close()


def fig3(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
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
                ax.axis("off")
                continue

            sns.lineplot(
                data=subdf,
                x="collapsed_fraction",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                markers=True,
                dashes=False,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            format_fraction_axis(ax)
            format_accuracy_axis(ax)
            standard_legend(ax)

    fig.suptitle(
        "Accuracy Change vs Collapsed Fraction\n"
        "(Approximation Error Proxy)",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_collapsed_fraction_vs_accuracy.png")
    plt.close()

def fig4(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
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

            ax.set_title(f"{arch} on {ds}")
            format_fraction_axis(ax)
            ax.set_ylabel("FLOPs Reduction (%)\n(higher is better)")
            standard_legend(ax)

    fig.suptitle(
        "FLOPs Reduction vs Collapsed Fraction\n"
        "(Compression–Efficiency Relationship)",
        fontsize=14,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_collapsed_fraction_vs_flops.png")
    plt.close()

def fig5(df: pd.DataFrame):
    max_collapse = (
        df.dropna(subset=["collapsed_fraction"])
          .groupby(["architecture", "dataset"])
          .collapsed_fraction
          .max()
          .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=max_collapse,
        x="architecture",
        y="collapsed_fraction",
        hue="dataset",
        ax=ax,
    )

    ax.set_title(
        "Maximum Achievable Model Collapsibility\n"
        "Across Architectures and Datasets"
    )
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Maximum Collapsed Fraction\n(higher = more compression)")
    ax.legend(title="Dataset")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_max_collapsibility.png")
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

# from pathlib import Path
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd


# def plot_expname_overlay_by_dataset(
#     df: pd.DataFrame,
#     out_dir: Path,
# ):
#     """
#     For each architecture:
#       - one figure
#       - rows = datasets
#       - x-axis = experiment group
#       - quantized vs non-quantized overlaid per experiment group
#       - experiment groups sorted by number of parameters
#         (largest model first, biggest collapse last)
#     """

#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     # Canonical experiment grouping
#     df = df.copy()
#     df["exp_group"] = (
#         df["exp_name"]
#         .str.replace("_quant", "", regex=False)
#         .str.replace("_JF", "", regex=False)
#         .str.replace("_Kevin", "", regex=False)
#         .str.strip()
#     )

#     for architecture, df_arch in df.groupby("architecture"):

#         datasets = sorted(df_arch["dataset"].unique())
#         n_rows = len(datasets)

#         # ------------------------------------------------
#         # SORT experiment groups by mean parameter count
#         # (largest → smallest)
#         # ------------------------------------------------
#         exp_order = (
#             df_arch.groupby("exp_group")["params"]
#             .mean()
#             .sort_values(ascending=False)
#             .index
#         )

#         fig, axes = plt.subplots(
#             n_rows,
#             1,
#             figsize=(max(10, 0.55 * len(exp_order)), 3.5 * n_rows),
#             sharex=True,
#             squeeze=False,
#         )

#         for i, dataset in enumerate(datasets):
#             ax = axes[i, 0]
#             g = df_arch[df_arch["dataset"] == dataset]

#             if g.empty:
#                 ax.axis("off")
#                 continue

#             sns.pointplot(
#                 data=g,
#                 x="exp_group",
#                 y="accuracy",
#                 hue="is_quantized",
#                 order=exp_order,
#                 dodge=0.35,
#                 markers=["o", "s"],
#                 linestyles=["-", "--"],
#                 errorbar=None,
#                 ax=ax,
#             )

#             ax.set_title(dataset, loc="left", fontsize=11)
#             ax.set_ylabel("Top-1 Accuracy")
#             ax.grid(True, axis="y", linestyle="--", alpha=0.5)

#             if i != n_rows - 1:
#                 ax.set_xlabel("")
#             else:
#                 ax.set_xlabel("Experiment Group (sorted by #params →)")
#                 ax.set_xticklabels(
#                     ax.get_xticklabels(),
#                     rotation=45,
#                     ha="right",
#                 )

#             # Show legend only on top row
#             if i == 0:
#                 ax.legend(title="Quantized")
#             else:
#                 ax.get_legend().remove()

#         fig.suptitle(
#             f"{architecture}: Accuracy vs Experiment Group\n"
#             "Ordered by Model Size (Largest → Smallest)",
#             fontsize=14,
#             y=1.02,
#         )

#         plt.tight_layout()
#         plt.savefig(out_dir / f"{architecture}_accuracy_vs_expgroup_sorted.png")
#         plt.close()


from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_expname_multi_metric(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory"],
    out_dir: Path = Path("./plots"),
):
    """
    For each architecture:
      - Rows = datasets
      - Columns = metrics
      - X-axis = experiment group (largest → smallest)
      - Hue = quantized / non-quantized
      - Each metric gets its own column
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Canonical experiment grouping (ignore quantization suffixes for sorting)
    df = df.copy()
    df["exp_group"] = (
        df["exp_name"]
        .str.replace("_quant", "", regex=False)
        .str.replace("_JF", "", regex=False)
        .str.replace("_Kevin", "", regex=False)
        .str.strip()
    )

    # Add boolean column for quantization
    df["is_quantized"] = df["exp_name"].str.contains("_quant")

    for architecture, df_arch in df.groupby("architecture"):

        datasets = sorted(df_arch["dataset"].unique())
        n_rows = len(datasets)
        n_cols = len(metrics)

        # -------------------------------
        # Determine x-axis ordering (largest → smallest)
        # -------------------------------
        exp_order = (
            df_arch.groupby("exp_group")["params"]
            .mean()
            .sort_values(ascending=False)
            .index
        )

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(5, 0.6 * len(exp_order)) * n_cols, 3.5 * n_rows),
            squeeze=False,
            sharex='col'
        )

        for i, dataset in enumerate(datasets):
            g_dataset = df_arch[df_arch["dataset"] == dataset]

            for j, metric in enumerate(metrics):
                ax = axes[i, j]

                if g_dataset.empty or metric not in g_dataset.columns:
                    ax.axis("off")
                    continue

                sns.pointplot(
                    data=g_dataset,
                    x="exp_group",
                    y=metric,
                    hue="is_quantized",
                    order=exp_order,
                    dodge=0.35,
                    markers=["o", "s"],
                    linestyles=["-", "--"],
                    errorbar=None,
                    ax=ax,
                )

                if i == 0:
                    ax.set_title(metric.capitalize(), fontsize=11)
                if i != n_rows - 1:
                    ax.set_xlabel("")
                else:
                    ax.set_xlabel("Experiment Group (Largest → Smallest)")
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

                if j > 0:
                    ax.set_ylabel("")
                else:
                    ax.set_ylabel(dataset, rotation=0, labelpad=50, fontsize=10, va='center')

                # Only show legend on top-left
                if i == 0 and j == 0:
                    ax.legend(title="Quantized")
                else:
                    ax.get_legend().remove()

                ax.grid(True, axis="y", linestyle="--", alpha=0.5)

        fig.suptitle(
            f"{architecture}: Metrics vs Experiment Group\n"
            "Rows = Datasets | Columns = Metrics | Hue = Quantized",
            fontsize=14,
            y=1.02,
        )

        plt.tight_layout()
        plt.savefig(out_dir / f"{architecture}_multi_metric.png")
        plt.close()

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
    plot_expname_multi_metric(
        df,
        out_dir=FIG_DIR / "expname_ablations",
    )
