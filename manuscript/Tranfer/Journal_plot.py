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
    palette="colorblind",
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

DATASET_ORDER = ["cifar10_", "cifar100_", "tinyimagenet", "imagenet", "ConvNeXt"]

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
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "exp_name": exp,
                    "posthoc_or_posttrain": infer_posthoc_or_posttrain(exp),
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
    return "unknown" 

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
    if "convnext" in name:
        return "ConvNeXt"
    return "UnknownArch"

def infer_posthoc_or_posttrain(exp_name: str) -> str:
    n = exp_name.lower()
    if "jf" in n:
        return "Post-Prune (JF)"  # Explicit Label
    if "kevin" in n:
        return "No-Prune (Kevin)" # Explicit Label
    if "original" in n or "baseline" in n:
        return "Baseline"
    return "Unknown"

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
        | df["exp_name"].str.lower().str.contains("baseline")
    )
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            warnings.warn(f"No baseline for {ds}-{arch}")
            # Still add rows, but without relative metrics
            for _, r in g.iterrows():
                out.append(r)
            continue

        for _, r in g.iterrows():
            row = r.copy()
            # Avoid division by zero if params are missing
            if baseline["params"]:
                row["d_acc"] =  r["accuracy"] - baseline["accuracy"] 
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
                row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
                row["collapsed_fraction"] = row["d_params"] / 100.0
            out.append(row)

    return pd.DataFrame(out)

# =========================
# Plotting Helpers
# =========================
def format_accuracy_axis(ax):
    ax.set_ylabel("Accuracy Change (%)\n(higher is better)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

def format_reduction_axis(ax, label):
    ax.set_xlabel(f"{label} Reduction (%)\n(higher is better)")

def format_fraction_axis(ax):
    ax.set_xlabel("Collapsed Fraction\n(higher = more compression)")

def standard_legend(ax):
    ax.legend(
        title="Configuration",
        frameon=True,
        loc="best",
    )

# =========================
# Figures 1-5 (Standard)
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
                ax.axis("off")
                continue

            sns.scatterplot(
                data=subdf,
                x="d_params",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                s=100, # size
                alpha=0.8,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1:
                format_reduction_axis(ax, "Parameter")
            if j == 0:
                format_accuracy_axis(ax)
            
            # Simplified legend for individual plots
            if i==0 and j==0:
                standard_legend(ax)
            else:
                ax.get_legend().remove()

    fig.suptitle("Accuracy vs Parameter Reduction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_params_vs_accuracy.png", bbox_inches='tight')
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

            sns.scatterplot(
                data=subdf,
                x="d_flops",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                s=100,
                alpha=0.8,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1:
                format_reduction_axis(ax, "FLOPs")
            if j == 0:
                format_accuracy_axis(ax)
            
            if i==0 and j==0:
                standard_legend(ax)
            else:
                ax.get_legend().remove()

    fig.suptitle("Accuracy vs FLOPs Reduction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_flops_vs_accuracy.png", bbox_inches='tight')
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

            sns.scatterplot(
                data=subdf,
                x="collapsed_fraction",
                y="d_acc",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                s=100,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1:
                format_fraction_axis(ax)
            if j == 0:
                format_accuracy_axis(ax)

            if i==0 and j==0:
                standard_legend(ax)
            else:
                ax.get_legend().remove()

    fig.suptitle("Accuracy vs Collapsed Fraction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_collapsed_fraction_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig4(df: pd.DataFrame):
    # Only if collapsed_fraction and d_flops exist
    if "collapsed_fraction" not in df.columns: return

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

            sns.scatterplot(
                data=subdf,
                x="collapsed_fraction",
                y="d_flops",
                hue="posthoc_or_posttrain",
                style="is_quantized",
                s=100,
                ax=ax,
            )

            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1:
                format_fraction_axis(ax)
            if j == 0:
                ax.set_ylabel("FLOPs Reduction (%)")
            
            if i==0 and j==0:
                standard_legend(ax)
            else:
                ax.get_legend().remove()

    fig.suptitle("FLOPs Reduction vs Collapsed Fraction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_collapsed_fraction_vs_flops.png", bbox_inches='tight')
    plt.close()

def fig5(df: pd.DataFrame):
    if "collapsed_fraction" not in df.columns: return
    
    max_collapse = (
        df.dropna(subset=["collapsed_fraction"])
          .groupby(["architecture", "dataset"])
          .collapsed_fraction
          .max()
          .reset_index()
    )

    if max_collapse.empty: return

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=max_collapse,
        x="architecture",
        y="collapsed_fraction",
        hue="dataset",
        ax=ax,
    )

    ax.set_title("Maximum Model Collapsibility")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Max Collapsed Fraction")
    ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_max_collapsibility.png", bbox_inches='tight')
    plt.close()


# =========================
# Tables
# =========================
def tab1(df: pd.DataFrame):
    comparison_data = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None: continue

        g_valid = g.dropna(subset=["collapsed_fraction"])
        if g_valid.empty: continue

        max_collapse = g_valid.loc[g_valid["collapsed_fraction"].idxmax()]

        comparison_data.append(
            {
                "Dataset": ds,
                "Arch": arch,
                "Base Acc": baseline["accuracy"],
                "Max Col Acc": max_collapse["accuracy"],
                "Acc Drop": max_collapse["d_acc"],
                "Col Fraction": max_collapse["collapsed_fraction"],
            }
        )

    if not comparison_data: return

    comparison_df = pd.DataFrame(comparison_data).sort_values(["Dataset", "Arch"])
    
    # Save LaTeX table
    table_path = TABLE_DIR / "tab1_baseline_vs_max_collapse.tex"
    with open(table_path, "w") as f:
        f.write(
            comparison_df.to_latex(
                index=False,
                float_format="%.2f",
                caption="Baseline vs. Max Collapse.",
                label="tab:baseline_vs_max_collapse",
            )
        )
    print(f"Table 1 saved to {table_path}")

def tab2(df: pd.DataFrame):
    efficiency_data = []

    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None: continue

        for _, r in g.iterrows():
            if r["model_type"] != "collapsed": continue

            efficiency_data.append(
                {
                    "Dataset": ds,
                    "Arch": arch,
                    "Exp": r["exp_name"],
                    "Comp Ratio (%)": r.get("d_params", 0),
                    "Acc Drop (%)": r.get("d_acc", 0),
                    "FLOPs Red (%)": r.get("d_flops", 0),
                }
            )

    if not efficiency_data: return

    efficiency_df = pd.DataFrame(efficiency_data).sort_values(["Dataset", "Arch"])

    table_path = TABLE_DIR / "tab2_model_efficiency.tex"
    with open(table_path, "w") as f:
        f.write(
            efficiency_df.to_latex(
                index=False,
                float_format="%.2f",
                caption="Efficiency metrics.",
                label="tab:model_efficiency",
            )
        )
    print(f"Table 2 saved to {table_path}")


# =========================
# UPDATED: Split Plotting
# =========================
def plot_expname_multi_metric(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory"],
    out_dir: Path = Path("./plots"),
):
    """
    Plots metrics with distinct visual splits for:
    1. Prune (JF) vs No-Prune (Kevin) -> mapped to Hue
    2. Quantized vs FP32 -> mapped to Style (markers)
    3. Model Groups -> mapped to X-axis
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()

    # 1. Clean exp_group so JF and Kevin share the same X-tick
    # Remove _quant, _JF, _Kevin to get the "Base" configuration name
    df["exp_group"] = (
        df["exp_name"]
        .str.replace("_quant", "", regex=False)
        .str.replace("_JF", "", regex=False) 
        .str.replace("_Kevin", "", regex=False)
        .str.strip()
        .str.strip("_")
    )

    # 2. Filter out baselines from this specific ablation plot to avoid clutter
    # (Optional, but usually ablations focus on the collapsed models)
    df_ablation = df[df["model_type"] == "collapsed"].copy()

    for architecture, df_arch in df_ablation.groupby("architecture"):

        datasets = sorted(df_arch["dataset"].unique())
        n_rows = len(datasets)
        n_cols = len(metrics)

        # Sort exp_groups by average parameter count (Largest -> Smallest)
        # This ensures the X-axis is ordered by model size
        if "params" in df_arch.columns:
            exp_order = (
                df_arch.groupby("exp_group")["params"]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        else:
            exp_order = sorted(df_arch["exp_group"].unique())

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(max(6, 0.8 * len(exp_order)) * n_cols, 4.0 * n_rows),
            squeeze=False,
            sharex='col' # Share X per column
        )

        for i, dataset in enumerate(datasets):
            g_dataset = df_arch[df_arch["dataset"] == dataset]

            for j, metric in enumerate(metrics):
                ax = axes[i, j]

                if g_dataset.empty or metric not in g_dataset.columns:
                    ax.axis("off")
                    continue

                # === THE SPLIT ===
                # Hue = Prune vs No Prune (posthoc_or_posttrain)
                # Style = Quantized vs Not (is_quantized)
                sns.pointplot(
                    data=g_dataset,
                    x="exp_group",
                    y=metric,
                    hue="posthoc_or_posttrain", 
                    style="is_quantized",
                    order=exp_order,
                    dodge=0.4,       # Separate the hues clearly side-by-side
                    join=False,      # Don't connect lines across groups (confusing)
                    markers=["o", "s", "^", "D"], 
                    scale=1.2,
                    errorbar=None,
                    ax=ax,
                )

                # Titles and Labels
                if i == 0:
                    ax.set_title(metric.capitalize(), fontsize=13, fontweight='bold')
                
                if i == n_rows - 1:
                    ax.set_xlabel("Model Config (Size Descending)", fontsize=10)
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                else:
                    ax.set_xlabel("")

                if j == 0:
                    ax.set_ylabel(f"{dataset}\nValue", fontsize=11)
                else:
                    ax.set_ylabel("")

                # Grid
                ax.grid(True, axis="y", linestyle="--", alpha=0.5)

                # Legend Management: Only show on top-left plot to save space
                if i == 0 and j == 0:
                    ax.legend(title="Method / Quant", fontsize=8, loc='upper right')
                else:
                    if ax.get_legend():
                        ax.get_legend().remove()

        fig.suptitle(
            f"{architecture}: Ablation Study\n"
            "Split by Pruning Method (Color) and Quantization (Shape)",
            fontsize=16,
            y=1.02,
        )

        plt.tight_layout()
        save_path = out_dir / f"{architecture}_ablation_split.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved ablation plot: {save_path}")
        plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        
        print("\nLoaded Data Summary:")
        print(df.head())

        print("\n==============================")
        print(" GENERATING FIGURES ")
        print("==============================")
        
        # Standard Figures
        fig1(df)
        fig2(df)
        fig3(df)
        fig4(df)
        fig5(df)
        
        # Tables
        tab1(df)
        tab2(df)

        # The Requested Update
        plot_expname_multi_metric(
            df,
            out_dir=FIG_DIR / "expname_ablations",
        )
        
        print("\nDone.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()