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

# =========================
# Plotting Style
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
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
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
# Data Loading & Utilities
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

def infer_dataset_from_path(p: Path) -> str:
    name = p.parent.parent.name.lower()
    for ds in DATASET_ORDER:
        if ds in name:
            return ds
    return "unknown" 

def infer_architecture_from_path(p: Path) -> str:
    name = p.parent.parent.name.lower()
    if "regnet" in name: return "RegNetX"
    if "vgg" in name: return "VGG16"
    if "inception" in name: return "InceptionNet"
    if "xception" in name: return "Xception"
    if "mobilenet" in name: return "MobileNet"
    if "convnext" in name: return "ConvNeXt"
    return "UnknownArch"

def infer_posthoc_or_posttrain(exp_name: str) -> str:
    n = exp_name.lower()
    if "jf" in n: return "Post-Prune (JF)"
    if "kevin" in n: return "No-Prune (Kevin)"
    if "original" in n or "baseline" in n: return "Baseline"
    return "Unknown"

def infer_model_type(exp_name: str) -> str:
    n = exp_name.lower()
    if "original" in n or "baseline" in n: return "baseline"
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
            for _, r in g.iterrows(): out.append(r)
            continue

        for _, r in g.iterrows():
            row = r.copy()
            if baseline["params"]:
                row["d_acc"] =  r["accuracy"] - baseline["accuracy"] 
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
                row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
                row["collapsed_fraction"] = row["d_params"] / 100.0
            out.append(row)
    return pd.DataFrame(out)

def save_plot_source_data(df: pd.DataFrame, filename: str):
    """Saves the data used for a specific plot to CSV and prints a preview."""
    filepath = TABLE_DIR / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"\n[Data Export] Saved source data for {filename} to {filepath}")
    
    desired_cols = ["dataset", "architecture", "exp_name", "d_acc", "d_params", "collapsed_fraction", "accuracy_delta"]
    existing_cols = [c for c in desired_cols if c in df.columns]
    
    if existing_cols:
        print(df[existing_cols].head(3).to_string())
    else:
        print(df.head(3).to_string())

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
    ax.legend(title="Configuration", frameon=True, loc="best")

# =========================
# Figures 1-5 (Existing)
# =========================
def fig1(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())
    
    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )

    plot_data_accum = []

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["d_params", "d_acc"])
            
            subdf = subdf.sort_values("d_params")
            plot_data_accum.append(subdf)

            if subdf.empty:
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
            if i == len(architectures)-1: format_reduction_axis(ax, "Parameter")
            if j == 0: format_accuracy_axis(ax)
            if i==0 and j==0: standard_legend(ax)
            else: 
                if ax.get_legend(): ax.get_legend().remove()

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig1_source_data")

    fig.suptitle("Accuracy vs Parameter Reduction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_params_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig2(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )

    plot_data_accum = []

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["d_flops", "d_acc"])
            
            subdf = subdf.sort_values("d_flops")
            plot_data_accum.append(subdf)

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
            if i == len(architectures)-1: format_reduction_axis(ax, "FLOPs")
            if j == 0: format_accuracy_axis(ax)
            if i==0 and j==0: standard_legend(ax)
            else: 
                if ax.get_legend(): ax.get_legend().remove()

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig2_source_data")

    fig.suptitle("Accuracy vs FLOPs Reduction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_flops_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig3(df: pd.DataFrame):
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )

    plot_data_accum = []

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["collapsed_fraction", "d_acc"])

            subdf = subdf.sort_values("collapsed_fraction")
            plot_data_accum.append(subdf)

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
            if i == len(architectures)-1: format_fraction_axis(ax)
            if j == 0: format_accuracy_axis(ax)
            if i==0 and j==0: standard_legend(ax)
            else: 
                if ax.get_legend(): ax.get_legend().remove()

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig3_source_data")

    fig.suptitle("Accuracy vs Collapsed Fraction", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_collapsed_fraction_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig4(df: pd.DataFrame):
    if "collapsed_fraction" not in df.columns: return

    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )

    plot_data_accum = []

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df[
                (df["architecture"] == arch) &
                (df["dataset"] == ds)
            ].dropna(subset=["collapsed_fraction", "d_flops"])

            subdf = subdf.sort_values("collapsed_fraction")
            plot_data_accum.append(subdf)

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
            if i == len(architectures)-1: format_fraction_axis(ax)
            if j == 0: ax.set_ylabel("FLOPs Reduction (%)")
            if i==0 and j==0: standard_legend(ax)
            else: 
                if ax.get_legend(): ax.get_legend().remove()

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig4_source_data")

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

    save_plot_source_data(max_collapse, "fig5_source_data")

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
# NEW: Figure 6 (Pareto Frontier)
# =========================
def fig6_pareto_frontier(df: pd.DataFrame):
    """
    Plots the 'Optimal Trade-off' curve (Pareto Frontier) for each dataset/architecture.
    It highlights the best models (highest accuracy for a given compression) 
    and fades out suboptimal ones.
    """
    architectures = sorted(df["architecture"].unique())
    datasets = sorted(df["dataset"].unique())

    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )
    
    plot_data_accum = []

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
            
            # 1. Identify Pareto Frontier
            # Sort by compression (ascending d_params)
            # We want to maximize d_acc for any given d_params
            subdf = subdf.sort_values("d_params")
            pareto_points = []
            current_max_acc = -np.inf
            
            # Simple heuristic: scan from right (high compression) to left? 
            # Actually, standard pareto for trade-off:
            # We want Max(d_params) AND Max(d_acc).
            # A point is on the frontier if no other point has BOTH higher params AND higher accuracy.
            # Simplified for plotting: Just the "Upper Envelope"
            
            # Sort by d_params descending (highest compression first)
            sorted_points = subdf.sort_values("d_params", ascending=False)
            current_max_acc = -np.inf
            
            for _, row in sorted_points.iterrows():
                if row["d_acc"] >= current_max_acc:
                    pareto_points.append(row)
                    current_max_acc = row["d_acc"]
            
            pareto_df = pd.DataFrame(pareto_points).sort_values("d_params")
            plot_data_accum.append(pareto_df)

            # 2. Plot ALL points faintly
            sns.scatterplot(
                data=subdf, x="d_params", y="d_acc",
                color="lightgray", alpha=0.5, s=30, ax=ax, legend=False
            )

            # 3. Plot Pareto Frontier strongly
            sns.lineplot(
                data=pareto_df, x="d_params", y="d_acc",
                color="black", linewidth=2, linestyle="--", 
                marker="o", label="Pareto Frontier", ax=ax
            )
            
            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1: format_reduction_axis(ax, "Parameter")
            if j == 0: format_accuracy_axis(ax)
            
            # Legend only on first
            if i==0 and j==0: 
                ax.legend(loc="best", fontsize=8)
            else:
                if ax.get_legend(): ax.get_legend().remove()

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig6_pareto_source")

    fig.suptitle("Pareto Frontier: Optimal Accuracy-Compression Trade-off", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_pareto_frontier.png", bbox_inches='tight')
    plt.close()


# =========================
# NEW: Figure 7 (Method Delta)
# =========================
def fig7_method_delta(df: pd.DataFrame):
    """
    Bar chart showing the Accuracy Delta (Method A - Method B).
    Specifically: Post-Prune (JF) minus No-Prune (Kevin).
    Positive = Post-Prune is better. Negative = No-Prune is better.
    """
    
    # 1. Prepare Data: Match JF and Kevin experiments
    # We need a common key. We used 'exp_group' earlier which strips the suffix.
    df = df.copy()
    df["exp_base"] = (
        df["exp_name"]
        .str.replace("_JF", "", regex=False)
        .str.replace("_Kevin", "", regex=False)
        .str.strip()
    )
    
    # Filter for only the two methods we care about
    valid_methods = ["Post-Prune (JF)", "No-Prune (Kevin)"]
    df_methods = df[df["posthoc_or_posttrain"].isin(valid_methods)]
    
    # Pivot to align them side-by-side
    pivot_cols = ["dataset", "architecture", "exp_base", "is_quantized"]
    df_pivot = df_methods.pivot_table(
        index=pivot_cols,
        columns="posthoc_or_posttrain",
        values="accuracy"
    ).reset_index()
    
    # Calculate Delta
    if "Post-Prune (JF)" in df_pivot.columns and "No-Prune (Kevin)" in df_pivot.columns:
        df_pivot["accuracy_delta"] = df_pivot["Post-Prune (JF)"] - df_pivot["No-Prune (Kevin)"]
    else:
        print("Skipping Fig 7: Missing matching data for JF vs Kevin comparison.")
        return

    # Drop NaNs (unmatched experiments)
    df_pivot = df_pivot.dropna(subset=["accuracy_delta"])
    
    # Save Source Data
    save_plot_source_data(df_pivot, "fig7_method_delta_source")

    # 2. Plotting
    architectures = sorted(df_pivot["architecture"].unique())
    datasets = sorted(df_pivot["dataset"].unique())
    
    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.5 * len(datasets), 4.5 * len(architectures)),
        sharex=True, sharey=True, squeeze=False,
    )

    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            subdf = df_pivot[
                (df_pivot["architecture"] == arch) &
                (df_pivot["dataset"] == ds)
            ].sort_values("exp_base")
            
            if subdf.empty:
                ax.axis("off")
                continue
            
            # Color bars: Blue if positive (JF wins), Red if negative (Kevin wins)
            colors = ["#1f77b4" if x >= 0 else "#d62728" for x in subdf["accuracy_delta"]]
            
            sns.barplot(
                data=subdf, x="exp_base", y="accuracy_delta",
                ax=ax, palette=colors, hue="exp_base", legend=False # hue to avoid warning
            )
            
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{arch} on {ds}")
            
            if j == 0:
                ax.set_ylabel("Acc Delta (JF - Kevin)\n(>0 means Post-Prune wins)")
            else:
                ax.set_ylabel("")
                
            if i == len(architectures)-1:
                ax.set_xlabel("Experiment Group")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

    fig.suptitle("Method Comparison: Post-Prune (JF) vs No-Prune (Kevin)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_method_delta.png", bbox_inches='tight')
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
        comparison_data.append({
            "Dataset": ds, "Arch": arch,
            "Base Acc": baseline["accuracy"], "Max Col Acc": max_collapse["accuracy"],
            "Acc Drop": max_collapse["d_acc"], "Col Fraction": max_collapse["collapsed_fraction"],
        })

    if not comparison_data: return
    comparison_df = pd.DataFrame(comparison_data).sort_values(["Dataset", "Arch"])
    
    save_plot_source_data(comparison_df, "tab1_baseline_max_collapse")

    table_path = TABLE_DIR / "tab1_baseline_vs_max_collapse.tex"
    with open(table_path, "w") as f:
        f.write(comparison_df.to_latex(index=False, float_format="%.2f"))
    print(f"Table 1 saved to {table_path}")

def tab2(df: pd.DataFrame):
    efficiency_data = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None: continue
        for _, r in g.iterrows():
            if r["model_type"] != "collapsed": continue
            efficiency_data.append({
                "Dataset": ds, "Arch": arch, "Exp": r["exp_name"],
                "Comp Ratio (%)": r.get("d_params", 0),
                "Acc Drop (%)": r.get("d_acc", 0),
                "FLOPs Red (%)": r.get("d_flops", 0),
            })

    if not efficiency_data: return
    efficiency_df = pd.DataFrame(efficiency_data).sort_values(["Dataset", "Arch"])
    
    save_plot_source_data(efficiency_df, "tab2_model_efficiency")

    table_path = TABLE_DIR / "tab2_model_efficiency.tex"
    with open(table_path, "w") as f:
        f.write(efficiency_df.to_latex(index=False, float_format="%.2f"))
    print(f"Table 2 saved to {table_path}")


# =========================
# Multi Metric Plot
# =========================
def plot_expname_multi_metric(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory"],
    out_dir: Path = Path("./plots"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["exp_group"] = (
        df["exp_name"]
        .str.replace("_quant", "", regex=False)
        .str.replace("_JF", "", regex=False) 
        .str.replace("_Kevin", "", regex=False)
        .str.strip()
        .str.strip("_")
    )

    df_ablation = df[df["model_type"] == "collapsed"].copy()

    for architecture, df_arch in df_ablation.groupby("architecture"):

        datasets = sorted(df_arch["dataset"].unique())
        n_rows = len(datasets)
        n_cols = len(metrics)

        if "params" in df_arch.columns:
            exp_order_index = (
                df_arch.groupby("exp_group")["params"]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        else:
            exp_order_index = sorted(df_arch["exp_group"].unique())
            
        df_arch["exp_group"] = pd.Categorical(
            df_arch["exp_group"], categories=exp_order_index, ordered=True
        )

        save_plot_source_data(df_arch, f"{architecture}_ablation_source")

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(6, 0.9 * len(exp_order_index)) * n_cols, 4.0 * n_rows),
            squeeze=False, sharex='col'
        )

        for i, dataset in enumerate(datasets):
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()
            g_dataset = g_dataset.sort_values("exp_group")

            for j, metric in enumerate(metrics):
                ax = axes[i, j]

                if g_dataset.empty or metric not in g_dataset.columns:
                    ax.axis("off")
                    continue

                sns.lineplot(
                    data=g_dataset,
                    x="exp_group",
                    y=metric,
                    hue="posthoc_or_posttrain", 
                    style="is_quantized",
                    markers=True,
                    dashes=True,
                    linewidth=2.5,
                    ax=ax,
                )

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

                ax.grid(True, axis="y", linestyle="--", alpha=0.5)

                if i == 0 and j == 0:
                    ax.legend(title="Method / Quant", fontsize=8, loc='upper right')
                else:
                    if ax.get_legend(): ax.get_legend().remove()

        fig.suptitle(
            f"{architecture}: Ablation Study\n"
            "Split by Pruning Method (Color) and Quantization (Shape/Dash)",
            fontsize=16, y=1.02,
        )

        plt.tight_layout()
        save_path = out_dir / f"{architecture}_ablation_split.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


# =========================
# Main
# =========================
if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        
        print("\n==============================")
        print(" GENERATING FIGURES & DATA ")
        print("==============================")
        
        fig1(df)
        fig2(df)
        fig3(df)
        fig4(df)
        fig5(df)
        
        # New Figures
        fig6_pareto_frontier(df)
        fig7_method_delta(df)
        
        tab1(df)
        tab2(df)

        plot_expname_multi_metric(
            df,
            out_dir=FIG_DIR / "expname_ablations",
        )
        
        print("\nDone. Check the 'tables/' folder for CSV data sources.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()