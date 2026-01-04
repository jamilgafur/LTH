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
            rows.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "exp_name": exp,
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
            row["d_acc"] = baseline["accuracy"] - r["accuracy"]
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
    # this method saves into the FIG_DIR
    # For each architecture as a row and dataset as a column in a single subplot
    # plot a line plot d_params vs d_acc, for is_quantized and non-quantized 
    # use df.unique to get all architectures, and DATASET_ORDER for datasets
    # x axis log scale

    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
    )
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j] if len(architectures) > 1 else axes[j]
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                ax.set_title(f"{arch} - {ds} (no data)")
                continue
            sns.lineplot(
                data=subdf,
                x="d_params",
                y="d_acc",
                hue="is_quantized",
                style="model_type",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_title(f"{arch} - {ds}")
            ax.set_xlabel("Parameter Reduction (%)")
            ax.set_ylabel("Accuracy Change (%)")
            ax.legend(title="Quantized / Model Type")
    plt.tight_layout()
    fig_path = FIG_DIR / "fig1_dparams_vs_dacc.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()
    # print out the data and statistics used in the figure to the terminal
    print("\nData used in Figure 1:")
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print(subdf[["d_params", "d_acc", "is_quantized", "model_type"]].to_string(index=False))
    # print a latex table for each archtirecture and dataset the baseline accuracy, maximum collapsible fraction, corresponding exp_name and accuracy drop
    print("\nLatex Tables:")
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            baseline = find_baseline(subdf)
            if baseline is None:
                continue
            max_collapse = subdf.loc[subdf["collapsed_fraction"].idxmax()]
            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print("\\begin{table}[h]")
            print("\\centering")
            print("\\begin{tabular}{lcc}")
            print("\\hline")
            print(" & Baseline & Max Collapse \\\\")
            print("\\hline")
            print(f"Exp Name & {baseline['exp_name']} & {max_collapse['exp_name']} \\\\")
            print(f"Accuracy & {baseline['accuracy']:.2f} & {max_collapse['accuracy']:.2f} \\\\")
            print(f"Collapsed Fraction & 0.00 & {max_collapse['collapsed_fraction']:.2f} \\\\")
            print(f"Accuracy Drop & 0.00 & {max_collapse['d_acc']:.2f} \\\\")
            print("\\hline")
            print("\\end{tabular}")
            print(f"\\caption{{Baseline and Maximum Collapse for {arch} on {ds}}}")
            print("\\end{table}")

def fig2(df: pd.DataFrame):
    # In here we show the FLOPS reduction, memory reduction and parameter reduction as their own columns in a subplot
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()
    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
    )
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j] if len(architectures) > 1 else axes[j]
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                ax.set_title(f"{arch} - {ds} (no data)")
                continue
            sns.lineplot(
                data=subdf,
                x="d_flops",
                y="d_acc",
                hue="is_quantized",
                style="model_type",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_title(f"{arch} - {ds}")
            ax.set_xlabel("FLOPS Reduction (%)")
            ax.set_ylabel("Accuracy Change (%)")
            ax.legend(title="Quantized / Model Type")
    plt.tight_layout()
    fig_path = FIG_DIR / "fig2_dflops_vs_dacc.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()
    # print out the data and statistics used in the figure to the terminal
    print("\nData used in Figure 2:")
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print(subdf[["d_flops", "d_acc", "is_quantized", "model_type"]].to_string(index=False))

def fig3(df: pd.DataFrame):
    # Surrogate Approximation Error (Fig. 3): A scatter plot correlating the local approximation error of the surrogate with the final downstream accuracy.
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()
    
    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
    )
    
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j] if len(architectures) > 1 else axes[j]
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                ax.set_title(f"{arch} - {ds} (no data)")
                continue

            # Use collapsed_fraction as a proxy for the approximation error
            sns.scatterplot(
                data=subdf,
                x="collapsed_fraction",  # Collapsed fraction as a proxy for approximation error
                y="d_acc",
                hue="is_quantized",
                style="model_type",
                ax=ax,
                markers=True,
            )
            ax.set_title(f"{arch} - {ds}")
            ax.set_xlabel("Collapsed Fraction (Proxy for Approximation Error)")
            ax.set_ylabel("Accuracy Change (%)")
            ax.legend(title="Quantized / Model Type")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig3_approximation_error_vs_accuracy.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()
    
    # Print out the data and statistics used in the figure to the terminal
    print("\nData used in Figure 3:")
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print(subdf[["collapsed_fraction", "d_acc", "is_quantized", "model_type"]].to_string(index=False))

def fig4(df: pd.DataFrame):
    # Efficiency Trade-offs (Fig. 4): Line or scatter plots showing the reduction in FLOPs, latency, and activation memory footprint as more of the network is collapsed.
    architectures = df["architecture"].unique()
    datasets = df["dataset"].unique()
    
    fig, axes = plt.subplots(
        len(architectures),
        len(datasets),
        figsize=(5 * len(datasets), 4 * len(architectures)),
        sharex=True,
        sharey=True,
    )
    
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            ax = axes[i, j] if len(architectures) > 1 else axes[j]
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                ax.set_title(f"{arch} - {ds} (no data)")
                continue

            sns.lineplot(
                data=subdf,
                x="collapsed_fraction",  # Fraction of network collapsed
                y="d_flops",
                hue="is_quantized",
                style="model_type",
                markers=True,
                dashes=False,
                ax=ax,
            )
            ax.axhline(0, color="gray", linestyle="--")
            ax.set_title(f"{arch} - {ds}")
            ax.set_xlabel("Collapsed Fraction")
            ax.set_ylabel("FLOPS Reduction (%)")
            ax.legend(title="Quantized / Model Type")

    plt.tight_layout()
    fig_path = FIG_DIR / "fig4_flops_vs_collapsed_fraction.png"
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()
    
    # Print out the data and statistics used in the figure to the terminal
    print("\nData used in Figure 4:")
    for i, arch in enumerate(architectures):
        for j, ds in enumerate(datasets):
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            print(f"\nArchitecture: {arch}, Dataset: {ds}")
            print(subdf[["collapsed_fraction", "d_flops", "is_quantized", "model_type"]].to_string(index=False))

def fig5(df: pd.DataFrame):
    # Cross-Architecture Comparisons (Fig. 5): A bar chart comparing the maximum collapsible fraction across various models.
    max_collapse_data = []
    
    for arch in df["architecture"].unique():
        for ds in df["dataset"].unique():
            subdf = df[(df["architecture"] == arch) & (df["dataset"] == ds)]
            if subdf.empty:
                continue
            max_collapse = subdf.loc[subdf["collapsed_fraction"].idxmax()]
            max_collapse_data.append(
                {
                    "architecture": arch,
                    "dataset": ds,
                    "max_collapsed_fraction": max_collapse["collapsed_fraction"],
                }
            )

    max_collapse_df = pd.DataFrame(max_collapse_data)
    
    # Bar plot for max collapsed fraction comparison
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
    plt.savefig(fig_path)
    print(f"Figure saved to {fig_path}")
    plt.close()
    
    # Print out the data and statistics used in the figure to the terminal
    print("\nData used in Figure 5:")
    print(max_collapse_df[["architecture", "dataset", "max_collapsed_fraction"]].to_string(index=False))

def tab1(df: pd.DataFrame):
    # Tabular Summary for Baseline vs Max Collapse (Tab. 2)
    # This table compares the baseline performance with the maximum collapsed fraction for each dataset and architecture.
    comparison_data = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            continue
        max_collapse = g.loc[g["collapsed_fraction"].idxmax()]
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
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nTabular Summary (Tab. 2): Baseline vs Max Collapse Comparison:")
    print(comparison_df.to_string(index=False))

    # Save as latex table
    table_path = TABLE_DIR / "tab1_baseline_vs_max_collapse.tex"
    with open(table_path, "w") as f:
        f.write(comparison_df.to_latex(index=False, float_format="%.2f"))
    print(f"Table saved to {table_path}")

def tab2(df: pd.DataFrame):
    # Tabular Summary of Model Efficiency (Tab. 3)
    # This table shows the compression ratio, accuracy drop, and parameter reduction for collapsed models.
    efficiency_data = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            continue
        for _, r in g.iterrows():
            if r["model_type"] == "collapsed":
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
    
    efficiency_df = pd.DataFrame(efficiency_data)
    print("\nTabular Summary (Tab. 3): Model Efficiency (Compression Ratio, Accuracy Drop, and Parameter Reduction):")
    print(efficiency_df.to_string(index=False))

    # Save as latex table
    table_path = TABLE_DIR / "tab2_model_efficiency.tex"
    with open(table_path, "w") as f:
        f.write(efficiency_df.to_latex(index=False, float_format="%.2f"))
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
