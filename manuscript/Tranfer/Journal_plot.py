from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List
import numpy as np
import logging
import re

# Import custom modules (ensure these match your project structure)
from manuscript.Tranfer.utils import load_dataset
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Set the logging level for matplotlib to WARNING or ERROR
logging.getLogger('matplotlib').setLevel(logging.WARNING)

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
    # Look for both master_metrics and the per-experiment json files
    files = list(RESULTS_DIR.rglob("*metrics.json"))
    # Also look for the centralized consolidated metrics
    files += list(RESULTS_DIR.rglob("metrics_consolidated/*.json"))
    
    files = list(set(files)) # Remove duplicates
    print(f"[•] Found {len(files)} metrics JSON files.")

    rows = []

    for p in files:
        try:
            with open(p) as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            print(f"[!] Could not decode JSON: {p}")
            continue

        for exp_name, metrics in raw.items():
            # Basic metric extraction
            row = {
                "params": metrics.get("param_count", None),
                "inference_time": metrics.get("inference_time", None),
                "flops": metrics.get("flops", None),
                "memory": metrics.get("total_size_mb", None),
                "accuracy": metrics.get("final_accuracy", None),
                "metadata": metrics.get("metadata", {}),
                "cka": metrics.get("cka", None),
                "history": metrics.get("history", None),
                "file_path": str(p)
            }
            
            # --- Inference Logic Updates ---
            path_str = str(p).lower()
            
            # 1. Dataset & Architecture
            row["dataset"] = infer_dataset_from_string(exp_name) or infer_dataset_from_string(path_str)
            row["architecture"] = infer_architecture_from_string(exp_name) or infer_architecture_from_string(path_str)
            
            # 2. Experiment Specifics
            row["exp_name"] = exp_name
            row["break_group"] = infer_break_group(path_str)
            row["posthoc_or_posttrain"] = infer_posthoc_or_posttrain(path_str)
            row["model_type"] = infer_model_type(path_str)
            row["is_quantized"] = infer_isquant(exp_name)
            
            # 3. Collapse Specifics
            if "collapse" in metrics:
                row["collapse_start"] = metrics["collapse"][0]
                row["collapse_end"] = metrics["collapse"][1]
            elif "post_collapse" in path_str:
                # Fallback: extract from path like post_collapse_layer1_layer3
                parts = path_str.split("post_collapse_")[-1].split("_")
                if len(parts) >= 2:
                    row["collapse_start"] = parts[0]
                    row["collapse_end"] = parts[1]

            rows.append(row)
            
    return pd.DataFrame(rows)

def infer_dataset_from_string(s: str) -> str:
    s = s.lower()
    if "cifar100" in s: return "cifar100" # Check 100 before 10
    if "cifar10" in s: return "cifar10"
    if "tinyimagenet" in s: return "tinyimagenet"
    if "imagenet" in s: return "imagenet"
    return None

def infer_architecture_from_string(s: str) -> str:
    s = s.lower()
    if "regnet" in s: return "RegNetX"
    if "vgg" in s: return "VGG16"
    if "inception" in s: return "InceptionNet"
    if "xception" in s: return "Xception"
    if "mobilenet" in s: return "MobileNet"
    if "convnext" in s: return "ConvNeXt"
    return None

def infer_break_group(path_str: str) -> str:
    # Look for patterns like "break3", "break_3", "group3"
    match = re.search(r"break_?(\d+)", path_str)
    if match:
        return f"Break {match.group(1)}"
    return "Unknown Break"

def infer_posthoc_or_posttrain(path_str: str) -> str:
    """
    Decides if this is the initialized model (Post-Train / Kevin) 
    or the pruned model (Post-Hoc / JF) based on directory names.
    """
    path_str = path_str.lower()
    
    # Logic based on main_1.py directory naming
    if "_pruned" in path_str:
        return "Post-Hoc (JF)"
    if "_initialized" in path_str:
        return "Post-Train (Kevin)"
    
    # Fallback for baseline or unclear paths
    if "baseline" in path_str:
        return "Baseline"
        
    return "Unknown"

def infer_model_type(path_str: str) -> str:
    if "post_collapse" in path_str:
        return "collapsed"
    if "baseline" in path_str:
        return "baseline"
    return "other"

def infer_isquant(exp_name: str) -> bool:
    return "quant" in exp_name.lower()

def find_baseline(df: pd.DataFrame):
    """
    Finds the baseline row for a specific group (dataset/arch/break_group).
    Prioritizes the 'initialized' baseline if both exist, as that is the standard comparison point.
    """
    # Filter for baseline rows
    mask = df["model_type"] == "baseline"
    baselines = df[mask]
    
    if baselines.empty:
        return None
        
    # If we have multiple baselines (e.g. one for JF, one for Kevin), 
    # we usually compare against the Post-Train (Kevin/Initialized) as the "Dense" reference 
    # OR we compare strictly within groups.
    
    # For normalization, let's pick the one with highest accuracy as the "Golden Standard"
    return baselines.sort_values("accuracy", ascending=False).iloc[0]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    # Group by dataset, architecture, AND break_group to ensure fair comparison
    # (e.g. don't compare Break 3 collapse against Break 4 baseline)
    for (ds, arch, brk), g in df.groupby(["dataset", "architecture", "break_group"]):
        baseline = find_baseline(g)
        
        if baseline is None:
            # Try to find a generic baseline for this dataset/arch ignoring break group
            # This handles cases where baseline might be shared
            backup_baseline = find_baseline(df[(df["dataset"]==ds) & (df["architecture"]==arch)])
            if backup_baseline is not None:
                baseline = backup_baseline
            else:
                # warnings.warn(f"No baseline for {ds}-{arch}-{brk}")
                pass

        for _, r in g.iterrows():
            row = r.copy()
            if baseline is not None and baseline["params"]:
                row["d_acc"] = r["accuracy"] - baseline["accuracy"]
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
                row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
                row["collapsed_fraction"] = row["d_params"] / 100.0
            else:
                # If no baseline, we can't calculate deltas
                row["d_acc"] = np.nan
                row["d_params"] = np.nan 
                row["collapsed_fraction"] = np.nan
                
            out.append(row)
    return pd.DataFrame(out)

def save_plot_source_data(df: pd.DataFrame, filename: str):
    filepath = TABLE_DIR / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    print(f"\n[Data Export] Saved source data for {filename} to {filepath}")

# =========================
# Plotting Helpers (Unchanged)
# =========================
def format_accuracy_axis(ax):
    ax.set_ylabel("Accuracy Change (%)\n(higher is better)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

def format_reduction_axis(ax, label):
    ax.set_xlabel(f"{label} Reduction (%)\n(higher is better)")

def format_fraction_axis(ax):
    ax.set_xlabel("Collapsed Fraction\n(higher = more compression)")

def standard_legend(ax):
    ax.legend(title="Config", frameon=True, loc="best", fontsize=8)

# =========================
# Figures 1-5
# =========================
def fig1(df: pd.DataFrame):
    """Accuracy vs Parameter Reduction"""
    df_plot = df.dropna(subset=["d_params", "d_acc"]).sort_values("d_params")
    if df_plot.empty: return

    g = sns.relplot(
        data=df_plot,
        x="d_params", y="d_acc",
        col="dataset", row="architecture",
        hue="posthoc_or_posttrain", style="break_group",
        kind="line", markers=True, dashes=False,
        height=3, aspect=1.2,
        facet_kws={'sharex': True, 'sharey': False} # Allow y-axis to vary per arch
    )
    
    g.set_titles("{row_name} | {col_name}")
    g.set_xlabels("Param Reduction (%)")
    g.set_ylabels("Acc Change (%)")
    
    for ax in g.axes.flat:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    save_plot_source_data(df_plot, "fig1_source_data")
    plt.savefig(FIG_DIR / "fig1_params_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig2(df: pd.DataFrame):
    """Accuracy vs FLOPs Reduction"""
    df_plot = df.dropna(subset=["d_flops", "d_acc"]).sort_values("d_flops")
    if df_plot.empty: return

    g = sns.relplot(
        data=df_plot,
        x="d_flops", y="d_acc",
        col="dataset", row="architecture",
        hue="posthoc_or_posttrain", style="break_group",
        kind="line", markers=True, dashes=False,
        height=3, aspect=1.2,
        facet_kws={'sharex': True, 'sharey': False}
    )

    g.set_titles("{row_name} | {col_name}")
    g.set_xlabels("FLOPs Reduction (%)")
    g.set_ylabels("Acc Change (%)")

    for ax in g.axes.flat:
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    save_plot_source_data(df_plot, "fig2_source_data")
    plt.savefig(FIG_DIR / "fig2_flops_vs_accuracy.png", bbox_inches='tight')
    plt.close()

def fig4(df: pd.DataFrame):
    """FLOPs Reduction vs Collapsed Fraction (Linearity check)"""
    df_plot = df.dropna(subset=["collapsed_fraction", "d_flops"]).sort_values("collapsed_fraction")
    if df_plot.empty: return

    g = sns.relplot(
        data=df_plot,
        x="collapsed_fraction", y="d_flops",
        col="dataset", row="architecture",
        hue="posthoc_or_posttrain",
        kind="line", markers=True,
        height=3, aspect=1.2
    )
    
    save_plot_source_data(df_plot, "fig4_source_data")
    plt.savefig(FIG_DIR / "fig4_collapsed_fraction_vs_flops.png", bbox_inches='tight')
    plt.close()

def fig6(df: pd.DataFrame):
    """Pareto Frontier"""
    architectures = sorted(df["architecture"].dropna().unique())
    datasets = sorted(df["dataset"].dropna().unique())

    if not architectures or not datasets: return

    fig, axes = plt.subplots(
        len(architectures), len(datasets),
        figsize=(5.0 * len(datasets), 4.0 * len(architectures)),
        sharex=True, sharey=False, squeeze=False,
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
            
            # Identify Pareto points
            subdf = subdf.sort_values("d_params", ascending=False) # High compression first
            pareto_points = []
            current_max_acc = -np.inf
            
            for _, row in subdf.iterrows():
                if row["d_acc"] >= current_max_acc:
                    pareto_points.append(row)
                    current_max_acc = row["d_acc"]
            
            pareto_df = pd.DataFrame(pareto_points).sort_values("d_params")
            plot_data_accum.append(pareto_df)

            # Plot All points
            sns.scatterplot(
                data=subdf, x="d_params", y="d_acc",
                hue="posthoc_or_posttrain", style="break_group",
                alpha=0.6, s=60, ax=ax, legend=(i==0 and j==0)
            )

            # Plot Frontier
            sns.lineplot(
                data=pareto_df, x="d_params", y="d_acc",
                color="black", linewidth=2, linestyle="--", 
                label="Pareto Frontier" if i==0 and j==0 else None, ax=ax,
                errorbar=None
            )
            
            ax.set_title(f"{arch} on {ds}")
            ax.axhline(0, color="gray", alpha=0.5, linestyle=":")
            
            if i == len(architectures)-1: format_reduction_axis(ax, "Parameter")
            if j == 0: format_accuracy_axis(ax)

    if plot_data_accum:
        save_plot_source_data(pd.concat(plot_data_accum), "fig6_pareto_source")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_pareto_frontier.png", bbox_inches='tight')
    plt.close()
def fig5(df: pd.DataFrame):
    """
    Bar chart of Maximum Model Collapsibility per architecture/dataset.
    """
    if "collapsed_fraction" not in df.columns: return
    
    # Get the max collapse observed for each Arch/Dataset pair
    # We ignore break_group here to find the absolute max across all runs
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
        palette="viridis"
    )

    ax.set_title("Maximum Model Collapsibility Observed")
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Max Collapsed Fraction (0-1)")
    ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.axhline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_max_collapsibility.png", bbox_inches='tight')
    plt.close()

def fig7(df: pd.DataFrame):
    """
    Bar chart showing the Accuracy Delta (Method A - Method B).
    Updated for new labels: "Post-Hoc (JF)" vs "Post-Train (Kevin)".
    """
    df = df.copy()
    
    # 1. Create a common key to match experiments (strip the method suffix)
    # We use 'exp_name' but need to remove the method identifiers
    df["exp_base"] = (
        df["exp_name"]
        .str.replace("_JF", "", regex=False)
        .str.replace("_Kevin", "", regex=False)
        .str.replace("_pruned", "", regex=False)
        .str.replace("_initialized", "", regex=False)
        .str.strip()
    )
    
    # 2. Filter for the two methods we compare
    # Note: Ensure these match the outputs of infer_posthoc_or_posttrain
    valid_methods = ["Post-Hoc (JF)", "Post-Train (Kevin)"]
    df_methods = df[df["posthoc_or_posttrain"].isin(valid_methods)]
    
    if df_methods.empty:
        print("[!] Fig7 Skipped: No matching Post-Hoc/Post-Train pairs found.")
        return

    # 3. Pivot to align them side-by-side
    # We must include 'break_group' in the index so we don't mix different breaks
    pivot_cols = ["dataset", "architecture", "exp_base", "break_group", "is_quantized"]
    
    try:
        df_pivot = df_methods.pivot_table(
            index=pivot_cols,
            columns="posthoc_or_posttrain",
            values="accuracy"
        ).reset_index()
    except Exception as e:
        print(f"[!] Fig7 Pivot Failed: {e}")
        return

    # 4. Calculate Delta (Post-Hoc - Post-Train)
    if "Post-Hoc (JF)" in df_pivot.columns and "Post-Train (Kevin)" in df_pivot.columns:
        df_pivot["accuracy_delta"] = df_pivot["Post-Hoc (JF)"] - df_pivot["Post-Train (Kevin)"]
    else:
        print("[!] Fig7 Skipped: Missing one of the required columns after pivot.")
        return

    df_pivot = df_pivot.dropna(subset=["accuracy_delta"])
    save_plot_source_data(df_pivot, "fig7_method_delta_source")

    # 5. Plotting
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
            
            # Color: Blue if Positive (JF better), Red if Negative (Kevin better)
            colors = ["#1f77b4" if x >= 0 else "#d62728" for x in subdf["accuracy_delta"]]
            
            sns.barplot(
                data=subdf, x="exp_base", y="accuracy_delta",
                ax=ax, palette=colors, hue="exp_base", legend=False
            )
            
            ax.axhline(0, color="black", linewidth=1)
            ax.set_title(f"{arch} on {ds}")
            
            if j == 0:
                ax.set_ylabel("Acc Delta (JF - Kevin)")
            else:
                ax.set_ylabel("")
                
            if i == len(architectures)-1:
                ax.set_xlabel("Experiment Group")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    fig.suptitle("Method Comparison: Post-Hoc (JF) vs Post-Train (Kevin)", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_method_delta.png", bbox_inches='tight')
    plt.close()

def fig8(df: pd.DataFrame, metrics: list[str] = ["accuracy", "params", "flops", "memory"]):
    """
    Ablation Study: Plots metrics across sorted experiment groups.
    """
    out_dir = FIG_DIR / "expname_ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    # Create a cleaner display name for the x-axis
    df["display_group"] = df["exp_name"].apply(lambda x: x.split("_pretrain")[0])

    # Filter only for collapsed models to see the trend
    df_ablation = df[df["model_type"] == "collapsed"].copy()

    for architecture, df_arch in df_ablation.groupby("architecture"):
        datasets = sorted(df_arch["dataset"].unique())
        n_rows = len(datasets)
        n_cols = len(metrics)
        
        # Sort experiments by parameter count (descending) to show "compression progress"
        if "params" in df_arch.columns:
            exp_order = (
                df_arch.groupby("display_group")["params"]
                .mean()
                .sort_values(ascending=False)
                .index
            )
        else:
            exp_order = sorted(df_arch["display_group"].unique())

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(max(6, 1.2 * len(exp_order)) * n_cols, 4.0 * n_rows),
            squeeze=False, sharex='col'
        )

        for i, dataset in enumerate(datasets):
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()
            
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                if g_dataset.empty or metric not in g_dataset.columns:
                    ax.axis("off")
                    continue

                sns.lineplot(
                    data=g_dataset,
                    x="display_group", y=metric,
                    hue="posthoc_or_posttrain", 
                    style="break_group", # Differentiate breaks
                    markers=True, dashes=False,
                    ax=ax
                )

                if i == 0: ax.set_title(metric.capitalize(), fontweight='bold')
                if i == n_rows - 1:
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                else:
                    ax.set_xlabel("")

        fig.suptitle(f"{architecture}: Ablation Study", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / f"{architecture}_ablation.png", bbox_inches='tight')
        plt.close()

def tab1(df: pd.DataFrame):
    """
    Table 1: Baseline Accuracy vs Max Collapsed Accuracy vs Drop.
    """
    comparison_data = []

    # Group by dataset, architecture AND break_group to be precise
    for (ds, arch, brk), g in df.groupby(["dataset", "architecture", "break_group"]):
        baseline = find_baseline(g)
        if baseline is None: continue
        
        # Find best collapsed model (highest collapsed fraction)
        g_valid = g.dropna(subset=["collapsed_fraction"])
        if g_valid.empty: continue
        
        max_collapse = g_valid.loc[g_valid["collapsed_fraction"].idxmax()]
        
        comparison_data.append({
            "Dataset": ds, "Arch": arch, "Break": brk,
            "Base Acc": baseline["accuracy"], 
            "Max Col Acc": max_collapse["accuracy"],
            "Acc Drop": max_collapse["d_acc"], 
            "Col Fraction": max_collapse["collapsed_fraction"],
        })

    if not comparison_data: return
    comparison_df = pd.DataFrame(comparison_data).sort_values(["Dataset", "Arch"])
    
    save_plot_source_data(comparison_df, "tab1_baseline_max_collapse")

    table_path = TABLE_DIR / "tab1_baseline_vs_max_collapse.tex"
    with open(table_path, "w") as f:
        f.write(comparison_df.to_latex(index=False, float_format="%.2f"))
    print(f"[•] Table 1 saved to {table_path}")

def tab2(df: pd.DataFrame):
    """
    Table 2: Efficiency metrics for all collapsed models.
    """
    efficiency_data = []
    
    # We just dump the collapsed rows with relevant metrics
    collapsed_df = df[df["model_type"] == "collapsed"].copy()
    
    for _, r in collapsed_df.iterrows():
        efficiency_data.append({
            "Dataset": r["dataset"], 
            "Arch": r["architecture"], 
            "Break": r.get("break_group", "N/A"),
            "Exp": r["exp_name"],
            "Comp Ratio (%)": r.get("d_params", 0),
            "Acc Drop (%)": r.get("d_acc", 0),
            "FLOPs Red (%)": r.get("d_flops", 0),
        })

    if not efficiency_data: return
    efficiency_df = pd.DataFrame(efficiency_data).sort_values(["Dataset", "Arch", "Break"])
    
    save_plot_source_data(efficiency_df, "tab2_model_efficiency")

    table_path = TABLE_DIR / "tab2_model_efficiency.tex"
    with open(table_path, "w") as f:
        f.write(efficiency_df.to_latex(index=False, float_format="%.2f"))
    print(f"[•] Table 2 saved to {table_path}")
# =========================
# Diagnostic Figures (PWCCA)
# =========================
def pwcca_distance(X, Y, epsilon=1e-10):
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Covariance
    Cxx = X.T @ X + epsilon * np.eye(X.shape[1])
    Cyy = Y.T @ Y + epsilon * np.eye(Y.shape[1])
    Cxy = X.T @ Y

    # Whitening
    Ux, Sx, _ = np.linalg.svd(Cxx)
    Uy, Sy, _ = np.linalg.svd(Cyy)

    Sx_inv_sqrt = np.diag(1.0 / np.sqrt(Sx + epsilon))
    Sy_inv_sqrt = np.diag(1.0 / np.sqrt(Sy + epsilon))

    T = Sx_inv_sqrt @ Ux.T @ Cxy @ Uy @ Sy_inv_sqrt

    # CCA
    _, s, Vt = np.linalg.svd(T)
    alpha = np.sum(np.abs(Vt), axis=1)
    alpha /= np.sum(alpha)

    return float(np.sum(alpha * s))

def extract_representation(model, dataloader, device, max_batches=5):
    model.eval()
    activations = []

    def hook_fn(_, __, output):
        activations.append(output.detach().cpu())

    # Find the last convolutional or linear layer
    target_module = None
    for m in reversed(list(model.modules())):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            target_module = m
            break
            
    if target_module is None: return None

    handle = target_module.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= max_batches: break
            model(x.to(device))

    handle.remove()
    acts = torch.cat(activations, dim=0)
    return acts.flatten(start_dim=1).numpy()

if __name__ == "__main__":
    df = load_results()
    
    if df.empty:
        print("[!] No results found. Check your directory structure.")
    else:
        df = normalize(df)
        print(f"[•] Loaded {len(df)} rows.")

        print("\n==============================")
        print(" GENERATING FIGURES & DATA ")
        print("==============================")
        
        fig1(df)
        fig2(df)
        fig4(df)
        fig5(df)  # Restored
        fig6(df)
        fig7(df)  # Restored
        fig8(df)  # Restored
        
        # fig10(device="cuda" if torch.cuda.is_available() else "cpu")
        
        tab1(df)  # Restored
        tab2(df)  # Restored

        print("\nDone. Check the 'tables/' folder for CSV data sources.")
    df = load_results()
    
    if df.empty:
        print("[!] No results found. Check your directory structure.")
    else:
        df = normalize(df)
        print(f"[•] Loaded {len(df)} rows. Sample:")
        print(df[["dataset", "architecture", "break_group", "posthoc_or_posttrain"]].head())

        print("\n==============================")
        print(" GENERATING FIGURES & DATA ")
        print("==============================")
        
        fig1(df)
        fig2(df)
        fig4(df)
        fig6(df)
        
        # PWCCA requires live model loading, can be slow
        # fig10(device="cuda" if torch.cuda.is_available() else "cpu")

        print("\nDone. Check the 'tables/' folder for CSV data sources.")