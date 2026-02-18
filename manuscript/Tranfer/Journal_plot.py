from __future__ import annotations

import glob
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg

# Local imports (Assumed available based on original file)
from manuscript.Tranfer.utils import load_dataset
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from collapse import collapse_only

# =========================
# Configuration & Style
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

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

RESULTS_DIR = Path("./")
FIG_DIR = Path("./figures")
TABLE_DIR = Path("./tables")

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

DATASET_ORDER = ["cifar10_", "cifar100_", "tinyimagenet", "imagenet", "ConvNeXt"]

# =========================
# Data Loading & Utilities
# =========================

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

def infer_model_type(exp_name: str) -> str:
    n = exp_name.lower()
    if "original" in n or "baseline" in n: return "baseline"
    return "collapsed"

def infer_isquant(exp_name: str) -> bool:
    return "quant" in exp_name.lower()

def infer_posthoc_or_posttrain(exp_name: str, architecture: str) -> str:
    """
    Determines the method group.
    - Baseline: If 'original' or 'baseline' in name.
    - Pruned/No-Prune: ONLY for VGG16 and RegNetX.
    - Collapsed: For all other architectures (averages JF/Kevin).
    """
    n = exp_name.lower()
    
    if "original" in n or "baseline" in n:
        return "Baseline"
        
    # Only distinguish JF vs Kevin for specific architectures
    if architecture in ["VGG16", "RegNetX"]:
        if "jf" in n: return "Pruned (JF)"
        if "kevin" in n: return "No-Prune (Kevin)"
        
    # For ConvNeXt, MobileNet, etc., merge them into one group
    return "Collapsed"

def clean_exp_name(exp_name: str) -> str:
    """
    Standardizes experiment names by removing suffixes and meta-tags.
    Used to group 'Stage 2-7' and 'Stage 2-7 (Quant)' together.
    """
    n = exp_name
    # Remove suffixes
    n = n.replace("_quant", "").replace("_JF", "").replace("_Kevin", "")
    
    # Remove architecture prefixes if present
    for arch in ["RegNetX_400MF_", "VGG16_", "MobileNet_", "ConvNeXt_", "InceptionNet_", "XceptionNet_"]:
        n = n.replace(arch, "")
    
    # Standardize Block/Stage format
    n = n.replace("Block ", "Block-").replace("Stage ", "Stage-") 
    n = n.replace(" Only", "")
    
    # Handle Baseline/Original
    if "Original" in n or "Baseline" in n: 
        return "Original"
        
    return n.strip()

def find_baseline(df: pd.DataFrame):
    mask = (
        df["exp_name"].str.lower().str.contains("original")
        | df["exp_name"].str.lower().str.contains("baseline")
    )
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def load_results() -> pd.DataFrame:
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    if not files:
        # Fallback if no subdirectories found (e.g. flat file)
        if (RESULTS_DIR / "merged_metrics.json").exists():
            files = [RESULTS_DIR / "merged_metrics.json"]
        else:
            raise FileNotFoundError("No merged_metrics.json files found")

    rows = []

    for p in files:
        dataset = infer_dataset_from_path(p)
        arch = infer_architecture_from_path(p)
        
        try:
            with open(p) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p}: {e}")
            continue

        for exp_name, metrics in raw.items():
            # Basic inferences
            method_group = infer_posthoc_or_posttrain(exp_name, arch)
            is_quant = infer_isquant(exp_name)
            
            # Name cleaning for plotting
            base_name = clean_exp_name(exp_name)
            display_name = f"{base_name}\n(Quant)" if is_quant else base_name

            rows.append(
                {
                    "dataset": dataset,
                    "architecture": arch,
                    "exp_name": exp_name,
                    "base_name": base_name,
                    "display_name": display_name,
                    "posthoc_or_posttrain": method_group,
                    "model_type": infer_model_type(exp_name),
                    "is_quantized": is_quant,

                    # Core metrics
                    "accuracy": metrics.get("final_accuracy"),
                    "params": metrics.get("param_count"),
                    "flops": metrics.get("flops"),
                    "memory": metrics.get("total_size_mb"),
                }
            )

    return pd.DataFrame(rows)

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
# Figures 1-7
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

def fig6(df: pd.DataFrame):
    """
    Plots the 'Optimal Trade-off' curve (Pareto Frontier).
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
            
            # Identify Pareto Frontier
            subdf = subdf.sort_values("d_params")
            pareto_points = []
            
            # Sort by d_params descending (highest compression first)
            sorted_points = subdf.sort_values("d_params", ascending=False)
            current_max_acc = -np.inf
            
            for _, row in sorted_points.iterrows():
                if row["d_acc"] >= current_max_acc:
                    pareto_points.append(row)
                    current_max_acc = row["d_acc"]
            
            pareto_df = pd.DataFrame(pareto_points).sort_values("d_params")
            plot_data_accum.append(pareto_df)

            # Plot ALL points faintly
            sns.scatterplot(
                data=subdf, x="d_params", y="d_acc",
                color="lightgray", alpha=0.5, s=30, ax=ax, legend=False
            )

            # Plot Pareto Frontier strongly
            sns.lineplot(
                data=pareto_df, x="d_params", y="d_acc",
                color="black", linewidth=2, linestyle="--", 
                marker="o", label="Pareto Frontier", ax=ax
            )
            
            ax.set_title(f"{arch} on {ds}")
            if i == len(architectures)-1: format_reduction_axis(ax, "Parameter")
            if j == 0: format_accuracy_axis(ax)
            
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

def fig7(df: pd.DataFrame):
    """
    Bar chart showing the Accuracy Delta (Method A - Method B).
    """
    df = df.copy()
    # Use clean base name from load_results
    df["exp_base"] = df["base_name"]
    
    valid_methods = ["Post-Prune (JF)", "No-Prune (Kevin)"]
    df_methods = df[df["posthoc_or_posttrain"].isin(valid_methods)]
    
    pivot_cols = ["dataset", "architecture", "exp_base", "is_quantized"]
    df_pivot = df_methods.pivot_table(
        index=pivot_cols,
        columns="posthoc_or_posttrain",
        values="accuracy"
    ).reset_index()
    
    if "Post-Prune (JF)" in df_pivot.columns and "No-Prune (Kevin)" in df_pivot.columns:
        df_pivot["accuracy_delta"] = df_pivot["Post-Prune (JF)"] - df_pivot["No-Prune (Kevin)"]
    else:
        print("Skipping Fig 7: Missing matching data for JF vs Kevin comparison.")
        return

    df_pivot = df_pivot.dropna(subset=["accuracy_delta"])
    save_plot_source_data(df_pivot, "fig7_method_delta_source")

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
            
            colors = ["#1f77b4" if x >= 0 else "#d62728" for x in subdf["accuracy_delta"]]
            
            sns.barplot(
                data=subdf, x="exp_base", y="accuracy_delta",
                ax=ax, palette=colors, hue="exp_base", legend=False
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
# Figure 8 (Updated)
# =========================
def fig8(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory"],
    out_dir: Path = Path("./figures/individual_plots"),
):
    """
    Generates improved INDIVIDUAL plot files.
    - Separates Quantized vs FP32 on X-axis.
    - Groups "Collapsed" models.
    - Applies robust hatching for Quantization.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    metric_titles = {
        "accuracy": "Accuracy (%)",
        "params": "Parameters",
        "flops": "FLOPs",
        "memory": "Memory (MB)"
    }
    
    palette = {
        "Baseline": "#333333",      # Dark Grey
        "Pruned (JF)": "#1f77b4",   # Blue
        "No-Prune (Kevin)": "#ff7f0e", # Orange
        "Collapsed": "#2ca02c"      # Green
    }

    for architecture, df_arch in df.groupby("architecture"):
        for dataset in df_arch["dataset"].unique():
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()

            if g_dataset.empty: continue

            # Determine Sort Order: Rank by non-quantized performance (or params) first
            # 1. Calculate rank based on 'params' (or fallback to accuracy) of base_name
            if "params" in g_dataset.columns:
                base_name_rank = g_dataset.groupby("base_name")["params"].max().sort_values(ascending=False)
            else:
                base_name_rank = g_dataset.groupby("base_name")["accuracy"].max().sort_values(ascending=True)
            
            rank_map = {name: i for i, name in enumerate(base_name_rank.index)}
            
            # 2. Assign rank and sort (primary: rank, secondary: is_quantized)
            g_dataset["rank"] = g_dataset["base_name"].map(rank_map)
            g_dataset.sort_values(["rank", "is_quantized"], ascending=[True, True], inplace=True)
            
            sort_order = g_dataset["display_name"].unique().tolist()

            for metric in metrics:
                if metric not in g_dataset.columns:
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))

                sns.barplot(
                    data=g_dataset,
                    x="display_name",
                    y=metric,
                    hue="posthoc_or_posttrain",
                    order=sort_order,
                    palette=palette,
                    edgecolor="black",
                    linewidth=1.0,
                    ax=ax,
                    errorbar=None 
                )

                # Hatching Logic: check x-tick labels
                locs = ax.get_xticks()
                labels = [l.get_text() for l in ax.get_xticklabels()]
                
                for patch in ax.patches:
                    x_center = patch.get_x() + patch.get_width() / 2
                    
                    # Find closest tick index
                    if len(locs) > 0:
                        closest_idx = min(range(len(locs)), key=lambda i: abs(locs[i] - x_center))
                        lbl = labels[closest_idx]
                        
                        if "(Quant)" in lbl:
                            patch.set_hatch("///")
                            patch.set_edgecolor("black")
                            patch.set_linewidth(1.0)

                ax.set_ylabel(metric_titles.get(metric, metric), fontsize=12, fontweight='bold')
                ax.set_xlabel("")
                ax.set_title(f"{architecture} - {dataset} ({metric})", fontsize=14, fontweight='bold')
                
                ax.legend(title="Method", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
                
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                
                filename = f"{architecture}_{dataset}_{metric}.png".replace(" ", "_")
                plt.savefig(out_dir / filename, bbox_inches='tight')
                plt.close()
                print(f"[Plot] Saved {filename}")

# =========================
# Figure 10: PWCCA
# =========================
def pwcca_distance(X, Y, epsilon=1e-10):
    # Center
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Covariance matrices
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

    hook = None
    for m in reversed(list(model.modules())):
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
            hook = m.register_forward_hook(hook_fn)
            break

    if hook is None:
        raise RuntimeError("No suitable layer found for PWCCA")

    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            if i >= max_batches:
                break
            model(x.to(device))

    hook.remove()
    acts = torch.cat(activations, dim=0)
    return acts.flatten(start_dim=1).numpy()


def get_collapse_range(arch_name, exp_name):
    """
    Maps experiment names (e.g., 'Stage 1 (Full)') to layer ranges 
    (e.g., ('stage1.block1_1', 'stage1.block1_2')) by importing directly from transfer.py.
    """
    # 1. Clean exp_name to match keys in the config dictionaries
    # Removes prefixes like "final_JF_" and suffixes like "_quant" or ".pt"
    name = exp_name.replace("final_JF_", "").replace("final_Kevin_", "") \
                   .replace("_quant", "").replace(".pt", "").strip()
    
    # 2. Import experiment configurations
    try:
        from transfer import (
            Vgg_common, 
            RegNetX_common, 
            ConvNeXt_common, 
            InceptionNet_common, 
            XceptionNet_common, 
            mobileNet_common
        )
    except ImportError:
        print("[!] Could not import experiment configs from transfer.py. Ensure it is in the path.")
        return None

    # 3. Map architecture strings to the imported dictionaries
    mappings = {
        "VGG16": Vgg_common,
        "RegNetX": RegNetX_common,
        "ConvNeXt": ConvNeXt_common,
        "InceptionNet": InceptionNet_common,
        "XceptionNet": XceptionNet_common,
        "MobileNet": mobileNet_common
    }

    # 4. Fuzzy match the architecture name (e.g., "RegNetX_400MF" -> "RegNetX")
    arch_key = None
    an_lower = arch_name.lower()
    
    if "convnext" in an_lower: arch_key = "ConvNeXt"
    elif "vgg" in an_lower: arch_key = "VGG16"
    elif "regnet" in an_lower: arch_key = "RegNetX"
    elif "inception" in an_lower: arch_key = "InceptionNet"
    elif "xception" in an_lower: arch_key = "XceptionNet"
    elif "mobile" in an_lower: arch_key = "MobileNet"
    
    # 5. Look up the specific experiment range
    if arch_key and arch_key in mappings:
        experiment_dict = mappings[arch_key]
        
        # Exact match attempt first
        if name in experiment_dict:
            val = experiment_dict[name]
            # Ensure it is returned as a list of tuples (or list of lists) for collapse_only
            return [val] if isinstance(val, tuple) else val
            
        # Fuzzy match experiment name (e.g., matching "Stage 1 (Full)" inside a longer string)
        for key, val in experiment_dict.items():
            if key in name: 
                # If val is a tuple (start, end), wrap in list. If list of tuples, return as is.
                return [val] if isinstance(val, tuple) else val
                
    return None

def fig10(
    results_dir: Path = RESULTS_DIR,
    out_dir: Path = FIG_DIR / "pwcca",
    device: str = "cuda",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model_map = {
        "VGG16": VGG16,
        "RegNetX_400MF": RegNetX_400MF,
        "ConvNeXt": ConvNeXt,
        "InceptionNet": InceptionNet,
        "XceptionNet": XceptionNet,
        "MobileNet": MobileNet
    }

    checkpoint_dirs = list(results_dir.glob("*/checkpoints"))
    if not checkpoint_dirs:
        print("[!] No checkpoint directories found")
        return

    for ckpt_dir in checkpoint_dirs:
        ckpt_files = sorted(ckpt_dir.glob("final*.pt"))
        if not ckpt_files:
            continue

        dir_name = ckpt_dir.parent.name
        print(f"[•] Processing {dir_name}")

        try:
            model_str = next((m for m in model_map.keys() if m in dir_name), None)
            
            if "tinyimagenet" in dir_name.lower(): ds_name = "tinyimagenet"
            elif "cifar100" in dir_name.lower(): ds_name = "Cifar100"
            elif "cifar10" in dir_name.lower(): ds_name = "Cifar10"
            elif "imagenet" in dir_name.lower(): ds_name = "imagenet"
            else: ds_name = "Cifar10"

            if not model_str:
                print(f"[!] Could not infer architecture from {dir_name}, skipping.")
                continue

            # Load Dataset info
            train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name=ds_name, model_name=model_str)
            one_batch = next(iter(train_loader))[0]
            ModelClass = model_map[model_str]
            
        except Exception as e:
            print(f"[!] Setup failed for {dir_name}: {e}")
            continue

        baseline = None
        others = []

        for ckpt_path in ckpt_files:
            try:
                # 1. Initialize Baseline Model
                model = ModelClass(num_classes=num_classes, one_batch=one_batch).to(device)
                name = ckpt_path.stem

                # 2. Apply Structural Collapse (Fixes the state_dict error)
                if "Original" not in name and "Baseline" not in name:
                    collapse_range = get_collapse_range(model_str, name)
                    if collapse_range:
                        # print(f"   [+] Applying structure collapse for {name}: {collapse_range}")
                        model = collapse_only(
                            model=model,
                            compression_set=collapse_range,
                            input_shape=one_batch.shape,
                            device=device,
                            debug=False 
                        )
                
                # 3. Load Weights
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                elif "model" in ckpt and isinstance(ckpt["model"], dict):
                    model.load_state_dict(ckpt["model"])
                else:
                    model.load_state_dict(ckpt)
                
                model.eval()
                
                if "Kevin" in name and "Original" in name and "quant" not in name:
                    baseline = (name, model)
                else:
                    others.append((name, model))

            except RuntimeError as e:
                # Catch mismatch errors specifically to print cleaner logs
                if "Error(s) in loading state_dict" in str(e):
                    print(f"[!] Arch mismatch for {name}. Ensure get_collapse_range covers this experiment.")
                else:
                    print(f"[!] Error loading {ckpt_path.name}: {e}")
                continue
            except Exception as e:
                print(f"[!] Unexpected error loading {ckpt_path.name}: {e}")
                continue

        if baseline is None:
            print("[!] No baseline found, skipping")
            continue

        base_name, base_model = baseline
        print(f"[✓] Baseline: {base_name}")

        try:
            base_repr = extract_representation(base_model, test_loader, device)
            pwcca_scores = []

            for name, model in others:
                try:
                    repr_other = extract_representation(model, test_loader, device)
                    score = pwcca_distance(base_repr, repr_other)
                    pwcca_scores.append({"model": name, "pwcca": score})
                except Exception as e:
                    print(f"[!] Failed PWCCA for {name}: {e}")

            if not pwcca_scores:
                continue

            pwcca_scores.sort(key=lambda x: x["pwcca"], reverse=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(
                [x["model"] for x in pwcca_scores],
                [x["pwcca"] for x in pwcca_scores],
            )
            ax.set_ylabel("PWCCA Similarity")
            ax.set_xlabel("Model Variant")
            ax.set_title(f"PWCCA Drift from Baseline\n{dir_name}", fontsize=14)
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", linestyle="--", alpha=0.5)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            save_path = out_dir / f"{dir_name}_pwcca.png"
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"[✓] Saved {save_path}")

        except Exception as e:
            print(f"[!] Analysis failed for {dir_name}: {e}")
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
        fig4(df)
        fig5(df)
        fig6(df)
        fig7(df)
        
        fig8(
            df,
            out_dir=FIG_DIR / "expname_ablations",
        )
        print("\n[•] Generating Diagnostic Figures from saved data...")
      
        fig10(
            results_dir=RESULTS_DIR,
            out_dir=FIG_DIR / "pwcca",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        tab1(df)
        tab2(df)

        print("\nDone. Check the 'tables/' folder for CSV data sources.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()