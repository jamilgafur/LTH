from __future__ import annotations

import glob
import json
import warnings
import re
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports (Assumed available based on original file)
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from collapse import collapse_only

# =========================
# Configuration & Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Enhanced Journal-level styling
sns.set_theme(
    context="paper",
    style="ticks",
    palette="colorblind",
    font_scale=1.2,
)

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "legend.fontsize": 10,
    "legend.title_fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
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
    if "regnet" in name: return "RegNetX_400MF"
    if "vgg" in name: return "VGG16"
    if "inception" in name: return "InceptionNet"
    if "xception" in name: return "XceptionNet"
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
    - Not Pruned: Merged group for VGG16/RegNetX (now includes failed 'pruned' runs).
    - Collapsed: For all other architectures.
    """
    n = exp_name.lower()
    
    if "original" in n or "baseline" in n:
        return "Baseline"
        
    # Per user request: 'Pruned' runs failed and ran without pruning. 
    # We map them to 'Not Pruned' so the data is kept but correctly labeled.
    if "VGG16" in architecture or "RegNetX" in architecture:
        # Check for both traditional 'not pruned' and the 'pruned' strings
        is_pruning_related = any(x in n for x in ["jf", "pruned", "kevin", "no-prune", "not pruned"])
        if is_pruning_related:
            return "Retrained"
        
    return "Collapsed"

def clean_exp_name(exp_name: str) -> str:
    n = exp_name
    n = re.sub(r'(?i)[_-]?quant|\(quant\)', '', n)
    n = re.sub(r'(?i)[_-]?jf|\(jf\)|[_-]?kevin|\(kevin\)|no-prune|not pruned|pruned', '', n)
    
    for arch in ["RegNetX_400MF_", "VGG16_", "MobileNet_", "ConvNeXt_", "InceptionNet_", "XceptionNet_"]:
        n = n.replace(arch, "")
        
    n = n.replace("Block ", "Block-").replace("Stage ", "Stage-") 
    n = n.replace(" Only", "")
    n = n.strip(" -_()")
    
    if "Original" in n or "Baseline" in n: 
        return "Original Model"
    return n.strip()

def find_baseline(df: pd.DataFrame):
    mask = (
        df["exp_name"].str.lower().str.contains("original")
        | df["exp_name"].str.lower().str.contains("baseline")
    )
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def load_results() -> pd.DataFrame:
    logger.info(f"Scanning for metrics files in {RESULTS_DIR.resolve()}")
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    
    if not files:
        if (RESULTS_DIR / "merged_metrics.json").exists():
            files = [RESULTS_DIR / "merged_metrics.json"]
        else:
            raise FileNotFoundError("No merged_metrics.json files found")

    logger.info(f"Found {len(files)} metrics file(s).")
    rows = []

    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower():
            dataset = "tinyimagenet"
            
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch":
            arch = infer_architecture_from_path(Path(p.name))
            
        logger.debug(f"Processing file: {p.name} | Inferred Arch: {arch} | Inferred Dataset: {dataset}")
        
        try:
            with open(p) as f:
                raw = json.load(f)
        except Exception as e:
            logger.error(f"Skipping {p} due to read error: {e}")
            continue

        for exp_name, metrics in raw.items():
            method_group = infer_posthoc_or_posttrain(exp_name, arch)
            is_quant = infer_isquant(exp_name)
            
            base_name = clean_exp_name(exp_name)
            display_name = f"{base_name}\n(Quant)" if is_quant else base_name

            rows.append({
                "dataset": dataset,
                "architecture": arch,
                "exp_name": exp_name,
                "base_name": base_name,
                "display_name": display_name,
                "posthoc_or_posttrain": method_group,
                "model_type": infer_model_type(exp_name),
                "is_quantized": is_quant,
                "accuracy": metrics.get("final_accuracy"),
                "params": metrics.get("param_count"),
                "flops": metrics.get("flops"),
                "memory": metrics.get("total_size_mb"),
            })

    logger.info(f"Loaded {len(rows)} total experiment configurations.")
    return pd.DataFrame(rows)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Normalizing metrics against baselines...")
    out = []
    
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            logger.warning(f"No baseline found for Architecture: {arch} on Dataset: {ds}. Metrics will not be normalized.")
            for _, r in g.iterrows(): out.append(r)
            continue

        logger.debug(f"Found baseline for {arch} on {ds}: '{baseline['exp_name']}' with Acc: {baseline['accuracy']}")

        for _, r in g.iterrows():
            row = r.copy()
            # Added pd.notnull and > 0 check to prevent division by zero or NaN propagation
            if pd.notnull(baseline.get("params")) and baseline["params"] > 0:
                row["d_acc"] = r["accuracy"] - baseline["accuracy"] 
                row["acc_drop"] = baseline["accuracy"] - r["accuracy"] 
                row["baseline_acc"] = baseline["accuracy"] 
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                
                if pd.notnull(baseline.get("flops")) and baseline["flops"] > 0:
                    row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
                if pd.notnull(baseline.get("memory")) and baseline["memory"] > 0:
                    row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
            out.append(row)
            
    return pd.DataFrame(out)

# =========================
# Plotting Helpers & Main Generation
# =========================

def fig1(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory"], 
    out_dir: Path = Path("./figures/individual_plots"),
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating Individual Bar Plots & Tables in {out_dir}")
    
    metric_titles = {
        "accuracy": "Accuracy (%)",
        "params": "Params (M)",
        "flops": "GFLOPs",
        "memory": "Memory (MB)",
    }
    
    # Inside fig1 function in Journal_plot.py
    palette = {
    "Baseline": "#333333",      
    "Retrained": "#ff7f0e",  # Mapping legacy 'pruned' runs to Retrained color
    "Not Pruned": "#ff7f0e",    
    "Collapsed": "#2ca02c"      
}

    group_cols = ["dataset", "architecture", "base_name", "display_name", "posthoc_or_posttrain", "is_quantized"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    df_agg = df.groupby(group_cols, dropna=False)[available_metrics].mean(numeric_only=True).reset_index()

    for architecture, df_arch in df_agg.groupby("architecture"):
        for dataset in df_arch["dataset"].unique():
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()

            if g_dataset.empty: continue

            if "params" in g_dataset.columns:
                base_name_rank = g_dataset.groupby("base_name")["params"].max().sort_values(ascending=False)
            else:
                base_name_rank = g_dataset.groupby("base_name")["accuracy"].max().sort_values(ascending=True)
            
            rank_map = {name: i for i, name in enumerate(base_name_rank.index)}
            g_dataset["rank"] = g_dataset["base_name"].map(rank_map)
            g_dataset.sort_values(["rank", "is_quantized"], ascending=[True, True], inplace=True)
            
            sort_order = g_dataset["display_name"].unique().tolist()

            # --- LaTeX Table Generation ---
            table_df = g_dataset.copy()
            if "params" in table_df.columns: table_df["params"] = table_df["params"] / 1e6
            if "flops" in table_df.columns: table_df["flops"] = table_df["flops"] / 1e9

            rename_map = {"display_name": "Model", "posthoc_or_posttrain": "Type"}
            rename_map.update(metric_titles)
            table_df.rename(columns=rename_map, inplace=True)
            
            table_cols = ["Model", "Type"] + [metric_titles.get(m, m) for m in available_metrics]
            float_fmt = lambda x: f"{x:.4f}" if x < 10 else f"{x:.2f}"
            
            df_unquant = table_df[table_df["is_quantized"] == False][table_cols]
            df_quant = table_df[table_df["is_quantized"] == True][table_cols]

            if not df_unquant.empty:
                tex_filename_unq = f"{architecture}_{dataset}_unquantized_table.tex".replace(" ", "_")
                df_unquant.to_latex(
                    out_dir / tex_filename_unq, index=False, float_format=float_fmt,
                    caption=f"Unquantized performance metrics for {architecture} on {dataset}.",
                    label=f"tab:{architecture}_{dataset}_unquant", escape=True,
                    column_format="ll" + "c" * len(available_metrics)
                )
                logger.debug(f"Saved LaTeX Table: {tex_filename_unq}")

            if not df_quant.empty:
                tex_filename_q = f"{architecture}_{dataset}_quantized_table.tex".replace(" ", "_")
                df_quant.to_latex(
                    out_dir / tex_filename_q, index=False, float_format=float_fmt,
                    caption=f"Quantized performance metrics for {architecture} on {dataset}.",
                    label=f"tab:{architecture}_{dataset}_quant", escape=True,
                    column_format="ll" + "c" * len(available_metrics)
                )
                logger.debug(f"Saved LaTeX Table: {tex_filename_q}")

            # --- Plot Generation ---
            for metric in metrics:
                if metric not in g_dataset.columns or g_dataset[metric].isnull().all():
                    logger.debug(f"Skipping plot for {metric} (No data) for {architecture} on {dataset}")
                    continue

                fig, ax = plt.subplots(figsize=(12, 6))

                sns.barplot(
                    data=g_dataset, x="display_name", y=metric, hue="posthoc_or_posttrain",
                    order=sort_order, palette=palette, edgecolor="black", linewidth=1.2, ax=ax, errorbar=None 
                )

                locs = ax.get_xticks()
                labels = [l.get_text() for l in ax.get_xticklabels()]
                
                for patch in ax.patches:
                    x_center = patch.get_x() + patch.get_width() / 2
                    if len(locs) > 0:
                        closest_idx = min(range(len(locs)), key=lambda i: abs(locs[i] - x_center))
                        lbl = labels[closest_idx]
                        if "(Quant)" in lbl:
                            patch.set_hatch("///")
                            patch.set_edgecolor("black")
                            patch.set_linewidth(1.2)

                ax.set_ylabel(metric_titles.get(metric, metric))
                ax.set_xlabel("")
                ax.set_title(f"{architecture} - {dataset} ({metric_titles.get(metric, metric)})")
                sns.despine(ax=ax)
                
                ax.legend(title="Method", loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
                
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, axis="y", linestyle=":", alpha=0.6, zorder=0)
                plt.tight_layout()
                
                filename = f"{architecture}_{dataset}_{metric}.png".replace(" ", "_")
                plt.savefig(out_dir / filename)
                plt.close()
                logger.info(f"Saved Bar Plot: {filename}")


from adjustText import adjust_text

def fig2_correlation_and_pareto(
    df: pd.DataFrame, 
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/correlation_plots")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating Correlation & Pareto Plots in {out_dir}")

    df_agg = df.groupby(["dataset", "architecture", "base_name"]).mean(numeric_only=True).reset_index()

    for (dataset, arch), g_metrics in df_agg.groupby(["dataset", "architecture"]):
        csv_filename = f"{arch}_{dataset}_experiment_block_stats.csv"
        csv_path = stats_dir / csv_filename
        
        if not csv_path.exists():
            continue
            
        df_heuristics = pd.read_csv(csv_path)
        df_merged = pd.merge(df_heuristics, g_metrics, left_on="Experiment", right_on="base_name", how="inner")
        
        if df_merged.empty: continue
            
        baseline_acc = g_metrics["baseline_acc"].max() if "baseline_acc" in g_metrics.columns else 100.0
            
        # =========================================================
        # 1. The "Proof" Plot - Variance vs Accuracy Drop
        # =========================================================
        fig, ax = plt.subplots(figsize=(8, 6)) # Standardized size for IEEE/ACM double-column

        # Use distinct markers based on whether accuracy dropped or improved
        df_merged['acc_improved'] = df_merged['acc_drop'] < 0
        markers = {True: '^', False: 'o'} # Triangles for improvement, circles for degradation

        sns.scatterplot(
            data=df_merged, x="Median Variance", y="acc_drop", 
            hue="acc_drop", palette="coolwarm", size="d_params", sizes=(60, 250), 
            style="acc_improved", markers=markers, edgecolor="black", linewidth=1, ax=ax, legend=False
        )
        
        # Smart text adjustment to prevent overlapping
        texts = []
        for i in range(df_merged.shape[0]):
            texts.append(ax.text(
                df_merged["Median Variance"].iloc[i], 
                df_merged["acc_drop"].iloc[i], 
                df_merged["Experiment"].iloc[i], 
                size=9, color='black', zorder=10
            ))
        
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.7), ax=ax)

        ax.set_xscale('symlog', linthresh=10.0) 
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
        
        y_max = max(df_merged["acc_drop"].max() * 1.1, 10.0)
        ax.axhspan(-5, 0, color='#e6f4ea', alpha=0.4, zorder=0) # Green safe zone
        ax.axhspan(0, y_max, color='#fce8e6', alpha=0.4, zorder=0) # Red danger zone

        ax.set_title(f"Heuristic Validation: {arch} on {dataset}", fontweight='bold')
        ax.set_ylabel(r"$\Delta$ Accuracy (\%) $\rightarrow$ Lower is Better", fontweight='bold')
        ax.set_xlabel("Block Median Variance (SymLog) $\rightarrow$ Represents Info Bottleneck", fontweight='bold')
        sns.despine()
        ax.grid(True, linestyle=":", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{arch}_{dataset}_heuristic_proof.pdf", dpi=300) # Save as PDF for vector scaling
        plt.close()

        # =========================================================
        # 2. The "Value" Plot - Pareto Efficiency Curve
        # =========================================================
        fig, ax = plt.subplots(figsize=(8, 6))

        # Calculate mathematical Pareto Frontier
        # Sort by parameters removed (descending), keep tracking the max accuracy
        pareto_df = df_merged.sort_values("d_params", ascending=False)
        pareto_front_x, pareto_front_y = [0], [baseline_acc] # Baseline starts the frontier
        max_acc_seen = -float('inf')

        for _, row in pareto_df.iterrows():
            if row["accuracy"] >= max_acc_seen:
                pareto_front_x.append(row["d_params"])
                pareto_front_y.append(row["accuracy"])
                max_acc_seen = row["accuracy"]

        # Draw the Pareto curve (Step line)
        ax.step(pareto_front_x, pareto_front_y, where='pre', color='darkorange', 
                linestyle='-', linewidth=2, zorder=1, label="Pareto Frontier")

        sns.scatterplot(
            data=df_merged, x="d_params", y="accuracy", 
            hue="Median Variance", palette="viridis", s=120, 
            edgecolor="black", linewidth=1, ax=ax, legend=False, zorder=5
        )

        ax.scatter([0], [baseline_acc], color="gold", marker="*", s=400, 
                   edgecolor="black", linewidth=1.5, label="Baseline Model", zorder=6)

        texts = []
        for i in range(df_merged.shape[0]):
            texts.append(ax.text(
                df_merged["d_params"].iloc[i], 
                df_merged["accuracy"].iloc[i], 
                df_merged["Experiment"].iloc[i], 
                size=9, color='black', zorder=10
            ))
        
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.8, alpha=0.7), ax=ax)

        ax.set_title(f"Efficiency Frontier: {arch} on {dataset}", fontweight='bold')
        ax.set_ylabel("Final Accuracy (\%)", fontweight='bold')
        ax.set_xlabel("Parameters Removed (\%) $\rightarrow$ Higher is Better", fontweight='bold')
        
        ax.axhline(baseline_acc, color='black', linestyle=':', alpha=0.7)
        ax.axhline(baseline_acc - 2.0, color='red', linestyle='--', alpha=0.5, label="2% Degradation Limit")
        
        ax.legend(loc="lower left", frameon=True, fancybox=True, shadow=False, edgecolor='black')
        sns.despine()
        ax.grid(True, linestyle=":", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"{arch}_{dataset}_pareto_efficiency.pdf", dpi=300)
        plt.close()


def fig3_v2t_heuristic_validation(
    df: pd.DataFrame, 
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/heuristic_validation")
):
    """
    Creates a high-level scientific plot proving the V2T Heuristic.
    X-axis: Median Variance of the collapsed block.
    Y-axis: Accuracy Drop (0 = no drop, higher = worse).
    Color: Topology Type (Single-Path vs Multi-Path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating V2T Heuristic Validation Map in {out_dir}")

    # 1. Define Topologies for categorization
    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]
    single_path_archs = ["VGG16", "MobileNet"]

    # 2. Prepare aggregated data
    df_agg = df.groupby(["dataset", "architecture", "base_name"]).mean(numeric_only=True).reset_index()
    all_merged_data = []

    for (dataset, arch), g_metrics in df_agg.groupby(["dataset", "architecture"]):
        csv_filename = f"{arch}_{dataset}_experiment_block_stats.csv"
        csv_path = stats_dir / csv_filename
        
        if csv_path.exists():
            df_heuristics = pd.read_csv(csv_path)
            merged = pd.merge(df_heuristics, g_metrics, left_on="Experiment", right_on="base_name")
            merged["Topology"] = "Multi-Path" if arch in multi_path_archs else "Single-Path"
            all_merged_data.append(merged)

    if not all_merged_data:
        logger.warning("No heuristic stats found. Skipping Fig 3.")
        return

    full_df = pd.concat(all_merged_data)

    # 3. Create the Visualization
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # We use a custom 'Performance Impact' metric for the Y-axis
    # (Accuracy Drop / Parameters Removed) to show "Value per Param"
    full_df['efficiency'] = full_df['acc_drop'] / (full_df['d_params'] + 1e-6)

    sns.scatterplot(
        data=full_df, 
        x="Median Variance", 
        y="acc_drop", 
        hue="Topology", 
        style="architecture",
        size="d_params", 
        sizes=(50, 400),
        alpha=0.7, 
        edgecolor="black", 
        linewidth=1, 
        ax=ax
    )

    # 4. Annotate the "Safe Zones" defined in your Method.tex
    # Single-Path Safe Zone: Low Variance
    ax.axvspan(0.1, 10, color='green', alpha=0.1, label="Single-Path Safe Zone")
    # Multi-Path Safe Zone: High Variance Spikes
    ax.axvspan(100, full_df["Median Variance"].max(), color='orange', alpha=0.1, label="Multi-Path Safe Zone")

    # Formatting
    ax.set_xscale('log')
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_title("V2T Heuristic Validation: Variance vs. Stability", fontsize=16)
    ax.set_xlabel("Median Activation Variance (Log Scale)", fontweight='bold')
    ax.set_ylabel(r"Accuracy Drop ($\Delta\%$) $\rightarrow$ Lower is Better", fontweight='bold')
    
    # Move legend outside
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, edgecolor='black')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(out_dir / "V2T_heuristic_validation_map.png", dpi=300)
    plt.close()


def fig4_heuristic_search_space_map(
    df: pd.DataFrame, 
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/search_space")
):
    """
    Generates a 'Ladder Plot' showing sequential collapse candidates over layer depth.
    - X-axis: Sequential Layer Index.
    - Y-axis: Specific Experiment (Collapse Range).
    - Background: Normalized Variance Heatmap/Line.
    """
    from transfer import EXPERIMENTS # Assumes transfer.py is in the same path
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating Heuristic Search Space Maps in {out_dir}")

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): continue
        
        # 1. Load Layer Data
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = layer_df['Variance'].values
        
        # 2. Get Experiment Ranges for this model
        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: continue

        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # 3. Plot Variance Landscape (Background)
        ax1.plot(range(len(layers)), variances, color='gray', alpha=0.3, label='Layer Variance')
        ax1.fill_between(range(len(layers)), 0, variances, color='gray', alpha=0.1)
        ax1.set_yscale('symlog')
        ax1.set_ylabel("Activation Variance (log)", fontweight='bold')
        ax1.set_xlabel("Sequential Layer Depth", fontweight='bold')

        # 4. Plot 'Ladder' Lines for Collapse Ranges
        # Secondary axis for stacking experiments
        ax2 = ax1.twinx()
        exp_list = [name for name, range_val in model_exps.items() if range_val is not None]
        
        for i, exp_name in enumerate(exp_list):
            layer_range = model_exps[exp_name]
            # Handle list of ranges or single tuple
            ranges = layer_range if isinstance(layer_range, list) else [layer_range]
            
            for start_layer, end_layer in ranges:
                # Find indices in the CSV
                try:
                    start_idx = next(i for i, n in enumerate(layers) if start_layer in n)
                    end_idx = next(i for i, n in reversed(list(enumerate(layers))) if end_layer in n)
                    
                    # Color line based on accuracy drop from g_metrics
                    match = g_metrics[g_metrics['base_name'] == exp_name]
                    drop = match['acc_drop'].iloc[0] if not match.empty else 0
                    color = 'green' if drop < 2 else 'orange' if drop < 10 else 'red'
                    
                    # Draw horizontal line for the block
                    ax2.hlines(y=i, xmin=start_idx, xmax=end_idx, linewidth=6, color=color, alpha=0.8)
                    ax2.plot(start_idx, i, marker='|', color='black', markersize=10) # Start cap
                    ax2.plot(end_idx, i, marker='|', color='black', markersize=10)   # End cap
                except (StopIteration, ValueError):
                    continue

        ax2.set_yticks(range(len(exp_list)))
        ax2.set_yticklabels(exp_list, fontsize=9)
        ax2.set_ylabel("Collapse Candidates (Ladder)", fontweight='bold')
        ax2.set_ylim(-1, len(exp_list))
        
        # 5. Heuristic Annotations
        is_multi = arch in ["RegNetX_400MF", "ConvNeXt", "InceptionNet", "XceptionNet"]
        title_type = "Multi-Path: Targeting High Spikes" if is_multi else "Single-Path: Targeting Low Stability"
        ax1.set_title(f"Heuristic Search Space Map: {arch}\n{title_type}", fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig(out_dir / f"{arch}_{dataset}_search_space_ladder.pdf")
        plt.close()

import argparse

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots for specific models and datasets.")
    parser.add_argument("--model", type=str, default=None, help="Target model architecture to plot (e.g., InceptionNet)")
    parser.add_argument("--dataset", type=str, default=None, help="Target dataset to plot (e.g., tinyimagenet)")
    args = parser.parse_args()

    raw = load_results()
    
    if args.model:
        logger.info(f"Filtering down to architecture: {args.model}")
        raw = raw[raw["architecture"] == args.model]
    if args.dataset:
        logger.info(f"Filtering down to dataset: {args.dataset}")
        raw = raw[raw["dataset"] == args.dataset]
        
    if raw.empty:
        logger.error(f"No data found matching Model='{args.model}' and Dataset='{args.dataset}'. Exiting.")
        exit(0)
        
    df = normalize(raw)
    
    logger.info("==============================")
    logger.info(" GENERATING FIGURES & DATA ")
    logger.info("==============================")
    
    # 1. Generate Individual Bar Charts & Tables
    fig1(df)
    
    # 2. Generate Scatter Proof & Pareto Efficiency Curves
    fig2_correlation_and_pareto(df)
    fig3_v2t_heuristic_validation(df)
    fig4_heuristic_search_space_map(df)
    
    logger.info("Execution completed successfully.")
