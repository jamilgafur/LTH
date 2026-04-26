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

# Removed adjustText import

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
    n = exp_name.lower()
    if "original" in n or "baseline" in n:
        return "Baseline"
    if "VGG16" in architecture or "RegNetX" in architecture:
        is_legacy_run = any(x in n for x in ["jf", "pruned", "kevin", "no-prune", "not pruned"])
        if is_legacy_run:
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
    if "Original" in n or "Baseline" in n: return "Original Model"
    return n.strip()

def find_baseline(df: pd.DataFrame):
    mask = (df["exp_name"].str.lower().str.contains("original") | df["exp_name"].str.lower().str.contains("baseline"))
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def load_results() -> pd.DataFrame:
    logger.info(f"Scanning for metrics files in {RESULTS_DIR.resolve()}")
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    if not files:
        if (RESULTS_DIR / "merged_metrics.json").exists(): files = [RESULTS_DIR / "merged_metrics.json"]
        else: raise FileNotFoundError("No merged_metrics.json files found")
    rows = []
    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower(): dataset = "tinyimagenet"
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch": arch = infer_architecture_from_path(Path(p.name))
        try:
            with open(p) as f: raw = json.load(f)
        except Exception: continue
        for exp_name, metrics in raw.items():
            method_group = infer_posthoc_or_posttrain(exp_name, arch)
            is_quant = infer_isquant(exp_name)
            base_name = clean_exp_name(exp_name)
            rows.append({
                "dataset": dataset, "architecture": arch, "exp_name": exp_name,
                "base_name": base_name, "display_name": f"{base_name}\n(Quant)" if is_quant else base_name,
                "posthoc_or_posttrain": method_group, "model_type": infer_model_type(exp_name),
                "is_quantized": is_quant, "accuracy": metrics.get("final_accuracy"),
                "params": metrics.get("param_count"), "flops": metrics.get("flops"),
                "memory": metrics.get("total_size_mb"),
            })
    return pd.DataFrame(rows)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            for _, r in g.iterrows(): out.append(r)
            continue
        for _, r in g.iterrows():
            row = r.copy()
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
# Figure Generations (Updated to PNG and no adjustText)
# =========================

def fig1(df: pd.DataFrame, metrics: list[str] = ["accuracy", "params", "flops", "memory"], out_dir: Path = Path("./figures/individual_plots")):
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Baseline": "#333333", "Retrained": "#ff7f0e", "Not Pruned": "#ff7f0e", "Collapsed": "#2ca02c"}
    group_cols = ["dataset", "architecture", "base_name", "display_name", "posthoc_or_posttrain", "is_quantized"]
    available_metrics = [m for m in metrics if m in df.columns]
    df_agg = df.groupby(group_cols, dropna=False)[available_metrics].mean(numeric_only=True).reset_index()

    for architecture, df_arch in df_agg.groupby("architecture"):
        for dataset in df_arch["dataset"].unique():
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()
            if g_dataset.empty: continue
            for metric in available_metrics:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(data=g_dataset, x="display_name", y=metric, hue="posthoc_or_posttrain", palette=palette, edgecolor="black", ax=ax)
                plt.xticks(rotation=45, ha="right")
                plt.savefig(out_dir / f"{architecture}_{dataset}_{metric}.png") # Changed to .png
                plt.close()

def fig2_correlation_and_pareto(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/correlation_plots")):
    out_dir.mkdir(parents=True, exist_ok=True)
    df_agg = df.groupby(["dataset", "architecture", "base_name"]).mean(numeric_only=True).reset_index()
    for (dataset, arch), g_metrics in df_agg.groupby(["dataset", "architecture"]):
        csv_filename = f"{arch}_{dataset}_experiment_block_stats.csv"
        csv_path = stats_dir / csv_filename
        if not csv_path.exists(): continue
        df_heuristics = pd.read_csv(csv_path)
        df_merged = pd.merge(df_heuristics, g_metrics, left_on="Experiment", right_on="base_name", how="inner")
        if df_merged.empty: continue
        baseline_acc = g_metrics["baseline_acc"].max() if "baseline_acc" in g_metrics.columns else 100.0

        fig, ax = plt.subplots(figsize=(8, 6))
        pareto_df = df_merged.sort_values("d_params", ascending=False)
        pareto_front_x, pareto_front_y = [0], [baseline_acc]
        max_acc_seen = -float('inf')
        for _, row in pareto_df.iterrows():
            if row["accuracy"] >= max_acc_seen:
                pareto_front_x.append(row["d_params"]); pareto_front_y.append(row["accuracy"])
                max_acc_seen = row["accuracy"]

        ax.step(pareto_front_x, pareto_front_y, where='pre', color='darkorange', linewidth=2, label="Pareto Frontier")
        sns.scatterplot(data=df_merged, x="d_params", y="accuracy", hue="Median Variance", palette="viridis", s=120, ax=ax)
        
        # Standard annotation without adjustText
        for _, row in df_merged.iterrows():
            ax.text(row["d_params"], row["accuracy"], row["Experiment"], size=8, alpha=0.8)

        ax.set_title(f"Efficiency Frontier: {arch}")
        plt.savefig(out_dir / f"{arch}_{dataset}_pareto_efficiency.png") # Changed to .png
        plt.close()

def fig3_v2t_heuristic_validation(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/heuristic_validation")):
    out_dir.mkdir(parents=True, exist_ok=True)
    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]
    all_merged_data = []
    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        csv_path = stats_dir / f"{arch}_{dataset}_experiment_block_stats.csv"
        if csv_path.exists():
            df_h = pd.read_csv(csv_path)
            merged = pd.merge(df_h, g_metrics, left_on="Experiment", right_on="base_name")
            merged["Topology"] = "Multi-Path" if arch in multi_path_archs else "Single-Path"
            all_merged_data.append(merged)
    if not all_merged_data: return
    full_df = pd.concat(all_merged_data)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=full_df, x="Median Variance", y="acc_drop", hue="Topology", style="architecture", size="d_params", ax=ax)
    ax.axvspan(0.1, 10, color='green', alpha=0.1, label="Single-Path Safe Zone")
    ax.axvspan(100, full_df["Median Variance"].max(), color='orange', alpha=0.1, label="Multi-Path Safe Zone")
    ax.set_xscale('log')
    plt.savefig(out_dir / "V2T_heuristic_validation_map.png") # Changed to .png
    plt.close()

def fig4_heuristic_search_space_map(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/search_space")):
    """
    Enhanced Journal Plot: Synchronized Variance Profile and Heuristic Accuracy Map.
    Directly answers: 'Where is variance low, and how did that choice impact accuracy?'
    """
    from transfer import EXPERIMENTS 
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Define a color mapping for accuracy impact
    def get_acc_color(drop):
        if drop < 1.5: return "#2ca02c" # Green: Safe/Excellent
        if drop < 5.0: return "#ff7f0e" # Orange: Moderate
        return "#d62728"                # Red: Collapse/Poor

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): 
            logger.warning(f"Skipping {arch}_{dataset}: No layer stats found.")
            continue
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = layer_df['Variance'].values
        
        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: continue
        
        # Create a two-panel synchronized figure
        fig, (ax_var, ax_heur) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                             gridspec_kw={'height_ratios': [1, 2]})
        plt.subplots_adjust(hspace=0.08)

        # --- TOP PANEL: Activation Variance (The Heuristic Input) ---
        ax_var.plot(range(len(layers)), variances, color='#555555', linewidth=1.5, alpha=0.8, label="Activation Variance")
        ax_var.fill_between(range(len(layers)), variances, color='gray', alpha=0.1)
        ax_var.set_yscale('log')
        ax_var.set_ylabel("Variance ($\sigma^2$)")
        ax_var.set_title(f"Heuristic Selection Guide: {arch} on {dataset}", loc='left', pad=20)
        
        # Highlight "Safe Zones" (Heuristic logic: low variance = low redundancy)
        variance_threshold = np.percentile(variances, 25)
        ax_var.axhline(y=variance_threshold, color='green', linestyle='--', alpha=0.3, label="Low Variance Threshold")
        ax_var.legend(loc='upper right', frameon=False)

        # --- BOTTOM PANEL: Heuristic Decisions & Outcomes ---
        exp_list = [n for n, r in model_exps.items() if r is not None]
        
        for i, exp_name in enumerate(exp_list):
            ranges = model_exps[exp_name]
            ranges = ranges if isinstance(ranges, list) else [ranges]
            
            # Fetch accuracy data for this specific experiment
            exp_results = g_metrics[g_metrics['base_name'] == exp_name]
            
            if not exp_results.empty:
                acc_drop = exp_results['acc_drop'].iloc[0]
                final_acc = exp_results['accuracy'].iloc[0]
                color = get_acc_color(acc_drop)
                label_text = f"{final_acc:.1f}% (-{acc_drop:.1f}%)"
            else:
                color = 'gray'
                label_text = "N/A"

            # Draw the horizontal collapse range
            for start_layer, end_layer in ranges:
                try:
                    s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                    e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                    
                    # Plot the bar
                    line = ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=12, color=color, alpha=0.9)
                    
                    # Add accuracy label at the end of the bar
                    ax_heur.text(e_idx + 0.5, i, label_text, va='center', fontsize=9, fontweight='bold', color=color)
                except StopIteration:
                    continue
        
        ax_heur.set_yticks(range(len(exp_list)))
        ax_heur.set_yticklabels([clean_exp_name(e) for e in exp_list], fontsize=10)
        ax_heur.set_xlabel("Network Depth (Layer Index)")
        ax_heur.set_ylabel("Collapsed Layer Heuristics")
        
        # Clean up aesthetics
        sns.despine(ax=ax_var)
        sns.despine(ax=ax_heur)
        ax_heur.grid(axis='x', alpha=0.2)

        save_path = out_dir / f"{arch}_{dataset}_decision_map.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Generated unified decision map: {save_path}")

        
if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        fig1(df)
        fig2_correlation_and_pareto(df)
        fig3_v2t_heuristic_validation(df)
        fig4_heuristic_search_space_map(df)
        logger.info("All journal figures generated successfully as PNG.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)