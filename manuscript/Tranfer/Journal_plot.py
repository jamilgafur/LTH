from __future__ import annotations

import glob
import json
import warnings
import re
import os
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
# Configuration & Journal Style
# =========================
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# Enhanced Journal-level styling
sns.set_theme(
    context="paper",
    style="ticks", # 'ticks' is often preferred for academic journals over 'whitegrid'
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
    if "regnet" in name: return "RegNetX_400MF" # Match the exact names from transfer.py
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
    - Pruned/Not Pruned: ONLY for VGG16 and RegNetX.
    - Collapsed: For all other architectures (forces averaging later).
    """
    n = exp_name.lower()
    
    if "original" in n or "baseline" in n:
        return "Baseline"
        
    # Only distinguish distinct methods for specific architectures
    if "VGG16" in architecture or "RegNetX" in architecture:
        if "jf" in n or ("pruned" in n and "no" not in n and "not" not in n): 
            return "Pruned"
        if "kevin" in n or "no-prune" in n or "not pruned" in n: 
            return "Not Pruned"
        
    # For ConvNeXt, MobileNet, etc., merge them into one group for averaging
    return "Collapsed"

def clean_exp_name(exp_name: str) -> str:
    n = exp_name
    
    # Strip JF/Kevin and Quant text rigorously so that non-VGG/RegNetX models map 
    # to the exact same base_name to be averaged properly.
    n = re.sub(r'(?i)[_-]?quant|\(quant\)', '', n)
    n = re.sub(r'(?i)[_-]?jf|\(jf\)|[_-]?kevin|\(kevin\)|no-prune|not pruned|pruned', '', n)
    
    for arch in ["RegNetX_400MF_", "VGG16_", "MobileNet_", "ConvNeXt_", "InceptionNet_", "XceptionNet_"]:
        n = n.replace(arch, "")
        
    n = n.replace("Block ", "Block-").replace("Stage ", "Stage-") 
    n = n.replace(" Only", "")
    n = n.strip(" -_()")
    
    if "Original" in n or "Baseline" in n: 
        return "Original Model" # Normalize baseline name
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
        if (RESULTS_DIR / "merged_metrics.json").exists():
            files = [RESULTS_DIR / "merged_metrics.json"]
        else:
            raise FileNotFoundError("No merged_metrics.json files found")

    rows = []

    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower():
            dataset = "tinyimagenet" # Fallback if folder structure is flat
            
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch":
            # Fallback extraction from filename if folder structure doesn't match
            arch = infer_architecture_from_path(Path(p.name))
            
        try:
            with open(p) as f:
                raw = json.load(f)
        except Exception as e:
            print(f"Skipping {p}: {e}")
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
                row["d_acc"] = r["accuracy"] - baseline["accuracy"] 
                row["acc_drop"] = baseline["accuracy"] - r["accuracy"] # Added explicitly for scatter plots
                row["baseline_acc"] = baseline["accuracy"] # Added to plot pareto threshold
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
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
    """
    Generates improved INDIVIDUAL plot files AND LaTeX tables.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    metric_titles = {
        "accuracy": "Accuracy (%)",
        "params": "Params (M)",
        "flops": "GFLOPs",
        "memory": "Memory (MB)",
    }
    
    palette = {
        "Baseline": "#333333",      # Dark Grey
        "Pruned": "#1f77b4",        # Blue
        "Not Pruned": "#ff7f0e",    # Orange
        "Collapsed": "#2ca02c"      # Green
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
            
            table_cols = ["Model", "Type"] + [metric_titles[m] for m in available_metrics]
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

            if not df_quant.empty:
                tex_filename_q = f"{architecture}_{dataset}_quantized_table.tex".replace(" ", "_")
                df_quant.to_latex(
                    out_dir / tex_filename_q, index=False, float_format=float_fmt,
                    caption=f"Quantized performance metrics for {architecture} on {dataset}.",
                    label=f"tab:{architecture}_{dataset}_quant", escape=True,
                    column_format="ll" + "c" * len(available_metrics)
                )

            # --- Plot Generation ---
            for metric in metrics:
                if metric not in g_dataset.columns or g_dataset[metric].isnull().all():
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
                print(f"[Plot] Saved {filename}")


def fig2_correlation_and_pareto(
    df: pd.DataFrame, 
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/correlation_plots")
):
    """
    Merges normalized accuracy metrics with heuristic statistics (Median Variance) 
    to generate the Scatter Proof Plot and the Pareto Efficiency Curve.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # We want to average the results (e.g., combining quant/non-quant drops) to get a clean scatter plot
    df_agg = df.groupby(["dataset", "architecture", "base_name"]).mean(numeric_only=True).reset_index()

    for (dataset, arch), g_metrics in df_agg.groupby(["dataset", "architecture"]):
        
        # Load the corresponding heuristic stats file generated by transfer.py
        csv_filename = f"{arch}_{dataset}_experiment_block_stats.csv"
        csv_path = stats_dir / csv_filename
        
        if not csv_path.exists():
            print(f"[Skip] Heuristic stats not found for {arch} on {dataset} ({csv_path})")
            continue
            
        try:
            df_heuristics = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[!] Failed to read {csv_path}: {e}")
            continue
            
        # Merge metrics with heuristics
        # CSV has "Experiment", df has "base_name"
        df_merged = pd.merge(df_heuristics, g_metrics, left_on="Experiment", right_on="base_name", how="inner")
        
        if df_merged.empty:
            print(f"[!] Merge failed for {arch} on {dataset}. Check naming alignments.")
            continue
            
        # Optional: Grab baseline accuracy for thresholding
        baseline_acc = g_metrics["baseline_acc"].max() if "baseline_acc" in g_metrics.columns else 100.0
            
        # =========================================================
        # 1. The "Proof" Plot - Variance vs Accuracy Drop
        # =========================================================
        fig, ax = plt.subplots(figsize=(10, 7))

        sns.scatterplot(
            data=df_merged, x="Median Variance", y="acc_drop", 
            hue="acc_drop", palette="coolwarm", size="d_params", sizes=(50, 300), 
            edgecolor="black", linewidth=1, ax=ax, legend="brief"
        )

        for i in range(df_merged.shape[0]):
            ax.text(
                df_merged["Median Variance"].iloc[i], 
                df_merged["acc_drop"].iloc[i] + 0.5, 
                df_merged["Experiment"].iloc[i], 
                horizontalalignment='center', size='small', color='black', alpha=0.7
            )

        ax.set_xscale('symlog', linthresh=10.0) 
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        
        y_max = max(df_merged["acc_drop"].max() * 1.1, 10.0)
        ax.axhspan(-5, 2, color='#e6f4ea', alpha=0.3, zorder=0) 
        ax.axhspan(2, y_max, color='#fce8e6', alpha=0.3, zorder=0) 

        ax.set_title(f"Heuristic Validation: Variance vs Network Failure\n{arch} | {dataset}", fontweight='bold', pad=15)
        ax.set_ylabel("Accuracy Drop (%) -> Lower is Better", fontweight='bold')
        ax.set_xlabel("Block Median Variance (SymLog Scale) -> Predicts Information Bottleneck", fontweight='bold')
        sns.despine()
        
        plt.tight_layout()
        proof_filename = out_dir / f"{arch}_{dataset}_heuristic_proof.png".replace(" ", "_")
        plt.savefig(proof_filename, dpi=300)
        plt.close()
        print(f"[Plot] Saved Scatter Proof: {proof_filename.name}")

        # =========================================================
        # 2. The "Value" Plot - Pareto Efficiency Curve
        # =========================================================
        fig, ax = plt.subplots(figsize=(10, 7))

        sns.scatterplot(
            data=df_merged, x="d_params", y="accuracy", 
            hue="Median Variance", palette="viridis", s=150, 
            edgecolor="black", linewidth=1.5, ax=ax
        )

        ax.scatter([0], [baseline_acc], color="gold", marker="*", s=500, edgecolor="black", label="Baseline Model")

        for i in range(df_merged.shape[0]):
            ax.text(
                df_merged["d_params"].iloc[i], 
                df_merged["accuracy"].iloc[i] - 1.5, 
                df_merged["Experiment"].iloc[i], 
                horizontalalignment='center', size='small', color='black', alpha=0.7
            )

        ax.set_title(f"Efficiency Frontier: Compression vs Accuracy\n{arch} | {dataset}", fontweight='bold', pad=15)
        ax.set_ylabel("Final Accuracy (%)", fontweight='bold')
        ax.set_xlabel("Parameters Removed (%) -> Higher is Better", fontweight='bold')
        ax.legend(loc='lower left')
        
        ax.axhline(baseline_acc, color='black', linestyle='-', alpha=0.5)
        ax.axhline(baseline_acc - 2.0, color='red', linestyle='--', alpha=0.5, label="2% Degradation Limit")
        
        sns.despine()
        plt.tight_layout()
        pareto_filename = out_dir / f"{arch}_{dataset}_pareto_efficiency.png".replace(" ", "_")
        plt.savefig(pareto_filename, dpi=300)
        plt.close()
        print(f"[Plot] Saved Pareto Efficiency: {pareto_filename.name}")


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
        
        # 1. Generate Individual Bar Charts & Tables
        fig1(df)
        
        # 2. Generate Scatter Proof & Pareto Efficiency Curves
        fig2_correlation_and_pareto(df)
               
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()