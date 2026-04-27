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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration & Logging
# =========================
logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Silence Matplotlib font spam while keeping our debug logs
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

DATASET_NAME_MAP = {
    "tinyimagenet": "TinyImageNet",
    "cifar10_": "CIFAR-10",
    "cifar100_": "CIFAR-100",
    "imagenet": "ImageNet"
}

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

def infer_run_id(exp_name: str) -> str:
    """Isolate different runs to prevent cross-matching baselines."""
    n = exp_name.lower()
    if "jf" in n: return "JF"
    if "kevin" in n: return "Kevin"
    return "Default"

def clean_exp_name(exp_name: str) -> str:
    n = exp_name
    n = re.sub(r'(?i)[_-]?quant|\(quant\)', '', n)
    n = re.sub(r'(?i)[_-]?jf|\(jf\)|[_-]?kevin|\(kevin\)|no-prune|not pruned|pruned', '', n)
    for arch in ["RegNetX_400MF_", "VGG16_", "MobileNet_", "ConvNeXt_", "InceptionNet_", "XceptionNet_"]:
        n = n.replace(arch, "")
    n = n.replace("Block ", "Block-").replace("Stage ", "Stage-") 
    n = n.replace(" Only", "")
    
    # FIXED: Removed parentheses from strip to prevent breaking "(Full)" and "(1-11)"
    n = n.strip(" -_")
    
    if "Original" in n or "Baseline" in n: return "Original Model"
    return n.strip()

def standardize_heuristic_name(name: str) -> str:
    """Fuzzy matching helper to link extracted names with EXPERIMENT keys."""
    name = name.replace("-", " ").strip().lower()
    # Safely close parentheses if they were somehow cut off
    if "(" in name and ")" not in name:
        name += ")"
    return re.sub(r'\s+', ' ', name)

def find_baseline(df: pd.DataFrame):
    mask = (df["exp_name"].str.lower().str.contains("original") | df["exp_name"].str.lower().str.contains("baseline"))
    m = df[mask].sort_values("exp_name")
    return None if m.empty else m.iloc[0]

def load_results() -> pd.DataFrame:
    logger.info(f"Scanning for metrics files in {RESULTS_DIR.resolve()}")
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    
    if not files:
        if (RESULTS_DIR / "merged_metrics.json").exists(): files = [RESULTS_DIR / "merged_metrics.json"]
        else: 
            logger.warning("No merged_metrics.json files found. Returning empty dataframe.")
            return pd.DataFrame() 

    rows = []
    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower(): dataset = "tinyimagenet"
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch": arch = infer_architecture_from_path(Path(p.name))
        
        try:
            with open(p) as f: raw = json.load(f)
        except Exception as e: 
            logger.error(f"Failed to load JSON {p}: {e}")
            continue
            
        for exp_name, metrics in raw.items():
            method_group = infer_posthoc_or_posttrain(exp_name, arch)
            is_quant = infer_isquant(exp_name)
            base_name = clean_exp_name(exp_name)
            run_id = infer_run_id(exp_name)
            acc = metrics.get("final_accuracy")
            params = metrics.get("param_count")
            
            rows.append({
                "dataset": dataset, "architecture": arch, "run_id": run_id, 
                "exp_name": exp_name, "base_name": base_name, 
                "display_name": f"{base_name}\n(Quant)" if is_quant else base_name,
                "posthoc_or_posttrain": method_group, "model_type": infer_model_type(exp_name),
                "is_quantized": is_quant, "accuracy": acc,
                "params": params, "flops": metrics.get("flops"),
                "memory": metrics.get("total_size_mb"),
            })
            
    df = pd.DataFrame(rows)
    logger.info(f"Loaded {len(df)} total experiment rows.")
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = []
    
    # FIXED: Group by run_id as well to isolate Kevin vs JF baselines
    for (ds, arch, run_id), g in df.groupby(["dataset", "architecture", "run_id"]):
        baseline = find_baseline(g)
        if baseline is None:
            logger.debug(f"No Baseline for {arch}/{ds} ({run_id} run). Skipping deltas.")
            for _, r in g.iterrows(): out.append(r)
            continue
            
        logger.debug(f"Normalizing {arch}/{ds} ({run_id}) against Baseline: {baseline['accuracy']:.2f}%")
        
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
# Figure Generations
# =========================

def fig1(df: pd.DataFrame, metrics: list[str] = ["accuracy", "params", "flops", "memory"], out_dir: Path = Path("./figures/individual_plots")):
    if df.empty: return
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Baseline": "#333333", "Retrained": "#ff7f0e", "Not Pruned": "#ff7f0e", "Collapsed": "#2ca02c"}
    group_cols = ["dataset", "architecture", "base_name", "display_name", "posthoc_or_posttrain", "is_quantized"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    df_agg = df.groupby(group_cols, dropna=False)[available_metrics].mean(numeric_only=True).reset_index()

    for architecture, df_arch in df_agg.groupby("architecture"):
        for dataset in df_arch["dataset"].unique():
            g_dataset = df_arch[df_arch["dataset"] == dataset].copy()
            if g_dataset.empty: continue
            
            # Sort baseline to left
            g_dataset['is_baseline'] = g_dataset['base_name'].apply(lambda x: 0 if 'Original' in x or 'Baseline' in x else 1)
            g_dataset = g_dataset.sort_values(by=['is_baseline', 'display_name'])
            
            clean_ds = DATASET_NAME_MAP.get(dataset, dataset)

            for metric in available_metrics:
                fig, ax = plt.subplots(figsize=(14, 6))
                sns.barplot(data=g_dataset, x="display_name", y=metric, hue="posthoc_or_posttrain", 
                            palette=palette, edgecolor="black", ax=ax, dodge=False)
                
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, title="Optimization Strategy", loc='upper right', bbox_to_anchor=(1.0, 1.0))
                ax.set_title(f"{metric.capitalize()} Comparison: {architecture} on {clean_ds}")
                ax.set_xlabel("Model Configurations")
                ax.set_ylabel(metric.capitalize())
                plt.xticks(rotation=45, ha="right", rotation_mode="anchor")
                plt.savefig(out_dir / f"{architecture}_{dataset}_{metric}.png")
                plt.close()

def fig4_heuristic_search_space_map(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/search_space")):
    try:
        from transfer import EXPERIMENTS 
    except ImportError:
        logger.warning("Could not import EXPERIMENTS from transfer.py. Skipping Fig 4.")
        return
        
    if df.empty: return
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def get_acc_color(drop):
        if pd.isna(drop): return "#e0e0e0" 
        # FIXED: Account for models that perform better than baseline (negative drop)
        color_drop = max(0, drop)
        if color_drop < 1.5: return "#2ca02c"    
        if color_drop < 5.0: return "#ff7f0e"    
        return "#d62728"                   

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"Building Search Space Map for {arch} on {dataset}...")
        
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): 
            logger.warning(f"  -> Skipping. Missing variance file: {csv_path}")
            continue
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = layer_df['Variance'].values
        
        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: 
            continue
            
        # Pre-process base_names for fuzzy matching
        # Averages across runs (e.g. JF and Kevin) if both exist for the same base_name
        g_agg = g_metrics.groupby("base_name", as_index=False).mean(numeric_only=True)
        metrics_lookup = {standardize_heuristic_name(r['base_name']): r for _, r in g_agg.iterrows()}
        
        clean_ds = DATASET_NAME_MAP.get(dataset, dataset)
        fig, (ax_var, ax_heur) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                             gridspec_kw={'height_ratios': [1, 2]})
        plt.subplots_adjust(hspace=0.08)

        # Top Panel
        ax_var.plot(range(len(layers)), variances, color='#555555', linewidth=1.5, alpha=0.8, label="Activation Variance")
        ax_var.fill_between(range(len(layers)), variances, color='gray', alpha=0.1)
        ax_var.set_yscale('log')
        ax_var.set_ylabel("Variance ($\sigma^2$)")
        ax_var.set_title(f"Heuristic Search Space Guide: {arch} on {clean_ds}", loc='left', pad=20, fontweight='bold')
        
        variance_threshold = np.percentile(variances, 25)
        ax_var.axhline(y=variance_threshold, color='green', linestyle='--', alpha=0.4, label="Low Variance Threshold")
        ax_var.legend(loc='lower right', bbox_to_anchor=(1.0, 1.05), frameon=True, ncol=2)

        # Bottom Panel
        exp_list = [n for n, r in model_exps.items() if r is not None]
        
        for i, exp_name in enumerate(exp_list):
            ranges = model_exps[exp_name]
            ranges = ranges if isinstance(ranges, list) else [ranges]
            
            clean_target = standardize_heuristic_name(exp_name)
            match = metrics_lookup.get(clean_target)
            
            if match is not None:
                acc_drop = match['acc_drop']
                final_acc = match['accuracy']
                color = get_acc_color(acc_drop)
                label_text = f"{final_acc:.1f}% ($\Delta$ {-acc_drop:+.1f}%)"
                lw = 12
                txt_color = color
            else:
                color = get_acc_color(np.nan)
                label_text = "N/A"
                lw = 6 
                txt_color = '#aaaaaa'

            for start_layer, end_layer in ranges:
                try:
                    s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                    e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                    ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=lw, color=color, alpha=0.9)
                    ax_heur.text(e_idx + 0.5, i, label_text, va='center', fontsize=9, fontweight='bold', color=txt_color)
                except StopIteration:
                    continue
        
        ax_heur.set_yticks(range(len(exp_list)))
        ax_heur.set_yticklabels([clean_exp_name(e) for e in exp_list], fontsize=10)
        ax_heur.set_xlabel("Network Depth (Layer Index)")
        ax_heur.set_ylabel("Collapsed Layer Candidates")
        
        sns.despine(ax=ax_var)
        sns.despine(ax=ax_heur)
        ax_heur.grid(axis='x', alpha=0.15)

        save_path = out_dir / f"{arch}_{dataset}_decision_map.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    try:
        logger.info("Initializing plot generation script...")
        raw = load_results()
        df = normalize(raw)
        fig1(df)
        fig4_heuristic_search_space_map(df)
        logger.info("Script completed. All targeted journals generated successfully.")
    except Exception as e:
        logger.critical(f"FATAL ERROR during execution: {e}", exc_info=True)