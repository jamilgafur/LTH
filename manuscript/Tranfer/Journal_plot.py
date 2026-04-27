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

# =========================
# Configuration & Logging
# =========================
# CHANGED: Log level set to DEBUG to output everything

logging.basicConfig(
    level=logging.DEBUG, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
# Set Matplotlib's logger to INFO or WARNING level
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING) # Silences image processing logs too
logger = logging.getLogger(__name__)

# Make sure Pandas prints out all the columns when debugging
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

# Map raw dataset strings to Journal-ready text
DATASET_NAME_MAP = {
    "tinyimagenet": "TinyImageNet",
    "cifar10_": "CIFAR-10",
    "cifar100_": "CIFAR-100",
    "imagenet": "ImageNet"
}

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
        else: 
            logger.warning("No merged_metrics.json files found returning empty dataframe.")
            return pd.DataFrame() 

    logger.debug(f"Found files: {[str(f) for f in files]}")
    rows = []
    
    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower(): dataset = "tinyimagenet"
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch": arch = infer_architecture_from_path(Path(p.name))
        
        try:
            with open(p) as f: raw = json.load(f)
            logger.debug(f"Loaded {len(raw)} experiments from {p.name} (Arch: {arch}, Dataset: {dataset})")
        except Exception as e: 
            logger.error(f"Failed to load JSON {p}: {e}")
            continue
            
        for exp_name, metrics in raw.items():
            method_group = infer_posthoc_or_posttrain(exp_name, arch)
            is_quant = infer_isquant(exp_name)
            base_name = clean_exp_name(exp_name)
            
            # Extract metrics safely
            acc = metrics.get("final_accuracy")
            params = metrics.get("param_count")
            
            logger.debug(f"  -> Extracted: {exp_name} | Acc: {acc} | Params: {params}")
            
            rows.append({
                "dataset": dataset, "architecture": arch, "exp_name": exp_name,
                "base_name": base_name, "display_name": f"{base_name}\n(Quant)" if is_quant else base_name,
                "posthoc_or_posttrain": method_group, "model_type": infer_model_type(exp_name),
                "is_quantized": is_quant, "accuracy": acc,
                "params": params, "flops": metrics.get("flops"),
                "memory": metrics.get("total_size_mb"),
            })
            
    df = pd.DataFrame(rows)
    logger.info(f"\n{'='*20} RAW DATAFRAME LOADED {'='*20}\n{df.head(15)}\nTotal rows: {len(df)}")
    return df

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    out = []
    
    logger.info(f"\n{'='*20} STARTING NORMALIZATION {'='*20}")
    
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        logger.debug(f"\nNormalizing Group: Architecture='{arch}', Dataset='{ds}' (Rows: {len(g)})")
        
        baseline = find_baseline(g)
        if baseline is None:
            logger.warning(f"  -> NO BASELINE FOUND for {arch} on {ds}. Skipping delta calculations.")
            for _, r in g.iterrows(): out.append(r)
            continue
            
        logger.debug(f"  -> Found Baseline: '{baseline['exp_name']}' | Acc: {baseline['accuracy']} | Params: {baseline['params']}")
        
        for _, r in g.iterrows():
            row = r.copy()
            if pd.notnull(baseline.get("params")) and baseline["params"] > 0:
                row["d_acc"] = r["accuracy"] - baseline["accuracy"] 
                row["acc_drop"] = baseline["accuracy"] - r["accuracy"] 
                row["baseline_acc"] = baseline["accuracy"] 
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
                
                logger.debug(f"    -> Row: {r['base_name']} | Raw Acc: {r['accuracy']} | Delta Acc: {row['d_acc']:.2f} | Acc Drop: {row['acc_drop']:.2f}")
                
                if pd.notnull(baseline.get("flops")) and baseline["flops"] > 0:
                    row["d_flops"] = 100 * (1 - r["flops"] / baseline["flops"])
                if pd.notnull(baseline.get("memory")) and baseline["memory"] > 0:
                    row["d_memory"] = 100 * (1 - r["memory"] / baseline["memory"])
            else:
                logger.debug(f"    -> Missing params in baseline, skipping delta for {r['base_name']}")
                
            out.append(row)
            
    norm_df = pd.DataFrame(out)
    logger.info(f"\n{'='*20} NORMALIZED DATAFRAME {'='*20}\n{norm_df[['architecture', 'base_name', 'accuracy', 'acc_drop']].head(15)}\n")
    return norm_df

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
            
            logger.debug(f"\n[FIG 1 DEBUG] Sorted Plotting Data for {architecture} on {dataset}:")
            logger.debug(f"\n{g_dataset[['display_name', 'is_baseline', 'accuracy', 'posthoc_or_posttrain']]}")
            
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
        if drop < 1.5: return "#2ca02c"    
        if drop < 5.0: return "#ff7f0e"    
        return "#d62728"                   

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"\n{'='*20} BUILDING FIG 4 FOR {arch} on {dataset} {'='*20}")
        
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): 
            logger.warning(f"  -> Skipping. Missing variance file: {csv_path}")
            continue
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = layer_df['Variance'].values
        logger.debug(f"  -> Loaded Layer Stats: {len(layers)} layers")
        
        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: 
            logger.warning(f"  -> Skipping. No heuristics defined in EXPERIMENTS for {arch}/{dataset}")
            continue
            
        logger.debug(f"  -> Found {len(model_exps)} target heuristics in EXPERIMENTS.")
        
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
            
            exp_results = g_metrics[g_metrics['base_name'] == exp_name]
            
            if not exp_results.empty:
                acc_drop = exp_results['acc_drop'].iloc[0]
                final_acc = exp_results['accuracy'].iloc[0]
                color = get_acc_color(acc_drop)
                label_text = f"{final_acc:.1f}% ($\Delta$ {-acc_drop:+.1f}%)"
                lw = 12
                txt_color = color
                logger.debug(f"    -> Plotted: '{exp_name}' | Final Acc: {final_acc:.2f} | Drop: {acc_drop:.2f}")
            else:
                color = get_acc_color(np.nan)
                label_text = "N/A"
                lw = 6 
                txt_color = '#aaaaaa'
                logger.debug(f"    -> Plotted: '{exp_name}' | MISSING DATA (N/A)")

            for start_layer, end_layer in ranges:
                try:
                    s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                    e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                    ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=lw, color=color, alpha=0.9)
                    ax_heur.text(e_idx + 0.5, i, label_text, va='center', fontsize=9, fontweight='bold', color=txt_color)
                except StopIteration:
                    logger.error(f"    -> ERROR matching layers '{start_layer}' or '{end_layer}' for heuristic '{exp_name}'")
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