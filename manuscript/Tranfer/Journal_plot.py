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
    # 1. Pre-load heuristic CSVs to map ACS scores
    heuristic_files = list(RESULTS_DIR.rglob("*_heuristics.csv"))
    acs_data = {}
    for hf in heuristic_files:
        # Rely on your existing inference functions to group them
        arch = infer_architecture_from_path(hf)
        ds = infer_dataset_from_path(hf)
        try:
            df_h = pd.read_csv(hf)
            acs_data[(arch, ds)] = df_h
        except Exception as e:
            print(f"Skipping heuristics {hf}: {e}")

    # 2. Load merged metrics JSONs
    files = list(RESULTS_DIR.rglob("*merged_metrics.json"))
    if not files:
        if (RESULTS_DIR / "merged_metrics.json").exists():
            files = [RESULTS_DIR / "merged_metrics.json"]
        else:
            raise FileNotFoundError("No merged_metrics.json files found")

    rows = []

    for p in files:
        dataset = infer_dataset_from_path(p)
        arch = infer_architecture_from_path(p)
        
        # Retrieve the matching heuristic dataframe for this architecture/dataset
        h_df = acs_data.get((arch, dataset), None)
        
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

            # ---------------------------------------------------------
            # NEW: Map the Adaptive Collapse Score (ACS) to the Experiment
            # ---------------------------------------------------------
            avg_acs = None
            if h_df is not None and not h_df.empty:
                # Use your existing function to get the target layer tuples
                collapse_range = get_collapse_range(arch, exp_name)
                
                if collapse_range:
                    scores = []
                    layer_names = h_df['layer'].tolist()
                    
                    # collapse_range is a list of tuples: [('start_layer', 'end_layer')]
                    for start_layer, end_layer in collapse_range:
                        start_idx, end_idx = -1, -1
                        
                        # Fuzzy match layer names to find start/end indices in the CSV
                        for i, lname in enumerate(layer_names):
                            if start_layer in lname: start_idx = i
                            if end_layer in lname: end_idx = i
                            
                        if start_idx != -1 and end_idx != -1:
                            if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
                            
                            # Slice the heuristics dataframe and get the mean ACS score
                            subset = h_df.iloc[start_idx : end_idx + 1]
                            scores.append(subset['collapse_score'].mean())
                            
                    if scores:
                        avg_acs = np.mean(scores)

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
                    
                    # NEW: Add ACS Score
                    "acs_score": avg_acs,
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


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
def fig1(
    df: pd.DataFrame,
    metrics: list[str] = ["accuracy", "params", "flops", "memory", "acs_score"], # <-- NEW: Added acs_score
    out_dir: Path = Path("./figures/individual_plots"),
):
    """
    Generates improved INDIVIDUAL plot files AND LaTeX tables.
    - Converts Params to Millions (M) and FLOPs to GFLOPs (G).
    - Removes duplicate rows for cleaner tables.
    - Saves a LaTeX table summary including ACS scores.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()

    # Updated titles to reflect new units and ACS
    metric_titles = {
        "accuracy": "Accuracy (%)",
        "params": "Params (M)",
        "flops": "GFLOPs",
        "memory": "Memory (MB)",
        "acs_score": "Collapse Score (ACS)" # <-- NEW
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

            # Determine Sort Order
            if "params" in g_dataset.columns:
                base_name_rank = g_dataset.groupby("base_name")["params"].max().sort_values(ascending=False)
            else:
                base_name_rank = g_dataset.groupby("base_name")["accuracy"].max().sort_values(ascending=True)
            
            rank_map = {name: i for i, name in enumerate(base_name_rank.index)}
            
            g_dataset["rank"] = g_dataset["base_name"].map(rank_map)
            g_dataset.sort_values(["rank", "is_quantized"], ascending=[True, True], inplace=True)
            
            sort_order = g_dataset["display_name"].unique().tolist()

            # --- Save Data as LaTeX Table ---
            table_cols = ["display_name", "posthoc_or_posttrain"] + [m for m in metrics if m in g_dataset.columns]
            table_df = g_dataset[table_cols].copy()
            
            # 1. Scale metrics for better readability
            if "params" in table_df.columns:
                table_df["params"] = table_df["params"] / 1e6
            if "flops" in table_df.columns:
                table_df["flops"] = table_df["flops"] / 1e9

            # 2. Drop duplicates (cleaning up repeated entries)
            table_df = table_df.drop_duplicates(subset=["display_name", "posthoc_or_posttrain"])

            # 3. Rename columns using the metric_titles dictionary
            rename_map = {"display_name": "Model", "posthoc_or_posttrain": "Type"}
            rename_map.update(metric_titles)
            table_df.rename(columns=rename_map, inplace=True)
            
            tex_filename = f"{architecture}_{dataset}_table.tex".replace(" ", "_")
            
            # 4. Export to LaTeX (Formats floats dynamically)
            table_df.to_latex(
                out_dir / tex_filename,
                index=False,
                float_format=lambda x: "%.4f" % x if x < 10 else "%.2f" % x, # High precision for ACS
                caption=f"Performance metrics and ACS for {architecture} on {dataset}.",
                label=f"tab:{architecture}_{dataset}",
                escape=True
            )
            print(f"[Table] Saved {tex_filename}")
            # -------------------------------------

            for metric in metrics:
                if metric not in g_dataset.columns or g_dataset[metric].isnull().all():
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
                            patch.set_linewidth(1.0)

                ax.set_ylabel(metric_titles.get(metric, metric), fontsize=12, fontweight='bold')
                ax.set_xlabel("")
                ax.set_title(f"{architecture} - {dataset} ({metric_titles.get(metric, metric)})", fontsize=14, fontweight='bold')
                
                # Dynamic Y-Limit for ACS Score to ensure 0-1 scale is visible
                if metric == "acs_score":
                    ax.set_ylim(0, 1.1)
                
                ax.legend(title="Method", loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
                
                plt.xticks(rotation=45, ha="right")
                plt.grid(True, axis="y", linestyle="--", alpha=0.3)
                plt.tight_layout()
                
                filename = f"{architecture}_{dataset}_{metric}.png".replace(" ", "_")
                plt.savefig(out_dir / filename, bbox_inches='tight')
                plt.close()
                print(f"[Plot] Saved {filename}")

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
               
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()