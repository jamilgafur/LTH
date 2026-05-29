# Journal_plot.py
from __future__ import annotations
import warnings
# Suppress tight_layout warnings caused by GridSpec overlaps
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

import json
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration & Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

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
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

import argparse

def get_stats_dir(epochs, pretrain):
    # This ensures Journal_plot.py looks exactly where transfer.py saved the data
    return Path(f"./runs/plots/Layer_Statistics_ep{epochs}_pre{pretrain}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Journal Plots")

    parser.add_argument('--pretrain', type=int, default=None, 
                        help="Specific epoch integer to filter the Pretrain phase.")
    parser.add_argument('--epoch', type=int, default=None, 
                        help="Specific epoch integer to filter the regular/finetuning phases.")
    
    return parser.parse_args()
args = parse_args()
epochs = args.epoch
pretrain = args.pretrain
RESULTS_DIR = Path(f"./*epochs{epochs}*pretrain{pretrain}*")

# --- UPDATE: Make output directories dynamic based on parameters ---
FIG_DIR = Path(f"./figures_ep{epochs}_pre{pretrain}")
TABLE_DIR = Path(f"./tables_ep{epochs}_pre{pretrain}")
DIAGNOSTICS_DIR = Path(f"./diagnostics_ep{epochs}_pre{pretrain}")

for d in [FIG_DIR, TABLE_DIR, DIAGNOSTICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
    
DATASET_ORDER = ["cifar10_", "cifar100_", "tinyimagenet", "imagenet", "ConvNeXt"]

# =========================
# Data Loading & Utilities
# =========================

def format_dataset_name(ds: str) -> str:
    mapping = {"tinyimagenet": "TinyImageNet", "cifar10_": "CIFAR-10", "cifar100_": "CIFAR-100", "imagenet": "ImageNet"}
    return mapping.get(ds, ds.capitalize())

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
    if "original" in n or "baseline" in n or "control" in n:
        return "Baseline"
    if "vgg16" in architecture.lower() or "regnetx" in architecture.lower():
        return "Retrained"
    return "Collapsed"

def clean_exp_name(exp_name: str) -> str:
    n = exp_name
    n = re.sub(r'(?i)[_\-\s\(]*quant(ized)?[\)]*', '', n)
    n = re.sub(r'(?i)[_\-\s\(]?(jf|kevin|no-prune|not pruned|pruned)\b\)?', '', n)
    for arch in ["RegNetX_400MF", "VGG16", "MobileNet", "ConvNeXt", "InceptionNet", "XceptionNet"]:
        n = re.compile(re.escape(arch + "_"), re.IGNORECASE).sub('', n)
        n = re.compile(re.escape(arch), re.IGNORECASE).sub('', n)
    n = n.replace("Block ", "Block-").replace("Stage ", "Stage-") 
    n = n.replace(" Only", "").replace("()", "") 
    n = n.strip(" -_")
    
    if "Original" in n or "Baseline" in n or "Control" in n: 
        if "Continuted" in n: return "Control (Continued)"
        return "Control Model"
    return n.strip()

def load_results() -> pd.DataFrame:
    logger.info(f"Scanning for metrics files matching epochs={epochs}, pretrain={pretrain}")
    
    # 1. Search for all merged_metrics.json files recursively
    all_metrics = list(Path(".").rglob("*merged_metrics.json"))
    
    # 2. Filter them to only include those matching the current epochs and pretrain strings
    files = [
        p for p in all_metrics 
        if f"epochs{epochs}" in str(p) and f"pretrain{pretrain}" in str(p)
    ]
    
    if not files:
        # Fallback just in case it's in the root directory
        root_file = Path("merged_metrics.json")
        if root_file.exists(): 
            files = [root_file]
        else: 
            raise FileNotFoundError(f"No merged_metrics.json files found matching epochs{epochs} and pretrain{pretrain}")
    
    rows = []
    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower(): dataset = "tinyimagenet"
        if dataset == "unknown" and "cifar100" in str(p).lower(): dataset = "cifar100_"
        if dataset == "unknown" and "cifar10" in str(p).lower(): dataset = "cifar10_"
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
            
            rows.append({
                "dataset": dataset, "architecture": arch, "exp_name": exp_name,
                "base_name": base_name, "display_name": f"{base_name}\n(Quant)" if is_quant else base_name,
                "posthoc_or_posttrain": method_group, "model_type": infer_model_type(exp_name),
                "is_quantized": is_quant, "accuracy": metrics.get("final_accuracy"),
                "params": metrics.get("param_count"), "flops": metrics.get("flops"),
                "memory": metrics.get("total_size_mb"),
                "acc_curve": metrics.get("accuracies", []),
                "loss_curve": metrics.get("losses", []),
            })
            
    logger.info(f"Successfully parsed {len(rows)} experiment rows.")
    return pd.DataFrame(rows)

def find_baseline(df: pd.DataFrame):
    # Matches transfer.py phases (including "control")
    mask = (df["exp_name"].str.lower().str.contains("original") | 
            df["exp_name"].str.lower().str.contains("baseline") |
            df["exp_name"].str.lower().str.contains("control"))
            
    b_df = df[mask & (df["is_quantized"] == False)]
    if b_df.empty: b_df = df[mask]
    if b_df.empty: return None
    
    # Crucial: Phase 2 continued control is the true baseline for Stage 2 comparisons
    control_cont = b_df[b_df["exp_name"].str.lower().str.contains("continuted")]
    if not control_cont.empty:
        return control_cont.iloc[0] # Avoids .mean() dropping object columns
        
    return b_df.iloc[0] # Avoids .mean() dropping object columns

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        # Use our strict baseline finder
        baseline = find_baseline(g)
        if baseline is None:
            for _, r in g.iterrows(): out.append(r)
            continue
            
        b_acc = baseline["accuracy"]
        b_params = baseline["params"]
        
        # Isolate the baseline curves with robust mask
        baseline_mask = (g["exp_name"].str.lower().str.contains("baseline") | 
                         g["exp_name"].str.lower().str.contains("control") | 
                         g["exp_name"].str.lower().str.contains("original"))
                         
        b_df_curves = g[baseline_mask & (g["is_quantized"] == False)]
        if b_df_curves.empty: b_df_curves = g[baseline_mask]
        
        b_loss_curve = b_df_curves.iloc[0].get("loss_curve", []) if not b_df_curves.empty else []

        for _, r in g.iterrows():
            row = r.copy()
            if pd.notnull(b_params) and b_params > 0:
                row["d_acc"] = r["accuracy"] - b_acc 
                row["baseline_acc"] = b_acc 
                row["d_params"] = 100 * (1 - r["params"] / b_params)
                
                # --- Curve Dynamic Math ---
                r_loss = r.get("loss_curve", [])
                
                if len(r_loss) > 5 and len(b_loss_curve) > 5:
                    # 1. Asymptotic Loss Delta (Difference in last 5 epochs)
                    cand_asymptote = np.mean(r_loss[-5:])
                    base_asymptote = np.mean(b_loss_curve[-5:])
                    row["d_asymptotic_loss"] = cand_asymptote - base_asymptote
                    
                    # 2. Trajectory Correlation (Pearson R)
                    min_len = min(len(r_loss), len(b_loss_curve))
                    if min_len > 10:
                        corr_matrix = np.corrcoef(r_loss[:min_len], b_loss_curve[:min_len])
                        row["loss_correlation"] = corr_matrix[0, 1]
                else:
                    row["d_asymptotic_loss"] = None
                    row["loss_correlation"] = None

            out.append(row)
            
    # Drop the heavy raw arrays before returning so the dataframe stays light
    df_out = pd.DataFrame(out)
    if "acc_curve" in df_out.columns:
        df_out = df_out.drop(columns=["acc_curve", "loss_curve"])
    return df_out
# =========================
# Figure Generations
# =========================

def fig1(df: pd.DataFrame, metrics: list[str] = ["accuracy", "params", "flops", "memory"], out_dir: Path = Path("./figures/individual_plots")):
    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Baseline": "#333333", "Retrained": "#ff7f0e", "Collapsed": "#2ca02c"}
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
                plt.savefig(out_dir / f"{architecture}_{dataset}_{metric}.png")
                logger.info(f"[FIG1] Saved {metric} plot for {architecture}/{dataset} at {out_dir / f'{architecture}_{dataset}_{metric}.png'}")
                plt.close()


# ========================= FIG 2 ========================= #

def fig2_methodology_bav_regions(epochs, pretrain, out_dir=Path("./figures/methodology")):
    logger.info(f"Starting FIG2 generation. Target epochs: {epochs}, pretrain: {pretrain}")
    
    # 1. Robust Path Discovery: Search specifically for the 'Layer_Statistics' subfolders
    # Structure: .../epochs{X}_pretrain{Y}/**/Layer_Statistics
    root_plots_dir = Path("./runs/plots")
    search_pattern = f"**/epochs{epochs}_pretrain{pretrain}/**/Layer_Statistics"
    search_paths = list(root_plots_dir.glob(search_pattern))
    
    if not search_paths:
        logger.error(f"No Layer_Statistics directories found matching {search_pattern}")
        return

    for stats_dir in search_paths:
        # Extract Architecture and Dataset from the path hierarchy
        # Pattern: runs/plots/{arch}/{dataset}/epochs.../Layer_Statistics
        try:
            dataset = stats_dir.parents[1].name
            arch = stats_dir.parents[2].name
        except IndexError:
            logger.warning(f"Could not infer arch/dataset from path {stats_dir}, using 'unknown'")
            arch, dataset = "unknown", "unknown"
            
        stat_files = list(stats_dir.glob("*_layer_stats.csv"))
        if not stat_files:
            logger.warning(f"No CSVs found in {stats_dir}")
            continue

        for csv_path in stat_files:
            try:
                layer_df = pd.read_csv(csv_path)
                if layer_df.empty or 'Variance' not in layer_df.columns:
                    continue
                    
                layers = layer_df['Layer'].tolist()
                variances = np.maximum(layer_df['Variance'].values, 1e-6)
            except Exception as e:
                logger.error(f"Failed to read CSV {csv_path}: {e}")
                continue

            # --- Calculation Logic ---
            h_vals, sigma_bars = [], []
            window_size = 3  
            for i, sigma_i in enumerate(variances):
                start = max(0, i - window_size)
                end = min(len(variances), i + window_size + 1)
                local_vars = variances[start:end]
                sigma_bar = np.mean(local_vars) if len(local_vars) > 0 else np.mean(variances)
                sigma_bars.append(max(sigma_bar, 1e-12))
                
                diff = sigma_i - sigma_bars[-1]
                h = max(diff / sigma_bars[-1], -1.0) if diff < 0 else min(diff / sigma_bars[-1], 1.0)
                h_vals.append(h)
                
            # --- JSON Region Matching ---
            json_pattern = f"*{arch}*{dataset}*discovered_regions.json"
            json_files = list(Path(".").glob(json_pattern))
            verified_idx_ranges = []
            
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        config = json.load(f)
                    for k, v in config.items():
                        if k.startswith("Set_") and isinstance(v, (list, tuple)):
                            s_idx = next(idx for idx, n in enumerate(layers) if v[0] == n)
                            e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if v[-1] == n)
                            verified_idx_ranges.append((s_idx, e_idx))
                except Exception as e:
                    logger.error(f"Error parsing JSON: {e}")

            # --- Plotting ---
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
            x_vals = range(1, len(layers) + 1)
            
            ax1.plot(x_vals, variances, color='#4A4A4A', marker='o', markersize=4, label=r'Layer Variance ($\sigma_i$)')
            ax1.plot(x_vals, sigma_bars, color='#ff7f0e', linestyle='--', linewidth=2.5, label=r'Local Context Mean ($\bar{\sigma}$)')
            ax1.set_yscale('log')
            ax1.set_title(f"Dynamic Structural Redundancy Analysis\n{arch} on {dataset.capitalize()}", fontweight='bold')
            ax1.legend(loc='upper right', frameon=False)
            sns.despine(ax=ax1)

            # --- State Logic ---
            veto_idx = int(len(layers) * 0.25)
            final_states = ["DANGER"] * len(layers)
            for i in range(len(layers)):
                if i < veto_idx:
                    final_states[i] = "VETO"
                else:
                    in_verified = any(s_idx <= i <= e_idx for s_idx, e_idx in verified_idx_ranges)
                    final_states[i] = "VERIFIED" if in_verified else ("REJECTED" if h_vals[i] < 0 else "DANGER")

            # --- Bar Plot ---
            state_colors = {"VETO": "#999999", "VERIFIED": "#2ca02c", "REJECTED": "#ff7f0e", "DANGER": "#d62728"}
            ax2.bar(x_vals, h_vals, color=[state_colors[s] for s in final_states], alpha=0.85, edgecolor='black', linewidth=0.5)
            ax2.axhline(y=0, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=2)
            ax2.set_ylim(-1.1, 1.1)

            # --- Zone Shading ---
            zones = []
            if final_states:
                start_idx, current_state = 0, final_states[0]
                for i in range(1, len(final_states)):
                    if final_states[i] != current_state:
                        zones.append((start_idx, i - 1, current_state))
                        start_idx, current_state = i, final_states[i]
                zones.append((start_idx, len(final_states) - 1, current_state))

            for start, end, state in zones:
                span_start, span_end = (start + 1) - 0.5, (end + 1) + 0.5
                alpha_val = 0.4 if state == "VETO" else 0.15
                # CHANGED 'color' -> 'facecolor'
                ax1.axvspan(span_start, span_end, facecolor=state_colors.get(state, 'red'), alpha=alpha_val, edgecolor='none')
                ax2.axvspan(span_start, span_end, facecolor=state_colors.get(state, 'red'), alpha=alpha_val, label=state)

            ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
            plt.tight_layout()
            
            # --- Final Save ---
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"{arch}_{dataset}_bav_methodology_regions.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            logger.info(f"[FIG2] Saved methodology plot: {save_path}")

# ========================= FIG 3 ========================= #

def fig3_v2t_heuristic_validation(
    df: pd.DataFrame,
    epochs: int,
    pretrain: int,
    out_dir: Path = Path("./figures/heuristic_validation")
):
    out_dir.mkdir(parents=True, exist_ok=True)

    REQUIRED_COLS = {"dataset", "architecture", "posthoc_or_posttrain", "base_name", "is_quantized", "d_acc"}
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        logger.error(f"[FIG3] Missing required columns: {missing}")
        return

    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]
    all_merged_data = []

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        clean_metrics = g_metrics[g_metrics['posthoc_or_posttrain'].isin(['Collapsed', 'Retrained'])]
        if clean_metrics.empty: continue

        # --- Dynamic Path Resolution ---
        clean_ds = dataset.strip("_")
        arch_dir = Path(f"./runs/plots/{arch}")
        if not arch_dir.exists(): continue
        
        # Case-insensitive dataset folder match
        ds_dirs = [d for d in arch_dir.iterdir() if d.is_dir() and d.name.lower() == clean_ds.lower()]
        if not ds_dirs: continue
        
        target_dir = ds_dirs[0] / f"epochs{epochs}_pretrain{pretrain}"
        if not target_dir.exists(): continue

        # Scan all CSVs in this specific run to find the one with the Relative Variance stats
        csv_candidates = list(target_dir.rglob("*.csv"))
        df_h = pd.DataFrame()
        
        for c in csv_candidates:
            try:
                temp_df = pd.read_csv(c)
                # FIX: Changed "Median Variance" to "Relative Variance"
                if "Experiment" in temp_df.columns and "Relative Variance" in temp_df.columns:
                    df_h = temp_df
                    break
            except Exception: continue

        if df_h.empty: 
            logger.debug(f"[FIG3] No valid variance CSV found for {arch}/{dataset}")
            continue

        merged = pd.merge(df_h, clean_metrics, left_on="Experiment", right_on="base_name", how="inner")
        if merged.empty: continue

        merged["Topology"] = "Multi-Path" if arch in multi_path_archs else "Single-Path"
        all_merged_data.append(merged)

    if not all_merged_data: return

    full_df = pd.concat(all_merged_data, ignore_index=True)
    full_csv_path = out_dir / "fig3_full_merged_data.csv"
    full_df.to_csv(full_csv_path, index=False)

    full_df['Quantization State'] = full_df['is_quantized'].map({False: 'Unquantized', True: 'Quantized'})

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    for ax in axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
        ax.axhspan(0, 100, color='#2ca02c', alpha=0.05, zorder=0)
        ax.axhspan(-2.0, 0, color='#ff7f0e', alpha=0.05, zorder=0)
        ax.axhspan(-100, -2.0, color='#d62728', alpha=0.05, zorder=0)

    y_min, y_max = full_df['d_acc'].min() - 2, max(full_df['d_acc'].max() + 2, 5)
    for ax in axes: ax.set_ylim(y_min, y_max)

    # --- Single Path ---
    sp_df = full_df[full_df['Topology'] == 'Single-Path']
    if not sp_df.empty:
        # FIX: Changed 'Median Variance' to 'Relative Variance'
        sns.scatterplot(data=sp_df, x="Relative Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"}, s=250, alpha=0.9, edgecolor="black", color="#2ca02c", ax=axes[0], zorder=3)
        q1 = sp_df["Relative Variance"].quantile(0.3)
        axes[0].axvspan(1e-4, q1, color='#2ca02c', alpha=0.15, zorder=0, label=f"BAV Target: Bounded Growth (<{q1:.1f})")
        axes[0].set_xscale('symlog', linthresh=1e-2)
        axes[0].set_title("Single-Path: Target Bounded Growth", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylabel(r"$\Delta$ Accuracy (%) $\rightarrow$ Higher is Better", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Relative Activation Variance (SymLog Scale)", fontsize=12)
        axes[0].legend(loc="lower left", framealpha=0.9)

    # --- Multi Path ---
    mp_df = full_df[full_df['Topology'] == 'Multi-Path']
    if not mp_df.empty:
        # FIX: Changed 'Median Variance' to 'Relative Variance'
        sns.scatterplot(data=mp_df, x="Relative Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"}, s=250, alpha=0.9, edgecolor="black", color="#d62728", ax=axes[1], zorder=3)
        q1_mp = mp_df["Relative Variance"].quantile(0.4)
        axes[1].axvspan(1e-4, q1_mp, color='#2ca02c', alpha=0.15, zorder=0, label=f"BAV Target: Variance Valleys (<{q1_mp:.1f})")
        axes[1].set_xscale('symlog', linthresh=1e-2)
        axes[1].set_title("Multi-Path: Target Terminal Stabilization", fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel("Relative Activation Variance (SymLog Scale)", fontsize=12)
        axes[1].legend(loc="lower right", framealpha=0.9)

    sns.despine()
    plt.savefig(out_dir / "V2T_heuristic_validation_map.png")
    plt.close()
    logger.info(f"[FIG3] Saved V2T heuristic validation map at {out_dir / 'V2T_heuristic_validation_map.png'}")
# ========================= FIG 4 ========================= #
def fig4_comprehensive_search_space_map(
    df: pd.DataFrame,
    epochs: int,
    pretrain: int,
    out_dir: Path = Path("./figures/search_space")
):
    import matplotlib.gridspec as gridspec
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_acc_color(d_acc):
        if pd.isna(d_acc): return "#4C72B0" 
        if d_acc >= -2.0: return "#2ca02c"
        if d_acc >= -6.0: return "#ff7f0e"
        return "#d62728"

    def robust_match(target_name, is_quant_target, g_df):
        try:
            sub_df = g_df[g_df['is_quantized'] == is_quant_target]
            if sub_df.empty: return pd.DataFrame()
            m = sub_df[sub_df['base_name'] == target_name]
            if not m.empty: return m
            m = sub_df[sub_df['base_name'].str.lower() == target_name.lower()]
            if not m.empty: return m
            def squash(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
            st = squash(target_name)
            fuzzy = sub_df[sub_df['base_name'].apply(lambda x: squash(x) == st)]
            return fuzzy
        except Exception: return pd.DataFrame()

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        clean_ds = dataset.strip("_").lower()
        
        # --- Bulletproof Path Discovery for Layer Stats ---
        arch_dir = Path(f"./runs/plots/{arch}")
        if not arch_dir.exists(): continue
        
        # Case-insensitive dataset folder match
        ds_dirs = [d for d in arch_dir.iterdir() if d.is_dir() and d.name.lower() == clean_ds]
        if not ds_dirs: continue
        
        stats_dir = ds_dirs[0] / f"epochs{epochs}_pretrain{pretrain}" / "Layer_Statistics"
        if not stats_dir.exists(): continue
        
        all_csvs = list(stats_dir.glob("*_layer_stats.csv"))
        csv_candidates = [f for f in all_csvs if "normalized" not in f.name.lower()]
        
        if not csv_candidates: 
            continue
        csv_path = csv_candidates[0]
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()

        # --- FIX: Robust recursive JSON search (Case-Insensitive Match) ---
        potential_jsons = list(Path(".").rglob(f"{arch}*epochs{epochs}_pretrain{pretrain}*_discovered_regions.json"))
        all_jsons = [p for p in potential_jsons if clean_ds in p.name.lower()]
        
        if not all_jsons: 
            continue
            
        try:
            with open(all_jsons[0], 'r') as f:
                model_exps = json.load(f)
        except Exception: continue
        if not model_exps: continue

        baseline_mask = g_metrics['posthoc_or_posttrain'] == 'Baseline'
        if baseline_mask.any():
            base_row = g_metrics[baseline_mask].iloc[0]
            base_p, base_f, base_m = base_row.get('params', np.nan), base_row.get('flops', np.nan), base_row.get('memory', np.nan)
        else: base_p = base_f = base_m = np.nan

        for is_quant_target in [False, True]:
            valid_exps = []
            
            for exp_name, ranges in model_exps.items():
                cleaned_name = re.sub(r'(?i)[_\-\s\(]*quant(ized)?[\)]*', '', exp_name).strip(" -_")
                
                if "Original" in cleaned_name or "Baseline" in cleaned_name:
                    cleaned_name = "Original Model"
                
                m = robust_match(cleaned_name, is_quant_target=is_quant_target, g_df=g_metrics)
                
                if not m.empty:
                    exp_results = m.mean(numeric_only=True)
                    p_red = 100 * (1 - exp_results.get('params', np.nan) / base_p) if pd.notnull(base_p) else np.nan
                    f_red = 100 * (1 - exp_results.get('flops', np.nan) / base_f) if pd.notnull(base_f) else np.nan
                    m_red = 100 * (1 - exp_results.get('memory', np.nan) / base_m) if pd.notnull(base_m) else np.nan
                    
                    if p_red < -0.1 or f_red < -0.1 or m_red < -0.1:
                        continue 
                else:
                    exp_results = {'d_acc': np.nan}
                    p_red, f_red, m_red = np.nan, np.nan, np.nan
                        
                valid_exps.append((exp_name, ranges, exp_results, cleaned_name, p_red, f_red, m_red))
                    
            if not valid_exps: continue
            
            valid_exps = sorted(valid_exps, key=lambda x: (1 if x[1] is None else 0, x[2].get('d_acc', -100)))
            
            num_bars = len(valid_exps)
            file_suffix = "quantized" if is_quant_target else "unquantized"

            fig_height = max(3.5, 0.5 * num_bars + 1)
            y_limits = (-1, num_bars)
            
            fig_cand = plt.figure(figsize=(14, fig_height)) 
            gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.5], wspace=0.05)
            
            ax_heur = fig_cand.add_subplot(gs[0])
            ax_side = fig_cand.add_subplot(gs[1], sharey=ax_heur)
            ax_side.axis('off')
            
            for i, (orig_exp_name, ranges, exp_results, display_name, p_red, f_red, m_red) in enumerate(valid_exps):
                if ranges is None:
                    continue
                    
                if isinstance(ranges, tuple) or (isinstance(ranges, list) and len(ranges) > 0 and isinstance(ranges[0], str)):
                    ranges = [ranges]
                
                d_acc = exp_results.get('d_acc', np.nan)
                color = get_acc_color(d_acc)
                for start_layer, end_layer in ranges:
                    try:
                        s_idx = next(idx for idx, n in enumerate(layers) if start_layer == n) + 1
                        e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer == n) + 1
                        ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=16, color=color, alpha=0.85)
                    except StopIteration: continue
            
            ax_heur.set_xlim(0, len(layers) + 1)
            ax_heur.set_ylim(y_limits)
            ax_heur.set_yticks(range(len(valid_exps)))
            
            ax_heur.set_yticklabels([e[3] if e[1] is not None else f"{e[3]} (Control)" for e in valid_exps], fontsize=11, fontweight='bold')
            ax_heur.set_xlabel("Network Depth (Layer Index)", fontweight='bold', fontsize=11)
            ax_heur.set_title(f"Structural Candidates & Hardware Reductions", loc='left', pad=25, fontsize=14, fontweight='bold')
            sns.despine(ax=ax_heur)

            cols = [0.10, 0.35, 0.65, 0.90]
            headers = ["$\\Delta$ Acc", "Params $\\downarrow$", "FLOPs $\\downarrow$", "Memory $\\downarrow$"]
            
            header_y = len(valid_exps) 
            for x, h in zip(cols, headers):
                ax_side.text(x, header_y, h, ha='center', va='bottom', fontweight='bold', fontsize=11, color='#333333')
            
            for i, (_, ranges, exp_results, _, p_red, f_red, m_red) in enumerate(valid_exps):
                d_acc = exp_results.get('d_acc', np.nan)
                c = "#333333" if ranges is None else get_acc_color(d_acc)
                
                d_str = f"{d_acc:+.1f}%" if pd.notnull(d_acc) else "N/A"
                p_str = f"{p_red:.1f}%" if pd.notnull(p_red) else "N/A"
                f_str = f"{f_red:.1f}%" if pd.notnull(f_red) else "N/A"
                m_str = f"{m_red:.1f}%" if pd.notnull(m_red) else "N/A"
                
                for x, val in zip(cols, [d_str, p_str, f_str, m_str]):
                    fw = 'bold' if x == cols[0] else 'normal'
                    alpha = 0.6 if (d_acc < -6.0 and x != cols[0] and ranges is not None) else 1.0
                    ax_side.text(x, i, val, ha='center', va='center', color=c, fontweight=fw, fontsize=11, alpha=alpha)

            plt.tight_layout()
            cand_save_path = out_dir / f"{arch}_{dataset}_candidates_sidebar_{file_suffix}.png"
            fig_cand.savefig(cand_save_path, bbox_inches='tight')
            plt.close(fig_cand)

    logger.info("[FIG4] Candidate/Sidebar plots generated successfully.")
# ========================= FIG 5 ========================= #
def fig5_hardware_efficiency_profiles(
    df: pd.DataFrame,
    out_dir: Path = Path("./figures/hardware_efficiency")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_summary = []
    worst_summary = []
    all_tradeoff_data = []

    def format_dataset_name(ds: str) -> str:
        mapping = {"tinyimagenet": "TinyImageNet", "cifar10_": "CIFAR-10", "cifar100_": "CIFAR-100", "imagenet": "ImageNet"}
        return mapping.get(ds, ds.capitalize())

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"[FIG5] Processing Hardware Profiles for {arch}/{dataset}")
        
        # --- FIX: Recognize 'control' as the baseline trigger ---
        baseline_mask = (g_metrics['exp_name'].str.lower().str.contains('baseline') | 
                         g_metrics['exp_name'].str.lower().str.contains('control') |
                         g_metrics['exp_name'].str.lower().str.contains('original'))
        
        if not baseline_mask.any(): 
            logger.warning(f"[FIG5] No true 'Baseline Model' found for {arch}/{dataset}. Skipping.")
            continue
        # Ensure we grab the unquantized version of the baseline model as our control
        baseline_candidates = g_metrics[baseline_mask & (g_metrics['is_quantized'] == False)]
        if not baseline_candidates.empty:
            baseline_row = baseline_candidates.iloc[0]
        else:
            baseline_row = g_metrics[baseline_mask].iloc[0]

        base_params = baseline_row.get('params', np.nan)
        base_flops = baseline_row.get('flops', np.nan)
        base_memory = baseline_row.get('memory', np.nan)
        
        if pd.isna(base_params) or pd.isna(base_flops): continue

        # Filter out ANY baseline models (both Original and Baseline) from the candidate list
        candidates = g_metrics[(g_metrics['posthoc_or_posttrain'] != 'Baseline') & 
                               (g_metrics['is_quantized'] == False)].copy()
        if candidates.empty: continue
            
        candidates['Params Reduced (%)'] = 100 * (1 - (candidates['params'] / base_params))
        candidates['FLOPs Reduced (%)'] = 100 * (1 - (candidates['flops'] / base_flops))
        candidates['Memory Reduced (%)'] = 100 * (1 - (candidates['memory'] / base_memory))
        
        # --- STRICT FILTER: Reject inflated hardware profiles ---
        candidates = candidates[(candidates['Params Reduced (%)'] >= 0) & 
                                (candidates['FLOPs Reduced (%)'] >= 0) & 
                                (candidates['Memory Reduced (%)'] >= 0)]
                                
        if candidates.empty: continue
        
        candidates = candidates.sort_values(by='d_acc', ascending=False)

        table_df = candidates[['base_name', 'd_acc', 'Params Reduced (%)', 'FLOPs Reduced (%)', 'Memory Reduced (%)']].copy()
        table_df.columns = ['Candidate Block', 'Delta Acc (%)', 'Params Red. (%)', 'FLOPs Red. (%)', 'Memory Red. (%)']
        
        table_df.to_csv(out_dir / f"{arch}_{dataset}_all_candidates.csv", index=False)
        table_df.to_latex(out_dir / f"{arch}_{dataset}_all_candidates.tex", index=False, float_format="%.2f")

        melted_arch = candidates.melt(
            id_vars=['base_name', 'd_acc'], 
            value_vars=['Params Reduced (%)', 'FLOPs Reduced (%)', 'Memory Reduced (%)'],
            var_name='Metric', value_name='Reduction (%)'
        )
        melted_arch['Metric'] = melted_arch['Metric'].str.replace(' Reduced (%)', '')
        y_labels = [f"{row['base_name']}\n($\\Delta$ {row['d_acc']:+.1f}%)" for _, row in candidates.iterrows()]

        fig, ax = plt.subplots(figsize=(10, max(5, len(candidates) * 0.8)))
        sns.barplot(data=melted_arch, y='base_name', x='Reduction (%)', hue='Metric', 
                    palette=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black', ax=ax)
        
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel(""); ax.set_xlabel("Reduction Relative to Baseline (%)", fontweight='bold')
        ax.set_title(f"Hardware Resource Optimization: {arch}", pad=15, fontweight='bold', fontsize=14)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        ax.legend(title="", loc='lower right'); sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / f"{arch}_{dataset}_hardware_profile.png", bbox_inches='tight')
        logger.info(f"[FIG5] Saved hardware profile plot for {arch}/{dataset} at {out_dir / f'{arch}_{dataset}_hardware_profile.png'}")
        plt.close()

        best_cand = candidates.iloc[0]  
        worst_cand = candidates.iloc[-1] 
        
        for cand, target_list in zip([best_cand, worst_cand], [best_summary, worst_summary]):
            target_list.append({
                "Architecture": arch,
                "Delta_Acc": cand['d_acc'],
                "Params": cand['Params Reduced (%)'],
                "FLOPs": cand['FLOPs Reduced (%)'],
                "Memory": cand['Memory Reduced (%)']
            })

        for _, row in candidates.iterrows():
            all_tradeoff_data.append({
                "Architecture": arch,
                "Delta_Acc": row['d_acc'],
                "FLOPs_Reduction": row['FLOPs Reduced (%)']
            })

    def plot_unified(data_list, filename_suffix, title_prefix):
        if not data_list: return
        df_unified = pd.DataFrame(data_list).sort_values(by="Delta_Acc", ascending=False)
        melted = df_unified.melt(id_vars=['Architecture', 'Delta_Acc'], 
                                 value_vars=['Params', 'FLOPs', 'Memory'],
                                 var_name='Metric', value_name='Reduction')
        x_labels = [f"{r['Architecture']}\n($\\Delta$ {r['Delta_Acc']:+.1f}%)" for _, r in df_unified.iterrows()]

        fig, ax = plt.subplots(figsize=(12, 5.5))
        sns.barplot(data=melted, x='Architecture', y='Reduction', hue='Metric', 
                    palette=['#4C72B0', '#DD8452', '#55A868'], edgecolor='black', ax=ax, order=df_unified['Architecture'])
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_xlabel("Architecture & Accuracy Impact", fontweight='bold', fontsize=12)
        ax.set_ylabel("Reduction Relative to Baseline (%)", fontweight='bold', fontsize=12)
        ax.set_title(f"{title_prefix} Structural Collapse Efficiency by Architecture", pad=15, fontweight='bold', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        ax.legend(title="", loc='upper right'); sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / f"unified_{filename_suffix}.png", bbox_inches='tight')
        logger.info(f"[FIG5] Saved unified {filename_suffix} plot at {out_dir / f'unified_{filename_suffix}.png'}")
        plt.close()

    plot_unified(best_summary, "best_candidates", "Best-Performing")
    plot_unified(worst_summary, "worst_candidates", "Worst-Performing")

    if all_tradeoff_data:
        df_trade = pd.DataFrame(all_tradeoff_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
        ax.axvspan(0, 100, ymin=0.5, ymax=1, color='#e6f4ea', alpha=0.3, zorder=0, label="Ideal (Faster & More Accurate)")
        ax.axvspan(0, 100, ymin=0, ymax=0.5, color='#fce8e6', alpha=0.3, zorder=0, label="Degraded (Faster but Less Accurate)")

        sns.scatterplot(data=df_trade, x='FLOPs_Reduction', y='Delta_Acc', hue='Architecture', 
                        s=150, edgecolor='black', alpha=0.8, ax=ax, zorder=3)
        
        ax.set_xlabel("Computational Reduction (FLOPs Removed %)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Accuracy Impact ($\\Delta$ %)", fontweight='bold', fontsize=12)
        ax.set_title("Global Hardware Efficiency vs. Accuracy Trade-off", pad=15, fontweight='bold', fontsize=14)
        ax.legend(loc='lower left', framealpha=0.9)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / "global_tradeoff_scatter.png", bbox_inches='tight')
        plt.close()
        logger.info(f"[FIG5] Saved global trade-off scatter plot at {out_dir / 'global_tradeoff_scatter.png'}")

    logger.info("[FIG5] All 6 hardware deliverables generated successfully.")
# ========================= FIG 6 ========================= #

def fig6_training_curves(
    results_dir: Path = Path("./"),
    out_dir: Path = Path("./figures/learning_curves")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"[FIG6] Scanning for metrics files in {results_dir.resolve()} for training curves")
    files = list(results_dir.rglob("*merged_metrics.json"))
    
    if not files:
        if (results_dir / "merged_metrics.json").exists(): 
            files = [results_dir / "merged_metrics.json"]
        else: 
            logger.warning("[FIG6] No merged_metrics.json files found for training curves.")
            return

    for p in files:
        dataset = infer_dataset_from_path(p)
        if dataset == "unknown" and "tinyimagenet" in str(p).lower(): dataset = "tinyimagenet"
        if dataset == "unknown" and "cifar100" in str(p).lower(): dataset = "cifar100_"
        if dataset == "unknown" and "cifar10" in str(p).lower(): dataset = "cifar10_"
        
        arch = infer_architecture_from_path(p)
        if arch == "UnknownArch": arch = infer_architecture_from_path(Path(p.name))
        
        try:
            with open(p) as f: 
                raw = json.load(f)
        except Exception as e:
            continue
            
        if not raw: continue
        
        # Identify the Phase 1 Control to act as the branch root
        base_control_key = next((k for k in raw.keys() if "control" in k.lower() and "continuted" not in k.lower() and not infer_isquant(k)), None)
        base_acc = raw[base_control_key].get("accuracies", []) if base_control_key else []
        base_loss = raw[base_control_key].get("losses", []) if base_control_key else []
        base_epochs = len(base_acc)

        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        palette = sns.color_palette("husl", len(raw))
        has_data = False
        
        for idx, (exp_name, data) in enumerate(raw.items()):
            accuracies = data.get("accuracies", [])
            losses = data.get("losses", [])
            
            is_stage1_control = ("control" in exp_name.lower())
            display_name = clean_exp_name(exp_name)
            if infer_isquant(exp_name):
                display_name += " (Quant)"
                
            if accuracies:
                if not is_stage1_control and base_epochs > 0:
                    # Dynamically offset X and prepend the final Y value of the base control
                    x_acc = [base_epochs] + [base_epochs + x for x in range(1, len(accuracies) + 1)]
                    y_acc = [base_acc[-1]] + accuracies
                    # Draw a faint trailing shadow line of the original control for reference
                    axes[0].plot(range(1, base_epochs + 1), base_acc, color=palette[idx], linewidth=1.5, alpha=0.2, linestyle='--')
                else:
                    x_acc = list(range(1, len(accuracies) + 1))
                    y_acc = accuracies
                    
                axes[0].plot(x_acc, y_acc, label=display_name, color=palette[idx], linewidth=2.5, alpha=0.85)
                has_data = True
                
            if losses:
                if not is_stage1_control and base_epochs > 0 and base_loss:
                    x_loss = [base_epochs] + [base_epochs + x for x in range(1, len(losses) + 1)]
                    y_loss = [base_loss[-1]] + losses
                    axes[1].plot(range(1, base_epochs + 1), base_loss, color=palette[idx], linewidth=1.5, alpha=0.2, linestyle='--')
                else:
                    x_loss = list(range(1, len(losses) + 1))
                    y_loss = losses
                    
                axes[1].plot(x_loss, y_loss, label=display_name, color=palette[idx], linewidth=2.5, alpha=0.85)
                has_data = True
                
        if not has_data:
            plt.close()
            continue

        axes[0].set_title(f"Validation Accuracy Over Epochs\n{arch} | {format_dataset_name(dataset)}", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_xlabel("Epochs", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Accuracy (%)", fontsize=13, fontweight='bold')
        axes[0].legend(loc="lower right", framealpha=0.9, fontsize=11)
        
        axes[1].set_title(f"Loss Over Epochs\n{arch} | {format_dataset_name(dataset)}", fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel("Epochs", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Loss", fontsize=13, fontweight='bold')
        axes[1].legend(loc="upper right", framealpha=0.9, fontsize=11)

        sns.despine(bottom=False, left=False)
        plt.tight_layout()

        clean_ds = dataset.strip("_")
        save_path = out_dir / f"{arch}_{clean_ds}_training_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"[FIG6] Saved sequential training curves for {arch}/{dataset} at {save_path}")
        
def export_master_summary_json(df: pd.DataFrame, out_path: Path = Path("./master_results_summary.json")):
    """
    Consolidates the normalized DataFrame into a single, highly readable 
    JSON file structured by Architecture -> Dataset -> Experiment for quick lookup.
    """
    logger.info(f"[SUMMARY] Generating master lookup JSON...")
    
    summary_dict = {}
    
    # JSON does not handle NaNs cleanly; replace them with None (null)
    df_clean = df.replace({np.nan: None})
    
    for (arch, dataset), group in df_clean.groupby(["architecture", "dataset"]):
        if arch not in summary_dict:
            summary_dict[arch] = {}
            
        clean_ds = format_dataset_name(dataset)
        if clean_ds not in summary_dict[arch]:
            summary_dict[arch][clean_ds] = {}
            
        for _, row in group.iterrows():
            exp_key = row["exp_name"]
            if exp_key == "Control":
                continue
            # Extract and organize the metrics you need for the paper
            summary_dict[arch][clean_ds][exp_key] = {
                "base_name": row.get("base_name"),
                "posthoc_or_posttrain": row.get("posthoc_or_posttrain"),
                "is_quantized": row.get("is_quantized"),
                "accuracy": row.get("accuracy"),
                "d_acc": row.get("d_acc"),
                "d_asymptotic_loss": row.get("d_asymptotic_loss"), # NEW
                "loss_correlation": row.get("loss_correlation"),   # NEW
                "params": row.get("params"),
                "d_params_percent": row.get("d_params"), # Hardware reduction %
                "flops": row.get("flops"),
                "memory_mb": row.get("memory")
            }
            
    try:
        with open(out_path, "w") as f:
            json.dump(summary_dict, f, indent=4)
        logger.info(f"[SUMMARY] Master JSON successfully exported to {out_path.resolve()}")
    except Exception as e:
        logger.error(f"[SUMMARY] Failed to write master JSON: {e}")

# ========================= FIG 7 ========================= #

def fig7_convergence_metrics(
    df: pd.DataFrame,
    out_dir: Path = Path("./figures/convergence")
):
    """
    Generates a dual-panel bar chart proving that candidate models
    reach the exact same thermodynamic loss minimum as the baseline control,
    preventing the 'just train it longer' counter-argument.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure the dataframe actually has the new metrics before plotting
    if 'loss_correlation' not in df.columns or 'd_asymptotic_loss' not in df.columns:
        logger.warning("[FIG7] Convergence metrics not found in dataframe. Run normalize() with curve extraction.")
        return

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        # Filter to just the unquantized candidate runs (we compare FP32 training curves)
        candidates = g_metrics[(g_metrics['posthoc_or_posttrain'] != 'Baseline') & 
                               (g_metrics['is_quantized'] == False)].copy()
        
        # Drop any rows where curves were missing or too short to calculate
        candidates = candidates.dropna(subset=['loss_correlation', 'd_asymptotic_loss'])
        
        if candidates.empty:
            continue
            
        # Sort by delta accuracy to keep the x-axis consistent with Figure 5
        candidates = candidates.sort_values(by='d_acc', ascending=False)
        
        # Set up a stacked figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        x_labels = [f"{row['base_name']}\n($\\Delta$ {row['d_acc']:+.1f}%)" for _, row in candidates.iterrows()]
        x_pos = np.arange(len(x_labels))
        
        # --- Top Panel: Trajectory Correlation (r) ---
        sns.barplot(data=candidates, x='base_name', y='loss_correlation', ax=ax1, 
                    color='#4C72B0', edgecolor='black', alpha=0.9)
        
        ax1.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Baseline Control (r = 1.0)')
        
        # Auto-scale Y to show variation but emphasize proximity to 1.0
        min_corr = min(0.9, candidates['loss_correlation'].min() - 0.05)
        ax1.set_ylim(min_corr, 1.05)
        
        ax1.set_ylabel(r"Pearson Correlation ($r$)", fontweight='bold', fontsize=12)
        ax1.set_title(f"Convergence Trajectory & Asymptotic Stability\n{arch} | {format_dataset_name(dataset)}", 
                      fontweight='bold', fontsize=14, pad=15)
        ax1.legend(loc='lower left', framealpha=0.9)
        ax1.xaxis.grid(False); ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax1.set_axisbelow(True)
        
        # --- Bottom Panel: Asymptotic Loss Delta (ΔL) ---
        sns.barplot(data=candidates, x='base_name', y='d_asymptotic_loss', ax=ax2, 
                    color='#DD8452', edgecolor='black', alpha=0.9)
        
        ax2.axhline(0.0, color='black', linestyle='--', linewidth=2, label='Baseline Control Ceiling (0.0)')
        
        # Make the plot symmetrically bounded around 0 for visual clarity
        max_abs_val = candidates['d_asymptotic_loss'].abs().max()
        padding = max(0.02, max_abs_val * 0.2)
        ax2.set_ylim(-max_abs_val - padding, max_abs_val + padding)
        
        ax2.set_ylabel(r"$\Delta$ Final Loss ($\Delta \mathcal{L}_{final}$)", fontweight='bold', fontsize=12)
        ax2.set_xlabel("Candidate Architectural Sequence", fontweight='bold', fontsize=12)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontweight='bold', fontsize=11)
        ax2.legend(loc='upper right', framealpha=0.9)
        ax2.xaxis.grid(False); ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax2.set_axisbelow(True)
        
        sns.despine(ax=ax1)
        sns.despine(ax=ax2)
        plt.tight_layout()
        
        # Save output
        clean_ds = dataset.strip("_")
        save_path = out_dir / f"{arch}_{clean_ds}_convergence_metrics.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"[FIG7] Saved convergence metrics plot for {arch}/{dataset} at {save_path}")


# --- Updated __main__ block ---
if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        
        # Consistent dynamic output directories
        fig1(df, out_dir=FIG_DIR / "individual_plots")
        fig2_methodology_bav_regions(epochs, pretrain, out_dir=FIG_DIR / "methodology")
        
        # --- UPDATE: Pass epochs and pretrain here ---
        fig3_v2t_heuristic_validation(df, epochs, pretrain, out_dir=FIG_DIR / "heuristic_validation")
        fig4_comprehensive_search_space_map(df, epochs, pretrain, out_dir=FIG_DIR / "search_space")
        
        fig5_hardware_efficiency_profiles(df, out_dir=FIG_DIR / "hardware_efficiency")
        fig6_training_curves(out_dir=FIG_DIR / "learning_curves")
        fig7_convergence_metrics(df, out_dir=FIG_DIR / "convergence")
        
        export_master_summary_json(df, out_path=Path(f"./master_results_summary_ep{epochs}_pre{pretrain}.json"))
        logger.info("Script completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)