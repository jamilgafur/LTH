from __future__ import annotations

import json
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import argrelextrema

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
DIAGNOSTICS_DIR = Path("./diagnostics")

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
    if "original" in n or "baseline" in n:
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
    
    if "Original" in n or "Baseline" in n: return "Original Model"
    return n.strip()

def find_baseline(df: pd.DataFrame):
    mask = (df["exp_name"].str.lower().str.contains("original") | df["exp_name"].str.lower().str.contains("baseline"))
    b_df = df[mask & (df["is_quantized"] == False)]
    if b_df.empty: b_df = df[mask]
    if b_df.empty: return None
    return b_df.mean(numeric_only=True)

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
            })
            
    logger.info(f"Successfully parsed {len(rows)} experiment rows.")
    return pd.DataFrame(rows)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, arch), g in df.groupby(["dataset", "architecture"]):
        baseline = find_baseline(g)
        if baseline is None:
            for _, r in g.iterrows(): out.append(r)
            continue
            
        b_acc = baseline["accuracy"]
        for _, r in g.iterrows():
            row = r.copy()
            if pd.notnull(baseline.get("params")) and baseline["params"] > 0:
                row["d_acc"] = r["accuracy"] - b_acc 
                row["acc_drop"] = b_acc - r["accuracy"] 
                row["baseline_acc"] = b_acc 
                row["d_params"] = 100 * (1 - r["params"] / baseline["params"])
            out.append(row)
    return pd.DataFrame(out)

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
                plt.close()

# ========================= FIG 2 ========================= #

def fig2_methodology_bav_regions(
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/methodology")
):
    out_dir.mkdir(parents=True, exist_ok=True)
    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]

    stat_files = list(stats_dir.glob("*_layer_stats.csv"))
    if not stat_files:
        logger.warning("[FIG2] No layer stats CSVs found in the specified directory.")
        return

    for csv_path in stat_files:
        parts = csv_path.stem.replace("_layer_stats", "").split("_")
        arch = parts[0]
        dataset = "_".join(parts[1:])
        
        try:
            layer_df = pd.read_csv(csv_path)
            layers = layer_df['Layer'].tolist()
            variances = np.maximum(layer_df['Variance'].values, 1e-6)
        except Exception as e:
            logger.error(f"[FIG2] Failed to read {csv_path}: {e}")
            continue

        # Calculate macro-trend
        smoothed_var = pd.Series(variances).rolling(window=5, center=True, min_periods=1).median().values
        mu_net = np.mean(variances)
        median_net = np.median(variances)

        # Initialize Paper-Formatted Plot
        fig, ax = plt.subplots(figsize=(12, 4.5))

        # Replaced continuous line chart with discrete bar chart for layer variances 
        # and step plot for the macro trend to prevent false interpolation.
        x_vals = range(len(layers))
        ax.bar(x_vals, variances, color='#b0b0b0', alpha=0.6, label='Raw Activation Variance')
        ax.plot(x_vals, smoothed_var, color='#111111', linewidth=2.0, drawstyle='steps-mid', label=r'Macro Trend ($\tilde{V}_{trend}$)')

        ax.set_yscale('log')
        ax.set_ylabel("Activation Variance ($\sigma^2$)", fontweight='bold', fontsize=11)
        ax.set_xlabel("Network Depth (Layer Index)", fontweight='bold', fontsize=11)

        # Set Architecture-Specific Thresholds
        if arch in multi_path_archs:
            threshold = median_net
            target_label = r"Potential Target Zone ($\tilde{V}_{trend} < \tilde{V}_{net}$)"
            title_text = f"BAV Methodology (Multi-Path): {arch} on {dataset.capitalize()}"
        else:
            threshold = mu_net
            target_label = r"Potential Target Zone ($\tilde{V}_{trend} < \mu_{net}$)"
            title_text = f"BAV Methodology (Single-Path): {arch} on {dataset.capitalize()}"

        ax.set_title(title_text, pad=25, fontweight='bold', fontsize=14, loc='left')
        ax.axhline(y=threshold, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=1.5, label='Network Variance Threshold')

        # Identify Zones
        is_target = smoothed_var <= threshold
        veto_idx = int(len(layers) * 0.25)
        is_target[:veto_idx] = False

        # 1. Plot Foundational Veto
        ax.axvspan(0, max(0, veto_idx - 0.5), color='#e0e0e0', alpha=0.6, hatch='////', edgecolor='#999999', label="Foundational Veto (Depth < 25%)")

        # Compile contiguous Target/Danger zones
        zones = []
        if veto_idx < len(is_target):
            start_idx = veto_idx
            current_val = is_target[veto_idx]
            for i in range(veto_idx + 1, len(is_target)):
                if is_target[i] != current_val:
                    zones.append((start_idx, i, current_val))
                    start_idx = i
                    current_val = is_target[i]
            zones.append((start_idx, len(is_target)-1, current_val))

        added_target = False
        added_danger = False

        # 2. Plot Target and Danger Zones
        for start, end, is_good in zones:
            span_start = max(veto_idx, start - 0.5) if start > 0 else start
            span_end = min(len(layers) - 1, end + 0.5)

            if is_good:
                lbl = target_label if not added_target else ""
                ax.axvspan(span_start, span_end, color='#2ca02c', alpha=0.2, label=lbl)
                added_target = True
            else:
                lbl = "Danger Zone (Avoid)" if not added_danger else ""
                ax.axvspan(span_start, span_end, color='#d62728', alpha=0.1, label=lbl)
                added_danger = True

        # 3. Plot Boundary Anchors
        valleys = argrelextrema(smoothed_var, np.less, order=2)[0]
        added_valley = False
        for v in valleys:
            if v >= veto_idx:
                lbl = "Boundary Anchors (Local Minima)" if not added_valley else ""
                ax.axvline(x=v, color='black', linestyle=':', alpha=0.4, linewidth=1.5, label=lbl)
                added_valley = True

        # Legend formatting specifically for papers (horizontal, top outside)
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.22), ncol=3, fontsize=9, frameon=False)
        sns.despine(ax=ax)

        plt.tight_layout()
        save_path = out_dir / f"{arch}_{dataset}_bav_methodology_regions.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
    logger.info("[FIG2] Methodology region plots generated successfully.")

# ========================= FIG 3 ========================= #

def fig3_v2t_heuristic_validation(
    df: pd.DataFrame,
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
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

    logger.info(f"[FIG3] Starting with {len(df)} rows")

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"[FIG3] Processing: dataset={dataset}, arch={arch}, rows={len(g_metrics)}")

        clean_metrics = g_metrics[g_metrics['posthoc_or_posttrain'].isin(['Collapsed', 'Retrained'])]

        if clean_metrics.empty:
            logger.warning(f"[FIG3] Skipping {arch}/{dataset}: no valid posthoc/posttrain rows")
            continue

        csv_path = stats_dir / f"{arch}_{dataset}_experiment_block_stats.csv"

        if not csv_path.exists():
            logger.warning(f"[FIG3] Missing stats file: {csv_path}")
            continue

        try:
            df_h = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"[FIG3] Failed to read {csv_path}: {e}")
            continue

        if "Experiment" not in df_h.columns:
            logger.error(f"[FIG3] 'Experiment' column missing in {csv_path}")
            continue

        merged = pd.merge(
            df_h,
            clean_metrics,
            left_on="Experiment",
            right_on="base_name",
            how="inner"
        )

        logger.info(f"[FIG3] Merge result: {len(merged)} rows")

        if merged.empty:
            logger.warning(f"[FIG3] Empty merge for {arch}/{dataset}")
            continue

        merged["Topology"] = "Multi-Path" if arch in multi_path_archs else "Single-Path"

        # Save per-merge CSV (debug goldmine)
        debug_csv = out_dir / f"debug_merge_{arch}_{dataset}.csv"
        merged.to_csv(debug_csv, index=False)

        all_merged_data.append(merged)

    if not all_merged_data:
        logger.error("[FIG3] No data after merging. Exiting.")
        return

    full_df = pd.concat(all_merged_data, ignore_index=True)

    # Save FULL dataset CSV
    full_csv_path = out_dir / "fig3_full_merged_data.csv"
    full_df.to_csv(full_csv_path, index=False)
    logger.info(f"[FIG3] Saved full merged CSV → {full_csv_path}")

    # Map legend labels
    full_df['Quantization State'] = full_df['is_quantized'].map({
        False: 'Unquantized',
        True: 'Quantized'
    })

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plt.subplots_adjust(wspace=0.05)

    for ax in axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
        ax.axhspan(0, 100, color='#2ca02c', alpha=0.05, zorder=0)
        ax.axhspan(-2.0, 0, color='#ff7f0e', alpha=0.05, zorder=0)
        ax.axhspan(-100, -2.0, color='#d62728', alpha=0.05, zorder=0)

    y_min = full_df['d_acc'].min() - 2
    y_max = max(full_df['d_acc'].max() + 2, 5)

    for ax in axes:
        ax.set_ylim(y_min, y_max)

    # --- Single Path ---
    sp_df = full_df[full_df['Topology'] == 'Single-Path']
    if sp_df.empty:
        logger.warning("[FIG3] No Single-Path data")
    else:
        sns.scatterplot(
            data=sp_df,
            x="Median Variance",
            y="d_acc",
            style="Quantization State",
            markers={"Unquantized": "o", "Quantized": "X"},
            s=250,
            alpha=0.9,
            edgecolor="black",
            color="#2ca02c",
            ax=axes[0],
            zorder=3
        )
        # BAV Target: Single-Path targets Bounded Growth
        q1 = sp_df["Median Variance"].quantile(0.3)
        axes[0].axvspan(1e-4, q1, color='#2ca02c', alpha=0.15, zorder=0, label=f"BAV Target: Bounded Growth (<{q1:.1f})")
        axes[0].set_xscale('symlog', linthresh=1e-2)
        axes[0].set_title("Single-Path: Target Bounded Growth", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylabel(r"$\Delta$ Accuracy (%) $\rightarrow$ Higher is Better", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Median Activation Variance (SymLog Scale)", fontsize=12)
        axes[0].legend(loc="lower left", framealpha=0.9)

    # --- Multi Path ---
    mp_df = full_df[full_df['Topology'] == 'Multi-Path']
    if mp_df.empty:
        logger.warning("[FIG3] No Multi-Path data")
    else:
        sns.scatterplot(
            data=mp_df,
            x="Median Variance",
            y="d_acc",
            style="Quantization State",
            markers={"Unquantized": "o", "Quantized": "X"},
            s=250,
            alpha=0.9,
            edgecolor="black",
            color="#d62728",
            ax=axes[1],
            zorder=3
        )
        # BAV Target: Multi-Path targets Terminal Stabilization / Valleys
        q1_mp = mp_df["Median Variance"].quantile(0.4)
        axes[1].axvspan(1e-4, q1_mp, color='#2ca02c', alpha=0.15, zorder=0, label=f"BAV Target: Variance Valleys (<{q1_mp:.1f})")
        axes[1].set_xscale('symlog', linthresh=1e-2)
        axes[1].set_title("Multi-Path: Target Terminal Stabilization", fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel("Median Activation Variance (SymLog Scale)", fontsize=12)
        axes[1].legend(loc="lower right", framealpha=0.9)

    sns.despine()
    plt.savefig(out_dir / "V2T_heuristic_validation_map.png")
    plt.close()

    logger.info("[FIG3] Completed successfully")

# ========================= FIG 4 ========================= #
def fig4_comprehensive_search_space_map(
    df: pd.DataFrame,
    stats_dir: Path = Path("./runs/plots/Layer_Statistics"),
    out_dir: Path = Path("./figures/search_space")
):
    """
    Generates two separate files for the paper:
    1. The Variance Trend (The theory / macro-trend line) 
       - Updated to use piecewise function h(sigma_i, sigma_bar) and discrete bar mapping.
    2. The Candidate Bars + Hardware Sidebar integrated into one figure.
    """
    try:
        from transfer import EXPERIMENTS
    except ImportError:
        logger.error("[FIG4] Could not import EXPERIMENTS")
        return

    from scipy.signal import argrelextrema
    import matplotlib.gridspec as gridspec
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_acc_color(d_acc):
        if pd.isna(d_acc): return "#999999"
        if d_acc >= -2.0: return "#2ca02c"  # Green
        if d_acc >= -6.0: return "#ff7f0e"  # Orange
        return "#d62728"                    # Red

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
        except Exception as e:
            return pd.DataFrame()

    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]

    def format_dataset_name(ds: str) -> str:
        mapping = {"tinyimagenet": "TinyImageNet", "cifar10_": "CIFAR-10", "cifar100_": "CIFAR-100", "imagenet": "ImageNet"}
        return mapping.get(ds, ds.capitalize())

    # Implementation of piecewise function from whiteboard: h(sigma_i, sigma_bar)
    def piecewise_h(sigma_i, sigma_bar):
        diff = sigma_i - sigma_bar
        if diff < 0:
            return max(diff / sigma_bar, -1.0)
        else:
            return min(diff / sigma_bar, 1.0)
            
    vec_h = np.vectorize(piecewise_h)

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"[FIG4] Processing Trend and Candidate+Sidebar Maps for {arch}/{dataset}")

        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): continue
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = np.maximum(layer_df['Variance'].values, 1e-6)
        smoothed_var = pd.Series(variances).rolling(window=5, center=True, min_periods=1).median().values
        mu_net = np.mean(variances)
        median_net = np.median(variances)

        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: continue

        # Extract Baseline Metrics
        baseline_mask = g_metrics['posthoc_or_posttrain'] == 'Baseline'
        if baseline_mask.any():
            base_row = g_metrics[baseline_mask].iloc[0]
            base_p = base_row.get('params', np.nan)
            base_f = base_row.get('flops', np.nan)
            base_m = base_row.get('memory', np.nan)
        else:
            base_p = base_f = base_m = np.nan

        for is_quant_target in [False, True]:
            target_label = "Quantized" if is_quant_target else "Unquantized"
            valid_exps = []
            
            for exp_name, ranges in model_exps.items():
                if ranges is None: continue
                cleaned_name = re.sub(r'(?i)[_\-\s\(]*quant(ized)?[\)]*', '', exp_name).strip(" -_")
                m = robust_match(cleaned_name, is_quant_target=is_quant_target, g_df=g_metrics)
                
                if not m.empty:
                    exp_results = m.mean(numeric_only=True)
                    p_red = 100 * (1 - exp_results.get('params', np.nan) / base_p) if pd.notnull(base_p) else np.nan
                    f_red = 100 * (1 - exp_results.get('flops', np.nan) / base_f) if pd.notnull(base_f) else np.nan
                    m_red = 100 * (1 - exp_results.get('memory', np.nan) / base_m) if pd.notnull(base_m) else np.nan
                    valid_exps.append((exp_name, ranges, exp_results, cleaned_name, p_red, f_red, m_red))
                    
            if not valid_exps: continue
            valid_exps = sorted(valid_exps, key=lambda x: x[2].get('d_acc', -100))
            
            num_bars = len(valid_exps)
            x_vals = np.arange(len(layers))
            file_suffix = "quantized" if is_quant_target else "unquantized"

            # ==========================================
            # FILE 1: The Variance Trend Plot (Updated with piecewise h & Bar charts)
            # ==========================================
            fig_var, ax_var = plt.subplots(figsize=(10, 3.5))

            threshold = median_net if arch in multi_path_archs else mu_net
            
            # Apply h(sigma_i, sigma_bar) transformation
            h_variances = vec_h(variances, threshold)
            h_smoothed = vec_h(smoothed_var, threshold)

            # Replaced line chart with bar chart to remove false interpolation
            ax_var.bar(x_vals, h_variances, color='#999999', alpha=0.6, label='Normalized Raw Variance')
            # Changed macro trend to steps-mid
            ax_var.plot(x_vals, h_smoothed, color='#333333', linewidth=2.0, drawstyle='steps-mid', label='Macro Trend')
            
            # Since values are now clipped [-1, 1], we don't need log scale anymore.
            ax_var.set_ylim(-1.1, 1.1)
            ax_var.set_ylabel(r"Relative Variance $h(\sigma_i, \bar{\sigma})$", fontweight='bold')
            ax_var.set_xlabel("Network Depth (Layer Index)", fontweight='bold')
            ax_var.set_title(f"BAV Macro-Trend: {arch} ({target_label})", loc='left', pad=15, fontsize=14, fontweight='bold')
            ax_var.set_xlim(0, len(layers))
            
            # Since data is normalized relative to threshold, the new threshold line is at 0
            ax_var.axhline(y=0, color='blue', linestyle='--', alpha=0.5, label='Network Threshold (0)')
            
            is_target = h_smoothed <= 0
            veto_idx = int(len(layers) * 0.25)
            is_target[:veto_idx] = False 
            ax_var.axvspan(0, max(0, veto_idx - 0.5), color='gray', alpha=0.15, hatch='//', edgecolor='gray', label="Veto")
            
            ax_var.legend(loc='upper right', ncol=3, fontsize=9, framealpha=0.9)
            sns.despine(ax=ax_var)
            plt.tight_layout()
            
            var_save_path = out_dir / f"{arch}_{dataset}_variance_trend_{file_suffix}.png"
            fig_var.savefig(var_save_path, bbox_inches='tight')
            plt.close(fig_var)


            # ==========================================
            # FILE 2: Candidate Bars + Hardware Sidebar
            # ==========================================
            fig_height = max(3.5, 0.5 * num_bars + 1)
            y_limits = (-1, num_bars)
            
            fig_cand = plt.figure(figsize=(14, fig_height)) # Wide enough to hold both
            gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.5], wspace=0.05)
            
            ax_heur = fig_cand.add_subplot(gs[0])
            ax_side = fig_cand.add_subplot(gs[1], sharey=ax_heur)
            ax_side.axis('off')
            
            # --- Left Side: Candidate Bars ---
            for i, (orig_exp_name, ranges, exp_results, display_name, p_red, f_red, m_red) in enumerate(valid_exps):
                ranges = ranges if isinstance(ranges, list) else [ranges]
                d_acc = exp_results.get('d_acc', np.nan)
                color = get_acc_color(d_acc)
                for start_layer, end_layer in ranges:
                    try:
                        s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                        e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                        ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=16, color=color, alpha=0.85)
                    except StopIteration: continue
            
            ax_heur.set_xlim(0, len(layers))
            ax_heur.set_ylim(y_limits)
            ax_heur.set_yticks(range(len(valid_exps)))
            ax_heur.set_yticklabels([e[3] for e in valid_exps], fontsize=11, fontweight='bold')
            ax_heur.set_xlabel("Network Depth (Layer Index)", fontweight='bold', fontsize=11)
            ax_heur.set_title(f"Structural Candidates & Hardware Reductions", loc='left', pad=25, fontsize=14, fontweight='bold')
            sns.despine(ax=ax_heur)

            # --- Right Side: Hardware Sidebar Table ---
            cols = [0.10, 0.35, 0.65, 0.90]
            headers = ["$\\Delta$ Acc", "Params $\\downarrow$", "FLOPs $\\downarrow$", "Memory $\\downarrow$"]
            
            header_y = len(valid_exps) # Top row header position
            for x, h in zip(cols, headers):
                ax_side.text(x, header_y, h, ha='center', va='bottom', fontweight='bold', fontsize=11, color='#333333')
            
            for i, (_, _, exp_results, _, p_red, f_red, m_red) in enumerate(valid_exps):
                d_acc = exp_results.get('d_acc', np.nan)
                c = get_acc_color(d_acc)
                
                d_str = f"{d_acc:+.1f}%" if pd.notnull(d_acc) else "N/A"
                p_str = f"{p_red:.1f}%" if pd.notnull(p_red) else "N/A"
                f_str = f"{f_red:.1f}%" if pd.notnull(f_red) else "N/A"
                m_str = f"{m_red:.1f}%" if pd.notnull(m_red) else "N/A"
                
                for x, val in zip(cols, [d_str, p_str, f_str, m_str]):
                    fw = 'bold' if x == cols[0] else 'normal'
                    alpha = 0.6 if (d_acc < -6.0 and x != cols[0]) else 1.0
                    ax_side.text(x, i, val, ha='center', va='center', color=c, fontweight=fw, fontsize=11, alpha=alpha)

            plt.tight_layout()
            cand_save_path = out_dir / f"{arch}_{dataset}_candidates_sidebar_{file_suffix}.png"
            fig_cand.savefig(cand_save_path, bbox_inches='tight')
            plt.close(fig_cand)

    logger.info("[FIG4] Variance Trends and Candidate/Sidebar plots generated successfully.")

def fig5_hardware_efficiency_profiles(
    df: pd.DataFrame,
    out_dir: Path = Path("./figures/hardware_efficiency")
):
    """
    Generates comprehensive hardware efficiency reports:
    1. Per-model CSVs of all candidates.
    2. Per-model LaTeX tables.
    3. Per-model grouped bar charts.
    4. Unified Grouped Bar Chart of the BEST candidate per architecture.
    5. Unified Grouped Bar Chart of the WORST candidate per architecture.
    6. A global Accuracy vs. FLOPs Trade-off Scatter Plot (Pareto Frontier).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    best_summary = []
    worst_summary = []
    all_tradeoff_data = []

    def format_dataset_name(ds: str) -> str:
        mapping = {"tinyimagenet": "TinyImageNet", "cifar10_": "CIFAR-10", "cifar100_": "CIFAR-100", "imagenet": "ImageNet"}
        return mapping.get(ds, ds.capitalize())

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        logger.info(f"[FIG5] Processing Hardware Profiles for {arch}/{dataset}")
        
        # 1. Identify Baseline
        baseline_mask = g_metrics['posthoc_or_posttrain'] == 'Baseline'
        if not baseline_mask.any(): continue
            
        baseline_row = g_metrics[baseline_mask].iloc[0]
        base_params = baseline_row.get('params', np.nan)
        base_flops = baseline_row.get('flops', np.nan)
        base_memory = baseline_row.get('memory', np.nan)
        
        if pd.isna(base_params) or pd.isna(base_flops): continue

        # 2. Filter Candidates (Unquantized)
        candidates = g_metrics[(g_metrics['posthoc_or_posttrain'] != 'Baseline') & 
                               (g_metrics['is_quantized'] == False)].copy()
        if candidates.empty: continue
            
        # Calculate Reductions
        candidates['Params Reduced (%)'] = 100 * (1 - (candidates['params'] / base_params))
        candidates['FLOPs Reduced (%)'] = 100 * (1 - (candidates['flops'] / base_flops))
        candidates['Memory Reduced (%)'] = 100 * (1 - (candidates['memory'] / base_memory))
        candidates = candidates.sort_values(by='d_acc', ascending=False) # Best to worst

        # --- Deliverable 1 & 5: Per-Model CSV and LaTeX Tables ---
        table_df = candidates[['base_name', 'd_acc', 'Params Reduced (%)', 'FLOPs Reduced (%)', 'Memory Reduced (%)']].copy()
        table_df.columns = ['Candidate Block', 'Delta Acc (%)', 'Params Red. (%)', 'FLOPs Red. (%)', 'Memory Red. (%)']
        
        table_df.to_csv(out_dir / f"{arch}_{dataset}_all_candidates.csv", index=False)
        table_df.to_latex(out_dir / f"{arch}_{dataset}_all_candidates.tex", index=False, float_format="%.2f")

        # --- Deliverable 2: Per-Model Grouped Bar Chart ---
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
        
        ax.set_yticklabels(y_labels, fontsize=10, fontweight='bold')
        ax.set_ylabel(""); ax.set_xlabel("Reduction Relative to Baseline (%)", fontweight='bold')
        ax.set_title(f"Hardware Resource Optimization: {arch}", pad=15, fontweight='bold', fontsize=14)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        ax.legend(title="", loc='lower right'); sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / f"{arch}_{dataset}_hardware_profile.png", bbox_inches='tight')
        plt.close()

        # --- Prep for Unified Plots ---
        best_cand = candidates.iloc[0]  # Highest d_acc
        worst_cand = candidates.iloc[-1] # Lowest d_acc
        
        for cand, target_list in zip([best_cand, worst_cand], [best_summary, worst_summary]):
            target_list.append({
                "Architecture": arch,
                "Delta_Acc": cand['d_acc'],
                "Params": cand['Params Reduced (%)'],
                "FLOPs": cand['FLOPs Reduced (%)'],
                "Memory": cand['Memory Reduced (%)']
            })

        # Add to Trade-off data
        for _, row in candidates.iterrows():
            all_tradeoff_data.append({
                "Architecture": arch,
                "Delta_Acc": row['d_acc'],
                "FLOPs_Reduction": row['FLOPs Reduced (%)']
            })

    # --- Deliverables 3 & 4: Unified Best and Worst Plots ---
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
        
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_xlabel("Architecture & Accuracy Impact", fontweight='bold', fontsize=12)
        ax.set_ylabel("Reduction Relative to Baseline (%)", fontweight='bold', fontsize=12)
        ax.set_title(f"{title_prefix} Structural Collapse Efficiency by Architecture", pad=15, fontweight='bold', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        ax.legend(title="", loc='upper right'); sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / f"unified_{filename_suffix}.png", bbox_inches='tight')
        plt.close()

    plot_unified(best_summary, "BEST_hardware_efficiency", "Optimal (Best Case)")
    plot_unified(worst_summary, "WORST_hardware_efficiency", "Catastrophic (Worst Case)")

    # --- Deliverable 6: The Trade-off Scatter Plot (Pareto Frontier) ---
    if all_tradeoff_data:
        df_trade = pd.DataFrame(all_tradeoff_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Quadrant coloring
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, zorder=1)
        ax.axvspan(0, 100, ymin=0.5, ymax=1, color='#e6f4ea', alpha=0.3, zorder=0, label="Ideal (Faster & More Accurate)")
        ax.axvspan(0, 100, ymin=0, ymax=0.5, color='#fce8e6', alpha=0.3, zorder=0, label="Degraded (Faster but Less Accurate)")

        sns.scatterplot(data=df_trade, x='FLOPs_Reduction', y='Delta_Acc', hue='Architecture', 
                        s=150, edgecolor='black', alpha=0.8, ax=ax, zorder=3)
        
        ax.set_xlabel("Computational Reduction (FLOPs Removed %)", fontweight='bold', fontsize=12)
        ax.set_ylabel("Accuracy Impact ($\Delta$ %)", fontweight='bold', fontsize=12)
        ax.set_title("Global Hardware Efficiency vs. Accuracy Trade-off", pad=15, fontweight='bold', fontsize=14)
        ax.legend(loc='lower left', framealpha=0.9)
        sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / "global_tradeoff_scatter.png", bbox_inches='tight')
        plt.close()

    logger.info("[FIG5] All 6 hardware deliverables generated successfully.")

if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        fig1(df)
        fig2_methodology_bav_regions()
        fig3_v2t_heuristic_validation(df)
        fig4_comprehensive_search_space_map(df)
        fig5_hardware_efficiency_profiles(df)
        logger.info("Script completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)