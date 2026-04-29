# Journal_plot.py
from __future__ import annotations

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

    # Ignore normalized files
    stat_files = [f for f in stats_dir.glob("*_layer_stats.csv") if "normalized" not in f.name]
    
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

        # Exact mathematical parity with transfer.py
        h_vals = []
        sigma_bars = []
        
        for i, sigma_i in enumerate(variances):
            next_vars = variances[i+1 : i+6]
            sigma_bar = np.mean(next_vars) if len(next_vars) > 0 else np.mean(variances)
            sigma_bar = max(sigma_bar, 1e-12)
            sigma_bars.append(sigma_bar)
            
            diff = sigma_i - sigma_bar
            h = max(diff / sigma_bar, -1.0) if diff < 0 else min(diff / sigma_bar, 1.0)
            h_vals.append(h)
            
        h_vals = np.array(h_vals)
        sigma_bars = np.array(sigma_bars)

        # Two-Panel Layout
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1.5]})
        x_vals = range(len(layers))
        
        # --- TOP PANEL ---
        ax1.plot(x_vals, variances, color='#4A4A4A', marker='o', markersize=4, linestyle='-', linewidth=1.5, label=r'Layer Variance ($\sigma_i$)')
        ax1.plot(x_vals, sigma_bars, color='#ff7f0e', linestyle='--', linewidth=2.5, label=r'Local Context Mean ($\bar{\sigma}$)')
        ax1.set_ylabel("Raw Variance (Log)", fontweight='bold', fontsize=12)
        ax1.set_yscale('log')
        ax1.set_title(f"Dynamic Structural Redundancy Analysis\n{arch} on {dataset.capitalize()}", pad=15, fontweight='bold', fontsize=15, loc='center')
        ax1.legend(loc='upper right', frameon=False, fontsize=10)
        sns.despine(ax=ax1)

        # --- BOTTOM PANEL ---
        ax2.bar(x_vals, h_vals, color='#4A4A4A', alpha=0.8, edgecolor='black', linewidth=0.5, label='Relative Local Variance ($h$)')
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_ylabel(r"Relative Variance ($h$)", fontweight='bold', fontsize=12)
        ax2.set_xlabel("Network Depth (Layer Index)", fontweight='bold', fontsize=12)
        ax2.axhline(y=0, color='#1f77b4', linestyle='--', alpha=0.8, linewidth=2, label='Collapse Threshold (0)')

        # Precision Zone Mapping
        veto_idx = int(len(layers) * 0.25)
        zones = []
        if len(h_vals) > 0:
            start_idx = 0
            def check_state(idx):
                if idx < veto_idx: return "VETO"
                return "SAFE" if h_vals[idx] < 0 else "DANGER"

            current_state = check_state(0)
            for i in range(1, len(h_vals)):
                new_state = check_state(i)
                if new_state != current_state:
                    zones.append((start_idx, i - 1, current_state))
                    start_idx = i
                    current_state = new_state
            zones.append((start_idx, len(h_vals) - 1, current_state))

        for start, end, state in zones:
            span_start, span_end = start - 0.5, end + 0.5
            if state == "VETO":
                ax1.axvspan(span_start, span_end, color='#e0e0e0', alpha=0.4, hatch='////', edgecolor='none')
                ax2.axvspan(span_start, span_end, color='#e0e0e0', alpha=0.6, hatch='////', edgecolor='#999999', label="Foundational Veto (Depth < 25%)")
            elif state == "SAFE":
                ax1.axvspan(span_start, span_end, color='#2ca02c', alpha=0.1, edgecolor='none')
                ax2.axvspan(span_start, span_end, color='#2ca02c', alpha=0.2, label="Candidate Collapse Region ($h < 0$)")
            else:
                ax1.axvspan(span_start, span_end, color='#d62728', alpha=0.05, edgecolor='none')
                # LAtex Bug fixed here
                ax2.axvspan(span_start, span_end, color='#d62728', alpha=0.1, label=r"Feature Extraction Region ($h \geq 0$)")

        # Perfect Deduplication
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Ensure we drop any empty labels
        by_label = {k: v for k, v in by_label.items() if k}
        ax2.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=10, frameon=False)
        
        sns.despine(ax=ax2)
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

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        clean_metrics = g_metrics[g_metrics['posthoc_or_posttrain'].isin(['Collapsed', 'Retrained'])]
        if clean_metrics.empty: continue

        csv_path = stats_dir / f"{arch}_{dataset}_experiment_block_stats.csv"
        if not csv_path.exists(): continue

        try:
            df_h = pd.read_csv(csv_path)
        except Exception: continue

        if "Experiment" not in df_h.columns: continue

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
        sns.scatterplot(data=sp_df, x="Median Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"}, s=250, alpha=0.9, edgecolor="black", color="#2ca02c", ax=axes[0], zorder=3)
        q1 = sp_df["Median Variance"].quantile(0.3)
        axes[0].axvspan(1e-4, q1, color='#2ca02c', alpha=0.15, zorder=0, label=f"BAV Target: Bounded Growth (<{q1:.1f})")
        axes[0].set_xscale('symlog', linthresh=1e-2)
        axes[0].set_title("Single-Path: Target Bounded Growth", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylabel(r"$\Delta$ Accuracy (%) $\rightarrow$ Higher is Better", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Median Activation Variance (SymLog Scale)", fontsize=12)
        axes[0].legend(loc="lower left", framealpha=0.9)

    # --- Multi Path ---
    mp_df = full_df[full_df['Topology'] == 'Multi-Path']
    if not mp_df.empty:
        sns.scatterplot(data=mp_df, x="Median Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"}, s=250, alpha=0.9, edgecolor="black", color="#d62728", ax=axes[1], zorder=3)
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
    import matplotlib.gridspec as gridspec
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_acc_color(d_acc):
        if pd.isna(d_acc): return "#999999"
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
        
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists() or "normalized" in csv_path.name: continue
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()

        json_files = list(Path(".").glob(f"{arch}_{dataset}_*_discovered_regions.json"))
        if not json_files: continue
            
        try:
            with open(json_files[0], 'r') as f:
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
            file_suffix = "quantized" if is_quant_target else "unquantized"

            # ==========================================
            # FILE 2: Candidate Bars + Hardware Sidebar
            # (FILE 1 - Variance Plot removed to prevent redundancy with Methodology Plots)
            # ==========================================
            fig_height = max(3.5, 0.5 * num_bars + 1)
            y_limits = (-1, num_bars)
            
            fig_cand = plt.figure(figsize=(14, fig_height)) 
            gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.5], wspace=0.05)
            
            ax_heur = fig_cand.add_subplot(gs[0])
            ax_side = fig_cand.add_subplot(gs[1], sharey=ax_heur)
            ax_side.axis('off')
            
            for i, (orig_exp_name, ranges, exp_results, display_name, p_red, f_red, m_red) in enumerate(valid_exps):
                if isinstance(ranges, tuple) or (isinstance(ranges, list) and len(ranges) > 0 and isinstance(ranges[0], str)):
                    ranges = [ranges]
                
                d_acc = exp_results.get('d_acc', np.nan)
                color = get_acc_color(d_acc)
                for start_layer, end_layer in ranges:
                    try:
                        s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                        e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                        ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=16, color=color, alpha=0.85)
                    except StopIteration: continue
            
            ax_heur.set_xlim(-1, len(layers))
            ax_heur.set_ylim(y_limits)
            ax_heur.set_yticks(range(len(valid_exps)))
            ax_heur.set_yticklabels([e[3] for e in valid_exps], fontsize=11, fontweight='bold')
            ax_heur.set_xlabel("Network Depth (Layer Index)", fontweight='bold', fontsize=11)
            ax_heur.set_title(f"Structural Candidates & Hardware Reductions", loc='left', pad=25, fontsize=14, fontweight='bold')
            sns.despine(ax=ax_heur)

            cols = [0.10, 0.35, 0.65, 0.90]
            headers = ["$\\Delta$ Acc", "Params $\\downarrow$", "FLOPs $\\downarrow$", "Memory $\\downarrow$"]
            
            header_y = len(valid_exps) 
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

    logger.info("[FIG4] Candidate/Sidebar plots generated successfully.")

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
        
        ax.set_yticks(range(len(y_labels)))
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
        
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
        ax.set_xlabel("Architecture & Accuracy Impact", fontweight='bold', fontsize=12)
        ax.set_ylabel("Reduction Relative to Baseline (%)", fontweight='bold', fontsize=12)
        ax.set_title(f"{title_prefix} Structural Collapse Efficiency by Architecture", pad=15, fontweight='bold', fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7); ax.set_axisbelow(True)
        ax.legend(title="", loc='upper right'); sns.despine()
        plt.tight_layout()
        plt.savefig(out_dir / f"unified_{filename_suffix}.png", bbox_inches='tight')
        plt.close()

    # --- Deliverable 6: The Trade-off Scatter Plot (Pareto Frontier) ---
    if all_tradeoff_data:
        df_trade = pd.DataFrame(all_tradeoff_data)
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        df = raw.copy()
        # ---------------------------------------------------------
        # COMPUTE MISSING DELTAS (d_acc, d_params, d_flops, d_mem)
        # ---------------------------------------------------------
        if 'd_acc' not in df.columns:
            print("[INFO] Computing missing delta metrics...")
            
            # 1. Use the EXACT lowercase column names generated by load_results()
            group_cols = ['architecture', 'dataset', 'is_quantized']
            
            # 2. Find the baseline rows
            baselines = df[df['base_name'] == 'Original Model']
            
            if baselines.empty:
                print("[WARNING] Could not find 'Original Model' baselines to compute deltas.")
            else:
                # 3. Create a map of baseline values
                base_map = baselines.drop_duplicates(subset=group_cols).set_index(group_cols)[
                    ['accuracy', 'params', 'flops', 'memory']
                ].rename(columns={
                    'accuracy': 'base_acc',
                    'params': 'base_params',
                    'flops': 'base_flops',
                    'memory': 'base_mem'
                })
                
                # 4. Join back to the main dataframe
                df = df.join(base_map, on=group_cols)

                
                # 5. Compute the deltas
                df['d_acc'] = df['accuracy'] - df['base_acc']
                df['d_params'] = (1 - (df['params'] / df['base_params'].replace(0, 1))) * 100
                df['d_flops'] = (1 - (df['flops'] / df['base_flops'].replace(0, 1))) * 100 
                df['d_mem'] = (1 - (df['memory'] / df['base_mem'].replace(0, 1))) * 100

                
                # 6. Clean up NaNs
                df.fillna({'d_acc': 0, 'd_params': 0, 'd_flops': 0, 'd_mem': 0}, inplace=True)
        # ---------------------------------------------------------
        df = normalize(df)
        fig1(df)
        fig2_methodology_bav_regions()
        fig3_v2t_heuristic_validation(df)
        fig4_comprehensive_search_space_map(df)
        fig5_hardware_efficiency_profiles(df)
        logger.info("Script completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)