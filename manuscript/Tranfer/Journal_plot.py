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

def fig3_v2t_heuristic_validation(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/heuristic_validation")):
    out_dir.mkdir(parents=True, exist_ok=True)
    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]
    all_merged_data = []
    
    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        clean_metrics = g_metrics[g_metrics['posthoc_or_posttrain'].isin(['Collapsed', 'Retrained'])]
        if clean_metrics.empty: continue
            
        csv_path = stats_dir / f"{arch}_{dataset}_experiment_block_stats.csv"
        if csv_path.exists():
            df_h = pd.read_csv(csv_path)
            merged = pd.merge(df_h, clean_metrics, left_on="Experiment", right_on="base_name")
            merged["Topology"] = "Multi-Path" if arch in multi_path_archs else "Single-Path"
            all_merged_data.append(merged)
            
    if not all_merged_data: return
    full_df = pd.concat(all_merged_data)
    
    # Map booleans to clean strings for the legend
    full_df['Quantization State'] = full_df['is_quantized'].map({False: 'Unquantized', True: 'Quantized'})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    plt.subplots_adjust(wspace=0.05)
    
    for ax in axes:
        ax.axhline(0, color='black', linestyle='--', linewidth=2, zorder=1)
        ax.axhspan(0, 100, color='#2ca02c', alpha=0.05, zorder=0, label="Accuracy Improved")
        ax.axhspan(-2.0, 0, color='#ff7f0e', alpha=0.05, zorder=0, label="Minor Degradation (<2%)")
        ax.axhspan(-100, -2.0, color='#d62728', alpha=0.05, zorder=0, label="Severe Degradation")
        ax.set_ylim(full_df['d_acc'].min() - 2, max(full_df['d_acc'].max() + 2, 5))
    
    sp_df = full_df[full_df['Topology'] == 'Single-Path']
    if not sp_df.empty:
        sns.scatterplot(data=sp_df, x="Median Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"},
                        s=250, alpha=0.9, edgecolor="black", color="#2ca02c", ax=axes[0], zorder=3)
        # Matches method.tex: Single-Path target is Flat Flow
        q1 = sp_df["Median Variance"].quantile(0.3)
        axes[0].axvspan(1e-4, q1, color='#2ca02c', alpha=0.15, zorder=0, label=f"V2T Target: Flat Flow (<{q1:.1f})")
        axes[0].set_xscale('symlog', linthresh=1e-2)
        axes[0].set_title("Single-Path: Target Flat Representation", fontsize=16, fontweight='bold', pad=15)
        axes[0].set_ylabel(r"$\Delta$ Accuracy (%) $\rightarrow$ Higher is Better", fontsize=14, fontweight='bold')
        axes[0].set_xlabel("Median Activation Variance (SymLog Scale)", fontsize=12)
        axes[0].legend(loc="lower left", framealpha=0.9)
        
    mp_df = full_df[full_df['Topology'] == 'Multi-Path']
    if not mp_df.empty:
        sns.scatterplot(data=mp_df, x="Median Variance", y="d_acc", style="Quantization State", markers={"Unquantized": "o", "Quantized": "X"},
                        s=250, alpha=0.9, edgecolor="black", color="#d62728", ax=axes[1], zorder=3)
        # Matches method.tex: Multi-Path target is High Spikes
        q3 = mp_df["Median Variance"].quantile(0.6)
        axes[1].axvspan(q3, mp_df["Median Variance"].max() * 5, color='#d62728', alpha=0.15, zorder=0, label=f"V2T Target: High Spikes (>{q3:.1f})")
        axes[1].set_xscale('symlog', linthresh=1e-2)
        axes[1].set_title("Multi-Path: Target Overfitting Spikes", fontsize=16, fontweight='bold', pad=15)
        axes[1].set_xlabel("Median Activation Variance (SymLog Scale)", fontsize=12)
        axes[1].legend(loc="lower right", framealpha=0.9)

    sns.despine()
    plt.savefig(out_dir / "V2T_heuristic_validation_map.png")
    plt.close()

def fig4_heuristic_search_space_map(df: pd.DataFrame, stats_dir: Path = Path("./runs/plots/Layer_Statistics"), out_dir: Path = Path("./figures/search_space")):
    try:
        from transfer import EXPERIMENTS 
    except ImportError:
        logger.error("Could not import EXPERIMENTS from transfer.py. Skipping Fig4.")
        return
        
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def get_acc_color(d_acc):
        if d_acc >= -2.0: return "#2ca02c"
        if d_acc >= -6.0: return "#ff7f0e"
        return "#d62728"

    def robust_match(target_name, is_quant_target, g_df):
        sub_df = g_df[(g_df['is_quantized'] == is_quant_target) & (g_df['posthoc_or_posttrain'].isin(['Collapsed', 'Retrained']))]
        if sub_df.empty: sub_df = g_df[(g_df['is_quantized'] == is_quant_target)]
        if sub_df.empty: return pd.DataFrame()

        m = sub_df[sub_df['base_name'] == target_name]
        if not m.empty: return m
        
        m = sub_df[sub_df['base_name'].str.lower() == target_name.lower()]
        if not m.empty: return m
        
        def squash(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
        st = squash(target_name)
        
        fuzzy_matches = []
        for _, row in sub_df.iterrows():
            if squash(row['base_name']) == st:
                fuzzy_matches.append(row)
                
        if fuzzy_matches: return pd.DataFrame(fuzzy_matches)
        return pd.DataFrame()

    multi_path_archs = ["RegNetX_400MF", "InceptionNet", "ConvNeXt", "XceptionNet"]

    for (dataset, arch), g_metrics in df.groupby(["dataset", "architecture"]):
        csv_path = stats_dir / f"{arch}_{dataset}_layer_stats.csv"
        if not csv_path.exists(): continue
            
        layer_df = pd.read_csv(csv_path)
        layers = layer_df['Layer'].tolist()
        variances = np.maximum(layer_df['Variance'].values, 1e-6)
        
        mu_net = np.mean(variances)
        median_net = np.median(variances)
        std_net = np.std(variances)
        
        model_exps = EXPERIMENTS.get(arch, {}).get(dataset, {})
        if not model_exps: continue

        for is_quant_target in [False, True]:
            target_label = "Quantized" if is_quant_target else "Unquantized"
            logger.info(f"--- Generating Ladder Plot for {arch} on {dataset} ({target_label}) ---")
            
            valid_exps = []
            for exp_name, ranges in model_exps.items():
                if ranges is None: continue
                cleaned_name = clean_exp_name(exp_name)
                
                m = robust_match(cleaned_name, is_quant_target=is_quant_target, g_df=g_metrics)
                if not m.empty:
                    valid_exps.append((exp_name, ranges, m, cleaned_name))
                    
            if not valid_exps:
                logger.warning(f"  [EMPTY] No valid data remains for {arch} on {dataset} ({target_label}). Skipping plot.")
                continue
            
            # Sort by performance: highest d_acc (lowest drop) plots at the top (highest index)
            valid_exps = sorted(valid_exps, key=lambda x: x[2].mean(numeric_only=True)['d_acc'])
            
            num_bars = len(valid_exps)
            fig_height = max(8, 4 + 0.4 * num_bars)
            height_ratios = [1, max(2, fig_height / 3.5)]
            
            fig, (ax_var, ax_heur) = plt.subplots(2, 1, figsize=(14, fig_height), sharex=True, gridspec_kw={'height_ratios': height_ratios})
            plt.subplots_adjust(hspace=0.08)

            # Explicit x-array is required to enable fill_between interpolation
            x_vals = np.arange(len(layers))

            # TOP PANEL: Variance & Heuristics
            ax_var.plot(x_vals, variances, color='#555555', linewidth=1.5, alpha=0.8)
            ax_var.set_yscale('log')
            ax_var.set_ylabel("Variance ($\sigma^2$)")
            title_text = f"Heuristic Search Space Guide: {arch} on {format_dataset_name(dataset)} ({target_label})"
            ax_var.set_title(title_text, loc='left', pad=20, fontsize=16, fontweight='bold')
            
            y_min = max(1e-4, min(variances) * 0.5)
            y_max = max(variances) * 2.0
            ax_var.set_ylim(y_min, y_max)
            
            if arch in multi_path_archs:
                # Matches method.tex: Target regions above the median + std for Multi-Path
                var_thresh = median_net + std_net
                ax_var.axhline(y=var_thresh, color='#d62728', linestyle='--', alpha=0.5, label=r"Multi-Path Target ($V_m > \tilde{V} + \sigma$)")
                ax_var.fill_between(x_vals, var_thresh, y_max, where=(variances >= var_thresh), color='#d62728', alpha=0.15, interpolate=True)
            else:
                # Matches method.tex: Target regions under the global mean for Single-Path
                var_thresh = mu_net
                ax_var.axhline(y=var_thresh, color='#2ca02c', linestyle='--', alpha=0.5, label=r"Single-Path Target ($V_m < \mu_{net}$)")
                ax_var.fill_between(x_vals, y_min, var_thresh, where=(variances <= var_thresh), color='#2ca02c', alpha=0.15, interpolate=True)
            ax_var.legend(loc='upper right')

            # BOTTOM PANEL: Sorted Search Space Candidates
            for i, (orig_exp_name, ranges, exp_results, display_name) in enumerate(valid_exps):
                ranges = ranges if isinstance(ranges, list) else [ranges]
                
                exp_results = exp_results.mean(numeric_only=True)
                d_acc = exp_results['d_acc']
                final_acc = exp_results['accuracy']
                color = get_acc_color(d_acc)
                label_text = f"{final_acc:.1f}% ($\Delta$ {d_acc:+.1f}%)"

                for start_layer, end_layer in ranges:
                    try:
                        s_idx = next(idx for idx, n in enumerate(layers) if start_layer in n)
                        e_idx = next(idx for idx, n in reversed(list(enumerate(layers))) if end_layer in n)
                        ax_heur.hlines(y=i, xmin=s_idx, xmax=e_idx, linewidth=12, color=color, alpha=0.9)
                        ax_heur.text(e_idx + 0.5, i, label_text, va='center', fontsize=9, fontweight='bold', color=color)
                    except StopIteration: continue
            
            ax_heur.set_yticks(range(len(valid_exps)))
            ax_heur.set_yticklabels([e[3] for e in valid_exps], fontsize=10)
            ax_heur.set_xlabel("Network Depth (Layer Index)")
            ax_heur.set_ylabel("Collapsed Layer Candidates")
            sns.despine(ax=ax_var); sns.despine(ax=ax_heur)
            
            file_suffix = "quantized" if is_quant_target else "unquantized"
            save_path = out_dir / f"{arch}_{dataset}_decision_map_{file_suffix}.png"
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    try:
        raw = load_results()
        df = normalize(raw)
        fig1(df)
        fig3_v2t_heuristic_validation(df)
        fig4_heuristic_search_space_map(df)
        logger.info("Script completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)