# plots.py
import os
import matplotlib.pyplot as plt
import matplotlib
import logging
import pandas as pd
import seaborn as sns
import os
from typing import List, Dict
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import matplotlib.pyplot as plt
matplotlib.set_loglevel('ERROR')

def plot_experiment_heuristics(model_name, dataset_name, stats_csv_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # [✓] MOVED HERE: This breaks the circular import!
    from transfer import EXPERIMENTS 

    # Load the raw layer stats
    df_layers = pd.read_csv(stats_csv_path)
    layer_names = df_layers['Layer'].tolist()
    variances = dict(zip(df_layers['Layer'], df_layers['Variance']))
    activations = dict(zip(df_layers['Layer'], df_layers['Mean Activation']))

    exp_dict = EXPERIMENTS[model_name][dataset_name]
    
    exp_names, total_vars, avg_acts = [], [], []

    # Calculate Total Variance and Average Activation per experiment
    for exp_name, layer_range in exp_dict.items():
        if layer_range is None or exp_name == "Original Model":
            continue
            
        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        b_vars, b_acts = [], []
        
        for start_layer, end_layer in ranges:
            in_range = False
            for name in layer_names:
                if start_layer in name: in_range = True
                if in_range:
                    if name in variances: b_vars.append(variances[name])
                    if name in activations: b_acts.append(activations[name])
                if end_layer in name: break
                
        if b_vars and b_acts:
            exp_names.append(exp_name)
            total_vars.append(np.sum(b_vars)) # SUM of variance (Total Information)
            avg_acts.append(np.mean(b_acts))  # MEAN of activation (Average Volume)

    # Generate Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    df_plot = pd.DataFrame({"Experiment": exp_names, "Total Variance": total_vars, "Mean Activation": avg_acts})

    # Top Plot: Mean Activation
    sns.barplot(data=df_plot, x="Experiment", y="Mean Activation", color="#4C72B0", edgecolor="black", ax=ax1)
    ax1.set_title(f"Heuristic Profiling by Target Region: {model_name}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Avg Mean Activation", fontweight='bold')
    ax1.axhline(0, color='black', linewidth=1.5)

    # Bottom Plot: Total Variance
    sns.barplot(data=df_plot, x="Experiment", y="Total Variance", color="#C44E52", edgecolor="black", ax=ax2)
    ax2.set_ylabel("Total Sum of Variance", fontweight='bold')
    ax2.set_xlabel("Targeted Collapse Region", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"runs/plots/{model_name}_heuristic_target_summary.png", dpi=300)
    print(f"Saved runs/plots/{model_name}_heuristic_target_summary.png")


def plot_individual_layers(layer_activations, layer_variances, directory, model_name, dataset_name, exp_config=None):
    if not layer_activations:
        return
    layers = list(layer_activations.keys())
    activations = list(layer_activations.values())
    variances = list(layer_variances.values())

    df = pd.DataFrame({"Layer": layers, "Mean Activation": activations, "Variance": variances})
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats.csv"), index=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    regions = {}
    if exp_config:
        for key, val in exp_config.items():
            if "(Full)" in key and isinstance(val, tuple):
                start_layer, end_layer = val
                start_idx = next((i for i, n in enumerate(layers) if start_layer in n), None)
                end_idx = next((i for i, n in reversed(list(enumerate(layers))) if end_layer in n), None)
                if start_idx is not None and end_idx is not None:
                    clean_name = key.replace(" (Full)", "")
                    regions[clean_name] = (start_idx, end_idx)

    bg_colors = ['#eaf2f8', '#fdf2e9', '#e8f8f5', '#f5eef8', '#f4f6f7']
    for i, (region_name, (start, end)) in enumerate(regions.items()):
        color = bg_colors[i % len(bg_colors)]
        ax1.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        ax2.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        y_max = max(variances) if variances else 1
        ax2.text((start + end) / 2, y_max * 0.95, region_name, ha='center', va='top', fontsize=11, fontweight='bold', color='#555555', alpha=0.8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))

    sns.lineplot(data=df, x="Layer", y="Mean Activation", marker="o", color="steelblue", linewidth=2, ax=ax1, zorder=3)
    ax1.set_ylabel("Mean Activation", fontweight='bold', labelpad=10)
    ax1.set_title(f"Layer-wise Activation & Structural Stages\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=12)

    sns.lineplot(data=df, x="Layer", y="Variance", marker="s", color="crimson", linewidth=2, linestyle="--", ax=ax2, zorder=3)
    ax2.set_ylabel("Variance", fontweight='bold', labelpad=10)
    ax2.set_xlabel("Network Layer", fontweight='bold', labelpad=10)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=90, fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats_annotated.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_normalized_metrics(layer_activations, layer_variances, directory, model_name, dataset_name):
    if not layer_activations: return
    
    layers = list(layer_activations.keys())
    means = np.array(list(layer_activations.values()))
    vars_arr = np.array(list(layer_variances.values()))
    
    avg_var = np.mean(vars_arr)
    norm_vars = vars_arr / (avg_var + 1e-12)
    
    cvs = vars_arr / (np.abs(means) + 1e-12) 
    avg_cv = np.mean(cvs)
    norm_cvs = cvs / (avg_cv + 1e-12)
    
    df = pd.DataFrame({"Layer": layers, "Normalized Variance": norm_vars, "Normalized CV": norm_cvs})
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_normalized_layer_stats.csv"), index=False)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized Variance", color="coral", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-Variance (1.0)")
    ax.set_title(f"Normalized Layer Variance\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Variance / Avg Variance", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_variance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized CV", color="mediumpurple", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-CV (1.0)")
    ax.set_title(f"Normalized Coefficient of Variation (CV)\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Layer CV / Avg CV", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_cv.png"), dpi=300, bbox_inches='tight')
    plt.close()


def save_and_plot_metric(data, y_col, directory, title_prefix, ylabel, hline_val, hline_label, color_base, color_alt, model_name, dataset_name, invert_safe_zone=False):
    if not data: return
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_{y_col.replace(' ', '_')}.csv"), index=False)
    df.to_latex(os.path.join(directory, f"{model_name}_{dataset_name}.tex"), index=False, float_format="%.4f")

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 7))
    df['Color_Group'] = ['Control' if exp == 'Control' else 'Experiment' for exp in df['Experiment']]
    palette = {'Control': color_base, 'Experiment': color_alt}

    sns.barplot(data=df, x="Experiment", y=y_col, hue="Color_Group", palette=palette, dodge=False, edgecolor="black", linewidth=0.8, zorder=3, ax=ax)
    ax.legend_.remove()

    ymin, ymax = ax.get_ylim()
    if df[y_col].min() >= 0: ymin = 0.0  
    else: ymin = min(df[y_col].min() * 1.05, ymin)
    if hline_val > ymax: ymax = hline_val * 1.15
    ax.set_ylim(ymin, ymax)

    ax.axhline(0, color='black', linewidth=1.5, zorder=4) 
    ax.axhline(hline_val, color='crimson', linestyle='--', linewidth=2.5, zorder=4, label=hline_label)

    if invert_safe_zone:
        ax.axhspan(hline_val, ymax, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe (High Redundancy)')
        ax.axhspan(ymin, hline_val, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')
    else:
        ax.axhspan(ymin, hline_val, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe')
        ax.axhspan(hline_val, ymax, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')

    ax.set_title(f"{title_prefix}\n{model_name} | {dataset_name}", fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel("Structural Modification", fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)
    sns.despine(bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_experiment_{y_col.split(' ')[0]}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_paper_quality_scores(df, save_root_dir, model_name, dataset_name):
    """
    Generates a publication-ready bar chart for Collapse Scores.
    
    Changes:
    - Normalizes scores to 0.0 - 1.0 range relative to the min/max of the data.
    - Uses a continuous color gradient (Red -> Green) instead of discrete zones.
    - Removes hardcoded threshold lines.
    """
    # Create directory
    score_dir = os.path.join(save_root_dir, "collapse_score")
    os.makedirs(score_dir, exist_ok=True)
    
    # --- 1. Normalize Scores (Min-Max Scaling) ---
    # This ensures the plot always uses the full 0-1 vertical space,
    # making relative differences easier to see.
    min_score = df['collapse_score'].min()
    max_score = df['collapse_score'].max()
    
    # Avoid division by zero if all scores are identical
    # if max_score > min_score:
    #     df['norm_score'] = (df['collapse_score'] - min_score) / (max_score - min_score)
    # else:
    df['norm_score'] = df['collapse_score'] # Keep original if flat
    
    # --- 2. Setup Plot ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(max(10, len(df)*0.25), 5))
    
    # --- 3. Create Continuous Color Map ---
    # Map normalized scores to a Red-Yellow-Green gradient
    cmap = mcolors.LinearSegmentedColormap.from_list("safety_gradient", ["#e74c3c", "#f1c40f", "#2ecc71"])
    
    # We assign a specific color to each bar based on its normalized height
    bar_colors = [cmap(val) for val in df['norm_score']]
    
    # --- 4. Draw Bar Chart ---
    ax = sns.barplot(
        x="layer", 
        y="norm_score", 
        data=df, 
        palette=bar_colors,
        edgecolor="black", 
        linewidth=0.5
    )
    
    # --- 5. Formatting ---
    plt.title(f"Structural Stability Score \n{model_name} on {dataset_name}", fontsize=14, fontweight='bold', pad=15)
    plt.ylabel("Stability Score (0=Critical, 1=Safe)", fontsize=12, fontweight='bold')
    plt.xlabel("Layer Depth", fontsize=12)
    
    # X-Axis Ticks
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
    
    # Y-Axis Limits (Strict 0-1)
    plt.ylim(0, 1.05)
    
    # Add a Colorbar to act as a Legend
    sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label('Collapse Probability (Red=High Risk)', rotation=270, labelpad=15)

    plt.tight_layout()
    
    # Save High-Res
    filename = f"{model_name}_{dataset_name}_collapse_score_norm.png"
    save_path = os.path.join(score_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    [Saved] Normalized Research Plot -> {save_path}")
    
def table_failure_modes(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)
    failures = df[~df["accepted"]]
    failures.to_csv(os.path.join(save_dir, "table6_failure_cases.csv"), index=False)

def plot_failure_case(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)
    failures = df[~df["accepted"]].sort_values("delta_accuracy")

    if failures.empty:
        return

    f = failures.iloc[0]

    plt.figure(figsize=(7, 4))
    plt.plot(
        df["block_idx"],
        df["accuracy"],
        marker="o"
    )

    plt.axvline(f["block_idx"], color="red", linestyle="--")
    plt.annotate(
        f"Failure at block {f['block_idx']}",
        xy=(f["block_idx"], f["accuracy"]),
        xytext=(10, -15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->")
    )

    plt.xlabel("Block Index")
    plt.ylabel("Accuracy (%)")
    plt.title("Representative Failure Case")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig6_failure_case.svg"))
    plt.close()

def table_efficiency_comparison(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)
    cols = ["model", "collapsed_fraction", "params", "flops", "activation_mb"]
    df[cols].to_csv(os.path.join(save_dir, "table5_efficiency.csv"), index=False)

def plot_efficiency_vs_collapse(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    axs[0].plot(df["collapsed_fraction"], df["params"] / 1e6, marker="o")
    axs[0].set_ylabel("Parameters (M)")

    axs[1].plot(df["collapsed_fraction"], df["flops"] / 1e9, marker="o")
    axs[1].set_ylabel("FLOPs (G)")

    axs[2].plot(df["collapsed_fraction"], df["activation_mb"], marker="o")
    axs[2].set_ylabel("Activation Memory (MB)")
    axs[2].set_xlabel("Collapsed Depth Fraction")

    for ax in axs:
        ax.grid(alpha=0.3)

    plt.suptitle("Efficiency Effects of Depth Reduction")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig5_efficiency_vs_collapse.svg"))
    plt.close()

def table_collapsible_depth_stats(summary_table, save_dir):
    stats = summary_table.groupby("model")["max_collapsed_fraction"].agg(["mean", "std"])
    stats.to_csv(os.path.join(save_dir, "table4_collapsible_depth_stats.csv"))
    return stats

def plot_collapsible_depth_across_models(summary_table, save_dir):
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=summary_table,
        x="model",
        y="max_collapsed_fraction",
        hue="dataset"
    )

    plt.ylabel("Fraction of Collapsible Depth")
    plt.title("Consistency of Collapsible Depth Across Models & Datasets")
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig4_cross_model_consistency.svg"))
    plt.close()

def table_surrogate_summary(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)
    summary = df.groupby("accepted")[["surrogate_error", "delta_accuracy"]].agg(
        ["mean", "std"]
    )
    summary.to_csv(os.path.join(save_dir, "table3_surrogate_summary.csv"))
    return summary

def plot_surrogate_error_vs_accuracy(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=df,
        x="surrogate_error",
        y="delta_accuracy",
        hue="accepted",
        palette={True: "green", False: "red"},
        s=70,
    )

    plt.axhline(0, linestyle="--", color="gray")
    plt.xlabel("Surrogate Approximation Error (MSE)")
    plt.ylabel("Δ Test Accuracy (%)")
    plt.title("Surrogate Error vs Downstream Accuracy Change")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig3_surrogate_vs_accuracy.svg"))
    plt.close()

def table_block_statistics(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)
    cols = [
        "model", "dataset", "block_idx", "normalized_depth",
        "surrogate_error", "delta_accuracy", "accepted"
    ]
    out = df[cols]
    out.to_csv(os.path.join(save_dir, "table2_block_stats.csv"), index=False)
    return out

def plot_block_acceptance_by_depth(collapse_results, save_dir):
    df = pd.DataFrame(collapse_results)

    plt.figure(figsize=(9, 4))
    sns.scatterplot(
        data=df,
        x="normalized_depth",
        y="model",
        hue="accepted",
        style="accepted",
        palette={True: "green", False: "red"},
        s=80,
    )

    plt.xlabel("Normalized Network Depth")
    plt.ylabel("Architecture")
    plt.title("Block-Level Collapse Outcomes Across Depth")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig2_block_acceptance.svg"))
    plt.close()

def table_max_collapsible_depth(collapse_results, tau, save_dir):
    df = pd.DataFrame(collapse_results)

    rows = []
    for (model, dataset), g in df.groupby(["model", "dataset"]):
        baseline = g["baseline_accuracy"].iloc[0]
        valid = g[g["accuracy"] >= baseline - tau]
        max_frac = valid["collapsed_fraction"].max()
        acc_change = valid.loc[
            valid["collapsed_fraction"] == max_frac, "delta_accuracy"
        ].iloc[0]

        rows.append({
            "model": model,
            "dataset": dataset,
            "total_depth": g["block_idx"].max(),
            "max_collapsed_fraction": max_frac,
            "delta_accuracy": acc_change,
        })

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(save_dir, "table1_max_collapsible_depth.csv"), index=False)
    return table

def plot_accuracy_vs_collapsed_depth(
    collapse_results: List[Dict],
    tau: float,
    save_dir: str,
):
    df = pd.DataFrame(collapse_results)

    ensure_dir(save_dir)
    plt.figure(figsize=(8, 6))

    for (model, dataset), g in df.groupby(["model", "dataset"]):
        plt.plot(
            g["collapsed_fraction"],
            g["accuracy"],
            marker="o",
            label=f"{model} / {dataset}"
        )

    plt.axhline(
        y=df["baseline_accuracy"].iloc[0] - tau,
        linestyle="--",
        color="red",
        label=r"Accuracy tolerance $\tau$"
    )

    plt.xlabel("Fraction of Sequential Depth Collapsed")
    plt.ylabel("Top-1 Test Accuracy (%)")
    plt.title("Accuracy vs Collapsed Sequential Depth")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "fig1_accuracy_vs_depth.svg"))
    plt.close()

