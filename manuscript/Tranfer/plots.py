# plots.py
import os
import matplotlib.pyplot as plt
import matplotlib
import logging
from utils import ensure_dir
import pandas as pd
import seaborn as sns
import os
from typing import List, Dict
import matplotlib.colors as mcolors
import matplotlib.cm as cm

import matplotlib.pyplot as plt
matplotlib.set_loglevel('ERROR')

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
