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

import matplotlib.pyplot as plt
matplotlib.set_loglevel('ERROR')


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
