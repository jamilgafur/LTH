from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Configuration
# =========================
RESULTS_DIR = Path("./")
FIG_DIR = Path("./figures")
TABLE_DIR = Path("./tables")

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# Define dataset order (needed for inference)
DATASET_ORDER: List[str] = ["cifar10", "cifar100", "imagenet", "tinyimagenet"]

# =========================
# Utilities
# =========================

def infer_owner(exp_name: str) -> str:
    n = exp_name.lower()
    if "kevin" in n:
        return "Kevin"
    if "jf" in n:
        return "JF"
    return "Unknown"

def infer_dataset_from_path(p: Path) -> str:
    folder_name = p.parent.parent.name.lower()
    for ds in DATASET_ORDER:
        if ds in folder_name:
            return ds
    if "tiny" in folder_name and "imagenet" in folder_name:
        return "tinyimagenet"
    raise ValueError(f"Cannot infer dataset from path: {p}")

def infer_model_from_path(p: Path) -> str:
    """Extract the model architecture from the parent folder name."""
    folder_name = p.parent.parent.name.lower()
    if "vgg" in folder_name:
        return "VGG16"
    elif "inception" in folder_name:
        return "InceptionNet"
    else:
        return "Other"

def infer_model_type_from_exp_name(exp_name: str) -> str:
    n = exp_name.lower()
    if "original" in n or "baseline" in n:
        return "baseline"
    return "compressed"

# =========================
# Data Loading
# =========================
def load_and_merge_results(results_dir: Path) -> Dict[str, pd.DataFrame]:
    files = list(results_dir.rglob("*merged_metrics.json"))
    print(f"[DEBUG] Found files: {files}")
    if not files:
        raise FileNotFoundError("No *merged_metrics.json files found")

    datasets: Dict[str, List[Path]] = {ds: [] for ds in DATASET_ORDER}

    for f in files:
        dataset = infer_dataset_from_path(f)
        datasets[dataset].append(f)

    merged: Dict[str, pd.DataFrame] = {}
    for dataset, paths in datasets.items():
        if not paths:
            print(f"[DEBUG] Skipping dataset '{dataset}' (no files found)")
            continue

        rows = []
        for p in paths:
            with open(p, "r") as f:
                raw = json.load(f)
            
            model_arch = infer_model_from_path(p)

            for exp_name, metrics in raw.items():
                diag = metrics.get("diagnostics", {})
                rows.append({
                    "dataset": dataset,
                    "model": model_arch,
                    "exp_name": exp_name,
                    "owner": infer_owner(exp_name),
                    "model_type": infer_model_type_from_exp_name(exp_name),
                    "accuracy": metrics.get("final_accuracy"),
                    "params": metrics.get("param_count"),
                    "flops": metrics.get("flops"),
                    "memory": metrics.get("total_size_mb"),
                    "inference_time": metrics.get("inference_time"),
                    "epochs": metrics.get("total_epochs_trained"),
                    "per_layer_params_flops": diag.get("per_layer_params_flops"),
                    "activation_sizes": diag.get("activation_sizes"),
                    "memory_decomposition": diag.get("memory_decomposition"),
                })
        df = pd.DataFrame(rows)
        merged[dataset] = df

    return merged

# =========================
# Plotting functions
# =========================
sns.set(style="whitegrid")

def plot_metric_by_exp(df: pd.DataFrame, dataset: str, metric: str):
    """
    Bar plot per model: x-axis = exp_name, hue = owner, value = metric.
    """
    for model, df_model in df.groupby("model"):
        plt.figure(figsize=(max(6, len(df_model)*0.5), 4))
        sns.barplot(
            data=df_model,
            x="exp_name",
            y=metric,
            hue="owner",
            order=sorted(df_model['exp_name'].unique()),
            palette="Set2"
        )
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.replace("_", " ").capitalize())
        plt.xlabel("Experiment")
        plt.title(f"{metric.replace('_',' ').capitalize()} - {model} ({dataset})")
        plt.tight_layout()

        fig_path = FIG_DIR / f"{dataset}_{model}_{metric}.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] Saved plot: {fig_path}")

def plot_accuracy_vs_flops_by_exp(df: pd.DataFrame, dataset: str):
    """
    Scatter plot: Accuracy vs FLOPs per model, hue = owner, style = exp_name.
    """
    for model, df_model in df.groupby("model"):
        plt.figure(figsize=(6, 4))
        sns.scatterplot(
            data=df_model,
            x="flops",
            y="accuracy",
            hue="owner",
            style="exp_name",
            palette="Set2",
            s=100,
            alpha=0.8
        )
        plt.ylabel("Accuracy")
        plt.xlabel("FLOPs")
        plt.title(f"Accuracy vs FLOPs - {model} ({dataset})")
        plt.tight_layout()

        fig_path = FIG_DIR / f"{dataset}_{model}_accuracy_vs_flops.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] Saved scatter plot: {fig_path}")

def plot_multi_metric_per_exp(df: pd.DataFrame, dataset: str, metrics: List[str]):
    """
    Multi-metric bar plot per model: x-axis = exp_name, hue = owner.
    """
    for model, df_model in df.groupby("model"):
        df_melted = df_model.melt(id_vars=["exp_name", "owner"],
                                  value_vars=metrics,
                                  var_name="metric",
                                  value_name="value")
        plt.figure(figsize=(max(6, len(df_model)*0.5), 4))
        sns.barplot(
            data=df_melted,
            x="exp_name",
            y="value",
            hue="owner",
            ci=None,
            palette="Set2"
        )
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Metric Value")
        plt.xlabel("Experiment")
        plt.title(f"Metrics - {model} ({dataset})")
        plt.tight_layout()

        fig_path = FIG_DIR / f"{dataset}_{model}_multi_metric.png"
        plt.savefig(fig_path)
        plt.close()
        print(f"[INFO] Saved multi-metric plot: {fig_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    merged_datasets = load_and_merge_results(RESULTS_DIR)
    all_df = pd.concat(merged_datasets.values(), ignore_index=True)

    metrics_to_plot = ["accuracy", "params", "flops", "memory", "inference_time"]

    for dataset in all_df['dataset'].unique():
        df = all_df[all_df['dataset'] == dataset]
        if df.empty:
            continue

        # Bar plots per metric
        for metric in metrics_to_plot:
            plot_metric_by_exp(df, dataset, metric)

        # Scatter plot: accuracy vs FLOPs
        plot_accuracy_vs_flops_by_exp(df, dataset)

        # Multi-metric bar plot
        plot_multi_metric_per_exp(df, dataset, metrics_to_plot)
