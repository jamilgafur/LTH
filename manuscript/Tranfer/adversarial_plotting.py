"""Plotting utilities for adversarial analysis outputs."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from adversarial_reporting import ReportingSuite


class AdversarialPlotSuite:
    """Generate standard visualizations for attack and transfer results."""

    @classmethod
    def generate_plots(cls, output_dir: str, records: List[Dict], transfer_records: List[Dict]):
        os.makedirs(output_dir, exist_ok=True)

        if records:
            df = ReportingSuite.enrich_summary_dataframe(pd.DataFrame(records))
            cls._plot_attack_success_overview(output_dir, df)
            cls._plot_per_dataset_attack_views(output_dir, df)

        if transfer_records:
            records_df = ReportingSuite.enrich_summary_dataframe(pd.DataFrame(records)) if records else pd.DataFrame()
            tf_df = ReportingSuite.enrich_transfer_dataframe(pd.DataFrame(transfer_records), records_df)
            cls._plot_transferability(output_dir, tf_df)

    @staticmethod
    def _plot_attack_success_overview(output_dir: str, df: pd.DataFrame) -> None:
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x="model", y="attack_success_rate", hue="attack", palette="Set2", legend=True)
        plt.ylabel("Attack Success Rate", fontweight="bold")
        plt.xlabel("Model Architecture", fontweight="bold")
        plt.title("Adversarial Attack Success Rates Across Models", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attack_success_rates.png"), dpi=300)
        plt.close()

        for kind in df["kind"].unique():
            df_kind = df[df["kind"] == kind]
            plt.figure(figsize=(14, 6))
            sns.barplot(data=df_kind, x="model", y="attack_success_rate", hue="attack", palette="husl", legend=True)
            plt.ylabel("Attack Success Rate", fontweight="bold")
            plt.xlabel("Model Architecture", fontweight="bold")
            plt.title(f"Attack Success Rates - {kind} Models", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attack_success_rates_{kind}.png"), dpi=300)
            plt.close()

    @staticmethod
    def _plot_per_dataset_attack_views(output_dir: str, df: pd.DataFrame) -> None:
        for dataset in df["dataset"].unique():
            df_dataset = df[df["dataset"] == dataset]

            plt.figure(figsize=(12, 5))
            sns.boxplot(data=df_dataset, x="attack", y="attack_success_rate", hue="kind", palette="Set1")
            plt.ylabel("Attack Success Rate", fontweight="bold")
            plt.xlabel("Attack Method", fontweight="bold")
            plt.title(f"Attack Effectiveness Comparison - {dataset}", fontsize=14, fontweight="bold")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attack_comparison_{dataset}.png"), dpi=300)
            plt.close()

            direct_heatmap = df_dataset.pivot_table(
                index="model_label",
                columns="attack",
                values="attack_success_rate",
                aggfunc="mean",
            )
            if direct_heatmap is not None and not direct_heatmap.empty:
                plt.figure(figsize=(10, 7))
                sns.heatmap(
                    direct_heatmap,
                    annot=True,
                    fmt=".2%",
                    cmap="rocket_r",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Attack Success Rate"},
                )
                plt.title(f"Direct Attack Success Heatmap - {dataset}", fontsize=14, fontweight="bold")
                plt.xlabel("Attack Method", fontweight="bold")
                plt.ylabel("Model Variant", fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"direct_attack_success_heatmap_{dataset}.png"), dpi=300)
                plt.close()

            cls_rows = []
            for model_name in sorted(df_dataset["model"].unique()):
                for attack_name in sorted(df_dataset["attack"].unique()):
                    subset = df_dataset[(df_dataset["model"] == model_name) & (df_dataset["attack"] == attack_name)]
                    if {"Original", "Finetuned"}.issubset(set(subset["kind"])):
                        finetuned_value = float(subset[subset["kind"] == "Finetuned"]["attack_success_rate"].mean())
                        original_value = float(subset[subset["kind"] == "Original"]["attack_success_rate"].mean())
                        cls_rows.append(
                            {
                                "model": model_name,
                                "attack": attack_name,
                                "collapsed_minus_original": finetuned_value - original_value,
                            }
                        )

            if cls_rows:
                delta_df = pd.DataFrame(cls_rows)
                delta_heatmap = delta_df.pivot_table(
                    index="model", columns="attack", values="collapsed_minus_original", aggfunc="mean"
                )
                plt.figure(figsize=(10, 7))
                sns.heatmap(
                    delta_heatmap,
                    annot=True,
                    fmt=".2%",
                    cmap="coolwarm",
                    center=0,
                    cbar_kws={"label": "Collapsed - Original Attack Success"},
                )
                plt.title(f"Collapsed vs Original Susceptibility Delta - {dataset}", fontsize=14, fontweight="bold")
                plt.xlabel("Attack Method", fontweight="bold")
                plt.ylabel("Model Architecture", fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"collapsed_vs_original_delta_{dataset}.png"), dpi=300)
                plt.close()

    @staticmethod
    def _plot_transferability(output_dir: str, tf_df: pd.DataFrame) -> None:
        run_values = tf_df["run"].unique() if "run" in tf_df.columns else [None]

        for run in run_values:
            tf_subset_run = tf_df if run is None else tf_df[tf_df["run"] == run]
            run_suffix = f"_{run}" if run is not None else ""

            for dataset in tf_subset_run["dataset"].unique():
                for attack in tf_subset_run["source_attack"].unique():
                    tf_subset = tf_subset_run[
                        (tf_subset_run["dataset"] == dataset) & (tf_subset_run["source_attack"] == attack)
                    ]

                    pivot_data = tf_subset.pivot_table(
                        index="source_model",
                        columns="target_model",
                        values="transfer_success_rate",
                        aggfunc="mean",
                    )
                    if pivot_data is not None and not pivot_data.empty:
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(
                            pivot_data,
                            annot=True,
                            fmt=".2%",
                            cmap="mako",
                            vmin=0,
                            vmax=1,
                            cbar_kws={"label": "Transfer Success Rate"},
                        )
                        plt.title(f"Adversarial Transferability - {dataset} ({attack}){run_suffix}", fontsize=14, fontweight="bold")
                        plt.xlabel("Target Model", fontweight="bold")
                        plt.ylabel("Source Model", fontweight="bold")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"transferability_heatmap_{dataset}_{attack}{run_suffix}.png"), dpi=300)
                        plt.close()

                    full_pivot = tf_subset.pivot_table(
                        index="source_label",
                        columns="target_label",
                        values="transfer_success_rate",
                        aggfunc="mean",
                    )
                    if full_pivot is not None and not full_pivot.empty:
                        full_pivot.to_csv(os.path.join(output_dir, f"transferability_matrix_full_{dataset}_{attack}{run_suffix}.csv"))
                        plt.figure(figsize=(14, 10))
                        sns.heatmap(
                            full_pivot,
                            annot=True,
                            fmt=".2%",
                            cmap="mako",
                            vmin=0,
                            vmax=1,
                            cbar_kws={"label": "Transfer Success Rate"},
                        )
                        plt.title(f"Full Transferability Matrix - {dataset} ({attack}){run_suffix}", fontsize=14, fontweight="bold")
                        plt.xlabel("Target Model Variant", fontweight="bold")
                        plt.ylabel("Source Model Variant", fontweight="bold")
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"transferability_heatmap_full_{dataset}_{attack}{run_suffix}.png"), dpi=300)
                        plt.close()

                    pair_summary = tf_subset.groupby("pair_type", as_index=False)["normalized_transfer_rate"].mean()
                    if not pair_summary.empty:
                        plt.figure(figsize=(10, 5))
                        sns.barplot(data=pair_summary, x="pair_type", y="normalized_transfer_rate", palette="crest", legend=False)
                        plt.ylabel("Normalized Transfer Rate", fontweight="bold")
                        plt.xlabel("Source/Target Pair Type", fontweight="bold")
                        plt.title(f"Normalized Transferability by Pair Type - {dataset} ({attack}){run_suffix}", fontsize=14, fontweight="bold")
                        plt.xticks(rotation=20)
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f"transferability_pairtype_{dataset}_{attack}{run_suffix}.png"), dpi=300)
                        plt.close()
