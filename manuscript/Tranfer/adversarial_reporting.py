"""Reporting utilities for adversarial analysis.

This module owns dataframe enrichment, CSV merge helpers, and explainability
comparison tables so the main analysis script can stay focused on attack
execution and transfer evaluation.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


class ReportingSuite:
    """Encapsulates reporting and explainability table generation."""

    @staticmethod
    def merge_parallel_csvs(output_dir: str, base_name: str = "summary") -> None:
        pattern = os.path.join(output_dir, f"{base_name}_*.csv")
        csv_paths = glob.glob(pattern)
        if not csv_paths:
            return

        dfs = []
        for path in csv_paths:
            try:
                dfs.append(pd.read_csv(path))
            except Exception as exc:
                print(f"[WARN] Failed to read {path}: {exc}")

        if not dfs:
            return

        merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
        merged_path = os.path.join(output_dir, f"{base_name}.csv")
        merged.to_csv(merged_path, index=False)
        print(f"[INFO] Merged {len(csv_paths)} files into {merged_path}")

    @staticmethod
    def load_multiple_runs(root_dir: str) -> Tuple[List[Dict], List[Dict]]:
        summary_records: List[Dict] = []
        transfer_records: List[Dict] = []

        result_dirs = glob.glob(os.path.join(root_dir, "adversarial_results_*/"))
        if not result_dirs:
            print(f"[WARN] No result directories found under {root_dir}")
            return summary_records, transfer_records

        for run_dir in result_dirs:
            run_name = os.path.basename(os.path.normpath(run_dir))
            summary_path = os.path.join(run_dir, "summary.csv")
            transfer_path = os.path.join(run_dir, "transferability.csv")
            try:
                if os.path.exists(summary_path):
                    df_sum = pd.read_csv(summary_path)
                    df_sum["run"] = run_name
                    summary_records.extend(df_sum.to_dict(orient="records"))
                if os.path.exists(transfer_path):
                    df_tf = pd.read_csv(transfer_path)
                    df_tf["run"] = run_name
                    transfer_records.extend(df_tf.to_dict(orient="records"))
            except Exception as exc:
                print(f"[ERROR] Failed to load CSVs from {run_dir}: {exc}")

        print(
            f"[INFO] Loaded {len(summary_records)} summary rows and "
            f"{len(transfer_records)} transfer rows from {len(result_dirs)} runs"
        )
        return summary_records, transfer_records

    @staticmethod
    def model_kind_label(model_name: str, kind: str) -> str:
        return f"{model_name} ({kind})"

    @staticmethod
    def summarize_direct_metrics(clean_acc: float, adv_acc: float) -> dict:
        accuracy_drop = clean_acc - adv_acc
        attack_success_rate = 1.0 - adv_acc
        relative_accuracy_drop = accuracy_drop / clean_acc if clean_acc > 0 else np.nan
        robustness_ratio = adv_acc / clean_acc if clean_acc > 0 else np.nan
        return {
            "robust_accuracy": adv_acc,
            "clean_error_rate": 1.0 - clean_acc,
            "adv_error_rate": 1.0 - adv_acc,
            "attack_success_rate": attack_success_rate,
            "accuracy_drop": accuracy_drop,
            "relative_accuracy_drop": relative_accuracy_drop,
            "robustness_ratio": robustness_ratio,
        }

    @staticmethod
    def classify_transfer_pair(
        src_model: str,
        src_kind: str,
        tgt_model: str,
        tgt_kind: str,
    ) -> str:
        if src_model == tgt_model and src_kind == tgt_kind:
            return "self_same_kind"
        if src_model == tgt_model:
            return "same_arch_cross_kind"
        if src_kind == tgt_kind:
            return "cross_arch_same_kind"
        return "cross_arch_cross_kind"

    @classmethod
    def enrich_summary_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if "model_label" not in df.columns and {"model", "kind"}.issubset(df.columns):
            df["model_label"] = df.apply(
                lambda row: cls.model_kind_label(row["model"], row["kind"]),
                axis=1,
            )

        if "accuracy_drop" not in df.columns and {"clean_acc", "adv_acc"}.issubset(df.columns):
            df["accuracy_drop"] = df["clean_acc"] - df["adv_acc"]

        if "robust_accuracy" not in df.columns and "adv_acc" in df.columns:
            df["robust_accuracy"] = df["adv_acc"]

        if "clean_error_rate" not in df.columns and "clean_acc" in df.columns:
            df["clean_error_rate"] = 1.0 - df["clean_acc"]

        if "adv_error_rate" not in df.columns and "adv_acc" in df.columns:
            df["adv_error_rate"] = 1.0 - df["adv_acc"]

        if "attack_success_rate" not in df.columns and "adv_acc" in df.columns:
            df["attack_success_rate"] = 1.0 - df["adv_acc"]

        if "relative_accuracy_drop" not in df.columns and {"accuracy_drop", "clean_acc"}.issubset(df.columns):
            df["relative_accuracy_drop"] = np.where(
                df["clean_acc"] > 0,
                df["accuracy_drop"] / df["clean_acc"],
                np.nan,
            )

        if "robustness_ratio" not in df.columns and {"adv_acc", "clean_acc"}.issubset(df.columns):
            df["robustness_ratio"] = np.where(
                df["clean_acc"] > 0,
                df["adv_acc"] / df["clean_acc"],
                np.nan,
            )

        return df

    @classmethod
    def enrich_transfer_dataframe(
        cls,
        tf_df: pd.DataFrame,
        records_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if tf_df.empty:
            return tf_df

        if "source_label" not in tf_df.columns:
            tf_df["source_label"] = tf_df.apply(
                lambda row: cls.model_kind_label(row["source_model"], row["source_kind"]),
                axis=1,
            )

        if "target_label" not in tf_df.columns:
            tf_df["target_label"] = tf_df.apply(
                lambda row: cls.model_kind_label(row["target_model"], row["target_kind"]),
                axis=1,
            )

        if "transfer_success_rate" not in tf_df.columns and "transfer_acc" in tf_df.columns:
            tf_df["transfer_success_rate"] = 1.0 - tf_df["transfer_acc"]

        if "same_architecture" not in tf_df.columns:
            tf_df["same_architecture"] = tf_df["source_model"] == tf_df["target_model"]

        if "same_kind" not in tf_df.columns:
            tf_df["same_kind"] = tf_df["source_kind"] == tf_df["target_kind"]

        if "pair_type" not in tf_df.columns:
            tf_df["pair_type"] = tf_df.apply(
                lambda row: cls.classify_transfer_pair(
                    row["source_model"],
                    row["source_kind"],
                    row["target_model"],
                    row["target_kind"],
                ),
                axis=1,
            )

        if (
            records_df is not None
            and not records_df.empty
            and "source_attack_success_rate" not in tf_df.columns
        ):
            lookup = records_df[
                ["model", "dataset", "kind", "attack", "attack_success_rate"]
            ].rename(
                columns={
                    "model": "source_model",
                    "dataset": "dataset",
                    "kind": "source_kind",
                    "attack": "source_attack",
                    "attack_success_rate": "source_attack_success_rate",
                }
            )
            tf_df = tf_df.merge(
                lookup,
                on=["source_model", "dataset", "source_kind", "source_attack"],
                how="left",
            )

        if (
            "normalized_transfer_rate" not in tf_df.columns
            and {"transfer_success_rate", "source_attack_success_rate"}.issubset(tf_df.columns)
        ):
            tf_df["normalized_transfer_rate"] = np.where(
                tf_df["source_attack_success_rate"] > 0,
                tf_df["transfer_success_rate"] / tf_df["source_attack_success_rate"],
                np.nan,
            )

        return tf_df

    @classmethod
    def generate_comparison_tables(cls, output_dir: str, records: List[Dict]) -> None:
        if not records:
            return

        os.makedirs(output_dir, exist_ok=True)
        df = cls.enrich_summary_dataframe(pd.DataFrame(records))
        if df.empty:
            return

        if "param_count" not in df.columns:
            df["param_count"] = np.nan

        profile = (
            df.groupby(["model", "dataset", "kind"], as_index=False)
            .agg(
                clean_acc=("clean_acc", "mean"),
                robust_accuracy=("robust_accuracy", "mean"),
                attack_success_rate=("attack_success_rate", "mean"),
                param_count=("param_count", "mean"),
            )
        )

        orig_profile = profile[profile["kind"] == "Original"].rename(
            columns={
                "clean_acc": "original_clean_acc",
                "robust_accuracy": "original_robust_accuracy_mean",
                "attack_success_rate": "original_attack_success_rate_mean",
                "param_count": "original_param_count",
            }
        )[[
            "model",
            "dataset",
            "original_clean_acc",
            "original_robust_accuracy_mean",
            "original_attack_success_rate_mean",
            "original_param_count",
        ]]

        finetuned_profile = profile[profile["kind"] == "Finetuned"].rename(
            columns={
                "clean_acc": "collapsed_clean_acc",
                "robust_accuracy": "collapsed_robust_accuracy_mean",
                "attack_success_rate": "collapsed_attack_success_rate_mean",
                "param_count": "collapsed_param_count",
            }
        )[[
            "model",
            "dataset",
            "collapsed_clean_acc",
            "collapsed_robust_accuracy_mean",
            "collapsed_attack_success_rate_mean",
            "collapsed_param_count",
        ]]

        acc_param_df = orig_profile.merge(
            finetuned_profile,
            on=["model", "dataset"],
            how="outer",
        )
        acc_param_df["collapsed_minus_original_clean_acc"] = (
            acc_param_df["collapsed_clean_acc"] - acc_param_df["original_clean_acc"]
        )
        acc_param_df["params_reduction_percent"] = np.where(
            acc_param_df["original_param_count"] > 0,
            100.0
            * (
                1.0
                - acc_param_df["collapsed_param_count"]
                / acc_param_df["original_param_count"]
            ),
            np.nan,
        )
        acc_param_path = os.path.join(output_dir, "accuracy_parameter_comparison.csv")
        acc_param_df.to_csv(acc_param_path, index=False)
        print(f"[INFO] Saved: {acc_param_path}")

        by_attack = (
            df.groupby(["model", "dataset", "attack", "kind"], as_index=False)
            .agg(
                clean_acc=("clean_acc", "mean"),
                robust_accuracy=("robust_accuracy", "mean"),
                attack_success_rate=("attack_success_rate", "mean"),
                relative_accuracy_drop=("relative_accuracy_drop", "mean"),
                robustness_ratio=("robustness_ratio", "mean"),
                param_count=("param_count", "mean"),
            )
        )

        orig_attack = by_attack[by_attack["kind"] == "Original"].rename(
            columns={
                "clean_acc": "original_clean_acc",
                "robust_accuracy": "original_robust_accuracy",
                "attack_success_rate": "original_attack_success_rate",
                "relative_accuracy_drop": "original_relative_accuracy_drop",
                "robustness_ratio": "original_robustness_ratio",
                "param_count": "original_param_count",
            }
        )[[
            "model",
            "dataset",
            "attack",
            "original_clean_acc",
            "original_robust_accuracy",
            "original_attack_success_rate",
            "original_relative_accuracy_drop",
            "original_robustness_ratio",
            "original_param_count",
        ]]

        finetuned_attack = by_attack[by_attack["kind"] == "Finetuned"].rename(
            columns={
                "clean_acc": "collapsed_clean_acc",
                "robust_accuracy": "collapsed_robust_accuracy",
                "attack_success_rate": "collapsed_attack_success_rate",
                "relative_accuracy_drop": "collapsed_relative_accuracy_drop",
                "robustness_ratio": "collapsed_robustness_ratio",
                "param_count": "collapsed_param_count",
            }
        )[[
            "model",
            "dataset",
            "attack",
            "collapsed_clean_acc",
            "collapsed_robust_accuracy",
            "collapsed_attack_success_rate",
            "collapsed_relative_accuracy_drop",
            "collapsed_robustness_ratio",
            "collapsed_param_count",
        ]]

        explainability_df = orig_attack.merge(
            finetuned_attack,
            on=["model", "dataset", "attack"],
            how="inner",
        )
        explainability_df["collapsed_minus_original_attack_success_rate"] = (
            explainability_df["collapsed_attack_success_rate"]
            - explainability_df["original_attack_success_rate"]
        )
        explainability_df["collapsed_minus_original_robust_accuracy"] = (
            explainability_df["collapsed_robust_accuracy"]
            - explainability_df["original_robust_accuracy"]
        )
        explainability_df["collapsed_minus_original_relative_accuracy_drop"] = (
            explainability_df["collapsed_relative_accuracy_drop"]
            - explainability_df["original_relative_accuracy_drop"]
        )
        explainability_df["collapsed_minus_original_robustness_ratio"] = (
            explainability_df["collapsed_robustness_ratio"]
            - explainability_df["original_robustness_ratio"]
        )
        explainability_df["params_reduction_percent"] = np.where(
            explainability_df["original_param_count"] > 0,
            100.0
            * (
                1.0
                - explainability_df["collapsed_param_count"]
                / explainability_df["original_param_count"]
            ),
            np.nan,
        )

        explainability_path = os.path.join(
            output_dir,
            "collapsed_vs_original_explainability_by_attack.csv",
        )
        explainability_df.to_csv(explainability_path, index=False)
        print(f"[INFO] Saved: {explainability_path}")

        explainability_summary_df = (
            explainability_df.groupby(["model", "dataset"], as_index=False)
            .agg(
                original_clean_acc=("original_clean_acc", "mean"),
                collapsed_clean_acc=("collapsed_clean_acc", "mean"),
                original_param_count=("original_param_count", "mean"),
                collapsed_param_count=("collapsed_param_count", "mean"),
                params_reduction_percent=("params_reduction_percent", "mean"),
                mean_original_attack_success_rate=("original_attack_success_rate", "mean"),
                mean_collapsed_attack_success_rate=("collapsed_attack_success_rate", "mean"),
                mean_delta_attack_success_rate=("collapsed_minus_original_attack_success_rate", "mean"),
                mean_delta_robust_accuracy=("collapsed_minus_original_robust_accuracy", "mean"),
                mean_delta_relative_accuracy_drop=("collapsed_minus_original_relative_accuracy_drop", "mean"),
                mean_delta_robustness_ratio=("collapsed_minus_original_robustness_ratio", "mean"),
            )
        )
        explainability_summary_path = os.path.join(
            output_dir,
            "collapsed_vs_original_explainability_summary.csv",
        )
        explainability_summary_df.to_csv(explainability_summary_path, index=False)
        print(f"[INFO] Saved: {explainability_summary_path}")

    @classmethod
    def generate_comparison_tables_from_csv(cls, output_dir: str) -> bool:
        """Generate explainability and profile tables from summary CSV.

        Returns True when data was found and tables were generated.
        """
        summary_path = os.path.join(output_dir, "summary.csv")
        if not os.path.exists(summary_path):
            print(f"[WARN] No summary.csv found at {summary_path}; comparison tables skipped.")
            return False

        df_summary = pd.read_csv(summary_path)
        if df_summary.empty:
            print(f"[WARN] summary.csv at {summary_path} is empty; comparison tables skipped.")
            return False

        cls.generate_comparison_tables(output_dir, df_summary.to_dict(orient="records"))
        return True
