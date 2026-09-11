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


BASELINE_KIND = "Control_Continuted"
VARIANT_KINDS = [
    "Dynamic_Region_All_Combined",
    "Dynamic_Region_All_Combined_quant",
]


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
        # Clean up the temporary per‑run CSV files now that they have been merged.
        for path in csv_paths:
            try:
                os.remove(path)
            except OSError:
                # If removal fails (e.g., file in use), we simply continue.
                pass

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
        kind_map = {
            "Control_Continuted": "Control Continued",
            "Dynamic_Region_All_Combined": "Pruned Finetuned",
            "Dynamic_Region_All_Combined_quant": "Pruned Finetuned Quantized",
        }
        return f"{model_name} ({kind_map.get(kind, kind)})"

    @staticmethod
    def baseline_kind() -> str:
        return BASELINE_KIND

    @staticmethod
    def variant_kinds() -> list[str]:
        return list(VARIANT_KINDS)

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

        baseline_profile = profile[profile["kind"] == cls.baseline_kind()].rename(
            columns={
                "clean_acc": "baseline_clean_acc",
                "robust_accuracy": "baseline_robust_accuracy_mean",
                "attack_success_rate": "baseline_attack_success_rate_mean",
                "param_count": "baseline_param_count",
            }
        )[[
            "model",
            "dataset",
            "baseline_clean_acc",
            "baseline_robust_accuracy_mean",
            "baseline_attack_success_rate_mean",
            "baseline_param_count",
        ]]

        acc_param_frames = []
        for variant_kind in cls.variant_kinds():
            variant_profile = profile[profile["kind"] == variant_kind].rename(
                columns={
                    "clean_acc": "variant_clean_acc",
                    "robust_accuracy": "variant_robust_accuracy_mean",
                    "attack_success_rate": "variant_attack_success_rate_mean",
                    "param_count": "variant_param_count",
                }
            )[[
                "model",
                "dataset",
                "variant_clean_acc",
                "variant_robust_accuracy_mean",
                "variant_attack_success_rate_mean",
                "variant_param_count",
            ]]
            merged = baseline_profile.merge(variant_profile, on=["model", "dataset"], how="inner")
            if merged.empty:
                continue
            merged["baseline_kind"] = cls.baseline_kind()
            merged["variant_kind"] = variant_kind
            merged["variant_minus_baseline_clean_acc"] = (
                merged["variant_clean_acc"] - merged["baseline_clean_acc"]
            )
            merged["params_reduction_percent"] = np.where(
                merged["baseline_param_count"] > 0,
                100.0 * (1.0 - merged["variant_param_count"] / merged["baseline_param_count"]),
                np.nan,
            )
            acc_param_frames.append(merged)

        acc_param_df = pd.concat(acc_param_frames, ignore_index=True) if acc_param_frames else pd.DataFrame()
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

        baseline_attack = by_attack[by_attack["kind"] == cls.baseline_kind()].rename(
            columns={
                "clean_acc": "baseline_clean_acc",
                "robust_accuracy": "baseline_robust_accuracy",
                "attack_success_rate": "baseline_attack_success_rate",
                "relative_accuracy_drop": "baseline_relative_accuracy_drop",
                "robustness_ratio": "baseline_robustness_ratio",
                "param_count": "baseline_param_count",
            }
        )[[
            "model",
            "dataset",
            "attack",
            "baseline_clean_acc",
            "baseline_robust_accuracy",
            "baseline_attack_success_rate",
            "baseline_relative_accuracy_drop",
            "baseline_robustness_ratio",
            "baseline_param_count",
        ]]

        explainability_frames = []
        for variant_kind in cls.variant_kinds():
            variant_attack = by_attack[by_attack["kind"] == variant_kind].rename(
                columns={
                    "clean_acc": "variant_clean_acc",
                    "robust_accuracy": "variant_robust_accuracy",
                    "attack_success_rate": "variant_attack_success_rate",
                    "relative_accuracy_drop": "variant_relative_accuracy_drop",
                    "robustness_ratio": "variant_robustness_ratio",
                    "param_count": "variant_param_count",
                }
            )[[
                "model",
                "dataset",
                "attack",
                "variant_clean_acc",
                "variant_robust_accuracy",
                "variant_attack_success_rate",
                "variant_relative_accuracy_drop",
                "variant_robustness_ratio",
                "variant_param_count",
            ]]

            merged = baseline_attack.merge(variant_attack, on=["model", "dataset", "attack"], how="inner")
            if merged.empty:
                continue
            merged["baseline_kind"] = cls.baseline_kind()
            merged["variant_kind"] = variant_kind
            merged["variant_minus_baseline_attack_success_rate"] = (
                merged["variant_attack_success_rate"] - merged["baseline_attack_success_rate"]
            )
            merged["variant_minus_baseline_robust_accuracy"] = (
                merged["variant_robust_accuracy"] - merged["baseline_robust_accuracy"]
            )
            merged["variant_minus_baseline_relative_accuracy_drop"] = (
                merged["variant_relative_accuracy_drop"] - merged["baseline_relative_accuracy_drop"]
            )
            merged["variant_minus_baseline_robustness_ratio"] = (
                merged["variant_robustness_ratio"] - merged["baseline_robustness_ratio"]
            )
            merged["params_reduction_percent"] = np.where(
                merged["baseline_param_count"] > 0,
                100.0 * (1.0 - merged["variant_param_count"] / merged["baseline_param_count"]),
                np.nan,
            )
            explainability_frames.append(merged)

        explainability_df = pd.concat(explainability_frames, ignore_index=True) if explainability_frames else pd.DataFrame()

        explainability_path = os.path.join(
            output_dir,
            "collapsed_vs_original_explainability_by_attack.csv",
        )
        explainability_df.to_csv(explainability_path, index=False)
        print(f"[INFO] Saved: {explainability_path}")

        explainability_summary_df = (
            explainability_df.groupby(["model", "dataset", "baseline_kind", "variant_kind"], as_index=False)
            .agg(
                baseline_clean_acc=("baseline_clean_acc", "mean"),
                variant_clean_acc=("variant_clean_acc", "mean"),
                baseline_param_count=("baseline_param_count", "mean"),
                variant_param_count=("variant_param_count", "mean"),
                params_reduction_percent=("params_reduction_percent", "mean"),
                mean_baseline_attack_success_rate=("baseline_attack_success_rate", "mean"),
                mean_variant_attack_success_rate=("variant_attack_success_rate", "mean"),
                mean_delta_attack_success_rate=("variant_minus_baseline_attack_success_rate", "mean"),
                mean_delta_robust_accuracy=("variant_minus_baseline_robust_accuracy", "mean"),
                mean_delta_relative_accuracy_drop=("variant_minus_baseline_relative_accuracy_drop", "mean"),
                mean_delta_robustness_ratio=("variant_minus_baseline_robustness_ratio", "mean"),
            )
        )

        shap_summary_path = os.path.join(output_dir, "shap_original_vs_collapsed_summary.csv")
        if os.path.exists(shap_summary_path):
            shap_df = pd.read_csv(shap_summary_path)
            if not shap_df.empty:
                shap_summary_df = (
                    shap_df.groupby(["source_model", "dataset", "source_kind", "target_kind"], as_index=False)
                    .agg(
                        mean_shap_cosine_similarity=("cosine_similarity", "mean"),
                        mean_shap_pearson_r=("pearson_r", "mean"),
                        mean_shap_spearman_r=("spearman_r", "mean"),
                        mean_shap_l1_mean_abs_diff=("l1_mean_abs_diff", "mean"),
                        mean_shap_l2_distance=("l2_distance", "mean"),
                        mean_shap_topk_jaccard=("topk_jaccard", "mean"),
                        mean_shap_topk_ratio=("topk_ratio", "mean"),
                        shap_pair_count=("cosine_similarity", "size"),
                    )
                    .rename(
                        columns={
                            "source_model": "model",
                            "source_kind": "baseline_kind",
                            "target_kind": "variant_kind",
                        }
                    )
                )
                explainability_summary_df = explainability_summary_df.merge(
                    shap_summary_df,
                    on=["model", "dataset", "baseline_kind", "variant_kind"],
                    how="left",
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
