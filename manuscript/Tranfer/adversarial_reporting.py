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
    def _safe_read_csv(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}")
            return pd.DataFrame()

    @staticmethod
    def _split_tag_from_results_dir(result_dir: str) -> str:
        # Example: adversarial_results_ep100_pre300 -> epochs100_pretrain300
        name = os.path.basename(os.path.normpath(result_dir))
        parts = name.split("_")
        if len(parts) >= 4 and parts[-2].startswith("ep") and parts[-1].startswith("pre"):
            ep = parts[-2].replace("ep", "")
            pre = parts[-1].replace("pre", "")
            if ep.isdigit() and pre.isdigit():
                return f"epochs{ep}_pretrain{pre}"
        return name

    @classmethod
    def _load_summary_for_result_dir(cls, result_dir: str) -> pd.DataFrame:
        summary_path = os.path.join(result_dir, "summary.csv")
        df = cls._safe_read_csv(summary_path)
        if not df.empty:
            return cls.enrich_summary_dataframe(df)

        # Fallback for sharded runs where only summary_*.csv files are present.
        shard_paths = glob.glob(os.path.join(result_dir, "summary_*.csv"))
        if not shard_paths:
            return pd.DataFrame()
        frames = []
        for path in shard_paths:
            sdf = cls._safe_read_csv(path)
            if not sdf.empty:
                frames.append(sdf)
        if not frames:
            return pd.DataFrame()
        return cls.enrich_summary_dataframe(pd.concat(frames, ignore_index=True).drop_duplicates())

    @classmethod
    def generate_multi_run_kind_comparison_table(
        cls,
        output_dir: str,
        result_dirs: list[str],
    ) -> bool:
        """Build one CSV across multiple run folders with pairwise kind deltas.

        Includes robust/attack deltas and SHAP similarity deltas for
        Control_Continuted, Dynamic_Region_All_Combined, and
        Dynamic_Region_All_Combined_quant pairs within each model+dataset.
        """
        if not result_dirs:
            print("[WARN] No result directories provided for multi-run comparison table.")
            return False

        os.makedirs(output_dir, exist_ok=True)
        kind_order = [cls.baseline_kind(), *cls.variant_kinds()]
        rows: list[dict] = []

        for result_dir in result_dirs:
            run_tag = cls._split_tag_from_results_dir(result_dir)

            summary_df = cls._load_summary_for_result_dir(result_dir)
            if summary_df.empty:
                print(f"[WARN] Skipping {result_dir}: summary data unavailable.")
                continue

            by_kind = (
                summary_df.groupby(["model", "dataset", "kind"], as_index=False)
                .agg(
                    clean_acc=("clean_acc", "mean"),
                    robust_accuracy=("robust_accuracy", "mean"),
                    attack_success_rate=("attack_success_rate", "mean"),
                    relative_accuracy_drop=("relative_accuracy_drop", "mean"),
                    robustness_ratio=("robustness_ratio", "mean"),
                    param_count=("param_count", "mean"),
                )
            )

            shap_pairwise_path = os.path.join(result_dir, "shap_pairwise_similarity.csv")
            shap_df = cls._safe_read_csv(shap_pairwise_path)
            shap_summary = pd.DataFrame()
            if not shap_df.empty:
                required_cols = {
                    "source_model",
                    "target_model",
                    "dataset",
                    "source_kind",
                    "target_kind",
                }
                if required_cols.issubset(shap_df.columns):
                    shap_same_arch = shap_df[
                        (shap_df["source_model"] == shap_df["target_model"])
                        & (shap_df["source_kind"] != shap_df["target_kind"])
                    ].copy()
                    if not shap_same_arch.empty:
                        shap_summary = (
                            shap_same_arch.groupby(
                                ["source_model", "dataset", "source_kind", "target_kind"],
                                as_index=False,
                            )
                            .agg(
                                shap_cosine_similarity=("cosine_similarity", "mean"),
                                shap_pearson_r=("pearson_r", "mean"),
                                shap_spearman_r=("spearman_r", "mean"),
                                shap_l1_mean_abs_diff=("l1_mean_abs_diff", "mean"),
                                shap_l2_distance=("l2_distance", "mean"),
                                shap_topk_jaccard=("topk_jaccard", "mean"),
                                shap_pair_count=("cosine_similarity", "size"),
                            )
                            .rename(columns={"source_model": "model"})
                        )

            for (model_name, dataset_name), group_df in by_kind.groupby(["model", "dataset"]):
                kind_lookup = {
                    row["kind"]: row for _, row in group_df.iterrows()
                }

                for i, source_kind in enumerate(kind_order):
                    for target_kind in kind_order[i + 1 :]:
                        source_row = kind_lookup.get(source_kind)
                        target_row = kind_lookup.get(target_kind)
                        if source_row is None or target_row is None:
                            continue

                        row = {
                            "result_dir": os.path.basename(os.path.normpath(result_dir)),
                            "run_tag": run_tag,
                            "model": model_name,
                            "dataset": dataset_name,
                            "source_kind": source_kind,
                            "target_kind": target_kind,
                            "source_clean_acc": float(source_row["clean_acc"]),
                            "target_clean_acc": float(target_row["clean_acc"]),
                            "source_robust_accuracy": float(source_row["robust_accuracy"]),
                            "target_robust_accuracy": float(target_row["robust_accuracy"]),
                            "delta_robust_accuracy": float(target_row["robust_accuracy"] - source_row["robust_accuracy"]),
                            "source_attack_success_rate": float(source_row["attack_success_rate"]),
                            "target_attack_success_rate": float(target_row["attack_success_rate"]),
                            "delta_attack_success_rate": float(target_row["attack_success_rate"] - source_row["attack_success_rate"]),
                            "source_relative_accuracy_drop": float(source_row["relative_accuracy_drop"]),
                            "target_relative_accuracy_drop": float(target_row["relative_accuracy_drop"]),
                            "delta_relative_accuracy_drop": float(target_row["relative_accuracy_drop"] - source_row["relative_accuracy_drop"]),
                            "source_robustness_ratio": float(source_row["robustness_ratio"]),
                            "target_robustness_ratio": float(target_row["robustness_ratio"]),
                            "delta_robustness_ratio": float(target_row["robustness_ratio"] - source_row["robustness_ratio"]),
                            "source_param_count": float(source_row["param_count"]),
                            "target_param_count": float(target_row["param_count"]),
                            "params_reduction_percent": float(
                                100.0 * (1.0 - target_row["param_count"] / source_row["param_count"])
                            ) if source_row["param_count"] > 0 else np.nan,
                            "shap_cosine_similarity": np.nan,
                            "shap_pearson_r": np.nan,
                            "shap_spearman_r": np.nan,
                            "shap_l1_mean_abs_diff": np.nan,
                            "shap_l2_distance": np.nan,
                            "shap_topk_jaccard": np.nan,
                            "shap_pair_count": np.nan,
                        }

                        if not shap_summary.empty:
                            match = shap_summary[
                                (shap_summary["model"] == model_name)
                                & (shap_summary["dataset"] == dataset_name)
                                & (shap_summary["source_kind"] == source_kind)
                                & (shap_summary["target_kind"] == target_kind)
                            ]
                            if not match.empty:
                                best = match.iloc[0]
                                row.update(
                                    {
                                        "shap_cosine_similarity": float(best["shap_cosine_similarity"]),
                                        "shap_pearson_r": float(best["shap_pearson_r"]),
                                        "shap_spearman_r": float(best["shap_spearman_r"]),
                                        "shap_l1_mean_abs_diff": float(best["shap_l1_mean_abs_diff"]),
                                        "shap_l2_distance": float(best["shap_l2_distance"]),
                                        "shap_topk_jaccard": float(best["shap_topk_jaccard"]),
                                        "shap_pair_count": float(best["shap_pair_count"]),
                                    }
                                )

                        rows.append(row)

        if not rows:
            print("[WARN] Multi-run kind comparison table not generated: no rows available.")
            return False

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(output_dir, "multi_run_kind_comparison_table.csv")
        out_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved: {out_path}")
        return True

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
