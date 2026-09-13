"""Compute-cost tradeoff experiments and figures."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from adversarial_checkpointing import CheckpointManager
from adversarial_core import AdversarialCore
from adversarial_reporting import ReportingSuite


class ComputeTradeoffSuite:
    """Compute profiling and robustness-efficiency tradeoff analysis."""

    @staticmethod
    def _estimate_flops(model, sample_batch: torch.Tensor) -> float:
        try:
            from ptflops import get_model_complexity_info

            base_model = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                macs, _ = get_model_complexity_info(
                    base_model,
                    tuple(sample_batch.shape[1:]),
                    as_strings=False,
                    print_per_layer_stat=False,
                    verbose=False,
                )
            return float(macs)
        except Exception:
            return float("nan")

    @staticmethod
    def _measure_latency_and_throughput(model, loader, warmup: int = 5, timed: int = 20) -> tuple[float, float, float]:
        model.eval()
        device = next((model.module if hasattr(model, "module") else model).parameters()).device

        batches = []
        for idx, (imgs, _) in enumerate(loader):
            batches.append(imgs.to(device))
            if idx >= max(warmup + timed, 10):
                break

        for imgs in batches[:warmup]:
            with torch.no_grad():
                _ = model(imgs)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        seen = 0
        for imgs in batches[warmup : warmup + timed]:
            with torch.no_grad():
                _ = model(imgs)
            seen += imgs.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        mean_latency_ms = (elapsed / max(1, timed)) * 1000.0
        throughput = seen / max(elapsed, 1e-9)
        peak_mem_mb = (
            torch.cuda.max_memory_allocated(device) / (1024.0**2)
            if device.type == "cuda"
            else float("nan")
        )
        return mean_latency_ms, throughput, peak_mem_mb

    @classmethod
    def run(cls, output_dir: str, model_cache: dict, loader_cache: dict) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        rows: list[dict] = []

        summary_path = os.path.join(output_dir, "summary.csv")
        transfer_path = os.path.join(output_dir, "transferability.csv")
        explain_path = os.path.join(output_dir, "collapsed_vs_original_explainability_summary.csv")

        # If analyze was run from parallel shards, summary may still be split as
        # summary_*.csv files. Merge them before loading to avoid empty metrics.
        if not os.path.exists(summary_path):
            ReportingSuite.merge_parallel_csvs(output_dir)

        summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        transfer_df = pd.read_csv(transfer_path) if os.path.exists(transfer_path) else pd.DataFrame()
        explain_df = pd.read_csv(explain_path) if os.path.exists(explain_path) else pd.DataFrame()

        # Checkpoint and transfer files may differ only in case, whitespace, or the
        # historical Continued/Continuted spelling. Use private normalized keys for joins.
        def _key(value):
            return str(value).strip().lower().replace("continued", "continuted")

        if not summary_df.empty:
            summary_df["_model_key"] = summary_df["model"].map(_key)
            summary_df["_dataset_key"] = summary_df["dataset"].map(_key)
            summary_df["_kind_key"] = summary_df["kind"].map(_key)
        if not transfer_df.empty:
            transfer_df["_target_model_key"] = transfer_df["target_model"].map(_key)
            transfer_df["_dataset_key"] = transfer_df["dataset"].map(_key)
            transfer_df["_target_kind_key"] = transfer_df["target_kind"].map(_key)
        if not explain_df.empty:
            explain_df["_model_key"] = explain_df["model"].map(_key)
            explain_df["_dataset_key"] = explain_df["dataset"].map(_key)
            explain_df["_variant_kind_key"] = explain_df["variant_kind"].map(_key)

        for (model_name, dataset_name, kind), model in model_cache.items():
            # dataset_name may include a split tag; use the base name for loader lookup.
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)
            if base_dataset not in loader_cache:
                print(f"[WARN] Tradeoff: no loader for {model_name}/{dataset_name}; skipping {kind}")
                continue
            _, test_loader = loader_cache[base_dataset]
            sample_batch = next(iter(test_loader))[0]

            param_count = AdversarialCore.count_model_parameters(model)
            flops = cls._estimate_flops(model, sample_batch)
            latency_ms, throughput, peak_mem_mb = cls._measure_latency_and_throughput(model, test_loader)

            robust_acc = float("nan")
            attack_success = float("nan")
            transfer_success = float("nan")
            explain_delta = float("nan")
            explain_cosine = float("nan")
            explain_pearson = float("nan")
            explain_spearman = float("nan")
            explain_topk = float("nan")
            explain_l1 = float("nan")
            explain_l2 = float("nan")
            explain_pair_count = float("nan")

            if not summary_df.empty:
                s = summary_df[
                    (summary_df["_model_key"] == _key(model_name))
                    & (summary_df["_dataset_key"] == _key(dataset_name))
                    & (summary_df["_kind_key"] == _key(kind))
                ]
                if not s.empty:
                    robust_acc = float(s["adv_acc"].mean())
                    attack_success = float(s["attack_success_rate"].mean())

            if not transfer_df.empty:
                t = transfer_df[
                    (transfer_df["_target_model_key"] == _key(model_name))
                    & (transfer_df["_dataset_key"] == _key(dataset_name))
                    & (transfer_df["_target_kind_key"] == _key(kind))
                ]
                if not t.empty:
                    transfer_success = float(t["transfer_success_rate"].mean())

            if not explain_df.empty and kind != ReportingSuite.baseline_kind():
                e = explain_df[
                    (explain_df["_model_key"] == _key(model_name))
                    & (explain_df["_dataset_key"] == _key(dataset_name))
                    & (explain_df["_variant_kind_key"] == _key(kind))
                ]
                if not e.empty:
                    explain_delta = float(e["mean_delta_attack_success_rate"].mean())
                    if "mean_shap_cosine_similarity" in e.columns:
                        explain_cosine = float(e["mean_shap_cosine_similarity"].mean())
                    if "mean_shap_pearson_r" in e.columns:
                        explain_pearson = float(e["mean_shap_pearson_r"].mean())
                    if "mean_shap_spearman_r" in e.columns:
                        explain_spearman = float(e["mean_shap_spearman_r"].mean())
                    if "mean_shap_topk_jaccard" in e.columns:
                        explain_topk = float(e["mean_shap_topk_jaccard"].mean())
                    if "mean_shap_l1_mean_abs_diff" in e.columns:
                        explain_l1 = float(e["mean_shap_l1_mean_abs_diff"].mean())
                    if "mean_shap_l2_distance" in e.columns:
                        explain_l2 = float(e["mean_shap_l2_distance"].mean())
                    if "shap_pair_count" in e.columns:
                        explain_pair_count = float(e["shap_pair_count"].mean())

            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "kind": kind,
                    "model_label": f"{model_name} ({kind})",
                    "param_count": param_count,
                    "flops": flops,
                    "latency_ms": latency_ms,
                    "throughput_imgs_s": throughput,
                    "peak_memory_mb": peak_mem_mb,
                    "robust_accuracy": robust_acc,
                    "attack_success_rate": attack_success,
                    "transfer_success_rate": transfer_success,
                    "transfer_resistance": 1.0 - transfer_success if not np.isnan(transfer_success) else np.nan,
                    "explainability_delta_asr": explain_delta,
                    "explainability_shap_cosine_similarity": explain_cosine,
                    "explainability_shap_pearson_r": explain_pearson,
                    "explainability_shap_spearman_r": explain_spearman,
                    "explainability_shap_topk_jaccard": explain_topk,
                    "explainability_shap_l1_mean_abs_diff": explain_l1,
                    "explainability_shap_l2_distance": explain_l2,
                    "explainability_shap_pair_count": explain_pair_count,
                }
            )

        df = pd.DataFrame(rows)
        if df.empty:
            print("[WARN] Compute tradeoff: no rows generated.")
            return rows

        df["robustness_per_flop"] = np.where(df["flops"] > 0, df["robust_accuracy"] / df["flops"], np.nan)
        df["transfer_resistance_per_latency"] = np.where(
            df["latency_ms"] > 0,
            df["transfer_resistance"] / df["latency_ms"],
            np.nan,
        )

        csv_path = os.path.join(output_dir, "compute_profile.csv")
        df.to_csv(csv_path, index=False)
        print(f"[COST] Saved: {csv_path}")

        cls._build_tradeoff_summary(output_dir, df)
        cls._plot_tradeoff_figures(output_dir, df)
        return rows

    @staticmethod
    def _build_tradeoff_summary(output_dir: str, df: pd.DataFrame) -> None:
        pivot = df[[
            "model",
            "dataset",
            "kind",
            "param_count",
            "flops",
            "latency_ms",
            "throughput_imgs_s",
            "peak_memory_mb",
            "robust_accuracy",
            "attack_success_rate",
            "transfer_success_rate",
            "explainability_delta_asr",
            "robustness_per_flop",
            "transfer_resistance_per_latency",
        ]].copy()

        summary_path = os.path.join(output_dir, "tradeoff_summary.csv")
        pivot.to_csv(summary_path, index=False)
        print(f"[COST] Saved: {summary_path}")

    @staticmethod
    def _plot_tradeoff_figures(output_dir: str, df: pd.DataFrame) -> None:
        # Figure 1: Pareto (FLOPs vs robust accuracy)
        fig1_df = df[(df["flops"] > 0) & np.isfinite(df["flops"]) & np.isfinite(df["robust_accuracy"])].copy()
        x_col = "flops"
        x_label = "FLOPs (log scale)"
        x_scale = "log"

        # Fallback when FLOPs estimation failed (e.g., ptflops unavailable).
        if fig1_df.empty:
            fig1_df = df[np.isfinite(df["param_count"]) & np.isfinite(df["robust_accuracy"])].copy()
            x_col = "param_count"
            x_label = "Parameter Count (log scale)"

        plt.figure(figsize=(8, 6))
        if not fig1_df.empty:
            sns.scatterplot(data=fig1_df, x=x_col, y="robust_accuracy", hue="kind", style="model", s=120)
            plt.xscale(x_scale)
        else:
            plt.text(0.5, 0.5, "No valid FLOPs/robustness data", ha="center", va="center")
        plt.xlabel(x_label, fontweight="bold")
        plt.ylabel("Robust Accuracy", fontweight="bold")
        plt.title("Figure 1: Robustness-Compute Pareto", fontweight="bold")
        plt.tight_layout()
        # Include the output directory name in the figure filename so that each results
        # folder gets its own distinct copy when multiple pipelines are run.
        prefix = os.path.basename(output_dir)
        plt.savefig(os.path.join(output_dir, f"{prefix}_figure1_pareto_flops_vs_robust_accuracy.png"), dpi=300)
        plt.close()

        # Figure 2: Transfer resistance vs latency
        fig2_df = df[
            np.isfinite(df["latency_ms"]) & np.isfinite(df["transfer_resistance"])
        ].copy()
        plt.figure(figsize=(8, 6))
        if not fig2_df.empty:
            sizes = np.clip(fig2_df["param_count"] / max(1.0, fig2_df["param_count"].max()) * 700, 60, 700)
            plt.scatter(fig2_df["latency_ms"], fig2_df["transfer_resistance"], s=sizes, alpha=0.7)
            for _, row in fig2_df.iterrows():
                plt.annotate(row["model_label"], (row["latency_ms"], row["transfer_resistance"]), fontsize=7)
        else:
            plt.text(0.5, 0.5, "No valid transfer/latency data", ha="center", va="center")
        plt.xlabel("Latency (ms / batch)", fontweight="bold")
        plt.ylabel("Transfer Resistance (1 - transfer success)", fontweight="bold")
        plt.title("Figure 2: Transfer Resistance vs Latency", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_figure2_transfer_resistance_vs_latency.png"), dpi=300)
        plt.close()

        # Figure 3: Variant vs baseline deltas for core metrics.
        # Keep metrics in compatible units: accuracy/resistance are absolute deltas;
        # latency/FLOPs are percentage changes. Missing metrics are omitted, not plotted as zero.
        delta_rows = []
        for (model_name, dataset_name), g in df.groupby(["model", "dataset"]):
            baseline = g[g["kind"] == ReportingSuite.baseline_kind()]
            if baseline.empty:
                continue
            for variant_kind in ReportingSuite.variant_kinds():
                variant = g[g["kind"] == variant_kind]
                if variant.empty:
                    continue
                label = f"{model_name} | {dataset_name} | {variant_kind}"
                pairs = {
                    "robust_accuracy": ("Robust accuracy delta", False),
                    "transfer_resistance": ("Transfer resistance delta", False),
                    "latency_ms": ("Latency change (%)", True),
                    "flops": ("FLOPs change (%)", True),
                }
                for column, (metric, relative) in pairs.items():
                    b = pd.to_numeric(baseline[column], errors="coerce").mean()
                    v = pd.to_numeric(variant[column], errors="coerce").mean()
                    if not np.isfinite(b) or not np.isfinite(v):
                        continue
                    if relative:
                        if b == 0:
                            continue
                        delta = 100.0 * (v - b) / b
                    else:
                        delta = v - b
                    delta_rows.append({"label": label, "metric": metric, "delta": delta})

        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            metric_order = delta_df["metric"].unique().tolist()
            fig, axes = plt.subplots(len(metric_order), 1, figsize=(14, 3.2 * len(metric_order)), squeeze=False)
            for ax, metric in zip(axes[:, 0], metric_order):
                sub = delta_df[delta_df["metric"] == metric]
                sns.barplot(data=sub, x="label", y="delta", ax=ax, color="#0082c9")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_title(metric)
                ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=60, labelsize=7)
            fig.suptitle("Figure 3: Variant vs Control Continued Metric Deltas", fontweight="bold")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{prefix}_figure3_collapsed_original_deltas.png"), dpi=300)
            plt.close(fig)

        # Figure 4: normalized tradeoff heatmap. Keep rows unique and preserve NA cells.
        plot_cols = [
            "robust_accuracy", "transfer_resistance",
            "explainability_delta_asr",
            "explainability_shap_cosine_similarity",
            "explainability_shap_topk_jaccard",
            "flops", "latency_ms", "peak_memory_mb",
        ]
        plot_cols = [c for c in plot_cols if c in df.columns and df[c].notna().any()]
        if plot_cols:
            heat_df = df.copy()
            heat_df["row_label"] = (
                heat_df["model"].astype(str) + " | "
                + heat_df["dataset"].astype(str) + " | "
                + heat_df["kind"].astype(str)
            )
            mat = heat_df.set_index("row_label")[plot_cols]
            mat = mat[~mat.index.duplicated(keep="first")]
            mat_norm = (mat - mat.mean()) / (mat.std(ddof=0).replace(0, np.nan) + 1e-9)
            plt.figure(figsize=(12, max(7, 0.35 * len(mat_norm))))
            sns.heatmap(
                mat_norm, cmap="coolwarm", center=0, mask=mat_norm.isna(),
                annot=False, cbar_kws={"label": "Standardized score; blank = unavailable"}
            )
            plt.title("Figure 4: Normalized Compute-Performance Tradeoff", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{prefix}_figure4_tradeoff_heatmap.png"), dpi=300)
            plt.close()
        else:
            print("[WARN] Figure 4: no metrics contain valid values")
