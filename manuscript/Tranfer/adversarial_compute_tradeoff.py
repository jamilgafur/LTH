"""Compute-cost tradeoff experiments and figures."""

from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from adversarial_core import AdversarialCore


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

        summary_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        transfer_df = pd.read_csv(transfer_path) if os.path.exists(transfer_path) else pd.DataFrame()
        explain_df = pd.read_csv(explain_path) if os.path.exists(explain_path) else pd.DataFrame()

        for (model_name, dataset_name, kind), model in model_cache.items():
            # dataset_name may include a split tag; use the base name for loader lookup.
            base_dataset = dataset_name.split("|")[0]
            if base_dataset not in loader_cache:
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
                    (summary_df["model"] == model_name)
                    & (summary_df["dataset"] == dataset_name)
                    & (summary_df["kind"] == kind)
                ]
                if not s.empty:
                    robust_acc = float(s["adv_acc"].mean())
                    attack_success = float(s["attack_success_rate"].mean())

            if not transfer_df.empty:
                t = transfer_df[
                    (transfer_df["target_model"] == model_name)
                    & (transfer_df["dataset"] == dataset_name)
                    & (transfer_df["target_kind"] == kind)
                ]
                if not t.empty:
                    transfer_success = float(t["transfer_success_rate"].mean())

            if not explain_df.empty:
                e = explain_df[(explain_df["model"] == model_name) & (explain_df["dataset"] == dataset_name)]
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
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="flops", y="robust_accuracy", hue="kind", style="model", s=120)
        plt.xscale("log")
        plt.xlabel("FLOPs (log scale)", fontweight="bold")
        plt.ylabel("Robust Accuracy", fontweight="bold")
        plt.title("Figure 1: Robustness-Compute Pareto", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure1_pareto_flops_vs_robust_accuracy.png"), dpi=300)
        plt.close()

        # Figure 2: Transfer resistance vs latency
        plt.figure(figsize=(8, 6))
        sizes = np.clip(df["param_count"] / max(1.0, df["param_count"].max()) * 700, 60, 700)
        plt.scatter(df["latency_ms"], df["transfer_resistance"], s=sizes, alpha=0.7)
        for _, row in df.iterrows():
            plt.annotate(row["model_label"], (row["latency_ms"], row["transfer_resistance"]), fontsize=7)
        plt.xlabel("Latency (ms / batch)", fontweight="bold")
        plt.ylabel("Transfer Resistance (1 - transfer success)", fontweight="bold")
        plt.title("Figure 2: Transfer Resistance vs Latency", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure2_transfer_resistance_vs_latency.png"), dpi=300)
        plt.close()

        # Figure 3: Collapsed vs Original deltas for core metrics
        delta_rows = []
        for (model_name, dataset_name), g in df.groupby(["model", "dataset"]):
            orig = g[g["kind"] == "Original"]
            fin = g[g["kind"] == "Finetuned"]
            if orig.empty or fin.empty:
                continue
            delta_rows.extend(
                [
                    {"label": f"{model_name}-{dataset_name}", "metric": "robust_accuracy", "delta": float(fin["robust_accuracy"].mean() - orig["robust_accuracy"].mean())},
                    {"label": f"{model_name}-{dataset_name}", "metric": "transfer_resistance", "delta": float(fin["transfer_resistance"].mean() - orig["transfer_resistance"].mean())},
                    {"label": f"{model_name}-{dataset_name}", "metric": "latency_ms", "delta": float(fin["latency_ms"].mean() - orig["latency_ms"].mean())},
                    {"label": f"{model_name}-{dataset_name}", "metric": "flops", "delta": float(fin["flops"].mean() - orig["flops"].mean())},
                ]
            )
        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            plt.figure(figsize=(12, 6))
            sns.barplot(data=delta_df, x="label", y="delta", hue="metric")
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Finetuned - Original", fontweight="bold")
            plt.title("Figure 3: Collapsed vs Original Metric Deltas", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "figure3_collapsed_original_deltas.png"), dpi=300)
            plt.close()

        # Figure 4: normalized tradeoff heatmap
        plot_cols = [
            "robust_accuracy",
            "transfer_resistance",
            "explainability_delta_asr",
            "explainability_shap_cosine_similarity",
            "explainability_shap_topk_jaccard",
            "flops",
            "latency_ms",
            "peak_memory_mb",
        ]
        mat = df.set_index("model_label")[plot_cols]
        mat_norm = (mat - mat.mean()) / (mat.std(ddof=0) + 1e-9)
        plt.figure(figsize=(10, 7))
        sns.heatmap(mat_norm, cmap="coolwarm", center=0)
        plt.title("Figure 4: Normalized Compute-Performance Tradeoff", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure4_tradeoff_heatmap.png"), dpi=300)
        plt.close()
