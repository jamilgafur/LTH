"""Correlation analysis for transferability, similarity, and compute factors."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class CorrelationSuite:
    """Compute Pearson/Spearman correlations and generate correlation figures."""

    @staticmethod
    def _safe_pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        try:
            from scipy import stats as scipy_stats

            r, p = scipy_stats.pearsonr(x, y)
            return float(r), float(p)
        except Exception:
            if len(x) < 2:
                return float("nan"), float("nan")
            r = np.corrcoef(x, y)[0, 1]
            return float(r), float("nan")

    @staticmethod
    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        try:
            from scipy import stats as scipy_stats

            r, p = scipy_stats.spearmanr(x, y)
            return float(r), float(p)
        except Exception:
            if len(x) < 2:
                return float("nan"), float("nan")
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            r = np.corrcoef(rx, ry)[0, 1]
            return float(r), float("nan")

    @staticmethod
    def _bootstrap_ci(x: np.ndarray, y: np.ndarray, kind: str = "pearson", n_boot: int = 500) -> tuple[float, float]:
        if len(x) < 4:
            return float("nan"), float("nan")
        vals = []
        n = len(x)
        for _ in range(n_boot):
            idx = np.random.randint(0, n, n)
            xb, yb = x[idx], y[idx]
            if kind == "pearson":
                r, _ = CorrelationSuite._safe_pearson(xb, yb)
            else:
                r, _ = CorrelationSuite._safe_spearman(xb, yb)
            vals.append(r)
        return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))

    @staticmethod
    def _bh_fdr(pvals: list[float]) -> list[float]:
        p = np.array([v if np.isfinite(v) else 1.0 for v in pvals], dtype=float)
        n = len(p)
        order = np.argsort(p)
        ranked = p[order]
        adj = np.empty(n, dtype=float)
        prev = 1.0
        for i in range(n - 1, -1, -1):
            rank = i + 1
            val = min(prev, ranked[i] * n / rank)
            adj[i] = val
            prev = val
        out = np.empty(n, dtype=float)
        out[order] = np.clip(adj, 0, 1)
        return out.tolist()

    @staticmethod
    def _read_csv_if_valid(path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
        return df

    @staticmethod
    def _load_inputs(output_dir: str) -> pd.DataFrame:
        summary = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "summary.csv"))
        transfer = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "transferability.csv"))
        grad = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "gradient_similarity.csv"))
        cka = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "cka_similarity.csv"))
        shap_pairs = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "shap_pairwise_similarity.csv"))
        cost = CorrelationSuite._read_csv_if_valid(os.path.join(output_dir, "compute_profile.csv"))

        if transfer.empty:
            return pd.DataFrame()

        base = transfer.copy()

        if not grad.empty:
            g2 = grad.groupby(["source_model", "source_kind", "target_model", "target_kind", "dataset"], as_index=False)["gradient_similarity"].mean()
            base = base.merge(
                g2,
                on=["source_model", "source_kind", "target_model", "target_kind", "dataset"],
                how="left",
            )

        if not cka.empty:
            c2 = cka.groupby(["source_model", "source_kind", "target_model", "target_kind", "dataset"], as_index=False)["cka"].mean()
            base = base.merge(
                c2,
                on=["source_model", "source_kind", "target_model", "target_kind", "dataset"],
                how="left",
            )

        if not shap_pairs.empty:
            s2 = shap_pairs.groupby(
                ["source_model", "source_kind", "target_model", "target_kind", "dataset"],
                as_index=False,
            ).agg(
                shap_cosine_similarity=("cosine_similarity", "mean"),
                shap_pearson_r=("pearson_r", "mean"),
                shap_spearman_r=("spearman_r", "mean"),
                shap_l1_mean_abs_diff=("l1_mean_abs_diff", "mean"),
                shap_l2_distance=("l2_distance", "mean"),
                shap_topk_jaccard=("topk_jaccard", "mean"),
            )
            base = base.merge(
                s2,
                on=["source_model", "source_kind", "target_model", "target_kind", "dataset"],
                how="left",
            )

        if not summary.empty:
            s2 = summary.groupby(["model", "kind", "dataset"], as_index=False).agg(
                clean_acc=("clean_acc", "mean"),
                source_asr=("attack_success_rate", "mean"),
            )
            s2 = s2.rename(columns={"model": "source_model", "kind": "source_kind"})
            base = base.merge(s2, on=["source_model", "source_kind", "dataset"], how="left")

        if not cost.empty:
            csrc = cost[["model", "kind", "dataset", "flops", "latency_ms", "param_count"]].rename(
                columns={"model": "source_model", "kind": "source_kind", "flops": "source_flops", "latency_ms": "source_latency_ms", "param_count": "source_params"}
            )
            base = base.merge(csrc, on=["source_model", "source_kind", "dataset"], how="left")

        return base

    @classmethod
    def run(cls, output_dir: str) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        df = cls._load_inputs(output_dir)
        if df.empty:
            print("[WARN] Correlation analysis skipped: transferability.csv not found or empty.")
            return []

        numeric_cols = [
            "transfer_success_rate",
            "normalized_transfer_rate",
            "gradient_similarity",
            "cka",
            "shap_cosine_similarity",
            "shap_pearson_r",
            "shap_spearman_r",
            "shap_l1_mean_abs_diff",
            "shap_l2_distance",
            "shap_topk_jaccard",
            "clean_acc",
            "source_asr",
            "source_flops",
            "source_latency_ms",
            "source_params",
        ]

        pairs = [
            ("transfer_success_rate", "gradient_similarity"),
            ("transfer_success_rate", "cka"),
            ("transfer_success_rate", "source_asr"),
            ("transfer_success_rate", "clean_acc"),
            ("transfer_success_rate", "source_flops"),
            ("transfer_success_rate", "source_latency_ms"),
            ("transfer_success_rate", "source_params"),
            ("transfer_success_rate", "shap_cosine_similarity"),
            ("transfer_success_rate", "shap_pearson_r"),
            ("transfer_success_rate", "shap_spearman_r"),
            ("transfer_success_rate", "shap_topk_jaccard"),
            ("transfer_success_rate", "shap_l1_mean_abs_diff"),
            ("transfer_success_rate", "shap_l2_distance"),
            ("normalized_transfer_rate", "gradient_similarity"),
            ("normalized_transfer_rate", "cka"),
            ("normalized_transfer_rate", "shap_cosine_similarity"),
            ("normalized_transfer_rate", "shap_topk_jaccard"),
        ]

        rows = []
        for x_col, y_col in pairs:
            subset = df[[x_col, y_col]].dropna()
            if len(subset) < 4:
                continue
            x = subset[x_col].to_numpy(dtype=float)
            y = subset[y_col].to_numpy(dtype=float)

            pear_r, pear_p = cls._safe_pearson(x, y)
            spear_r, spear_p = cls._safe_spearman(x, y)
            p_low, p_high = cls._bootstrap_ci(x, y, kind="pearson")
            s_low, s_high = cls._bootstrap_ci(x, y, kind="spearman")

            rows.append(
                {
                    "x": x_col,
                    "y": y_col,
                    "n": len(subset),
                    "pearson_r": pear_r,
                    "pearson_p": pear_p,
                    "pearson_ci_low": p_low,
                    "pearson_ci_high": p_high,
                    "spearman_r": spear_r,
                    "spearman_p": spear_p,
                    "spearman_ci_low": s_low,
                    "spearman_ci_high": s_high,
                }
            )

        corr_df = pd.DataFrame(rows)
        if corr_df.empty:
            print("[WARN] Correlation analysis produced no valid pairs.")
            return []

        corr_df["pearson_p_fdr"] = cls._bh_fdr(corr_df["pearson_p"].tolist())
        corr_df["spearman_p_fdr"] = cls._bh_fdr(corr_df["spearman_p"].tolist())

        out_path = os.path.join(output_dir, "correlation_summary.csv")
        corr_df.to_csv(out_path, index=False)
        print(f"[CORR] Saved: {out_path}")

        cls._plot_scatter_panels(output_dir, df)
        available_numeric = [col for col in numeric_cols if col in df.columns]
        cls._plot_matrix(output_dir, df, available_numeric)
        cls._plot_effect_forest(output_dir, corr_df)
        cls._compute_partial_correlations(output_dir, df)

        return rows

    @classmethod
    def _plot_scatter_panels(cls, output_dir: str, df: pd.DataFrame) -> None:
        for y_col, fname, title in [
            ("gradient_similarity", "figure5_transfer_vs_gradient_similarity.png", "Figure 5: Transfer vs Gradient Similarity"),
            ("cka", "figure6_transfer_vs_cka.png", "Figure 6: Transfer vs CKA"),
            ("shap_cosine_similarity", "figure9_transfer_vs_shap_cosine.png", "Figure 9: Transfer vs SHAP Cosine Similarity"),
            ("shap_topk_jaccard", "figure10_transfer_vs_shap_topk_jaccard.png", "Figure 10: Transfer vs SHAP Top-k Jaccard"),
        ]:
            if y_col not in df.columns:
                continue
            sub = df[["transfer_success_rate", y_col, "source_attack"]].dropna()
            if sub.empty:
                continue
            g = sns.lmplot(
                data=sub,
                x=y_col,
                y="transfer_success_rate",
                col="source_attack",
                col_wrap=3,
                height=3.2,
                scatter_kws={"alpha": 0.5, "s": 25},
                line_kws={"color": "red"},
            )
            g.fig.subplots_adjust(top=0.86)
            g.fig.suptitle(title, fontweight="bold")
            g.savefig(os.path.join(output_dir, fname), dpi=300)
            plt.close(g.fig)

    @staticmethod
    def _plot_matrix(output_dir: str, df: pd.DataFrame, numeric_cols: list[str]) -> None:
        if not numeric_cols:
            return
        sub = df[numeric_cols].dropna()
        if sub.shape[0] < 4:
            return
        corr = sub.corr(method="spearman")
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
        plt.title("Figure 7: Correlation Matrix (Spearman)", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure7_correlation_matrix.png"), dpi=300)
        plt.close()

    @staticmethod
    def _plot_effect_forest(output_dir: str, corr_df: pd.DataFrame) -> None:
        sub = corr_df.sort_values("pearson_r")
        labels = [f"{r.x} vs {r.y}" for r in sub.itertuples(index=False)]
        y_idx = np.arange(len(sub))

        plt.figure(figsize=(10, max(4, 0.4 * len(sub))))
        plt.hlines(y_idx, sub["pearson_ci_low"], sub["pearson_ci_high"], color="gray", lw=2)
        plt.plot(sub["pearson_r"], y_idx, "o", color="navy")
        plt.axvline(0, color="black", lw=1)
        plt.yticks(y_idx, labels, fontsize=8)
        plt.xlabel("Pearson r (95% bootstrap CI)", fontweight="bold")
        plt.title("Figure 8: Correlation Effect Sizes", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "figure8_correlation_effect_sizes.png"), dpi=300)
        plt.close()

    @classmethod
    def _compute_partial_correlations(cls, output_dir: str, df: pd.DataFrame) -> None:
        # Partial corr of transfer_success_rate vs similarity metrics controlling for clean_acc + params
        rows = []
        for metric in [
            "gradient_similarity",
            "cka",
            "shap_cosine_similarity",
            "shap_pearson_r",
            "shap_spearman_r",
            "shap_topk_jaccard",
        ]:
            if metric not in df.columns:
                continue
            sub = df[["transfer_success_rate", metric, "clean_acc", "source_params"]].dropna()
            if len(sub) < 10:
                continue

            y = sub["transfer_success_rate"].to_numpy()
            x = sub[metric].to_numpy()
            c1 = sub["clean_acc"].to_numpy()
            c2 = np.log1p(sub["source_params"].to_numpy())

            # Residualize x and y against controls
            C = np.column_stack([np.ones(len(sub)), c1, c2])
            beta_y, *_ = np.linalg.lstsq(C, y, rcond=None)
            beta_x, *_ = np.linalg.lstsq(C, x, rcond=None)
            y_res = y - C @ beta_y
            x_res = x - C @ beta_x

            r, p = cls._safe_pearson(x_res, y_res)
            lo, hi = cls._bootstrap_ci(x_res, y_res, kind="pearson")
            rows.append(
                {
                    "metric": metric,
                    "n": len(sub),
                    "partial_r": r,
                    "partial_p": p,
                    "partial_ci_low": lo,
                    "partial_ci_high": hi,
                }
            )

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(output_dir, "partial_correlation_summary.csv"), index=False)
