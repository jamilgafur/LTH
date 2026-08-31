"""Advanced adversarial experiment suite."""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn

from adversarial_core import AdversarialCore
from adversarial_cka import CKASuite

EPSILON_VALUES = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]


class AdvancedExperimentSuite:
    """Collection of optional experiment phases (Exp 4/7/9/10/11)."""

    def __init__(
        self,
        instantiate_attack: Callable,
        model_kind_label: Callable,
        classify_transfer_pair: Callable,
    ) -> None:
        self.instantiate_attack = instantiate_attack
        self.model_kind_label = model_kind_label
        self.classify_transfer_pair = classify_transfer_pair

    def compute_gradient_similarity(
        self,
        model_s: nn.Module,
        model_t: nn.Module,
        dataloader,
        device: str,
        n_samples: int = 1000,
    ) -> float:
        """Compute average cosine similarity between input gradients."""
        model_s.eval()
        model_t.eval()
        similarities: list[float] = []
        count = 0

        for images, labels in dataloader:
            if count >= n_samples:
                break
            images, labels = images.to(device), labels.to(device)
            images = images.requires_grad_(True)

            out_s = model_s(images)
            loss_s = torch.nn.functional.cross_entropy(out_s, labels)
            grad_s = torch.autograd.grad(loss_s, images, create_graph=False)[0]

            out_t = model_t(images)
            loss_t = torch.nn.functional.cross_entropy(out_t, labels)
            grad_t = torch.autograd.grad(loss_t, images, create_graph=False)[0]

            g_s = grad_s.view(grad_s.size(0), -1)
            g_t = grad_t.view(grad_t.size(0), -1)
            cos = torch.nn.functional.cosine_similarity(g_s, g_t, dim=1)
            similarities.append(cos.mean().item())
            count += images.size(0)

        return float(np.mean(similarities)) if similarities else 0.0

    def gradient_similarity_phase(self, output_dir: str, model_cache: dict, loader_cache: dict) -> list[dict]:
        """Run pairwise gradient similarity analysis across models."""
        os.makedirs(output_dir, exist_ok=True)
        rows: list[dict] = []
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pairs = list(model_cache.keys())
        for dataset_name in set(k[1] for k in pairs):
            if dataset_name not in loader_cache:
                continue
            _, test_loader = loader_cache[dataset_name]
            dataset_pairs = [k for k in pairs if k[1] == dataset_name]

            for src_key in dataset_pairs:
                src_name, _, src_kind = src_key
                src_model = model_cache[src_key]
                for tgt_key in dataset_pairs:
                    tgt_name, _, tgt_kind = tgt_key
                    tgt_model = model_cache[tgt_key]
                    sim = self.compute_gradient_similarity(src_model, tgt_model, test_loader, device)
                    rows.append(
                        {
                            "source_model": src_name,
                            "source_kind": src_kind,
                            "source_label": self.model_kind_label(src_name, src_kind),
                            "target_model": tgt_name,
                            "target_kind": tgt_kind,
                            "target_label": self.model_kind_label(tgt_name, tgt_kind),
                            "dataset": dataset_name,
                            "gradient_similarity": sim,
                            "same_architecture": src_name == tgt_name,
                            "same_kind": src_kind == tgt_kind,
                            "pair_type": self.classify_transfer_pair(src_name, src_kind, tgt_name, tgt_kind),
                        }
                    )

        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, "gradient_similarity.csv")
        df.to_csv(csv_path, index=False)
        print(f"[EXP4] Saved: {csv_path}")
        return rows

    def epsilon_sensitivity_phase(
        self,
        output_dir: str,
        model_cache: dict,
        loader_cache: dict,
        attacks: list[str] | None = None,
        epsilon_values: list[float] = EPSILON_VALUES,
    ) -> list[dict]:
        """Evaluate attack success sensitivity as epsilon changes."""
        os.makedirs(output_dir, exist_ok=True)
        records: list[dict] = []

        selected_attacks = attacks if attacks else ["PGD", "FGSM", "BIM"]

        for (model_name, dataset_name, kind), model in model_cache.items():
            if dataset_name not in loader_cache:
                continue
            _, test_loader = loader_cache[dataset_name]
            clean_acc = AdversarialCore.evaluate_clean_accuracy(model, test_loader)

            for attack_name in selected_attacks:
                for eps in epsilon_values:
                    adv_bundle = AdversarialCore.generate_adversarial_dataset(
                        model,
                        test_loader,
                        attack_name,
                        epsilon=eps,
                    )
                    if adv_bundle is None:
                        continue

                    adv_loader = torch.utils.data.DataLoader(
                        torch.utils.data.TensorDataset(
                            adv_bundle["adversarial_images"],
                            adv_bundle["true_labels"],
                        ),
                        batch_size=256,
                        shuffle=False,
                    )
                    adv_acc = AdversarialCore.evaluate_clean_accuracy(model, adv_loader)
                    records.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "kind": kind,
                            "attack": attack_name,
                            "epsilon": float(eps),
                            "epsilon_255": float(round(eps * 255, 2)),
                            "clean_acc": float(clean_acc),
                            "adv_acc": float(adv_acc),
                            "attack_success_rate": float(1.0 - adv_acc),
                            "model_label": self.model_kind_label(model_name, kind),
                        }
                    )

        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "epsilon_sensitivity.csv")
        df.to_csv(csv_path, index=False)
        print(f"[EXP7] Saved: {csv_path}")

        if df.empty:
            print("[WARN] EXP7: no epsilon sensitivity records were generated.")
            return records

        delta_rows: list[dict] = []
        for (model_name, dataset_name, attack_name, eps), grp in df.groupby(
            ["model", "dataset", "attack", "epsilon"]
        ):
            orig = grp[grp["kind"] == "Original"]["attack_success_rate"]
            fine = grp[grp["kind"] == "Finetuned"]["attack_success_rate"]
            if not orig.empty and not fine.empty:
                delta_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "attack": attack_name,
                        "epsilon": float(eps),
                        "epsilon_255": float(round(eps * 255, 2)),
                        "asr_original": float(orig.mean()),
                        "asr_finetuned": float(fine.mean()),
                        "delta_asr": float(fine.mean()) - float(orig.mean()),
                    }
                )

        if delta_rows:
            delta_df = pd.DataFrame(delta_rows)
            delta_csv = os.path.join(output_dir, "epsilon_sensitivity_delta.csv")
            delta_df.to_csv(delta_csv, index=False)
            print(f"[EXP7] Saved: {delta_csv}")

        for dataset_name in df["dataset"].unique():
            for attack_name in df["attack"].unique():
                sub = df[(df["dataset"] == dataset_name) & (df["attack"] == attack_name)]
                if sub.empty:
                    continue
                plt.figure(figsize=(9, 6))
                for label, grp in sub.groupby("model_label"):
                    grp_sorted = grp.sort_values("epsilon_255")
                    plt.plot(
                        grp_sorted["epsilon_255"],
                        grp_sorted["attack_success_rate"],
                        marker="o",
                        label=label,
                    )
                plt.xlabel("Perturbation Budget epsilon (x1/255)", fontweight="bold")
                plt.ylabel("Attack Success Rate", fontweight="bold")
                plt.title(f"Epsilon Sensitivity - {dataset_name} ({attack_name})", fontsize=13, fontweight="bold")
                plt.legend(fontsize=7, ncol=2)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"epsilon_sensitivity_{dataset_name}_{attack_name}.png"),
                    dpi=300,
                )
                plt.close()

        return records

    def _load_multi_run_summaries(self, result_dirs: list[str]) -> pd.DataFrame:
        frames = []
        for run_dir in result_dirs:
            summary_path = os.path.join(run_dir, "summary.csv")
            if not os.path.exists(summary_path):
                print(f"[WARN] EXP9: summary.csv not found in {run_dir}")
                continue
            try:
                df = pd.read_csv(summary_path)
            except pd.errors.EmptyDataError:
                print(f"[WARN] EXP9: summary.csv is empty in {run_dir}")
                continue
            if df.empty:
                print(f"[WARN] EXP9: summary.csv has no rows in {run_dir}")
                continue
            df["run_dir"] = os.path.basename(run_dir)
            frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def statistical_significance_phase(
        self,
        output_dir: str,
        result_dirs: list[str] | None = None,
    ) -> list[dict]:
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            print("[WARN] EXP9: scipy not installed. Run: pip install scipy")
            return []

        os.makedirs(output_dir, exist_ok=True)

        if result_dirs:
            df_all = self._load_multi_run_summaries(result_dirs)
        else:
            summary_path = os.path.join(output_dir, "summary.csv")
            if not os.path.exists(summary_path):
                print(f"[WARN] EXP9: no summary.csv at {summary_path}")
                return []
            try:
                df_all = pd.read_csv(summary_path)
            except pd.errors.EmptyDataError:
                print(f"[WARN] EXP9: summary.csv is empty at {summary_path}")
                return []
            df_all["run_dir"] = "single_run"

        if df_all.empty:
            print("[WARN] EXP9: no data to analyse.")
            return []

        records: list[dict] = []
        for (model_name, dataset_name, attack_name), grp in df_all.groupby(["model", "dataset", "attack"]):
            orig = grp[grp["kind"] == "Original"]["attack_success_rate"].dropna().values
            fine = grp[grp["kind"] == "Finetuned"]["attack_success_rate"].dropna().values
            if len(orig) == 0 or len(fine) == 0:
                continue

            mean_orig = float(np.mean(orig))
            mean_fine = float(np.mean(fine))
            std_orig = float(np.std(orig, ddof=1)) if len(orig) > 1 else float("nan")
            std_fine = float(np.std(fine, ddof=1)) if len(fine) > 1 else float("nan")
            delta_asr = mean_fine - mean_orig

            if len(orig) == len(fine) and len(orig) > 1:
                t_stat, p_ttest = scipy_stats.ttest_rel(fine, orig)
            else:
                t_stat, p_ttest = float("nan"), float("nan")

            if len(result_dirs or []) >= 2 and len(orig) >= 2:
                _, p_kw = scipy_stats.kruskal(orig, fine)
            else:
                p_kw = float("nan")

            records.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "attack": attack_name,
                    "n_original": len(orig),
                    "n_finetuned": len(fine),
                    "mean_asr_original": mean_orig,
                    "std_asr_original": std_orig,
                    "mean_asr_finetuned": mean_fine,
                    "std_asr_finetuned": std_fine,
                    "delta_asr": delta_asr,
                    "t_statistic": t_stat,
                    "p_value_ttest": p_ttest,
                    "p_value_kruskal": p_kw,
                    "significant_at_0.05": (not np.isnan(p_ttest)) and (p_ttest < 0.05),
                    "effect_size_abs": abs(delta_asr),
                }
            )

        df_out = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, "statistical_significance.csv")
        df_out.to_csv(csv_path, index=False)

        if not df_out.empty:
            for dataset_name in df_out["dataset"].unique():
                sub = df_out[df_out["dataset"] == dataset_name].copy().sort_values("delta_asr")
                fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.35)))
                colors = ["#e74c3c" if s else "#2ecc71" for s in sub["significant_at_0.05"]]
                ax.barh(range(len(sub)), sub["delta_asr"], color=colors, edgecolor="black", linewidth=0.5)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_yticks(range(len(sub)))
                ax.set_yticklabels(sub["model"] + " / " + sub["attack"], fontsize=7)
                ax.set_xlabel("Delta ASR (Finetuned - Original)", fontweight="bold")
                ax.set_title(
                    f"Compression Effect on Attack Success Rate - {dataset_name}\n"
                    f"(red = p<0.05, green = not significant)",
                    fontsize=12,
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, f"statistical_significance_{dataset_name}.png"),
                    dpi=300,
                )
                plt.close()

        return records

    def cka_similarity_phase(
        self,
        output_dir: str,
        model_cache: dict,
        loader_cache: dict,
        max_samples: int = 512,
        max_layers: int = 8,
    ) -> list[dict]:
        """Run CKA using the dedicated CKA suite and guard empty outputs."""
        records = CKASuite.run(
            output_dir=output_dir,
            model_cache=model_cache,
            loader_cache=loader_cache,
            model_kind_label=self.model_kind_label,
            classify_transfer_pair=self.classify_transfer_pair,
            max_samples=max_samples,
            max_layers=max_layers,
        )
        if not records:
            print("[WARN] CKA phase produced no records; generated cka_similarity.csv may be empty.")
        return records

    @staticmethod
    def _extract_loader_images(loader, max_samples: int, device: str) -> torch.Tensor | None:
        chunks = []
        total = 0
        for images, _labels in loader:
            chunks.append(images.to(device))
            total += images.size(0)
            if total >= max_samples:
                break
        if not chunks:
            return None
        return torch.cat(chunks, dim=0)[:max_samples]

    @staticmethod
    def _extract_loader_samples(loader, max_samples: int, device: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        image_chunks = []
        label_chunks = []
        total = 0
        for images, labels in loader:
            images = images.to(device)
            take = min(images.size(0), max_samples - total)
            if take <= 0:
                break
            image_chunks.append(images[:take].detach().cpu())
            label_chunks.append(labels[:take].detach().cpu())
            total += take
            if total >= max_samples:
                break
        if not image_chunks:
            return None, None
        return torch.cat(image_chunks, dim=0), torch.cat(label_chunks, dim=0)

    @staticmethod
    def _safe_model_call(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = model(batch)
        return out if isinstance(out, torch.Tensor) else out[0]

    @staticmethod
    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return float("nan")
        rx = pd.Series(x).rank().to_numpy(dtype=float)
        ry = pd.Series(y).rank().to_numpy(dtype=float)
        return float(np.corrcoef(rx, ry)[0, 1])

    @staticmethod
    def _feature_vector_metrics(a: np.ndarray, b: np.ndarray, topk_ratio: float = 0.05) -> dict:
        a = a.astype(float)
        b = b.astype(float)

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        cosine = float(np.dot(a, b) / denom) if denom > 0 else float("nan")
        pearson = float(np.corrcoef(a, b)[0, 1]) if a.size >= 2 else float("nan")
        spearman = AdvancedExperimentSuite._safe_spearman(a, b)

        diff = a - b
        l1_mean = float(np.mean(np.abs(diff)))
        l2 = float(np.linalg.norm(diff))

        k = int(max(1, round(topk_ratio * a.size)))
        a_top = set(np.argpartition(np.abs(a), -k)[-k:])
        b_top = set(np.argpartition(np.abs(b), -k)[-k:])
        topk_intersection = len(a_top & b_top)
        topk_union = len(a_top | b_top)
        topk_jaccard = float(topk_intersection / topk_union) if topk_union > 0 else float("nan")

        return {
            "cosine_similarity": cosine,
            "pearson_r": pearson,
            "spearman_r": spearman,
            "l1_mean_abs_diff": l1_mean,
            "l2_distance": l2,
            "topk_ratio": float(topk_ratio),
            "topk_jaccard": topk_jaccard,
            "topk_intersection": int(topk_intersection),
            "topk_union": int(topk_union),
        }

    @staticmethod
    def _select_class_shap(shap_values, preds: np.ndarray) -> np.ndarray | None:
        if isinstance(shap_values, list):
            arr = np.stack([np.asarray(v) for v in shap_values], axis=1)
            if arr.ndim < 3:
                return None
            idx = np.arange(arr.shape[0])
            clipped = np.clip(preds, 0, arr.shape[1] - 1)
            return arr[idx, clipped]

        arr = np.asarray(shap_values)
        if arr.ndim >= 3 and arr.shape[0] == preds.shape[0]:
            # Some SHAP backends already return per-sample selected outputs.
            return arr
        if arr.ndim >= 3 and arr.shape[-1] > 1 and arr.shape[0] == preds.shape[0]:
            idx = np.arange(arr.shape[0])
            clipped = np.clip(preds, 0, arr.shape[-1] - 1)
            return arr[idx, ..., clipped]
        if arr.ndim >= 4 and arr.shape[1] == preds.shape[0]:
            # Shape [num_classes, N, ...]
            arr = np.moveaxis(arr, 0, 1)
            idx = np.arange(arr.shape[0])
            clipped = np.clip(preds, 0, arr.shape[1] - 1)
            return arr[idx, clipped]
        return None

    def _compute_shap_reference_vector(
        self,
        model: nn.Module,
        loader,
        device: str,
        max_samples: int,
        background_samples: int,
    ) -> tuple[np.ndarray | None, list[dict], dict | None]:
        try:
            import shap
        except Exception:
            print("[WARN] SHAP package not available. Install via: pip install shap")
            return None, [], None

        eval_images, eval_labels = self._extract_loader_samples(loader, max_samples=max_samples, device=device)
        if eval_images is None or eval_images.size(0) < 2 or eval_labels is None:
            return None, [], None

        bg_n = int(max(2, min(background_samples, eval_images.size(0))))
        background = eval_images[:bg_n]
        eval_batch = eval_images[:max_samples]
        eval_label_batch = eval_labels[:max_samples]

        base_model = model.module if hasattr(model, "module") else model
        base_model.eval()

        explainer = None
        for explainer_cls in (shap.DeepExplainer, shap.GradientExplainer):
            try:
                explainer = explainer_cls(base_model, background)
                break
            except Exception:
                explainer = None
        if explainer is None:
            print("[WARN] Could not initialize SHAP explainer for a model; skipping.")
            return None, [], None

        try:
            logits = self._safe_model_call(base_model, eval_batch)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            shap_values = explainer.shap_values(eval_batch)
        except Exception as exc:
            print(f"[WARN] SHAP attribution failed: {exc}")
            return None, [], None

        selected = self._select_class_shap(shap_values, preds)
        if selected is None:
            print("[WARN] Unsupported SHAP output format for this model; skipping.")
            return None, [], None

        selected_np = np.asarray(selected)
        sample_vectors = selected_np.reshape(selected_np.shape[0], -1)
        ref_vector = np.mean(np.abs(sample_vectors), axis=0)

        sample_stats: list[dict] = []
        class_examples: list[dict] = []
        seen_labels: set[int] = set()
        for sample_idx in range(sample_vectors.shape[0]):
            sv = sample_vectors[sample_idx]
            true_label = int(eval_label_batch[sample_idx].item())
            sample_stats.append(
                {
                    "sample_index": int(sample_idx),
                    "true_label": true_label,
                    "pred_class": int(preds[sample_idx]),
                    "attr_mean": float(np.mean(sv)),
                    "attr_mean_abs": float(np.mean(np.abs(sv))),
                    "attr_l1": float(np.sum(np.abs(sv))),
                    "attr_l2": float(np.linalg.norm(sv)),
                }
            )
            if true_label not in seen_labels:
                seen_labels.add(true_label)
                class_examples.append(
                    {
                        "sample_index": int(sample_idx),
                        "true_label": true_label,
                        "pred_class": int(preds[sample_idx]),
                        "input_image": np.asarray(eval_batch[sample_idx].detach().cpu().numpy(), dtype=np.float32),
                        "shap_map": np.asarray(selected_np[sample_idx], dtype=np.float32),
                        "attr_mean_abs": float(np.mean(np.abs(sv))),
                        "attr_l1": float(np.sum(np.abs(sv))),
                        "attr_l2": float(np.linalg.norm(sv)),
                    }
                )

        class_bundle = {
            "rows": class_examples,
            "sample_indices": np.array([row["sample_index"] for row in class_examples], dtype=int),
            "true_labels": np.array([row["true_label"] for row in class_examples], dtype=int),
            "pred_classes": np.array([row["pred_class"] for row in class_examples], dtype=int),
            "input_images": np.stack([row["input_image"] for row in class_examples], axis=0) if class_examples else None,
            "shap_maps": np.stack([row["shap_map"] for row in class_examples], axis=0) if class_examples else None,
        }

        return ref_vector, sample_stats, class_bundle

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(path_no_ext: str) -> None:
        """Save the current matplotlib figure as both PNG (300 dpi) and SVG."""
        plt.savefig(path_no_ext + ".png", dpi=300)
        plt.savefig(path_no_ext + ".svg")

    @staticmethod
    def _plot_shap_attribution_profiles(
        output_dir: str,
        ref_vectors: dict,
        model_kind_label_fn,
        topk: int = 30,
    ) -> None:
        """Plot per-model mean-absolute SHAP attribution bar charts and a
        combined overlay, saved as PNG + SVG."""
        datasets = sorted({k[1] for k in ref_vectors})
        for dataset_name in datasets:
            dataset_keys = [k for k in ref_vectors if k[1] == dataset_name]
            if not dataset_keys:
                continue

            # --- individual per-model attribution bars ---
            for key in dataset_keys:
                model_name, _, kind = key
                vec = ref_vectors[key]
                n = vec.size
                actual_k = min(topk, n)
                top_idx = np.argsort(vec)[::-1][:actual_k]
                top_vals = vec[top_idx]

                fig, ax = plt.subplots(figsize=(max(8, actual_k * 0.35), 4))
                ax.bar(range(actual_k), top_vals, color="#4c72b0", edgecolor="black", linewidth=0.4)
                ax.set_xticks(range(actual_k))
                ax.set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=7)
                ax.set_xlabel(f"Feature index (top-{actual_k} by |attribution|)", fontweight="bold")
                ax.set_ylabel("Mean |SHAP|", fontweight="bold")
                label = model_kind_label_fn(model_name, kind)
                ax.set_title(f"SHAP Attribution Profile – {label} – {dataset_name}", fontweight="bold")
                fig.tight_layout()
                safe = label.replace(" ", "_").replace("(", "").replace(")", "")
                AdvancedExperimentSuite._save_fig(
                    os.path.join(output_dir, f"shap_attribution_{dataset_name}_{safe}")
                )
                plt.close(fig)

            # --- combined overlay of all models' normalized profiles ---
            fig, ax = plt.subplots(figsize=(12, 5))
            for key in dataset_keys:
                model_name, _, kind = key
                vec = ref_vectors[key].astype(float)
                norm = vec / (vec.max() + 1e-12)
                label = model_kind_label_fn(model_name, kind)
                ax.plot(norm, linewidth=0.8, alpha=0.75, label=label)
            ax.set_xlabel("Feature index (flattened)", fontweight="bold")
            ax.set_ylabel("Normalized mean |SHAP|", fontweight="bold")
            ax.set_title(f"SHAP Attribution Profiles Overlay – {dataset_name}", fontweight="bold")
            ax.legend(fontsize=7, ncol=2, loc="upper right")
            fig.tight_layout()
            AdvancedExperimentSuite._save_fig(
                os.path.join(output_dir, f"shap_attribution_overlay_{dataset_name}")
            )
            plt.close(fig)

    @staticmethod
    def _to_display_image(image: np.ndarray) -> np.ndarray:
        arr = np.asarray(image, dtype=float)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if np.isfinite(arr_min) and np.isfinite(arr_max) and arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        return np.clip(arr, 0.0, 1.0)

    @staticmethod
    def _collapse_attribution_map(attr_map: np.ndarray) -> np.ndarray:
        arr = np.asarray(attr_map, dtype=float)
        if arr.ndim == 3:
            if arr.shape[0] in (1, 3):
                arr = np.mean(np.abs(arr), axis=0)
            elif arr.shape[-1] in (1, 3):
                arr = np.mean(np.abs(arr), axis=-1)
            else:
                arr = np.mean(np.abs(arr), axis=0)
        elif arr.ndim > 3:
            arr = np.mean(np.abs(arr.reshape(arr.shape[0], -1)), axis=0)
        return arr

    @classmethod
    def _save_shap_class_example_artifacts(
        cls,
        output_dir: str,
        dataset_name: str,
        model_name: str,
        class_rows: list[dict],
        original_kind: str,
        collapsed_kind: str,
    ) -> pd.DataFrame:
        output_rows = []
        input_images = []
        original_maps = []
        collapsed_maps = []
        delta_maps = []
        labels = []

        for row in class_rows:
            input_image = np.asarray(row["input_image"], dtype=np.float32)
            original_map = np.asarray(row["original_shap_map"], dtype=np.float32)
            collapsed_map = np.asarray(row["collapsed_shap_map"], dtype=np.float32)
            delta_map = collapsed_map - original_map

            input_images.append(input_image)
            original_maps.append(original_map)
            collapsed_maps.append(collapsed_map)
            delta_maps.append(delta_map)
            labels.append(int(row["true_label"]))

            output_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "true_label": int(row["true_label"]),
                    "class_name": row.get("class_name", str(row["true_label"])),
                    "sample_index": int(row["sample_index"]),
                    f"{original_kind.lower()}_pred_class": int(row["original_pred_class"]),
                    f"{collapsed_kind.lower()}_pred_class": int(row["collapsed_pred_class"]),
                    f"{original_kind.lower()}_attr_mean_abs": float(row["original_attr_mean_abs"]),
                    f"{collapsed_kind.lower()}_attr_mean_abs": float(row["collapsed_attr_mean_abs"]),
                    "delta_attr_mean_abs": float(np.mean(np.abs(delta_map))),
                    "delta_attr_l1": float(np.sum(np.abs(delta_map))),
                    "delta_attr_l2": float(np.linalg.norm(delta_map.reshape(-1))),
                }
            )

        csv_path = os.path.join(output_dir, f"shap_class_examples_{dataset_name}_{model_name}.csv")
        df = pd.DataFrame(output_rows)
        df.to_csv(csv_path, index=False)
        print(f"[EXP11] Saved: {csv_path}")

        npz_path = os.path.join(output_dir, f"shap_class_examples_{dataset_name}_{model_name}.npz")
        np.savez_compressed(
            npz_path,
            labels=np.asarray(labels, dtype=int),
            inputs=np.asarray(input_images, dtype=np.float32),
            original_shap_maps=np.asarray(original_maps, dtype=np.float32),
            collapsed_shap_maps=np.asarray(collapsed_maps, dtype=np.float32),
            delta_shap_maps=np.asarray(delta_maps, dtype=np.float32),
        )
        print(f"[EXP11] Saved: {npz_path}")

        return df

    @classmethod
    def _plot_shap_class_example_grid(
        cls,
        output_dir: str,
        dataset_name: str,
        model_name: str,
        class_rows: list[dict],
        class_names: list[str] | None,
        original_kind: str,
        collapsed_kind: str,
    ) -> None:
        if not class_rows:
            return

        class_rows = sorted(class_rows, key=lambda row: int(row["true_label"]))
        n_rows = len(class_rows)
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, max(4, 3.0 * n_rows)))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, row in enumerate(class_rows):
            class_id = int(row["true_label"])
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
            image = cls._to_display_image(row["input_image"])
            original_map = cls._collapse_attribution_map(row["original_shap_map"])
            collapsed_map = cls._collapse_attribution_map(row["collapsed_shap_map"])
            delta_map = collapsed_map - original_map

            ax = axes[row_idx, 0]
            ax.imshow(image)
            ax.set_title(f"Input | {class_name}")
            ax.axis("off")

            ax = axes[row_idx, 1]
            im = ax.imshow(original_map, cmap="magma")
            ax.set_title(f"{original_kind} SHAP")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            ax = axes[row_idx, 2]
            im = ax.imshow(collapsed_map, cmap="magma")
            ax.set_title(f"{collapsed_kind} SHAP")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

            ax = axes[row_idx, 3]
            im = ax.imshow(delta_map, cmap="coolwarm")
            ax.set_title(f"Delta ({collapsed_kind} - {original_kind})")
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        fig.suptitle(
            f"SHAP Example Changes by Class - {model_name} - {dataset_name}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        cls._save_fig(os.path.join(output_dir, f"shap_class_examples_{dataset_name}_{model_name}"))
        plt.close(fig)

    def explainability_similarity_phase(
        self,
        output_dir: str,
        model_cache: dict,
        loader_cache: dict,
        max_samples: int = 64,
        background_samples: int = 32,
        topk_ratio: float = 0.05,
    ) -> list[dict]:
        """Compute SHAP-based explainability similarity across model pairs on clean data."""
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        vector_rows: list[dict] = []
        sample_rows: list[dict] = []
        ref_vectors: dict[tuple[str, str, str], np.ndarray] = {}
        class_example_bundles: dict[tuple[str, str, str], dict] = {}

        for key, model in model_cache.items():
            model_name, dataset_name, kind = key
            if dataset_name not in loader_cache:
                continue
            _, test_loader = loader_cache[dataset_name]

            ref_vec, stats, class_bundle = self._compute_shap_reference_vector(
                model=model,
                loader=test_loader,
                device=device,
                max_samples=max_samples,
                background_samples=background_samples,
            )
            if ref_vec is None:
                continue

            ref_vectors[key] = ref_vec
            if class_bundle is not None:
                class_example_bundles[key] = class_bundle
            vector_rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "kind": kind,
                    "model_label": self.model_kind_label(model_name, kind),
                    "feature_dim": int(ref_vec.size),
                    **{f"feature_{i}": float(v) for i, v in enumerate(ref_vec)},
                }
            )
            for row in stats:
                sample_rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "kind": kind,
                        "model_label": self.model_kind_label(model_name, kind),
                        **row,
                    }
                )

        vectors_df = pd.DataFrame(vector_rows)
        vectors_path = os.path.join(output_dir, "shap_reference_vectors.csv")
        vectors_df.to_csv(vectors_path, index=False)
        print(f"[EXP11] Saved: {vectors_path}")

        sample_df = pd.DataFrame(sample_rows)
        sample_path = os.path.join(output_dir, "shap_sample_stats.csv")
        sample_df.to_csv(sample_path, index=False)
        print(f"[EXP11] Saved: {sample_path}")

        if not ref_vectors:
            print("[WARN] EXP11: no SHAP vectors generated; skipping similarity plots.")
            empty_pair = pd.DataFrame()
            empty_pair.to_csv(os.path.join(output_dir, "shap_pairwise_similarity.csv"), index=False)
            return []

        # --- Plot SHAP attribution profiles (PNG + SVG) ---
        self._plot_shap_attribution_profiles(
            output_dir, ref_vectors, self.model_kind_label
        )

        for dataset_name in sorted({dataset for (_model, dataset, _kind) in class_example_bundles}):
            model_names = sorted({model for (model, dataset, _kind) in class_example_bundles if dataset == dataset_name})
            for model_name in model_names:
                orig_key = (model_name, dataset_name, "Original")
                fin_key = (model_name, dataset_name, "Finetuned")
                if orig_key not in class_example_bundles or fin_key not in class_example_bundles:
                    continue

                orig_bundle = class_example_bundles[orig_key]
                fin_bundle = class_example_bundles[fin_key]
                orig_rows = {int(row["true_label"]): row for row in orig_bundle["rows"]}
                fin_rows = {int(row["true_label"]): row for row in fin_bundle["rows"]}
                class_ids = sorted(set(orig_rows) & set(fin_rows))
                if not class_ids:
                    continue

                test_loader = loader_cache[dataset_name][1]
                class_names = getattr(getattr(test_loader, "dataset", None), "classes", None)
                pair_rows: list[dict] = []

                for class_id in class_ids:
                    orig_row = orig_rows[class_id]
                    fin_row = fin_rows[class_id]
                    orig_vec = np.asarray(orig_row["shap_map"], dtype=np.float32).reshape(-1)
                    fin_vec = np.asarray(fin_row["shap_map"], dtype=np.float32).reshape(-1)
                    delta_vec = fin_vec - orig_vec
                    denom = float(np.linalg.norm(orig_vec) * np.linalg.norm(fin_vec))
                    delta_cos = float(np.dot(orig_vec, fin_vec) / denom) if denom > 0 else float("nan")

                    pair_rows.append(
                        {
                            "dataset": dataset_name,
                            "model": model_name,
                            "true_label": class_id,
                            "class_name": class_names[class_id] if class_names and class_id < len(class_names) else str(class_id),
                            "sample_index": int(orig_row["sample_index"]),
                            "original_pred_class": int(orig_row["pred_class"]),
                            "collapsed_pred_class": int(fin_row["pred_class"]),
                            "original_attr_mean_abs": float(orig_row["attr_mean_abs"]),
                            "collapsed_attr_mean_abs": float(fin_row["attr_mean_abs"]),
                            "delta_attr_mean_abs": float(np.mean(np.abs(delta_vec))),
                            "delta_attr_l1": float(np.sum(np.abs(delta_vec))),
                            "delta_attr_l2": float(np.linalg.norm(delta_vec)),
                            "delta_cosine_similarity": delta_cos,
                            "input_image": orig_row["input_image"],
                            "original_shap_map": orig_row["shap_map"],
                            "collapsed_shap_map": fin_row["shap_map"],
                        }
                    )

                if pair_rows:
                    pair_df = self._save_shap_class_example_artifacts(
                        output_dir=output_dir,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        class_rows=pair_rows,
                        original_kind="Original",
                        collapsed_kind="Finetuned",
                    )
                    self._plot_shap_class_example_grid(
                        output_dir=output_dir,
                        dataset_name=dataset_name,
                        model_name=model_name,
                        class_rows=pair_rows,
                        class_names=class_names,
                        original_kind="Original",
                        collapsed_kind="Finetuned",
                    )

        pair_rows: list[dict] = []
        keys = list(ref_vectors.keys())
        for src_key in keys:
            src_model, src_dataset, src_kind = src_key
            src_vec = ref_vectors[src_key]
            for tgt_key in keys:
                tgt_model, tgt_dataset, tgt_kind = tgt_key
                if src_dataset != tgt_dataset:
                    continue
                tgt_vec = ref_vectors[tgt_key]

                metrics = self._feature_vector_metrics(src_vec, tgt_vec, topk_ratio=topk_ratio)
                pair_rows.append(
                    {
                        "source_model": src_model,
                        "source_kind": src_kind,
                        "source_label": self.model_kind_label(src_model, src_kind),
                        "target_model": tgt_model,
                        "target_kind": tgt_kind,
                        "target_label": self.model_kind_label(tgt_model, tgt_kind),
                        "dataset": src_dataset,
                        "same_architecture": src_model == tgt_model,
                        "same_kind": src_kind == tgt_kind,
                        "pair_type": self.classify_transfer_pair(src_model, src_kind, tgt_model, tgt_kind),
                        **metrics,
                    }
                )

        pair_df = pd.DataFrame(pair_rows)
        pair_path = os.path.join(output_dir, "shap_pairwise_similarity.csv")
        pair_df.to_csv(pair_path, index=False)
        print(f"[EXP11] Saved: {pair_path}")

        collapsed_vs_original = pair_df[
            (pair_df["same_architecture"])
            & (pair_df["source_kind"] != pair_df["target_kind"])
        ].copy()
        collapsed_vs_original_path = os.path.join(output_dir, "shap_original_vs_collapsed_similarity.csv")
        collapsed_vs_original.to_csv(collapsed_vs_original_path, index=False)
        print(f"[EXP11] Saved: {collapsed_vs_original_path}")

        for dataset_name in pair_df["dataset"].unique():
            sub = pair_df[pair_df["dataset"] == dataset_name]
            for metric, fname_stem in [
                ("cosine_similarity", f"shap_cosine_heatmap_{dataset_name}"),
                ("pearson_r", f"shap_pearson_heatmap_{dataset_name}"),
                ("spearman_r", f"shap_spearman_heatmap_{dataset_name}"),
            ]:
                mat = sub.pivot(index="source_label", columns="target_label", values=metric)
                if mat.empty:
                    continue
                plt.figure(figsize=(12, 9))
                sns.heatmap(mat, annot=True, fmt=".3f", cmap="vlag", center=0)
                plt.title(f"SHAP {metric} Similarity - {dataset_name}", fontweight="bold")
                plt.xlabel("Target Model Variant", fontweight="bold")
                plt.ylabel("Source Model Variant", fontweight="bold")
                plt.tight_layout()
                self._save_fig(os.path.join(output_dir, fname_stem))
                plt.close()

            # Dedicated Original-vs-Collapsed cosine similarity heatmap
            ov_sub = sub[sub["same_architecture"] & (sub["source_kind"] != sub["target_kind"])]
            if not ov_sub.empty:
                ov_mat = ov_sub.pivot(
                    index="source_label", columns="target_label", values="cosine_similarity"
                )
                if not ov_mat.empty:
                    plt.figure(figsize=(max(6, len(ov_mat.columns) * 1.2), max(4, len(ov_mat) * 0.8)))
                    sns.heatmap(
                        ov_mat,
                        annot=True,
                        fmt=".3f",
                        cmap="RdYlGn",
                        center=0,
                        vmin=-1,
                        vmax=1,
                        linewidths=0.5,
                        cbar_kws={"label": "Cosine similarity"},
                    )
                    plt.title(
                        f"SHAP Cosine Similarity: Original vs Collapsed – {dataset_name}",
                        fontweight="bold",
                    )
                    plt.xlabel("Collapsed variant", fontweight="bold")
                    plt.ylabel("Original variant", fontweight="bold")
                    plt.tight_layout()
                    self._save_fig(
                        os.path.join(output_dir, f"shap_orig_vs_collapsed_cosine_{dataset_name}")
                    )
                    plt.close()

        if not collapsed_vs_original.empty:
            cv = (
                collapsed_vs_original.groupby(["dataset", "source_model"], as_index=False)
                .agg(
                    cosine_similarity=("cosine_similarity", "mean"),
                    pearson_r=("pearson_r", "mean"),
                    spearman_r=("spearman_r", "mean"),
                    l1_mean_abs_diff=("l1_mean_abs_diff", "mean"),
                    l2_distance=("l2_distance", "mean"),
                    topk_jaccard=("topk_jaccard", "mean"),
                )
            )
            cv.to_csv(os.path.join(output_dir, "shap_original_vs_collapsed_summary.csv"), index=False)

            melt = cv.melt(
                id_vars=["dataset", "source_model"],
                value_vars=["cosine_similarity", "pearson_r", "spearman_r", "topk_jaccard"],
                var_name="metric",
                value_name="value",
            )
            plt.figure(figsize=(12, 6))
            sns.barplot(data=melt, x="source_model", y="value", hue="metric", errorbar=None)
            plt.title("Original vs Collapsed SHAP Similarity Metrics", fontweight="bold")
            plt.xlabel("Model Architecture", fontweight="bold")
            plt.ylabel("Similarity", fontweight="bold")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            self._save_fig(os.path.join(output_dir, "shap_original_vs_collapsed_similarity_metrics"))
            plt.close()

        return pair_rows
