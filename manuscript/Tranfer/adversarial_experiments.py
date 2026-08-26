"""Advanced adversarial experiment suite."""

from __future__ import annotations

import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from adversarial_core import AdversarialCore
from adversarial_cka import CKASuite

EPSILON_VALUES = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]


class AdvancedExperimentSuite:
    """Collection of optional experiment phases (Exp 4/7/9/10)."""

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
