"""CKA feature similarity experiment utilities."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


class CKASuite:
    """Layer-wise CKA experiment implementation."""

    @staticmethod
    def _compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        X = X - X.mean(0, keepdim=True)
        Y = Y - Y.mean(0, keepdim=True)
        dot_xx = (X @ X.T).norm(p="fro") ** 2
        dot_yy = (Y @ Y.T).norm(p="fro") ** 2
        dot_xy = (X @ Y.T).norm(p="fro") ** 2
        denom = dot_xx.sqrt() * dot_yy.sqrt()
        return float(dot_xy / denom) if denom > 0 else float("nan")

    @staticmethod
    def _candidate_layers(model: nn.Module) -> list[str]:
        return [
            name
            for name, mod in model.named_modules()
            if isinstance(mod, (nn.Conv2d, nn.Linear))
        ]

    @staticmethod
    def _extract_representations(
        model: nn.Module,
        dataloader,
        layer_name: str,
        device: str,
        max_samples: int = 512,
    ) -> torch.Tensor | None:
        reps: list[torch.Tensor] = []
        count = 0

        def hook_fn(_module, _inp, out):
            reps.append(out.detach().cpu())

        target = dict(model.named_modules()).get(layer_name)
        if target is None:
            return None

        handle = target.register_forward_hook(hook_fn)
        model.eval()
        try:
            with torch.no_grad():
                for imgs, _ in dataloader:
                    if count >= max_samples:
                        break
                    model(imgs.to(device))
                    count += imgs.size(0)
        except Exception:
            pass
        finally:
            handle.remove()

        if not reps:
            return None
        out = torch.cat(reps, dim=0)[:max_samples]
        return out.view(out.size(0), -1)

    @classmethod
    def run(
        cls,
        output_dir: str,
        model_cache: dict,
        loader_cache: dict,
        model_kind_label,
        classify_transfer_pair,
        max_samples: int = 512,
        max_layers: int = 8,
    ) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        records: list[dict] = []
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
                src_base = src_model.module if hasattr(src_model, "module") else src_model
                all_layers = cls._candidate_layers(src_base)
                if not all_layers:
                    continue
                step = max(1, len(all_layers) // max_layers)
                probe_layers = all_layers[::step][:max_layers]

                for tgt_key in dataset_pairs:
                    tgt_name, _, tgt_kind = tgt_key
                    tgt_model = model_cache[tgt_key]
                    tgt_base = tgt_model.module if hasattr(tgt_model, "module") else tgt_model
                    tgt_layers = cls._candidate_layers(tgt_base)

                    for layer_name in probe_layers:
                        tgt_layer = layer_name if layer_name in tgt_layers else None
                        if tgt_layer is None and layer_name in all_layers:
                            idx = all_layers.index(layer_name)
                            if idx < len(tgt_layers):
                                tgt_layer = tgt_layers[idx]
                        if tgt_layer is None:
                            continue

                        X = cls._extract_representations(src_model, test_loader, layer_name, device, max_samples)
                        Y = cls._extract_representations(tgt_model, test_loader, tgt_layer, device, max_samples)
                        if X is None or Y is None or X.size(0) < 4 or Y.size(0) < 4:
                            continue

                        n = min(X.size(0), Y.size(0))
                        try:
                            cka_val = cls._compute_linear_cka(X[:n].float(), Y[:n].float())
                        except Exception:
                            cka_val = float("nan")

                        records.append(
                            {
                                "source_model": src_name,
                                "source_kind": src_kind,
                                "source_label": model_kind_label(src_name, src_kind),
                                "target_model": tgt_name,
                                "target_kind": tgt_kind,
                                "target_label": model_kind_label(tgt_name, tgt_kind),
                                "dataset": dataset_name,
                                "layer": layer_name,
                                "cka": cka_val,
                                "same_architecture": src_name == tgt_name,
                                "same_kind": src_kind == tgt_kind,
                                "pair_type": classify_transfer_pair(src_name, src_kind, tgt_name, tgt_kind),
                            }
                        )

        df = pd.DataFrame(records)
        df.to_csv(os.path.join(output_dir, "cka_similarity.csv"), index=False)

        for dataset_name in df["dataset"].unique():
            for src_label in df["source_label"].unique():
                sub = df[(df["dataset"] == dataset_name) & (df["source_label"] == src_label)]
                if sub.empty:
                    continue
                plt.figure(figsize=(11, 5))
                for tgt_label, grp in sub.groupby("target_label"):
                    plt.plot(range(len(grp)), grp["cka"].values, marker="o", label=f"-> {tgt_label}")
                plt.xlabel("Layer Index (shallow -> deep)", fontweight="bold")
                plt.ylabel("CKA", fontweight="bold")
                plt.ylim(0, 1.05)
                plt.title(f"Layer-wise CKA from {src_label} - {dataset_name}", fontsize=12, fontweight="bold")
                plt.legend(fontsize=7, ncol=2)
                plt.tight_layout()
                safe_label = src_label.replace(" ", "_").replace("/", "_")
                plt.savefig(os.path.join(output_dir, f"cka_layerwise_{dataset_name}_{safe_label}.png"), dpi=300)
                plt.close()

        if not df.empty:
            for dataset_name in df["dataset"].unique():
                sub = df[df["dataset"] == dataset_name]
                mean_cka = sub.groupby(["source_label", "target_label"])["cka"].mean().reset_index()
                pivot = mean_cka.pivot(index="source_label", columns="target_label", values="cka")
                plt.figure(figsize=(12, 9))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1, cbar_kws={"label": "Mean CKA (all layers)"})
                plt.title(f"Mean Layer-wise CKA - {dataset_name}", fontsize=14, fontweight="bold")
                plt.xlabel("Target Model Variant", fontweight="bold")
                plt.ylabel("Source Model Variant", fontweight="bold")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"cka_mean_heatmap_{dataset_name}.png"), dpi=300)
                plt.close()
                pivot.to_csv(os.path.join(output_dir, f"cka_mean_matrix_{dataset_name}.csv"))

        return records
