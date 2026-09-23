"""CKA feature similarity experiment utilities.

This module provides a ``CKASuite`` class that can compute linear CKA similarity
between model representations. Two entry points are offered:

* ``run`` – used by the existing ``adversarial_experiments`` pipeline. It
  accepts a pre‑populated ``model_cache`` and ``loader_cache`` and forwards to
  ``run_standalone`` for simplicity.
* ``run_standalone`` – a self‑contained implementation that discovers checkpoints
  on‑the‑fly, loads one model at a time, and writes incremental CSV/figure
  outputs after each model (required by the user’s incremental‑flush request).

Both methods share helper utilities for loading datasets, extracting layer
representations, computing linear CKA, and plotting heat‑maps (Figure 5).
"""

from __future__ import annotations

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from adversarial_checkpointing import CheckpointManager
from adversarial_core import AdversarialCore
from adversarial_reporting import ReportingSuite
# Import the atomic CSV helper from ``adversarial_reporting`` to avoid a circular import.
from adversarial_reporting import _write_locked_csv

try:
    from .collapse import _capture_preblock_activation
except ImportError:
    from collapse import _capture_preblock_activation


class CKASuite:
    """Utility suite for linear CKA similarity analysis.

    The class is deliberately lightweight – all heavy lifting is performed in
    static methods. ``FIGURE_PREFIX`` is used for naming output files.
    """

    FIGURE_PREFIX = "Figure_5"

    # ---------------------------------------------------------------------
    # Public entry points
    # ---------------------------------------------------------------------
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
        """Run CKA using an existing ``model_cache``.

        For compatibility with the original pipeline we simply forward to the
        ``run_standalone`` implementation, which discovers checkpoints itself.
        The additional ``model_cache`` and ``loader_cache`` arguments are ignored
        because the standalone version already handles incremental loading and
        flushing.
        """
        return cls.run_standalone(
            output_dir=output_dir,
            model_kind_label=model_kind_label,
            classify_transfer_pair=classify_transfer_pair,
            model_filter=None,
            dataset_filter=None,
            kind_filter=None,
            max_samples=max_samples,
            max_layers=max_layers,
        )

    @classmethod
    def run_standalone(
        cls,
        output_dir: str,
        model_kind_label,
        classify_transfer_pair,
        model_filter: str | None = None,
        dataset_filter: str | None = None,
        kind_filter: str | None = None,
        max_samples: int = 512,
        max_layers: int = 8,
    ) -> list[dict]:
        """Standalone CKA phase that loads one checkpoint model at a time.

        The implementation writes CSVs and figures after each model to avoid data
        loss.
        """
        os.makedirs(output_dir, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        loader_cache: dict = {}
        grouped: dict[str, list[tuple[str, str, str, str]]] = {}
        for model_name, dataset_name, kind, ckpt_path in CheckpointManager.discover_checkpoints():
            if not CheckpointManager.dataset_matches_output_dir(dataset_name, output_dir):
                continue
            if model_filter and model_filter != "ALL" and model_name != model_filter:
                continue
            if dataset_filter and dataset_filter != "ALL":
                base_check = CheckpointManager.base_dataset_name(dataset_name)
                if base_check.lower() != dataset_filter.lower():
                    continue
            if kind_filter and kind_filter != "ALL" and kind != kind_filter:
                continue
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)
            grouped.setdefault(base_dataset, []).append((model_name, dataset_name, kind, ckpt_path))

        records: list[dict] = []
        repr_bank: dict[str, dict[tuple[str, str, str], dict[str, torch.Tensor]]] = {}
        layer_bank: dict[tuple[str, str, str], list[str]] = {}

        def flush_outputs() -> None:
            df = pd.DataFrame(records)
            _write_locked_csv(os.path.join(output_dir, "cka_similarity.csv"), df, "cka_similarity")
            if {"source_model", "target_model", "source_kind", "target_kind", "dataset", "cka"}.issubset(df.columns):
                pairwise_df = df[df["source_model"].notna() & df["target_model"].notna()].copy()
                cls._plot_cka_from_dataframe(output_dir, pairwise_df)

        for base_dataset, checkpoints in grouped.items():
            try:
                train_loader, test_loader = cls._loader_for_dataset(base_dataset, loader_cache)
            except Exception as exc:
                print(f"[WARN] CKA standalone could not load dataset {base_dataset}: {exc}")
                continue

            repr_bank[base_dataset] = {}
            sample_batch = next(iter(train_loader))[0]
            input_shape = tuple(sample_batch.shape)

            for model_name, dataset_name, kind, ckpt_path in checkpoints:
                one_batch = next(iter(train_loader))[0]
                model = None
                try:
                    model = CheckpointManager.build_model_for_checkpoint(
                        model_name,
                        dataset_name,
                        kind,
                        10 if base_dataset == "Cifar10" else 100 if base_dataset == "Cifar100" else 200,
                        one_batch,
                        ckpt_path,
                        device,
                    )
                    AdversarialCore.robust_load_state_dict(model, ckpt_path)
                    if torch.cuda.is_available():
                        model = torch.nn.DataParallel(model)
                except Exception as exc:
                    print(f"[WARN] CKA standalone failed to load {model_name} ({kind}) on {dataset_name}: {exc}")
                    continue

                try:
                    base_model = model.module if hasattr(model, "module") else model
                    all_layers = cls._candidate_layers(base_model)
                    if not all_layers:
                        continue
                    step = max(1, len(all_layers) // max_layers)
                    probe_layers = all_layers[::step][:max_layers]
                    layer_bank[(model_name, dataset_name, kind)] = probe_layers

                    model_layers: dict[str, torch.Tensor] = {}
                    for layer_name in probe_layers:
                        reps = cls._extract_representations(model, test_loader, layer_name, device, max_samples)
                        if reps is not None and reps.size(0) >= 4:
                            model_layers[layer_name] = reps
                    repr_bank[base_dataset][(model_name, dataset_name, kind)] = model_layers
                finally:
                    try:
                        del model
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                flush_outputs()

            # Pairwise CKA across models of the same base dataset
            dataset_pairs = list(repr_bank[base_dataset].keys())
            for src_key in dataset_pairs:
                src_name, _, src_kind = src_key
                src_layers = layer_bank.get(src_key, [])
                src_reps = repr_bank[base_dataset].get(src_key, {})
                for tgt_key in dataset_pairs:
                    tgt_name, _, tgt_kind = tgt_key
                    tgt_layers = layer_bank.get(tgt_key, [])
                    tgt_reps = repr_bank[base_dataset].get(tgt_key, {})
                    for layer_name in src_layers:
                        tgt_layer = layer_name if layer_name in tgt_layers else None
                        if tgt_layer is None and layer_name in src_layers:
                            idx = src_layers.index(layer_name)
                            if idx < len(tgt_layers):
                                tgt_layer = tgt_layers[idx]
                        if tgt_layer is None:
                            continue
                        X = src_reps.get(layer_name)
                        Y = tgt_reps.get(tgt_layer)
                        if X is None or Y is None:
                            continue
                        n = min(X.size(0), Y.size(0))
                        if n < 4:
                            continue
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

            # Boundary CKA for compressed variants (optional)
            for model_name, dataset_name, kind, ckpt_path in checkpoints:
                if kind not in ReportingSuite.variant_kinds():
                    continue
                try:
                    compression_set = CheckpointManager.get_compression_set_for_checkpoint(
                        model_name, base_dataset, ckpt_path
                    )
                except Exception:
                    continue
                if base_dataset not in loader_cache:
                    continue
                train_loader, _ = loader_cache[base_dataset]
                sample_batch = next(iter(train_loader))[0]
                input_shape = tuple(sample_batch.shape)
                model = None
                try:
                    model = CheckpointManager.build_model_for_checkpoint(
                        model_name,
                        dataset_name,
                        kind,
                        10 if base_dataset == "Cifar10" else 100 if base_dataset == "Cifar100" else 200,
                        sample_batch,
                        ckpt_path,
                        device,
                    )
                    AdversarialCore.robust_load_state_dict(model, ckpt_path)
                    if torch.cuda.is_available():
                        model = torch.nn.DataParallel(model)
                except Exception:
                    continue
                try:
                    for block in compression_set:
                        if isinstance(block, dict):
                            start = block.get("start_layer_name") or block.get("start_layer")
                            end = block.get("end_layer_name") or block.get("end_layer")
                        else:
                            start, end = block[0], block[1]
                        if not start or not end:
                            continue
                        try:
                            x_in, y_out, _ = _capture_preblock_activation(
                                model, start, end, input_shape, [], None, device, debug=False
                            )
                            if x_in is None or y_out is None:
                                continue
                            X = x_in.view(x_in.size(0), -1).float()
                            Y = y_out.view(y_out.size(0), -1).float()
                            cka_val = CKASuite._compute_linear_cka(X, Y)
                        except Exception:
                            cka_val = float("nan")
                        records.append(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "kind": kind,
                                "block_start": start,
                                "block_end": end,
                                "boundary_cka": cka_val,
                            }
                        )
                finally:
                    try:
                        del model
                    except Exception:
                        pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                flush_outputs()

        # Final flush to ensure everything is written
        flush_outputs()
        return records

    # ---------------------------------------------------------------------
    # Helper utilities (private)
    # ---------------------------------------------------------------------
    @staticmethod
    def _loader_for_dataset(base_dataset: str, loader_cache: dict):
        """Return (train_loader, test_loader) for a given dataset name."""
        from pyPrune.utils import load_cifar10, load_cifar100, load_tiny_imagenet

        if base_dataset == "Cifar10":
            if "Cifar10" not in loader_cache:
                loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
            return loader_cache["Cifar10"]
        if base_dataset == "Cifar100":
            if "Cifar100" not in loader_cache:
                loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
            return loader_cache["Cifar100"]
        if base_dataset.lower() == "tinyimagenet":
            if "tinyimagenet" not in loader_cache:
                loader_cache["tinyimagenet"] = load_tiny_imagenet(batch_size=256, num_workers=4)
            return loader_cache["tinyimagenet"]
        raise ValueError(f"Unsupported dataset: {base_dataset}")

    @staticmethod
    def _candidate_layers(model: torch.nn.Module) -> list[str]:
        """Return a list of layer names suitable for probing (Conv2d and Linear)."""
        layers: list[str] = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                layers.append(name)
        return layers

    @staticmethod
    def _extract_representations(
        model: torch.nn.Module,
        loader,
        layer_name: str,
        device: str,
        max_samples: int,
    ) -> torch.Tensor | None:
        """Run ``model`` on ``loader`` and capture the output of ``layer_name``.

        The function uses a forward hook to collect activations.
        """
        activations: list[torch.Tensor] = []
        handle = None
        try:
            for name, module in model.named_modules():
                if name == layer_name:
                    handle = module.register_forward_hook(lambda m, i, o: activations.append(o.detach().cpu()))
                    break
            if handle is None:
                return None
            model.to(device)
            model.eval()
            total = 0
            with torch.no_grad():
                for images, _ in loader:
                    images = images.to(device)
                    _ = model(images)
                    total += images.size(0)
                    if total >= max_samples:
                        break
            if not activations:
                return None
            tensor = torch.cat(activations, dim=0)[:max_samples]
            return tensor
        finally:
            if handle is not None:
                handle.remove()

    @staticmethod
    def _compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute linear CKA between two representation matrices.

        Implements the standard linear CKA formula using centered Gram matrices.
        """
        X_centered = X - X.mean(0, keepdim=True)
        Y_centered = Y - Y.mean(0, keepdim=True)
        XTX = X_centered.t() @ X_centered
        YTY = Y_centered.t() @ Y_centered
        XY = X_centered.t() @ Y_centered
        numerator = (XY * XY).sum().item()
        denominator = (XTX * XTX).sum().item() * (YTY * YTY).sum().item()
        if denominator == 0:
            return float("nan")
        return numerator / (denominator ** 0.5)

    @classmethod
    def _plot_cka_from_dataframe(cls, output_dir: str, df: pd.DataFrame) -> None:
        """Generate Figure 5 heat‑maps from a CKA results DataFrame.

        Creates one figure per dataset and writes both PNG and SVG files. Also
        writes per‑model CSV matrices for downstream analysis.
        """
        if df.empty:
            print("[WARN] CKA: no records generated; skipping CKA plots.")
            return

        required_columns = {"source_model", "source_kind", "target_model", "target_kind", "dataset", "cka"}
        if not required_columns.issubset(df.columns):
            print("[WARN] CKA: pairwise columns missing; skipping CKA plots.")
            return

        df = df[df["source_model"].notna() & df["target_model"].notna()].copy()
        if df.empty:
            print("[WARN] CKA: no pairwise records generated; skipping CKA plots.")
            return

        variant_order = [ReportingSuite.baseline_kind(), *ReportingSuite.variant_kinds()]
        variant_label_map = {
            ReportingSuite.baseline_kind(): "original",
            ReportingSuite.variant_kinds()[0]: "pruned",
            ReportingSuite.variant_kinds()[1]: "pruned_quant",
        }

        for dataset_name in df["dataset"].unique():
            dataset_df = df[df["dataset"] == dataset_name].copy()
            if dataset_df.empty:
                continue

            model_names = sorted(dataset_df["source_model"].unique())
            n_models = len(model_names)
            ncols = 3
            nrows = (n_models + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
            fig.suptitle(
                f"Figure 5: CKA Similarity by Model and Variant - {dataset_name}",
                fontsize=15,
                fontweight="bold",
            )

            for idx, model_name in enumerate(model_names):
                ax = axes[idx // ncols, idx % ncols]
                model_df = dataset_df[dataset_df["source_model"] == model_name]
                if model_df.empty:
                    ax.axis("off")
                    continue

                summary = (
                    model_df.groupby(["source_kind", "target_kind"], as_index=False)["cka"]
                    .mean()
                    .pivot(index="source_kind", columns="target_kind", values="cka")
                )
                summary = summary.reindex(index=variant_order, columns=variant_order)
                sns.heatmap(
                    summary,
                    annot=True,
                    fmt=".3f",
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Mean CKA"},
                    ax=ax,
                )
                ax.set_title(model_name, fontweight="bold")
                ax.set_xlabel("Target kind")
                ax.set_ylabel("Source kind")
                ax.set_xticklabels(
                    [variant_label_map.get(v.get_text(), v.get_text()) for v in ax.get_xticklabels()],
                    rotation=25,
                    ha="right",
                )
                ax.set_yticklabels(
                    [variant_label_map.get(v.get_text(), v.get_text()) for v in ax.get_yticklabels()],
                    rotation=0,
                )

                model_matrix_path = os.path.join(
                    output_dir, f"Figure_5_cka_matrix_{dataset_name}_{model_name}.csv"
                )
                _write_locked_csv(model_matrix_path, summary.reset_index(), f"Figure_5_cka_matrix_{dataset_name}_{model_name}")

            total_axes = nrows * ncols
            for empty_idx in range(n_models, total_axes):
                fig.delaxes(axes[empty_idx // ncols, empty_idx % ncols])

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_{dataset_name}.png"), dpi=300)
            fig.savefig(os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_{dataset_name}.svg"))
            plt.close(fig)

            # Legacy CSV with source/target labels
            mean_cka = dataset_df.groupby(["source_label", "target_label"])["cka"].mean().reset_index()
            pivot = mean_cka.pivot(index="source_label", columns="target_label", values="cka")
            matrix_path = os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_matrix_{dataset_name}.csv")
            _write_locked_csv(matrix_path, pivot.reset_index(), f"{cls.FIGURE_PREFIX}_cka_matrix_{dataset_name}")
