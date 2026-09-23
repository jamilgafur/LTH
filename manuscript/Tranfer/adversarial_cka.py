"""CKA feature similarity experiment utilities."""

from __future__ import annotations

import os
import tempfile
import time
from collapse import _capture_preblock_activation
from adversarial_checkpointing import CheckpointManager
from adversarial_core import AdversarialCore
from adversarial_reporting import ReportingSuite
from pyPrune.utils import load_cifar10, load_cifar100, load_tiny_imagenet

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn


def _atomic_write_csv(path: str, df: pd.DataFrame) -> None:
    """Write CSV atomically to avoid partial writes from concurrent jobs."""
    path_obj = os.path.abspath(path)
    directory = os.path.dirname(path_obj) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=directory)
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path_obj)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _acquire_lock(lock_path: str, timeout_seconds: float = 60.0, stale_seconds: float = 300.0) -> bool:
    """Acquire a lock file with stale-lock cleanup."""
    lock_dir = os.path.dirname(lock_path) or "."
    os.makedirs(lock_dir, exist_ok=True)
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if (time.time() - os.path.getmtime(lock_path)) > stale_seconds:
                    os.unlink(lock_path)
                    print(f"[DEBUG] Removed stale lock {lock_path}")
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    return False


def _release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.unlink(lock_path)
    except OSError:
        pass


def _write_locked_csv(path: str, df: pd.DataFrame, lock_name: str) -> bool:
    lock_path = os.path.join(os.path.dirname(path) or ".", f".{lock_name}.lock")
    if not _acquire_lock(lock_path, timeout_seconds=120.0):
        print(f"[WARN] Could not acquire lock for {path}; skipping write.")
        return False
    try:
        _atomic_write_csv(path, df)
        try:
            open(os.path.join(os.path.dirname(path) or ".", f".{lock_name}_complete"), "a").close()
        except OSError:
            pass
        return True
    finally:
        _release_lock(lock_path)


class CKASuite:
    """Layer-wise CKA experiment implementation."""

    FIGURE_PREFIX = "Figure_5"

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
    def _loader_for_dataset(base_dataset: str, loader_cache: dict):
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

        # NOTE: If the model is wrapped in ``nn.DataParallel`` (or ``nn.DistributedDataParallel``),
        # the actual sub‑modules live under the ``.module`` attribute. ``named_modules()`` on the
        # wrapper prefixes all names with ``module.`` which means a plain ``layer_name`` (e.g.
        # ``features.0``) will not be found. To make the hook robust we unwrap the model when
        # searching for the target layer.
        base_model = model.module if hasattr(model, "module") else model
        target = dict(base_model.named_modules()).get(layer_name)
        if target is None:
            return None

        # Register the hook on the *unwrapped* module but keep the original ``model`` for the
        # forward pass (so DataParallel still handles scattering/gathering).
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
        return out.reshape(out.size(0), -1)

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
            # ``dataset_name`` may include a split tag (e.g. "Cifar10_epochs100_pretrain300").
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)
            if base_dataset not in loader_cache:
                continue
            _, test_loader = loader_cache[base_dataset]
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
        cka_path = os.path.join(output_dir, "cka_similarity.csv")
        _write_locked_csv(cka_path, df, "cka_similarity")

        # -----------------------------------------------------------------
        # Optional: compute CKA at exact collapsed block boundaries.
        # This provides a precise metric for H2 by comparing the representation
        # before and after each collapsed block in the collapsed variants.
        # -----------------------------------------------------------------
        boundary_records = []
        for (model_name, dataset_name, kind), model in model_cache.items():
            if kind not in ReportingSuite.variant_kinds():
                continue
            # Split tag handling (dataset_name may be "Cifar10_epochs100_pretrain300")
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)
            split_tag = CheckpointManager.split_tag_from_dataset_name(dataset_name)

            # Find the checkpoint path that matches this split.
            ckpt_candidates = CheckpointManager.get_checkpoint_path(model_name, base_dataset, kind)
            ckpt_path = None
            for p, tag in ckpt_candidates:
                if tag == split_tag:
                    ckpt_path = p
                    break
            if not ckpt_path:
                continue

            try:
                compression_set = CheckpointManager.get_compression_set_for_checkpoint(
                    model_name, base_dataset, ckpt_path
                )
            except Exception:
                continue

            # Use a real batch to infer input shape.
            if base_dataset not in loader_cache:
                continue
            train_loader, _ = loader_cache[base_dataset]
            sample_batch = next(iter(train_loader))[0]
            input_shape = tuple(sample_batch.shape)

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
                boundary_records.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "kind": kind,
                        "block_start": start,
                        "block_end": end,
                        "boundary_cka": cka_val,
                    }
                )

        if boundary_records:
            boundary_path = os.path.join(output_dir, "cka_boundary.csv")
            _write_locked_csv(boundary_path, pd.DataFrame(boundary_records), "cka_boundary")

        if df.empty:
            print("[WARN] CKA: no records generated; skipping CKA plots.")
            return records

        # Figure 5: model-separated CKA summaries that explicitly compare the
        # original / pruned / pruned_quant variants for each model.
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
                ax.set_xticklabels([variant_label_map.get(v.get_text(), v.get_text()) for v in ax.get_xticklabels()], rotation=25, ha="right")
                ax.set_yticklabels([variant_label_map.get(v.get_text(), v.get_text()) for v in ax.get_yticklabels()], rotation=0)

                model_matrix_path = os.path.join(output_dir, f"Figure_5_cka_matrix_{dataset_name}_{model_name}.csv")
                _write_locked_csv(model_matrix_path, summary.reset_index(), f"Figure_5_cka_matrix_{dataset_name}_{model_name}")

            total_axes = nrows * ncols
            for empty_idx in range(n_models, total_axes):
                fig.delaxes(axes[empty_idx // ncols, empty_idx % ncols])

            plt.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_{dataset_name}.png"), dpi=300)
            fig.savefig(os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_{dataset_name}.svg"))
            plt.close(fig)

            # Keep the legacy CSV for downstream consumers, but name it with the figure prefix.
            mean_cka = dataset_df.groupby(["source_label", "target_label"])['cka'].mean().reset_index()
            pivot = mean_cka.pivot(index="source_label", columns="target_label", values="cka")
            matrix_path = os.path.join(output_dir, f"{cls.FIGURE_PREFIX}_cka_matrix_{dataset_name}.csv")
            _write_locked_csv(matrix_path, pivot.reset_index(), f"{cls.FIGURE_PREFIX}_cka_matrix_{dataset_name}")

        return records

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
        """Standalone CKA phase that loads one checkpoint model at a time."""
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

            # Preserve boundary CKA for collapsed variants without keeping models alive.
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

        df = pd.DataFrame(records)
        cka_path = os.path.join(output_dir, "cka_similarity.csv")
        _write_locked_csv(cka_path, df, "cka_similarity")
        return records
