"""Core adversarial generation and transfer analysis utilities."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

try:
    import torchattacks
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "torchattacks is required for adversarial attacks. Install it via "
        "'pip install torchattacks' and re-run the script."
    ) from exc

# Added support for Tiny ImageNet (and ImageNet) dataset loaders
from pyPrune.utils import load_cifar10, load_cifar100, load_tiny_imagenet, load_imagenet

from adversarial_checkpointing import CheckpointManager
from adversarial_reporting import ReportingSuite


def _atomic_write_csv(path: str, df) -> None:
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


def _write_locked_csv(path: str, df, lock_name: str) -> bool:
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


def _append_locked_csv(path: str, df, lock_name: str) -> bool:
    """Append rows to a CSV under a file lock, writing the header once."""
    lock_path = os.path.join(os.path.dirname(path) or ".", f".{lock_name}.lock")
    if not _acquire_lock(lock_path, timeout_seconds=120.0):
        print(f"[WARN] Could not acquire lock for {path}; skipping append.")
        return False
    try:
        path_obj = os.path.abspath(path)
        directory = os.path.dirname(path_obj) or "."
        os.makedirs(directory, exist_ok=True)
        write_header = not os.path.exists(path_obj) or os.path.getsize(path_obj) == 0
        df.to_csv(path_obj, mode="a", header=write_header, index=False)
        return True
    finally:
        _release_lock(lock_path)


class AdversarialCore:
    """Shared core functionality for adversarial experiments."""

    @staticmethod
    def _default_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        return model.module if hasattr(model, "module") else model

    @staticmethod
    def robust_load_state_dict(model: nn.Module, ckpt_path: str):
        state = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected checkpoint format for {ckpt_path}: {type(state)}")

        sd = state.get("model_state_dict") or state.get("model") or state.get("state_dict") or state
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        return model.load_state_dict(sd, strict=False)

    @staticmethod
    def _disk_free_mb(path: str) -> float:
        """Return free disk space (MB) for the filesystem containing ``path``."""
        probe = path if os.path.exists(path) else os.path.dirname(path) or "."
        usage = shutil.disk_usage(probe)
        return usage.free / (1024.0 * 1024.0)

    @classmethod
    def safe_torch_save(cls, payload: dict, final_path: str) -> None:
        """Safely persist tensors by writing to temp storage then moving atomically."""
        final_dir = os.path.dirname(final_path) or "."
        os.makedirs(final_dir, exist_ok=True)

        tmp_root = os.environ.get("LTH_SAVE_TMPDIR") or os.environ.get("TMPDIR") or final_dir
        os.makedirs(tmp_root, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix="adv_tmp_", suffix=".pt", dir=tmp_root)
        os.close(fd)

        try:
            try:
                torch.save(payload, tmp_path)
            except Exception as exc:
                # Some HPC filesystems intermittently fail with zip writer mode.
                # Retry once using legacy serialization before giving up.
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                torch.save(payload, tmp_path, _use_new_zipfile_serialization=False)

            try:
                # Fast path: atomic move if temp and destination are on same filesystem.
                os.replace(tmp_path, final_path)
            except OSError:
                # Cross-device / flaky NFS path: copy into destination filesystem first,
                # then atomically replace final target there.
                last_copy_error = None
                for attempt in range(3):
                    staged_final = None
                    try:
                        fd2, staged_final = tempfile.mkstemp(
                            prefix="adv_stage_", suffix=".pt", dir=final_dir
                        )
                        os.close(fd2)
                        with open(tmp_path, "rb") as src, open(staged_final, "wb") as dst:
                            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
                            dst.flush()
                            os.fsync(dst.fileno())
                        os.replace(staged_final, final_path)
                        staged_final = None
                        last_copy_error = None
                        break
                    except OSError as copy_exc:
                        last_copy_error = copy_exc
                        # Backoff to ride out transient stale-handle errors on shared FS.
                        time.sleep(0.4 * (attempt + 1))
                    finally:
                        if staged_final and os.path.exists(staged_final):
                            try:
                                os.remove(staged_final)
                            except OSError:
                                pass

                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

                if last_copy_error is not None:
                    raise last_copy_error
        except Exception as exc:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

            final_free = cls._disk_free_mb(final_dir)
            tmp_free = cls._disk_free_mb(tmp_root)
            raise RuntimeError(
                "Failed to save adversarial artifact to "
                f"{final_path}. free_mb(final_fs)={final_free:.1f}, "
                f"free_mb(tmp_fs)={tmp_free:.1f}, tmp_root={tmp_root}. "
                f"Original error: {exc}"
            ) from exc


    @staticmethod
    def evaluate_model(model: nn.Module, loader, return_latency_ms: bool = False):
        """Evaluate accuracy and, optionally, mean inference latency in milliseconds."""
        model.eval()
        device = next(AdversarialCore._unwrap_model(model).parameters()).device
        correct = total = 0
        batch_latency_ms = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                start = time.perf_counter()
                outputs = model(imgs)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                batch_latency_ms.append(float(elapsed_ms))
                preds = outputs.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)

        accuracy = correct / total if total > 0 else 0.0
        if return_latency_ms:
            return accuracy, float(np.mean(batch_latency_ms)) if batch_latency_ms else 0.0
        return accuracy

    @staticmethod
    def evaluate_clean_accuracy(model: nn.Module, loader) -> float:
        accuracy, _ = AdversarialCore.evaluate_model(model, loader, return_latency_ms=True)
        return accuracy

    @staticmethod
    def get_available_attacks() -> list[str]:
        available = []
        for attack_name in ["PGD", "FGSM", "BIM", "APGD", "CW", "DeepFool", "Square", "AutoAttack"]:
            if hasattr(torchattacks, attack_name):
                available.append(attack_name)
        if not available:
            available = ["PGD", "FGSM"]
        print(f"[INFO] Available attacks in torchattacks: {available}")
        return available

    @staticmethod
    def get_attack_fallback_map() -> dict:
        return {"IFGSM": "BIM", "JSMA": "PGD", "PGD-L2": "PGD"}

    @staticmethod
    def instantiate_attack(attack_name: str, model: nn.Module, epsilon: float = 0.03, steps: int = 40):
        print(f"[ATTACK][SETUP] Initializing attack='{attack_name}' eps={epsilon:.6f} steps={steps}")
        try:
            if attack_name == "PGD":
                attack = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / steps, steps=steps)
            elif attack_name == "FGSM":
                attack = torchattacks.FGSM(model, eps=epsilon)
            elif attack_name == "BIM":
                attack = torchattacks.BIM(model, eps=epsilon, alpha=epsilon / steps, steps=steps)
            elif attack_name == "APGD":
                attack = torchattacks.APGD(model, eps=epsilon, steps=steps)
            elif attack_name == "Square":
                attack = torchattacks.Square(model, eps=epsilon, n_queries=5000)
            elif attack_name == "AutoAttack":
                attack = torchattacks.AutoAttack(model, norm="Linf", eps=epsilon, version="standard", verbose=False)
            elif attack_name == "CW":
                attack = torchattacks.CW(model, c=1, lr=0.01, steps=1000, kappa=0)
            elif attack_name == "DeepFool":
                attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
            else:
                raise ValueError(f"Unsupported attack: {attack_name}")
            print(f"[ATTACK][SETUP] Ready: {attack.__class__.__name__}")
            return attack
        except Exception as exc:
            print(f"[WARN] Failed to instantiate {attack_name}: {exc}. Skipping.")
            print(f"[ATTACK][ERROR] setup traceback:\n{traceback.format_exc().rstrip()}")
            return None

    @classmethod
    def generate_adversarial_dataset(
        cls,
        model: nn.Module,
        loader,
        attack_name: str,
        epsilon: float = 0.03,
        steps: int = 40,
    ) -> dict | None:
        model.eval()
        device = next(cls._unwrap_model(model).parameters()).device
        attack = cls.instantiate_attack(attack_name, model, epsilon, steps)
        if attack is None:
            return None

        clean_images = []
        adv_images = []
        true_labels = []
        clean_predictions = []
        adv_predictions = []
        running_seen = 0
        running_clean_correct = 0
        running_adv_correct = 0
        t0 = time.perf_counter()

        try:
            total_batches = len(loader) if hasattr(loader, "__len__") else -1
            # Use a lightweight DataLoader to avoid OOM in workers (especially for large models
            # like ConvNeXt on TinyImageNet). We create a single‑process loader with a modest batch size.
            from torch.utils.data import DataLoader
            # Preserve the original dataset if available; otherwise fall back to the iterator.
            dataset = getattr(loader, "dataset", None)
            batch_size = getattr(loader, "batch_size", 32)
            # Clamp batch size to a safe default (32) for attack generation.
            safe_batch = min(batch_size, 32)
            if dataset is not None:
                safe_loader = DataLoader(dataset, batch_size=safe_batch, shuffle=False, num_workers=0, pin_memory=False)
                iterator = iter(safe_loader)
            else:
                # If the loader is already an iterator (unlikely), just reuse it.
                iterator = iter(loader)

            for batch_idx, (imgs, lbls) in enumerate(iterator, start=1):
                imgs, lbls = imgs.to(device), lbls.to(device)
                with torch.no_grad():
                    clean_preds = model(imgs).argmax(dim=1)
                adv = attack(imgs, lbls)
                with torch.no_grad():
                    adv_preds = model(adv).argmax(dim=1)

                batch_size = lbls.size(0)
                running_seen += batch_size
                running_clean_correct += (clean_preds == lbls).sum().item()
                running_adv_correct += (adv_preds == lbls).sum().item()

                if batch_idx == 1 or batch_idx % 10 == 0 or (total_batches > 0 and batch_idx == total_batches):
                    clean_acc_running = running_clean_correct / max(1, running_seen)
                    adv_acc_running = running_adv_correct / max(1, running_seen)
                    print(
                        f"[ATTACK][BATCH] {attack_name} batch={batch_idx}/{total_batches if total_batches > 0 else '?'} "
                        f"seen={running_seen} clean_acc={clean_acc_running:.2%} adv_acc={adv_acc_running:.2%}"
                    )

                clean_images.append(imgs.cpu())
                adv_images.append(adv.cpu())
                true_labels.append(lbls.cpu())
                clean_predictions.append(clean_preds.cpu())
                adv_predictions.append(adv_preds.cpu())

            clean_acc_final = running_clean_correct / max(1, running_seen)
            adv_acc_final = running_adv_correct / max(1, running_seen)
            print(
                f"[ATTACK][DONE] attack='{attack_name}' samples={running_seen} "
                f"clean_acc={clean_acc_final:.2%} adv_acc={adv_acc_final:.2%} "
                f"asr={(1.0 - adv_acc_final):.2%} runtime={time.perf_counter() - t0:.2f}s"
            )
            return {
                "clean_images": torch.cat(clean_images),
                "adversarial_images": torch.cat(adv_images),
                "true_labels": torch.cat(true_labels),
                "source_clean_predictions": torch.cat(clean_predictions),
                "source_adversarial_predictions": torch.cat(adv_predictions),
            }
        except Exception as exc:
            print(f"[ERROR] Attack generation failed for {attack_name}: {exc}")
            print(f"[ATTACK][ERROR] traceback:\n{traceback.format_exc().rstrip()}")
            return None

    @staticmethod
    def load_adversarial_bundle(adv_path: str) -> dict:
        payload = torch.load(adv_path)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, tuple) and len(payload) == 2:
            adv_imgs, adv_lbls = payload
            return {
                "clean_images": None,
                "adversarial_images": adv_imgs,
                "true_labels": adv_lbls,
                "source_clean_predictions": None,
                "source_adversarial_predictions": None,
            }
        raise RuntimeError(f"Unsupported adversarial dataset format in {adv_path}")

    @classmethod
    def discover_existing_adversarial_artifacts(
        cls,
        output_dir: str,
        model_filter: str = None,
        dataset_filter: str = None,
        attack_filter: str = None,
        kind_filter: str = None,
    ) -> dict:
        """Reconstruct saved adversarial artifact mapping from an output directory."""
        adv_datasets: dict[tuple[str, str, str, str], str] = {}
        if not os.path.isdir(output_dir):
            return adv_datasets

        for name in sorted(os.listdir(output_dir)):
            if not name.endswith("_adv.pt"):
                continue

            adv_path = os.path.join(output_dir, name)
            try:
                payload = cls.load_adversarial_bundle(adv_path)
            except Exception as exc:
                print(f"[WARN] Could not read adversarial artifact {adv_path}: {exc}")
                continue

            model_name = payload.get("source_model") or payload.get("model")
            dataset_name = payload.get("dataset")
            kind = payload.get("kind")
            attack_name = payload.get("attack")
            if not all([model_name, dataset_name, kind, attack_name]):
                print(f"[WARN] Skipping adversarial artifact with missing metadata: {adv_path}")
                continue

            if model_filter and model_filter != "ALL" and model_name != model_filter:
                continue
            if dataset_filter and dataset_filter != "ALL":
                base_dataset = CheckpointManager.base_dataset_name(dataset_name).lower()
                if base_dataset != dataset_filter.lower():
                    continue
            if attack_filter and attack_filter != "ALL" and attack_name != attack_filter:
                continue
            if kind_filter and kind_filter != "ALL" and kind != kind_filter:
                continue

            adv_datasets[(model_name, dataset_name, kind, attack_name)] = adv_path

        print(f"[INFO] Discovered {len(adv_datasets)} saved adversarial artifacts in {output_dir}")
        return adv_datasets

    @staticmethod
    def count_model_parameters(model: nn.Module) -> int:
        return int(sum(p.numel() for p in model.parameters()))

    @classmethod
    def generate_attacks_phase(
        cls,
        output_dir: str,
        model_filter: str = None,
        dataset_filter: str = None,
        attack_filter: str = None,
        kind_filter: str = None,
    ):
        os.makedirs(output_dir, exist_ok=True)
        records = []
        adv_datasets = {}
        model_cache: dict[tuple[str, str, str], nn.Module] = {}
        loader_cache: dict[str, tuple] = {}
        device = cls._default_device()

        checkpoints = CheckpointManager.discover_checkpoints()
        available_attacks = cls.get_available_attacks()
        fallback_map = cls.get_attack_fallback_map()

        if attack_filter:
            if attack_filter in available_attacks:
                attacks = [attack_filter]
            elif attack_filter in fallback_map:
                fallback = fallback_map[attack_filter]
                print(f"[INFO] Requested attack '{attack_filter}' not available. Using '{fallback}'.")
                attacks = [fallback]
            else:
                print(f"[WARN] Requested attack '{attack_filter}' not available. Available: {available_attacks}")
                attacks = []
        else:
            attacks = available_attacks

        for model_name, dataset_name, kind, ckpt_path in checkpoints:
            if not CheckpointManager.dataset_matches_output_dir(dataset_name, output_dir):
                continue
            # ``*_FILTER`` arguments may be set to "ALL" by the shell scripts to indicate
            # no filtering. Treat both ``None`` and the literal string "ALL" as a no‑op.
            if model_filter and model_filter != "ALL" and model_name != model_filter:
                continue
            # Allow filtering by base dataset name (ignore split tag). ``ALL`` means no filter.
            if dataset_filter and dataset_filter != "ALL":
                # Perform case‑insensitive comparison because shell arguments may use
                # capitalised names (e.g., "TinyImageNet"). Internally dataset identifiers
                # are stored in lower‑case.
                base_check = CheckpointManager.base_dataset_name(dataset_name).lower()
                if base_check != dataset_filter.lower():
                    continue
            if kind_filter and kind_filter != "ALL" and kind != kind_filter:
                continue

            # Dataset strings may now include a split tag (e.g. "Cifar10_epochs100_pretrain300").
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)

            if base_dataset == "Cifar10":
                if "Cifar10" not in loader_cache:
                    loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
                train_loader, test_loader = loader_cache["Cifar10"]
                num_classes = 10
            elif base_dataset == "Cifar100":
                if "Cifar100" not in loader_cache:
                    loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
                train_loader, test_loader = loader_cache["Cifar100"]
                num_classes = 100
            elif base_dataset.lower() == "tinyimagenet":
                # Tiny ImageNet has 200 classes; use the same batch size / workers as CIFAR loaders.
                if "tinyimagenet" not in loader_cache:
                    loader_cache["tinyimagenet"] = load_tiny_imagenet(batch_size=256, num_workers=4)
                train_loader, test_loader = loader_cache["tinyimagenet"]
                num_classes = 200
            else:
                # Skip unsupported datasets.
                continue

            one_batch = next(iter(train_loader))[0]
            try:
                model = CheckpointManager.build_model_for_checkpoint(
                    model_name, dataset_name, kind, num_classes, one_batch, ckpt_path, device
                )
                load_result = cls.robust_load_state_dict(model, ckpt_path)
                print(
                    f"[DEBUG] load_state_dict result: missing={len(load_result.missing_keys)}, "
                    f"unexpected={len(load_result.unexpected_keys)}"
                )
            except Exception as exc:
                print(f"[WARN] Failed to load checkpoint for {model_name} ({kind}) on {dataset_name}: {exc}")
                continue

            param_count = cls.count_model_parameters(model)
            if device == "cuda":
                model = torch.nn.DataParallel(model)
            model_cache[(model_name, dataset_name, kind)] = model

            clean_acc, latency_ms = cls.evaluate_model(model, test_loader, return_latency_ms=True)
            for attack_name in attacks:
                adv_bundle = cls.generate_adversarial_dataset(model, test_loader, attack_name)
                if adv_bundle is None:
                    continue

                adv_imgs = adv_bundle["adversarial_images"]
                adv_lbls = adv_bundle["true_labels"]
                adv_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(adv_imgs, adv_lbls),
                    batch_size=256,
                    shuffle=False,
                )
                adv_acc = cls.evaluate_clean_accuracy(model, adv_loader)

                adv_path = os.path.join(output_dir, f"{model_name}_{dataset_name}_{kind}_{attack_name}_adv.pt")
                cls.safe_torch_save(
                    {
                        "source_model": model_name,
                        "dataset": dataset_name,
                        "kind": kind,
                        "attack": attack_name,
                        **adv_bundle,
                    },
                    adv_path,
                )
                adv_datasets[(model_name, dataset_name, kind, attack_name)] = adv_path

                records.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "kind": kind,
                        "attack": attack_name,
                        "model_label": ReportingSuite.model_kind_label(model_name, kind),
                        "param_count": param_count,
                        "latency_ms": float(latency_ms),
                        "clean_acc": clean_acc,
                        "adv_acc": adv_acc,
                        **ReportingSuite.summarize_direct_metrics(clean_acc, adv_acc),
                    }
                )

        return records, model_cache, loader_cache, adv_datasets

    @classmethod
    def compute_transfer_metrics(
        cls,
        source_model: nn.Module,
        target_model: nn.Module,
        adversarial_loader,
        source_clean_predictions=None,
        source_adversarial_predictions=None,
        true_labels=None,
        source_attack_success_rate: float | None = None,
    ) -> dict:
        """Compute aggregate transfer success between a source and target model."""
        target_acc = cls.evaluate_clean_accuracy(target_model, adversarial_loader)
        transfer_success_rate = 1.0 - target_acc

        conditioned_success_rate = np.nan
        if source_clean_predictions is not None and source_adversarial_predictions is not None and true_labels is not None:
            device = next(cls._unwrap_model(target_model).parameters()).device
            target_predictions = []
            with torch.no_grad():
                for imgs, _ in adversarial_loader:
                    imgs = imgs.to(device)
                    preds = target_model(imgs).argmax(dim=1).cpu()
                    target_predictions.append(preds)
            if target_predictions:
                target_preds = torch.cat(target_predictions)
                mask = (source_clean_predictions == true_labels) & (source_adversarial_predictions != true_labels)
                if mask.sum().item() > 0:
                    target_correct = (target_preds == true_labels)
                    conditioned_success_rate = 1.0 - float(target_correct[mask].float().mean())

        normalized_transfer_rate = np.nan
        normalized_conditioned_rate = np.nan
        if source_attack_success_rate is not None and float(source_attack_success_rate) > 0:
            normalized_transfer_rate = transfer_success_rate / float(source_attack_success_rate)
            if not np.isnan(conditioned_success_rate):
                normalized_conditioned_rate = conditioned_success_rate / float(source_attack_success_rate)

        return {
            "transfer_acc": float(target_acc),
            "transfer_success_rate": float(transfer_success_rate),
            "conditioned_transfer_success_rate": float(conditioned_success_rate) if not np.isnan(conditioned_success_rate) else np.nan,
            "source_attack_success_rate": float(source_attack_success_rate) if source_attack_success_rate is not None else np.nan,
            "normalized_transfer_rate": float(normalized_transfer_rate) if not np.isnan(normalized_transfer_rate) else np.nan,
            "normalized_conditioned_transfer_rate": float(normalized_conditioned_rate) if not np.isnan(normalized_conditioned_rate) else np.nan,
        }

    @classmethod
    def analyze_transferability_phase(
        cls,
        output_dir: str,
        model_cache: dict,
        adv_datasets: dict,
        transferability_output: str | None = None,
    ):
        import pandas as pd

        phase_start = time.perf_counter()
        summary_path = os.path.join(output_dir, "summary.csv")
        records_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        records_df = ReportingSuite.enrich_summary_dataframe(records_df)
        loader_cache: dict = {}
        checkpoint_catalog = list(CheckpointManager.discover_checkpoints())
        if transferability_output:
            transfer_path = transferability_output if os.path.isabs(transferability_output) else os.path.join(output_dir, transferability_output)
        else:
            transfer_path = os.path.join(output_dir, "transferability.csv")

        if os.path.exists(transfer_path):
            try:
                os.remove(transfer_path)
                print(f"[INFO] Removed stale transferability CSV before rebuild: {transfer_path}")
            except OSError as exc:
                print(f"[WARN] Could not remove stale transferability CSV {transfer_path}: {exc}")

        print(
            f"[INFO] Starting transferability analysis for {len(adv_datasets)} adversarial artifacts "
            f"across {len(checkpoint_catalog)} discovered checkpoints."
        )

        def _loader_for_dataset(dataset_name: str):
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)
            if base_dataset == "Cifar10":
                if "Cifar10" not in loader_cache:
                    loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
                return loader_cache["Cifar10"], 10
            if base_dataset == "Cifar100":
                if "Cifar100" not in loader_cache:
                    loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
                return loader_cache["Cifar100"], 100
            if base_dataset.lower() == "tinyimagenet":
                if "tinyimagenet" not in loader_cache:
                    loader_cache["tinyimagenet"] = load_tiny_imagenet(batch_size=256, num_workers=4)
                return loader_cache["tinyimagenet"], 200
            raise ValueError(f"Unsupported dataset for transfer analysis: {dataset_name}")

        def _source_attack_success_rate(src_model: str, src_dataset: str, src_kind: str, src_attack: str) -> float:
            if records_df.empty:
                return np.nan
            match = records_df[
                (records_df["model"] == src_model)
                & (records_df["dataset"] == src_dataset)
                & (records_df["kind"] == src_kind)
                & (records_df["attack"] == src_attack)
            ]
            if match.empty:
                return np.nan
            return float(match.iloc[0].get("attack_success_rate", np.nan))

        # Group source artifacts by base dataset so we can load one target model at a time.
        source_groups: dict[str, list[tuple[tuple[str, str, str, str], str]]] = {}
        for source_key, adv_path in adv_datasets.items():
            _src_model, src_dataset, _src_kind, _src_attack = source_key
            base_dataset = CheckpointManager.base_dataset_name(src_dataset)
            source_groups.setdefault(base_dataset, []).append((source_key, adv_path))

        rows_written = 0
        for dataset_index, (base_dataset, sources) in enumerate(source_groups.items(), start=1):
            dataset_start = time.perf_counter()
            print(
                f"[INFO] Dataset group {dataset_index}/{len(source_groups)}: {base_dataset} "
                f"with {len(sources)} adversarial artifacts"
            )

            try:
                (loader_pair, num_classes) = _loader_for_dataset(base_dataset)
            except Exception as exc:
                print(f"[ERROR] Could not load dataset helpers for {base_dataset}: {exc}")
                continue

            train_loader, _ = loader_pair
            one_batch = next(iter(train_loader))[0]

            dataset_checkpoints = [
                (model_name, dataset_name, kind, ckpt_path)
                for model_name, dataset_name, kind, ckpt_path in checkpoint_catalog
                if CheckpointManager.dataset_matches_output_dir(dataset_name, output_dir)
                and CheckpointManager.base_dataset_name(dataset_name) == base_dataset
            ]
            print(
                f"[INFO] Compatible target checkpoints for {base_dataset}: {len(dataset_checkpoints)}"
            )

            if not dataset_checkpoints:
                print(f"[WARN] No target checkpoints found for dataset {base_dataset}")
                continue

            for source_index, ((src_model, src_dataset, src_kind, src_attack), adv_path) in enumerate(sources, start=1):
                source_start = time.perf_counter()
                print(
                    f"[INFO] Transfer source {source_index}/{len(sources)} in {base_dataset}: "
                    f"model={src_model} dataset={src_dataset} kind={src_kind} attack={src_attack}"
                )
                try:
                    adv_bundle = cls.load_adversarial_bundle(adv_path)
                except Exception as exc:
                    print(f"[ERROR] Failed to load adversarial bundle at {adv_path}: {exc}")
                    continue

                true_labels = adv_bundle.get("true_labels")
                if true_labels is None:
                    print(f"[WARN] true_labels missing in bundle for {src_model}/{src_dataset}/{src_attack}")
                    continue

                adv_images = adv_bundle.get("adversarial_images")
                if adv_images is None:
                    print(f"[WARN] adversarial_images missing in bundle for {src_model}/{src_dataset}/{src_attack}")
                    continue

                adv_loader = torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(adv_images, true_labels),
                    batch_size=256,
                    shuffle=False,
                )

                src_clean_preds = adv_bundle.get("source_clean_predictions")
                src_adv_preds = adv_bundle.get("source_adversarial_predictions")
                source_attack_success_rate = _source_attack_success_rate(src_model, src_dataset, src_kind, src_attack)

                source_records = []
                for target_index, (tgt_model, tgt_dataset, tgt_kind, ckpt_path) in enumerate(dataset_checkpoints, start=1):
                    target_start = time.perf_counter()
                    tgt_model_obj = None
                    print(
                        f"[DEBUG] Loading target model {target_index}/{len(dataset_checkpoints)}: "
                        f"{tgt_model} ({tgt_kind}) from {ckpt_path}"
                    )
                    try:
                        tgt_model_obj = CheckpointManager.build_model_for_checkpoint(
                            tgt_model,
                            tgt_dataset,
                            tgt_kind,
                            num_classes,
                            one_batch,
                            ckpt_path,
                            device=cls._default_device(),
                        )
                        cls.robust_load_state_dict(tgt_model_obj, ckpt_path)
                        if torch.cuda.is_available():
                            tgt_model_obj = torch.nn.DataParallel(tgt_model_obj)

                        transfer_metrics = cls.compute_transfer_metrics(
                            source_model=None,
                            target_model=tgt_model_obj,
                            adversarial_loader=adv_loader,
                            source_clean_predictions=src_clean_preds,
                            source_adversarial_predictions=src_adv_preds,
                            true_labels=true_labels,
                            source_attack_success_rate=source_attack_success_rate,
                        )
                    except Exception as exc:
                        print(f"[ERROR] Transfer metric computation failed for target {tgt_model} ({tgt_kind}): {exc}")
                        continue
                    finally:
                        try:
                            if tgt_model_obj is not None:
                                del tgt_model_obj
                        except Exception:
                            pass
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    record = {
                        "source_model": src_model,
                        "source_kind": src_kind,
                        "source_label": ReportingSuite.model_kind_label(src_model, src_kind),
                        "source_attack": src_attack,
                        "target_model": tgt_model,
                        "target_kind": tgt_kind,
                        "target_label": ReportingSuite.model_kind_label(tgt_model, tgt_kind),
                        "dataset": src_dataset,
                        "transfer_acc": transfer_metrics["transfer_acc"],
                        "transfer_success_rate": transfer_metrics["transfer_success_rate"],
                        "conditioned_transfer_success_rate": transfer_metrics["conditioned_transfer_success_rate"],
                        "source_attack_success_rate": transfer_metrics["source_attack_success_rate"],
                        "normalized_transfer_rate": transfer_metrics["normalized_transfer_rate"],
                        "normalized_conditioned_transfer_rate": transfer_metrics["normalized_conditioned_transfer_rate"],
                        "same_architecture": src_model == tgt_model,
                        "same_kind": src_kind == tgt_kind,
                        "pair_type": ReportingSuite.classify_transfer_pair(src_model, src_kind, tgt_model, tgt_kind),
                    }
                    source_records.append(record)

                    if target_index == 1 or target_index % 10 == 0 or target_index == len(dataset_checkpoints):
                        print(
                            f"[DEBUG] Completed target {target_index}/{len(dataset_checkpoints)} for {src_model}/{src_attack} "
                            f"in {time.perf_counter() - target_start:.2f}s"
                        )

                if source_records:
                    source_df = pd.DataFrame(source_records)
                    if _append_locked_csv(transfer_path, source_df, "transferability"):
                        rows_written += len(source_df)
                        print(
                            f"[INFO] Appended {len(source_df)} transferability rows for "
                            f"{src_model}/{src_attack} to {transfer_path} in "
                            f"{time.perf_counter() - source_start:.2f}s"
                        )
                    else:
                        print(
                            f"[WARN] Failed to append transferability rows for {src_model}/{src_attack} "
                            f"to {transfer_path}"
                        )
                else:
                    print(
                        f"[WARN] No transferability rows generated for source "
                        f"{src_model}/{src_dataset}/{src_kind}/{src_attack}"
                    )

                print(
                    f"[INFO] Completed transfer source {source_index}/{len(sources)} for "
                    f"{src_model}/{src_attack} in {time.perf_counter() - source_start:.2f}s"
                )

            print(
                f"[INFO] Completed dataset group {base_dataset} in {time.perf_counter() - dataset_start:.2f}s"
            )

        if rows_written > 0:
            print(
                f"[INFO] Wrote transferability CSV to {transfer_path} with {rows_written} rows "
                f"in {time.perf_counter() - phase_start:.2f}s"
            )
        else:
            print("[WARN] No transferability rows were generated.")

        return rows_written

    @classmethod
    def rebuild_model_and_loader_cache(cls, args) -> tuple[dict, dict]:
        """Rebuild the model and loader cache for a single analysis shard.

        The analysis jobs already receive a shard-specific ``--dataset`` filter
        from the orchestrator, so we only need to cache checkpoints that belong
        to that dataset. This keeps the shard from loading unrelated models and
        makes the logging more useful when a shard ends up empty.
        """
        from collections import OrderedDict

        model_cache: OrderedDict = OrderedDict()
        loader_cache: dict = {}
        device = cls._default_device()
        checkpoints = list(CheckpointManager.discover_checkpoints())
        selected_dataset = None
        if getattr(args, "dataset", None):
            selected_dataset = CheckpointManager.base_dataset_name(args.dataset).lower()
        print(
            f"[INFO] rebuild_model_and_loader_cache starting on device={device} "
            f"with {len(checkpoints)} discovered checkpoints"
        )
        print(
            f"[INFO] Cache filters: model={getattr(args, 'model', None) or 'ALL'} "
            f"dataset={getattr(args, 'dataset', None) or 'ALL'} kind={getattr(args, 'kind', None) or 'ALL'}"
        )

        skipped_output_dir = 0
        skipped_model = 0
        skipped_dataset = 0
        skipped_kind = 0
        cached_count = 0

        for checkpoint_index, (model_name, dataset_name, kind, ckpt_path) in enumerate(checkpoints, start=1):
            if not CheckpointManager.dataset_matches_output_dir(dataset_name, args.output_dir):
                skipped_output_dir += 1
                continue
            if args.model and model_name != args.model:
                skipped_model += 1
                continue
            # Allow args.dataset to match base name, ignoring split tag.
            if selected_dataset:
                base_check = CheckpointManager.base_dataset_name(dataset_name)
                if base_check.lower() != selected_dataset:
                    skipped_dataset += 1
                    continue
            if args.kind and kind != args.kind:
                skipped_kind += 1
                continue

            # Dataset may include a split tag (e.g. "Cifar10_epochs100_pretrain300").
            base_dataset = CheckpointManager.base_dataset_name(dataset_name)

            if base_dataset == "Cifar10":
                if "Cifar10" not in loader_cache:
                    loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
                train_loader, _ = loader_cache["Cifar10"]
                num_classes = 10
            elif base_dataset == "Cifar100":
                if "Cifar100" not in loader_cache:
                    loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
                train_loader, _ = loader_cache["Cifar100"]
                num_classes = 100
            elif base_dataset.lower() == "tinyimagenet":
                # Tiny ImageNet loader (200 classes)
                if "tinyimagenet" not in loader_cache:
                    loader_cache["tinyimagenet"] = load_tiny_imagenet(batch_size=256, num_workers=4)
                train_loader, _ = loader_cache["tinyimagenet"]
                num_classes = 200
            else:
                # Skip unsupported datasets.
                continue

            checkpoint_start = time.perf_counter()
            print(
                f"[INFO] Cache rebuild checkpoint {checkpoint_index}/{len(checkpoints)}: "
                f"model={model_name} dataset={dataset_name} kind={kind}"
            )
            one_batch = next(iter(train_loader))[0]
            try:
                print(
                    f"[DEBUG] Building model skeleton for {model_name} ({dataset_name}, {kind}); "
                    f"checkpoint={ckpt_path}"
                )
                model = CheckpointManager.build_model_for_checkpoint(
                    model_name, dataset_name, kind, num_classes, one_batch, ckpt_path, device
                )
                print(f"[DEBUG] Model skeleton ready for {model_name} ({dataset_name}, {kind}); loading state dict")
                cls.robust_load_state_dict(model, ckpt_path)
                print(
                    f"[DEBUG] State dict loaded for {model_name} ({dataset_name}, {kind}) "
                    f"in {time.perf_counter() - checkpoint_start:.2f}s"
                )
            except Exception as exc:
                print(f"[WARN] Could not load {model_name}({kind}) on {dataset_name}: {exc}")
                continue

            if device == "cuda":
                print(f"[DEBUG] Wrapping {model_name} ({dataset_name}, {kind}) with DataParallel")
                model = torch.nn.DataParallel(model)
            model_cache[(model_name, dataset_name, kind)] = model
            cached_count += 1

            print(
                f"[INFO] Cached {model_name} ({dataset_name}, {kind}); total cached models={len(model_cache)}; "
                f"elapsed={time.perf_counter() - checkpoint_start:.2f}s"
            )

        print(
            f"[INFO] rebuild_model_and_loader_cache summary: cached={cached_count} "
            f"skipped_output_dir={skipped_output_dir} skipped_model={skipped_model} "
            f"skipped_dataset={skipped_dataset} skipped_kind={skipped_kind}"
        )

        if selected_dataset and cached_count == 0:
            print(
                f"[WARN] No checkpoints matched dataset filter '{args.dataset}'. "
                f"Check that the shard dataset name matches the saved checkpoint naming."
            )

        return model_cache, loader_cache
