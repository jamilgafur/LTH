"""Core adversarial generation and transfer analysis utilities."""

from __future__ import annotations

import os
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
    def evaluate_clean_accuracy(model: nn.Module, loader) -> float:
        model.eval()
        device = next(AdversarialCore._unwrap_model(model).parameters()).device
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total += lbls.size(0)
        return correct / total if total > 0 else 0.0

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
            for batch_idx, (imgs, lbls) in enumerate(loader, start=1):
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
            if model_filter and model_name != model_filter:
                continue
            # Allow filtering by base dataset name (ignore split tag)
            if dataset_filter:
                base_check = CheckpointManager.base_dataset_name(dataset_name)
                if base_check != dataset_filter:
                    continue
            if kind_filter and kind != kind_filter:
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

            clean_acc = cls.evaluate_clean_accuracy(model, test_loader)
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
                torch.save(
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
                        "clean_acc": clean_acc,
                        "adv_acc": adv_acc,
                        **ReportingSuite.summarize_direct_metrics(clean_acc, adv_acc),
                    }
                )

        return records, model_cache, loader_cache, adv_datasets

    @classmethod
    def analyze_transferability_phase(cls, output_dir: str, model_cache: dict, adv_datasets: dict):
        import pandas as pd

        transfer_records = []
        summary_path = os.path.join(output_dir, "summary.csv")
        records_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
        records_df = ReportingSuite.enrich_summary_dataframe(records_df)

        for (src_model, src_dataset, src_kind, src_attack), adv_path in adv_datasets.items():
            adv_bundle = cls.load_adversarial_bundle(adv_path)
            # Loader for evaluating target model on the full adversarial set (aggregate metric).
            adv_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    adv_bundle["adversarial_images"], adv_bundle["true_labels"]
                ),
                batch_size=256,
                shuffle=False,
            )

            # Extract per‑example source predictions for the conditioned metric.
            src_clean_preds = adv_bundle.get("source_clean_predictions")
            src_adv_preds = adv_bundle.get("source_adversarial_predictions")
            true_labels = adv_bundle["true_labels"]

            for (tgt_model, tgt_dataset, tgt_kind), tgt_model_obj in model_cache.items():
                if tgt_dataset != src_dataset:
                    continue

                # ----- Aggregate transfer success (existing) -----
                tgt_acc = cls.evaluate_clean_accuracy(tgt_model_obj, adv_loader)
                transfer_success_rate = 1.0 - tgt_acc

                # ----- Conditioned transfer success (new) -----
                conditioned_success_rate = np.nan
                if src_clean_preds is not None and src_adv_preds is not None:
                    # Gather target predictions per example.
                    device = next(tgt_model_obj.parameters()).device
                    tgt_preds_all = []
                    for imgs, _ in adv_loader:
                        imgs = imgs.to(device)
                        with torch.no_grad():
                            preds = tgt_model_obj(imgs).argmax(dim=1).cpu()
                        tgt_preds_all.append(preds)
                    tgt_preds = torch.cat(tgt_preds_all)

                    # Boolean mask: source correctly classified clean AND fooled on adversarial.
                    mask = (src_clean_preds == true_labels) & (src_adv_preds != true_labels)
                    if mask.sum().item() > 0:
                        # Transfer success = proportion where target misclassifies.
                        target_correct = (tgt_preds == true_labels)
                        conditioned_success_rate = 1.0 - float(target_correct[mask].float().mean())

                # ----- Normalized rates -----
                source_attack_success_rate = np.nan
                normalized_transfer_rate = np.nan
                normalized_conditioned_rate = np.nan
                if not records_df.empty:
                    match = records_df[
                        (records_df["model"] == src_model)
                        & (records_df["dataset"] == src_dataset)
                        & (records_df["kind"] == src_kind)
                        & (records_df["attack"] == src_attack)
                    ]
                    if not match.empty:
                        source_attack_success_rate = float(match.iloc[0]["attack_success_rate"])
                        if source_attack_success_rate > 0:
                            normalized_transfer_rate = transfer_success_rate / source_attack_success_rate
                            if not np.isnan(conditioned_success_rate):
                                normalized_conditioned_rate = conditioned_success_rate / source_attack_success_rate

                transfer_records.append(
                    {
                        "source_model": src_model,
                        "source_kind": src_kind,
                        "source_label": ReportingSuite.model_kind_label(src_model, src_kind),
                        "source_attack": src_attack,
                        "target_model": tgt_model,
                        "target_kind": tgt_kind,
                        "target_label": ReportingSuite.model_kind_label(tgt_model, tgt_kind),
                        "dataset": src_dataset,
                        "transfer_acc": tgt_acc,
                        "transfer_success_rate": transfer_success_rate,
                        "conditioned_transfer_success_rate": conditioned_success_rate,
                        "source_attack_success_rate": source_attack_success_rate,
                        "normalized_transfer_rate": normalized_transfer_rate,
                        "normalized_conditioned_transfer_rate": normalized_conditioned_rate,
                        "same_architecture": src_model == tgt_model,
                        "same_kind": src_kind == tgt_kind,
                        "pair_type": ReportingSuite.classify_transfer_pair(src_model, src_kind, tgt_model, tgt_kind),
                    }
                )

        return transfer_records

    @classmethod
    def rebuild_model_and_loader_cache(cls, args) -> tuple[dict, dict]:
        model_cache: dict = {}
        loader_cache: dict = {}
        device = cls._default_device()

        for model_name, dataset_name, kind, ckpt_path in CheckpointManager.discover_checkpoints():
            if args.model and model_name != args.model:
                continue
            # Allow args.dataset to match base name, ignoring split tag.
            if args.dataset:
                base_check = CheckpointManager.base_dataset_name(dataset_name)
                if base_check != args.dataset:
                    continue
            if args.kind and kind != args.kind:
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

            one_batch = next(iter(train_loader))[0]
            try:
                model = CheckpointManager.build_model_for_checkpoint(
                    model_name, dataset_name, kind, num_classes, one_batch, ckpt_path, device
                )
                cls.robust_load_state_dict(model, ckpt_path)
            except Exception as exc:
                print(f"[WARN] Could not load {model_name}({kind}) on {dataset_name}: {exc}")
                continue

            if device == "cuda":
                model = torch.nn.DataParallel(model)
            model_cache[(model_name, dataset_name, kind)] = model

        return model_cache, loader_cache
