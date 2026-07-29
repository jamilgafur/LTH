"""Adversarial analysis of pruned vs. original models.

This script generates adversarial examples using multiple attack methods,
evaluates direct attack success rates, and measures transferability across
model architectures and pruning states (collapsed vs. original).

Supported Attacks:
  - PGD (Projected Gradient Descent)
  - FGSM (Fast Gradient Sign Method)
  - IFGSM (Iterative FGSM)
  - BIM (Basic Iterative Method)
  - APGD (AutoAttack's PGD variant)
  - Square (Square Attack – query-based)
  - AutoAttack (Strong ensemble)
  - CW (Carlini-Wagner L2)
  - DeepFool

Requirements
------------
* ``torch`` and ``torchvision`` – already used in the repository.
* ``torchattacks`` – a lightweight library providing common attacks.
  Install with ``pip install torchattacks`` if not already available.
* ``pandas``, ``seaborn``, and ``matplotlib`` for result aggregation and plotting.

Usage
-----
Run from the repository root:

  # Full pipeline: generate all attacks and analyze transferability
  python manuscript/Tranfer/adversarial_analysis.py --mode full

  # Generate attacks only
  python manuscript/Tranfer/adversarial_analysis.py --mode generate --model VGG16 --dataset Cifar10 --attack PGD

  # Analyze transferability only (requires pre-generated attacks)
  python manuscript/Tranfer/adversarial_analysis.py --mode analyze

  # Plot results only
  python manuscript/Tranfer/adversarial_analysis.py --mode plot

Output Files
------------
``adversarial_results/`` directory contains:
* ``summary.csv`` – Direct attack and susceptibility metrics per model/dataset/kind/attack
* ``transferability.csv`` – Cross-model transferability rates and normalized transfer metrics
* Plots:
    - ``attack_success_rates.png`` – Attack success rate by model, attack, and kind
    - ``direct_attack_success_heatmap_*.png`` – Heatmaps of direct fooling rate by model and attack
    - ``collapsed_vs_original_delta_*.png`` – Heatmaps of collapsed minus original susceptibility
    - ``transferability_heatmap_full_*.png`` – Full source-to-target transfer matrices including kind
    - ``transferability_matrix_full_*.csv`` – Full source-to-target transfer matrices including kind
  - ``attack_comparison_*.png`` – Comparison of attack effectiveness by dataset
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import json
from typing import Dict, Tuple, List
import numpy as np

# Attempt to import torchattacks; provide a helpful message if missing.
try:
    import torchattacks
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torchattacks is required for adversarial attacks. Install it via "
        "'pip install torchattacks' and re-run the script."
    ) from e

# Import model definitions and utility functions from the existing codebase.
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.utils import load_cifar10, load_cifar100

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Checkpoint handling – we now support both finetuned (collapsed) and original
# (uncollapsed) checkpoints. The ``CHECKPOINT_BASES`` and ``CHECKPOINT_FILES``
# dictionaries from ``transfer.py`` define the locations of these files. We
# construct absolute paths for each model/dataset/kind combination.
# ---------------------------------------------------------------------------

import glob
import re
from transfer import CHECKPOINT_BASES, CHECKPOINT_FILES
from collapse import collapse_only, _wrap_pools_safe

# =========================================================================
# Robust Checkpoint Loading Helper
# =========================================================================

def robust_load_state_dict(model: nn.Module, ckpt_path: str):
    """Load a checkpoint robustly, handling multiple formats and DataParallel wrapping.
    
    Handles:
    - State dict wrapped under "model_state_dict", "model", or "state_dict" keys
    - State dict directly (plain dict of tensors)
    - DataParallel-saved checkpoints ("module." prefix in keys)
    - Mismatched architectures (loads with strict=False)
    
    Args:
        model: The PyTorch model to load into.
        ckpt_path: Path to the checkpoint file.
    
    Raises:
        RuntimeError: If checkpoint format is unexpected.
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # Extract the actual state dict from possible wrapper keys
    if isinstance(state, dict):
        sd = (
            state.get("model_state_dict")
            or state.get("model")
            or state.get("state_dict")
            or state
        )
    else:
        raise RuntimeError(
            f"Unexpected checkpoint format for {ckpt_path}: {type(state)}"
        )

    # Remove the DataParallel "module." prefix if it exists
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    # Load with strict=False to allow mismatched keys
    # (missing keys will be left at initialization, extra keys in checkpoint are ignored)
    return model.load_state_dict(sd, strict=False)

# Order of models and datasets as defined in ``transfer.py``.
MODEL_ORDER = ["VGG16", "RegNetX_400MF", "InceptionNet", "MobileNet", "XceptionNet", "ConvNeXt"]
DATASET_ORDER = ["Cifar10", "Cifar100", "imagenet", "tinyimagenet"]


def get_checkpoint_path(model: str, dataset: str, kind: str):
    import glob
    import os
    # Find directories matching this model and dataset
    pattern = f"{model}_{dataset}_*"
    dirs = glob.glob(pattern)
    if not dirs:
        return None
        
    # Prioritize epochs100_pretrain300 if it exists
    target_dir = dirs[0]
    for d in dirs:
        if "epochs100_pretrain300" in d:
            target_dir = d
            break
            
    base_dir = target_dir
    filename = "final_JF_Control.pt" if kind == "Original" else "final_JF_Dynamic_Region_All_Combined.pt"
    full_path = os.path.abspath(os.path.join(base_dir, "checkpoints", filename))
    
    if os.path.exists(full_path):
        return full_path
    return None


def get_model_kwargs(model_name: str, num_classes: int, one_batch: torch.Tensor | None) -> dict:
    kwargs = {"num_classes": num_classes}
    if one_batch is not None:
        kwargs["one_batch"] = one_batch
    if model_name == "InceptionNet":
        kwargs["aux_logits"] = False
    return kwargs


def get_discovered_regions_path(model_name: str, dataset_name: str, ckpt_path: str) -> str | None:
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    base_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = f"{model_name}_{dataset_name}_None_"

    if run_dir.startswith(prefix):
        budget = run_dir[len(prefix):]
        candidate = os.path.join(
            base_dir,
            f"{model_name}_{dataset_name}_{budget}_JF_discovered_regions.json",
        )
        if os.path.exists(candidate):
            return candidate

    matches = sorted(
        glob.glob(os.path.join(base_dir, f"{model_name}_{dataset_name}_epochs*_pretrain*_JF_discovered_regions.json"))
    )
    return matches[0] if matches else None


def get_compression_set_for_checkpoint(model_name: str, dataset_name: str, ckpt_path: str):
    json_path = get_discovered_regions_path(model_name, dataset_name, ckpt_path)
    if not json_path:
        raise FileNotFoundError(
            f"No discovered regions JSON found for {model_name}/{dataset_name} at {ckpt_path}"
        )

    with open(json_path, "r") as handle:
        regions = json.load(handle)

    compression_set = regions.get("Dynamic_Region_All_Combined")
    if not compression_set:
        raise KeyError(
            f"Dynamic_Region_All_Combined not found in {json_path}"
        )

    print(f"[DEBUG] Discovered regions path: {json_path}")
    print(f"[DEBUG] Collapse ranges loaded: {len(compression_set)}")
    return compression_set


def build_model_for_checkpoint(
    model_name: str,
    dataset_name: str,
    kind: str,
    num_classes: int,
    one_batch: torch.Tensor,
    ckpt_path: str,
    device: str,
) -> nn.Module:
    model_kwargs = get_model_kwargs(model_name, num_classes, one_batch)
    model = load_model(model_name, num_classes, one_batch=one_batch)
    _wrap_pools_safe(model)

    if kind == "Finetuned":
        compression_set = get_compression_set_for_checkpoint(model_name, dataset_name, ckpt_path)
        print(f"[DEBUG] Rebuilding collapsed architecture for {model_name} ({dataset_name}) before loading weights")
        model = collapse_only(
            model=model,
            compression_set=compression_set,
            input_shape=one_batch.shape,
            device=device,
            dry_run=False,
            debug=False,
            handle_skips=True,
        )
    else:
        model = model.to(device)

    return model

def discover_checkpoints():
    entries = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for kind in ("Finetuned", "Original"):
                path = get_checkpoint_path(model, dataset, kind)
                if path:
                    entries.append((model, dataset, kind, path))
    return entries

def load_model(model_name: str, num_classes: int, one_batch: torch.Tensor | None = None) -> nn.Module:
    """Instantiate a model architecture.

    The function maps the string name used in ``CHECKPOINT_BASES`` to the actual
    class imported from ``pyPrune.models``.
    """
    mapping = {
        "VGG16": VGG16,
        "RegNetX_400MF": RegNetX_400MF,
        "ConvNeXt": ConvNeXt,
        "InceptionNet": InceptionNet,
        "MobileNet": MobileNet,
        "XceptionNet": XceptionNet,
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported model: {model_name}")
    model_kwargs = get_model_kwargs(model_name, num_classes, one_batch)
    return mapping[model_name](**model_kwargs)


def evaluate_clean_accuracy(model: nn.Module, loader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)
    return correct / total if total > 0 else 0.0


def get_available_attacks() -> list[str]:
    """Detect which attacks are available in the installed torchattacks version.
    
    Returns:
        List of attack names that are available and can be instantiated.
    """
    available = []
    # Attacks to try
    candidate_attacks = ["PGD", "FGSM", "BIM", "APGD", "CW", "DeepFool", "Square", "AutoAttack"]
    
    for attack_name in candidate_attacks:
        if hasattr(torchattacks, attack_name):
            available.append(attack_name)
    
    if not available:
        # Fallback to at least PGD and FGSM
        available = ["PGD", "FGSM"]
    
    print(f"[INFO] Available attacks in torchattacks: {available}")
    return available


def get_attack_fallback_map() -> dict:
    """Return a mapping of unsupported attack names to their functional equivalents.
    
    ISSUE #5 FIX: Provide fallbacks for removed/unsupported attacks.
    - IFGSM (removed in newer torchattacks) → BIM (functionally identical)
    - Other deprecated attacks map to closest available variants.
    
    Returns:
        Dict mapping missing attack names to their replacements.
    """
    return {
        "IFGSM": "BIM",  # Iterative FGSM is the same as BIM
        "JSMA": "PGD",   # If JSMA is missing, use PGD
        "PGD-L2": "PGD", # L-infinity variant as fallback
    }


def instantiate_attack(attack_name: str, model: nn.Module, epsilon: float = 0.03, steps: int = 40):
    """Instantiate the appropriate attack from torchattacks.
    
    Supported attacks: PGD, FGSM, BIM, APGD, Square, AutoAttack, CW, DeepFool.
    """
    try:
        if attack_name == "PGD":
            return torchattacks.PGD(model, eps=epsilon, alpha=epsilon / steps, steps=steps)
        elif attack_name == "FGSM":
            return torchattacks.FGSM(model, eps=epsilon)
        elif attack_name == "BIM":
            return torchattacks.BIM(model, eps=epsilon, alpha=epsilon / steps, steps=steps)
        elif attack_name == "APGD":
            return torchattacks.APGD(model, eps=epsilon, steps=steps)
        elif attack_name == "Square":
            return torchattacks.Square(model, eps=epsilon, n_queries=5000)
        elif attack_name == "AutoAttack":
            return torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
        elif attack_name == "CW":
            return torchattacks.CW(model, c=1, lr=0.01, steps=1000, kappa=0)
        elif attack_name == "DeepFool":
            return torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")
    except Exception as e:
        print(f"[WARN] Failed to instantiate {attack_name}: {e}. Skipping.")
        return None


def generate_adversarial_dataset(model: nn.Module, loader, attack_name: str, epsilon: float = 0.03, steps: int = 40) -> dict | None:
    """Generate and package a single-source adversarial dataset.

    The saved bundle includes the clean images, adversarial images, ground-truth
    labels, and the source model predictions on both the clean and adversarial
    versions. This makes it possible to inspect the source behavior directly and
    then reuse the adversarial images for transfer studies against all other
    models trained on the same dataset.
    """
    model.eval()
    attack = instantiate_attack(attack_name, model, epsilon, steps)
    if attack is None:
        return None

    clean_images = []
    adv_images = []
    true_labels = []
    clean_predictions = []
    adv_predictions = []
    try:
        for imgs, lbls in loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            with torch.no_grad():
                clean_preds = model(imgs).argmax(dim=1)
            adv = attack(imgs, lbls)
            with torch.no_grad():
                adv_preds = model(adv).argmax(dim=1)

            clean_images.append(imgs.cpu())
            adv_images.append(adv.cpu())
            true_labels.append(lbls.cpu())
            clean_predictions.append(clean_preds.cpu())
            adv_predictions.append(adv_preds.cpu())

        return {
            "clean_images": torch.cat(clean_images),
            "adversarial_images": torch.cat(adv_images),
            "true_labels": torch.cat(true_labels),
            "source_clean_predictions": torch.cat(clean_predictions),
            "source_adversarial_predictions": torch.cat(adv_predictions),
        }
    except Exception as e:
        print(f"[ERROR] Attack generation failed for {attack_name}: {e}")
        return None


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


def evaluate_transferability(source_model: nn.Module, adv_loader, target_models: dict) -> dict:
    """Measure how adversarial examples transfer to other models.

    ``target_models`` is a mapping ``model_name -> nn.Module`` (already on CUDA).
    Returns a dict ``{target_name: accuracy}`` where accuracy is computed on the
    adversarial loader.
    """
    results = {}
    for name, tgt in target_models.items():
        acc = evaluate_clean_accuracy(tgt, adv_loader)
        results[name] = acc
    return results


def model_kind_label(model_name: str, kind: str) -> str:
    return f"{model_name} ({kind})"


def summarize_direct_metrics(clean_acc: float, adv_acc: float) -> dict:
    accuracy_drop = clean_acc - adv_acc
    attack_success_rate = 1.0 - adv_acc
    relative_accuracy_drop = accuracy_drop / clean_acc if clean_acc > 0 else np.nan
    robustness_ratio = adv_acc / clean_acc if clean_acc > 0 else np.nan
    return {
        "robust_accuracy": adv_acc,
        "clean_error_rate": 1.0 - clean_acc,
        "adv_error_rate": 1.0 - adv_acc,
        "attack_success_rate": attack_success_rate,
        "accuracy_drop": accuracy_drop,
        "relative_accuracy_drop": relative_accuracy_drop,
        "robustness_ratio": robustness_ratio,
    }


def classify_transfer_pair(src_model: str, src_kind: str, tgt_model: str, tgt_kind: str) -> str:
    if src_model == tgt_model and src_kind == tgt_kind:
        return "self_same_kind"
    if src_model == tgt_model:
        return "same_arch_cross_kind"
    if src_kind == tgt_kind:
        return "cross_arch_same_kind"
    return "cross_arch_cross_kind"


def enrich_summary_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "model_label" not in df.columns and {"model", "kind"}.issubset(df.columns):
        df["model_label"] = df.apply(lambda row: model_kind_label(row["model"], row["kind"]), axis=1)

    if "accuracy_drop" not in df.columns and {"clean_acc", "adv_acc"}.issubset(df.columns):
        df["accuracy_drop"] = df["clean_acc"] - df["adv_acc"]

    if "robust_accuracy" not in df.columns and "adv_acc" in df.columns:
        df["robust_accuracy"] = df["adv_acc"]

    if "clean_error_rate" not in df.columns and "clean_acc" in df.columns:
        df["clean_error_rate"] = 1.0 - df["clean_acc"]

    if "adv_error_rate" not in df.columns and "adv_acc" in df.columns:
        df["adv_error_rate"] = 1.0 - df["adv_acc"]

    if "attack_success_rate" not in df.columns and "adv_acc" in df.columns:
        df["attack_success_rate"] = 1.0 - df["adv_acc"]

    if "relative_accuracy_drop" not in df.columns and {"accuracy_drop", "clean_acc"}.issubset(df.columns):
        df["relative_accuracy_drop"] = np.where(df["clean_acc"] > 0, df["accuracy_drop"] / df["clean_acc"], np.nan)

    if "robustness_ratio" not in df.columns and {"adv_acc", "clean_acc"}.issubset(df.columns):
        df["robustness_ratio"] = np.where(df["clean_acc"] > 0, df["adv_acc"] / df["clean_acc"], np.nan)

    return df


def enrich_transfer_dataframe(tf_df: pd.DataFrame, records_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if tf_df.empty:
        return tf_df

    if "source_label" not in tf_df.columns:
        tf_df["source_label"] = tf_df.apply(
            lambda row: model_kind_label(row["source_model"], row["source_kind"]), axis=1
        )

    if "target_label" not in tf_df.columns:
        tf_df["target_label"] = tf_df.apply(
            lambda row: model_kind_label(row["target_model"], row["target_kind"]), axis=1
        )

    if "transfer_success_rate" not in tf_df.columns and "transfer_acc" in tf_df.columns:
        tf_df["transfer_success_rate"] = 1.0 - tf_df["transfer_acc"]

    if "same_architecture" not in tf_df.columns:
        tf_df["same_architecture"] = tf_df["source_model"] == tf_df["target_model"]

    if "same_kind" not in tf_df.columns:
        tf_df["same_kind"] = tf_df["source_kind"] == tf_df["target_kind"]

    if "pair_type" not in tf_df.columns:
        tf_df["pair_type"] = tf_df.apply(
            lambda row: classify_transfer_pair(
                row["source_model"], row["source_kind"], row["target_model"], row["target_kind"]
            ),
            axis=1,
        )

    if records_df is not None and not records_df.empty and "source_attack_success_rate" not in tf_df.columns:
        lookup = records_df[["model", "dataset", "kind", "attack", "attack_success_rate"]].rename(
            columns={
                "model": "source_model",
                "dataset": "dataset",
                "kind": "source_kind",
                "attack": "source_attack",
                "attack_success_rate": "source_attack_success_rate",
            }
        )
        tf_df = tf_df.merge(
            lookup,
            on=["source_model", "dataset", "source_kind", "source_attack"],
            how="left",
        )

    if "normalized_transfer_rate" not in tf_df.columns and {"transfer_success_rate", "source_attack_success_rate"}.issubset(tf_df.columns):
        tf_df["normalized_transfer_rate"] = np.where(
            tf_df["source_attack_success_rate"] > 0,
            tf_df["transfer_success_rate"] / tf_df["source_attack_success_rate"],
            np.nan,
        )

    return tf_df


def generate_attacks_phase(
    output_dir: str,
    model_filter: str = None,
    dataset_filter: str = None,
    attack_filter: str = None,
    kind_filter: str = None,
):
    """Phase 1: Generate adversarial examples and save datasets.
    
    Args:
        output_dir: Output directory for results.
        model_filter: Filter by model name (e.g., 'VGG16'). None means all.
        dataset_filter: Filter by dataset name (e.g., 'Cifar10'). None means all.
        attack_filter: Filter by attack name (e.g., 'PGD'). None means all.
        kind_filter: Filter by checkpoint kind ('Original' or 'Finetuned'). None means both.
    
    Returns:
        Tuple of (records_list, adv_datasets_dict).
    """
    os.makedirs(output_dir, exist_ok=True)
    records = []
    adv_datasets = {}  # Store paths to generated adversarial datasets

    checkpoints = discover_checkpoints()
    model_cache: dict[tuple[str, str, str], nn.Module] = {}
    loader_cache: dict[str, tuple] = {}

    # Get the attacks available in the installed torchattacks version
    available_attacks = get_available_attacks()
    fallback_map = get_attack_fallback_map()
    
    # Filter by user preference if specified
    if attack_filter:
        # ISSUE #5 FIX: Check fallback map if requested attack is not available
        if attack_filter in available_attacks:
            attacks = [attack_filter]
        elif attack_filter in fallback_map:
            fallback = fallback_map[attack_filter]
            print(f"[INFO] Requested attack '{attack_filter}' not available.")
            print(f"[INFO] Using functional equivalent: '{fallback}'")
            attacks = [fallback]
        else:
            print(f"[WARN] Requested attack '{attack_filter}' not available. Available: {available_attacks}")
            attacks = []
    else:
        attacks = available_attacks

    for model_name, dataset_name, kind, ckpt_path in checkpoints:
        # Apply filters
        if model_filter and model_name != model_filter:
            continue
        if dataset_filter and dataset_name != dataset_filter:
            continue
        if kind_filter and kind != kind_filter:
            continue

        # Load data loaders (cached per dataset).
        if dataset_name == "Cifar10":
            if "Cifar10" not in loader_cache:
                loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
            train_loader, test_loader = loader_cache["Cifar10"]
            num_classes = 10
        elif dataset_name == "Cifar100":
            if "Cifar100" not in loader_cache:
                loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
            train_loader, test_loader = loader_cache["Cifar100"]
            num_classes = 100
        else:
            continue

        # Load the source model.
        print(f"\n[DEBUG] Loading {model_name} ({kind}) on {dataset_name}")
        print(f"[DEBUG] Checkpoint path: {ckpt_path}")
        print(f"[DEBUG] Checkpoint exists: {os.path.exists(ckpt_path)}")

        one_batch = next(iter(train_loader))[0]
        try:
            model = build_model_for_checkpoint(
                model_name=model_name,
                dataset_name=dataset_name,
                kind=kind,
                num_classes=num_classes,
                one_batch=one_batch,
                ckpt_path=ckpt_path,
                device="cuda",
            )
            load_result = robust_load_state_dict(model, ckpt_path)
            print(
                f"[DEBUG] load_state_dict result: missing={len(load_result.missing_keys)}, "
                f"unexpected={len(load_result.unexpected_keys)}"
            )
            
            # ISSUE #1 FIX: Detect ConvNeXt state-dict mismatch early
            if len(load_result.missing_keys) > 0 or len(load_result.unexpected_keys) > 0:
                if model_name == "ConvNeXt" and kind == "Finetuned":
                    print(f"[WARN] ⚠ ConvNeXt ({kind}) has mismatched keys:")
                    print(f"       Missing: {load_result.missing_keys[:3]}... ({len(load_result.missing_keys)} total)")
                    print(f"       Unexpected: {load_result.unexpected_keys[:3]}... ({len(load_result.unexpected_keys)} total)")
                    print(f"[WARN] ⚠ This model's collapsed architecture does not match the saved checkpoint.")
                    print(f"[WARN] ⚠ Results from this model combination may be unreliable.")
                    print(f"[WARN] ⚠ Verify collapse regions JSON and retrain if needed.")
        except Exception as e:
            print(
                f"[WARN] Failed to load checkpoint for {model_name} ({kind}) on {dataset_name}: {e}"
            )
            # Skip this combination and move on to the next one
            continue
        model = torch.nn.DataParallel(model)
        model_cache[(model_name, dataset_name, kind)] = model

        # Clean accuracy - diagnostic check to verify weights were loaded correctly.
        clean_acc = evaluate_clean_accuracy(model, test_loader)
        print(f"[DEBUG] Clean accuracy after loading {kind} checkpoint: {clean_acc:.4f}")
        
        # Sanity check: if accuracy is too low (< 20% for 10-class CIFAR), weights likely didn't load
        if num_classes == 10 and clean_acc < 0.20:
            print(f"[WARN] *** SANITY CHECK FAILED: {kind} model has suspiciously low accuracy ({clean_acc:.2%})")
            print(f"[WARN] *** This suggests the checkpoint may not have loaded correctly!")
            print(f"[WARN] *** Checkpoint path: {ckpt_path}")

        # For each attack, generate adversarial examples.
        for attack_name in attacks:
            adv_bundle = generate_adversarial_dataset(model, test_loader, attack_name)
            if adv_bundle is None:
                continue

            adv_imgs = adv_bundle["adversarial_images"]
            adv_lbls = adv_bundle["true_labels"]

            adv_dataset = torch.utils.data.TensorDataset(adv_imgs, adv_lbls)
            adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=256, shuffle=False)
            adv_acc = evaluate_clean_accuracy(model, adv_loader)

            # Save adversarial dataset.
            adv_path = os.path.join(
                output_dir,
                f"{model_name}_{dataset_name}_{kind}_{attack_name}_adv.pt",
            )
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
                    "model_label": model_kind_label(model_name, kind),
                    "clean_acc": clean_acc,
                    "adv_acc": adv_acc,
                    **summarize_direct_metrics(clean_acc, adv_acc),
                }
            )
            print(f"[INFO] {model_name} ({kind}) {dataset_name} – {attack_name}: clean {clean_acc:.2%}, adv {adv_acc:.2%}")

    return records, model_cache, loader_cache, adv_datasets


def analyze_transferability_phase(output_dir: str, model_cache: dict, loader_cache: dict, adv_datasets: dict):
    """Phase 2: Evaluate transferability of adversarial examples across models."""
    transfer_records = []
    summary_path = os.path.join(output_dir, "summary.csv")
    records_df = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    records_df = enrich_summary_dataframe(records_df)

    for (src_model, src_dataset, src_kind, src_attack), adv_path in adv_datasets.items():
        # Load the pre-generated adversarial dataset.
        adv_bundle = load_adversarial_bundle(adv_path)
        adv_imgs = adv_bundle["adversarial_images"]
        adv_lbls = adv_bundle["true_labels"]
        adv_dataset = torch.utils.data.TensorDataset(adv_imgs, adv_lbls)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=256, shuffle=False)

        # Evaluate on all target models with the same dataset.
        for (tgt_model, tgt_dataset, tgt_kind), tgt_model_obj in model_cache.items():
            if tgt_dataset != src_dataset:
                continue
            
            tgt_acc = evaluate_clean_accuracy(tgt_model_obj, adv_loader)
            transfer_success_rate = 1.0 - tgt_acc
            source_attack_success_rate = np.nan
            normalized_transfer_rate = np.nan
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

            transfer_records.append(
                {
                    "source_model": src_model,
                    "source_kind": src_kind,
                    "source_label": model_kind_label(src_model, src_kind),
                    "source_attack": src_attack,
                    "target_model": tgt_model,
                    "target_kind": tgt_kind,
                    "target_label": model_kind_label(tgt_model, tgt_kind),
                    "dataset": src_dataset,
                    "transfer_acc": tgt_acc,
                    "transfer_success_rate": transfer_success_rate,
                    "source_attack_success_rate": source_attack_success_rate,
                    "normalized_transfer_rate": normalized_transfer_rate,
                    "same_architecture": src_model == tgt_model,
                    "same_kind": src_kind == tgt_kind,
                    "pair_type": classify_transfer_pair(src_model, src_kind, tgt_model, tgt_kind),
                }
            )

    return transfer_records


def generate_plots(output_dir: str, records: List[Dict], transfer_records: List[Dict]):
    """Phase 3: Generate comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Attack Success Rates (Accuracy Drop)
    if records:
        df = enrich_summary_dataframe(pd.DataFrame(records))
        
        # Overall attack success by model, dataset, and kind
        plt.figure(figsize=(14, 6))
        # ISSUE #7 FIX: Deprecated seaborn palette API – add legend parameter
        sns.barplot(data=df, x="model", y="attack_success_rate", hue="attack", palette="Set2", legend=True)
        plt.ylabel("Attack Success Rate", fontweight='bold')
        plt.xlabel("Model Architecture", fontweight='bold')
        plt.title("Adversarial Attack Success Rates Across Models", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attack_success_rates.png"), dpi=300)
        plt.close()
        print(f"[INFO] Saved: attack_success_rates.png")

        # Separate plots for Original vs Finetuned
        for kind in df["kind"].unique():
            df_kind = df[df["kind"] == kind]
            plt.figure(figsize=(14, 6))
            # ISSUE #7 FIX: Deprecated seaborn palette API usage
            sns.barplot(data=df_kind, x="model", y="attack_success_rate", hue="attack", palette="husl", legend=True)
            plt.ylabel("Attack Success Rate", fontweight='bold')
            plt.xlabel("Model Architecture", fontweight='bold')
            plt.title(f"Attack Success Rates – {kind} Models", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attack_success_rates_{kind}.png"), dpi=300)
            plt.close()
            print(f"[INFO] Saved: attack_success_rates_{kind}.png")

        # Attack effectiveness comparison by dataset
        for dataset in df["dataset"].unique():
            df_dataset = df[df["dataset"] == dataset]
            plt.figure(figsize=(12, 5))
            sns.boxplot(data=df_dataset, x="attack", y="attack_success_rate", hue="kind", palette="Set1")
            plt.ylabel("Attack Success Rate", fontweight='bold')
            plt.xlabel("Attack Method", fontweight='bold')
            plt.title(f"Attack Effectiveness Comparison – {dataset}", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attack_comparison_{dataset}.png"), dpi=300)
            plt.close()
            print(f"[INFO] Saved: attack_comparison_{dataset}.png")

            direct_heatmap = df_dataset.pivot_table(
                index="model_label",
                columns="attack",
                values="attack_success_rate",
                aggfunc="mean",
            )
            if direct_heatmap is not None and not direct_heatmap.empty:
                plt.figure(figsize=(10, 7))
                sns.heatmap(
                    direct_heatmap,
                    annot=True,
                    fmt=".2%",
                    cmap="rocket_r",
                    vmin=0,
                    vmax=1,
                    cbar_kws={"label": "Attack Success Rate"},
                )
                plt.title(f"Direct Attack Success Heatmap – {dataset}", fontsize=14, fontweight='bold')
                plt.xlabel("Attack Method", fontweight='bold')
                plt.ylabel("Model Variant", fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"direct_attack_success_heatmap_{dataset}.png"), dpi=300)
                plt.close()
                print(f"[INFO] Saved: direct_attack_success_heatmap_{dataset}.png")

            delta_rows = []
            for model_name in sorted(df_dataset["model"].unique()):
                for attack_name in sorted(df_dataset["attack"].unique()):
                    subset = df_dataset[(df_dataset["model"] == model_name) & (df_dataset["attack"] == attack_name)]
                    if {"Original", "Finetuned"}.issubset(set(subset["kind"])):
                        finetuned_value = float(subset[subset["kind"] == "Finetuned"]["attack_success_rate"].mean())
                        original_value = float(subset[subset["kind"] == "Original"]["attack_success_rate"].mean())
                        delta_rows.append(
                            {
                                "model": model_name,
                                "attack": attack_name,
                                "collapsed_minus_original": finetuned_value - original_value,
                            }
                        )

            if delta_rows:
                delta_df = pd.DataFrame(delta_rows)
                delta_heatmap = delta_df.pivot_table(
                    index="model",
                    columns="attack",
                    values="collapsed_minus_original",
                    aggfunc="mean",
                )
                plt.figure(figsize=(10, 7))
                sns.heatmap(
                    delta_heatmap,
                    annot=True,
                    fmt=".2%",
                    cmap="coolwarm",
                    center=0,
                    cbar_kws={"label": "Collapsed - Original Attack Success"},
                )
                plt.title(f"Collapsed vs Original Susceptibility Delta – {dataset}", fontsize=14, fontweight='bold')
                plt.xlabel("Attack Method", fontweight='bold')
                plt.ylabel("Model Architecture", fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"collapsed_vs_original_delta_{dataset}.png"), dpi=300)
                plt.close()
                print(f"[INFO] Saved: collapsed_vs_original_delta_{dataset}.png")

    # Plot 2: Transferability Heatmaps
    if transfer_records:
        records_df = enrich_summary_dataframe(pd.DataFrame(records)) if records else pd.DataFrame()
        tf_df = enrich_transfer_dataframe(pd.DataFrame(transfer_records), records_df)
        
        # Heatmap for each dataset and attack
        for dataset in tf_df["dataset"].unique():
            for attack in tf_df["source_attack"].unique():
                tf_subset = tf_df[(tf_df["dataset"] == dataset) & (tf_df["source_attack"] == attack)]
                
                # Coarse architecture-only view.
                pivot_data = tf_subset.pivot_table(
                    index="source_model",
                    columns="target_model",
                    values="transfer_success_rate",
                    aggfunc="mean"
                )
                
                if pivot_data is not None and not pivot_data.empty:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="mako", vmin=0, vmax=1,
                                cbar_kws={"label": "Transfer Success Rate"})
                    plt.title(f"Adversarial Transferability – {dataset} ({attack})", fontsize=14, fontweight='bold')
                    plt.xlabel("Target Model", fontweight='bold')
                    plt.ylabel("Source Model", fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"transferability_heatmap_{dataset}_{attack}.png"), dpi=300)
                    plt.close()
                    print(f"[INFO] Saved: transferability_heatmap_{dataset}_{attack}.png")

                # Full model-kind matrix requested for the paper.
                full_pivot = tf_subset.pivot_table(
                    index="source_label",
                    columns="target_label",
                    values="transfer_success_rate",
                    aggfunc="mean",
                )

                if full_pivot is not None and not full_pivot.empty:
                    full_pivot.to_csv(
                        os.path.join(output_dir, f"transferability_matrix_full_{dataset}_{attack}.csv")
                    )
                    plt.figure(figsize=(14, 10))
                    sns.heatmap(
                        full_pivot,
                        annot=True,
                        fmt=".2%",
                        cmap="mako",
                        vmin=0,
                        vmax=1,
                        cbar_kws={"label": "Transfer Success Rate"},
                    )
                    plt.title(f"Full Transferability Matrix – {dataset} ({attack})", fontsize=14, fontweight='bold')
                    plt.xlabel("Target Model Variant", fontweight='bold')
                    plt.ylabel("Source Model Variant", fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"transferability_heatmap_full_{dataset}_{attack}.png"), dpi=300)
                    plt.close()
                    print(f"[INFO] Saved: transferability_heatmap_full_{dataset}_{attack}.png")

                pair_summary = tf_subset.groupby("pair_type", as_index=False)["normalized_transfer_rate"].mean()
                if not pair_summary.empty:
                    plt.figure(figsize=(10, 5))
                    # ISSUE #7 FIX: Deprecated seaborn palette API usage
                    sns.barplot(data=pair_summary, x="pair_type", y="normalized_transfer_rate", palette="crest", legend=False)
                    plt.ylabel("Normalized Transfer Rate", fontweight='bold')
                    plt.xlabel("Source/Target Pair Type", fontweight='bold')
                    plt.title(f"Normalized Transferability by Pair Type – {dataset} ({attack})", fontsize=14, fontweight='bold')
                    plt.xticks(rotation=20)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"transferability_pairtype_{dataset}_{attack}.png"), dpi=300)
                    plt.close()
                    print(f"[INFO] Saved: transferability_pairtype_{dataset}_{attack}.png")


def main():
    parser = argparse.ArgumentParser(description="Adversarial robustness analysis of pruned models.")
    parser.add_argument("--mode", choices=["full", "generate", "analyze", "plot"], default="full",
                        help="Execution mode: full (all phases), generate (attacks only), analyze (transferability), plot (visualizations).")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., VGG16).")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., Cifar10).")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack (e.g., PGD).")
    parser.add_argument("--kind", choices=["Original", "Finetuned"], default=None,
                        help="Filter the source checkpoint kind for attack generation and source dataset selection.")
    parser.add_argument("--output-dir", type=str, default="adversarial_results", help="Output directory for results.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    records = []
    transfer_records = []
    model_cache = {}
    loader_cache = {}
    adv_datasets = {}

    # Execute phases based on mode
    if args.mode in ["full", "generate"]:
        print(f"\n{'='*70}")
        print(f"[PHASE 1] Generating adversarial examples")
        print(f"{'='*70}")
        records, model_cache, loader_cache, adv_datasets = generate_attacks_phase(
            args.output_dir, args.model, args.dataset, args.attack, args.kind
        )
        
        # Save records
        if records:
            df_records = pd.DataFrame(records)
            csv_path = os.path.join(args.output_dir, "summary.csv")
            df_records.to_csv(csv_path, index=False)
            print(f"[INFO] Summary written to {csv_path}")

    if args.mode in ["full", "analyze"]:
        print(f"\n{'='*70}")
        print(f"[PHASE 2] Analyzing transferability")
        print(f"{'='*70}")
        
        # If not in full mode, need to load existing data
        if args.mode == "analyze":
            summary_path = os.path.join(args.output_dir, "summary.csv")
            if os.path.exists(summary_path):
                df_summary = pd.read_csv(summary_path)
                records = df_summary.to_dict(orient="records")
            else:
                print(f"[ERROR] summary.csv not found. Run with --mode full or generate first.")
                return
            
            # Reconstruct model cache and loader cache (needs to load models again)
            checkpoints = discover_checkpoints()
            model_cache = {}
            loader_cache = {}
            adv_datasets = {}
            
            for model_name, dataset_name, kind, ckpt_path in checkpoints:
                if args.dataset and dataset_name != args.dataset:
                    continue
                if dataset_name == "Cifar10":
                    if "Cifar10" not in loader_cache:
                        loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
                    _, _ = loader_cache["Cifar10"]
                    num_classes = 10
                elif dataset_name == "Cifar100":
                    if "Cifar100" not in loader_cache:
                        loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
                    train_loader, _ = loader_cache["Cifar100"]
                    num_classes = 100
                else:
                    continue

                if dataset_name == "Cifar10":
                    train_loader, _ = loader_cache["Cifar10"]

                one_batch = next(iter(train_loader))[0]
                print(f"[DEBUG] Loading {model_name} ({kind}) on {dataset_name}")
                print(f"[DEBUG] Checkpoint path: {ckpt_path}")
                try:
                    model = build_model_for_checkpoint(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        kind=kind,
                        num_classes=num_classes,
                        one_batch=one_batch,
                        ckpt_path=ckpt_path,
                        device="cuda",
                    )
                    load_result = robust_load_state_dict(model, ckpt_path)
                    print(
                        f"[DEBUG] load_state_dict result: missing={len(load_result.missing_keys)}, "
                        f"unexpected={len(load_result.unexpected_keys)}"
                    )
                except Exception as e:
                    print(
                        f"[WARN] Failed to load checkpoint for {model_name} ({kind}) on {dataset_name}: {e}"
                    )
                    # Skip this combination and move on
                    continue
                model = torch.nn.DataParallel(model)
                model_cache[(model_name, dataset_name, kind)] = model
                
                # Locate pre-generated adversarial datasets
                available_attacks = get_available_attacks()
                for attack in available_attacks:
                    if args.model and model_name != args.model:
                        continue
                    if args.dataset and dataset_name != args.dataset:
                        continue
                    if args.kind and kind != args.kind:
                        continue
                    if args.attack and attack != args.attack:
                        continue
                    adv_path = os.path.join(args.output_dir, f"{model_name}_{dataset_name}_{kind}_{attack}_adv.pt")
                    if os.path.exists(adv_path):
                        adv_datasets[(model_name, dataset_name, kind, attack)] = adv_path
        
        transfer_records = analyze_transferability_phase(args.output_dir, model_cache, loader_cache, adv_datasets)
        
        if transfer_records:
            tf_df = pd.DataFrame(transfer_records)
            tf_csv = os.path.join(args.output_dir, "transferability.csv")
            tf_df.to_csv(tf_csv, index=False)
            print(f"[INFO] Transferability matrix saved to {tf_csv}")

    if args.mode in ["full", "plot"]:
        print(f"\n{'='*70}")
        print(f"[PHASE 3] Generating plots")
        print(f"{'='*70}")
        
        # If not in full mode, load existing CSVs
        if args.mode == "plot":
            summary_path = os.path.join(args.output_dir, "summary.csv")
            transfer_path = os.path.join(args.output_dir, "transferability.csv")
            records = []
            transfer_records = []
            
            if os.path.exists(summary_path):
                df_records = pd.read_csv(summary_path)
                records = df_records.to_dict(orient="records")
            
            if os.path.exists(transfer_path):
                df_transfer = pd.read_csv(transfer_path)
                transfer_records = df_transfer.to_dict(orient="records")
        
        generate_plots(args.output_dir, records, transfer_records)
        print(f"[INFO] All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
