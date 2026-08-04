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
* ``accuracy_parameter_comparison.csv`` – Original vs collapsed clean accuracy and parameter counts
* ``collapsed_vs_original_explainability_by_attack.csv`` – Per-attack explainability deltas (collapsed minus original)
* ``collapsed_vs_original_explainability_summary.csv`` – Mean explainability deltas aggregated across attacks
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
import time
import traceback
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
    import os

    # Strict prefix ensures Cifar10 never matches Cifar100 (e.g., "VGG16_Cifar10_")
    prefix = f"{model}_{dataset}_"
    
    # Filter for directories only that match the exact prefix
    dirs = [
        d for d in os.listdir(".")
        if os.path.isdir(d) and d.startswith(prefix)
    ]
    
    if not dirs:
        return None
        
    # Prioritize 'epochs100_pretrain300' directory if multiple runs exist
    target_dir = dirs[0]
    for d in dirs:
        if "epochs100_pretrain300" in d:
            target_dir = d
            break
            
    filename = "final_JF_Control.pt" if kind == "Original" else "final_JF_Dynamic_Region_All_Combined.pt"
    full_path = os.path.abspath(os.path.join(target_dir, "checkpoints", filename))
    
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
    import os
    import re
    import glob

    run_dir = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Extract exact epoch config from folder name (e.g., "epochs100_pretrain300")
    match = re.search(r"(epochs\d+_pretrain\d+)", run_dir)
    if match:
        epoch_tag = match.group(1)
        candidate = os.path.join(
            base_dir,
            f"{model_name}_{dataset_name}_{epoch_tag}_JF_discovered_regions.json"
        )
        if os.path.exists(candidate):
            return candidate

    # Fallback search if exact tag isn't in folder name
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
    print("\n" + "=" * 70)
    print("[CHECKPOINT DISCOVERY] Scanning for available models and datasets...")
    print("=" * 70)
    
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for kind in ("Finetuned", "Original"):
                path = get_checkpoint_path(model, dataset, kind)
                if path:
                    print(f"  [FOUND] Model: {model:<15} | Dataset: {dataset:<10} | Kind: {kind:<10}")
                    entries.append((model, dataset, kind, path))
                else:
                    # Debug log to track missing models
                    pass

    print("=" * 70)
    print(f"[CHECKPOINT DISCOVERY] Total valid checkpoint pairs found: {len(entries)}")
    print("=" * 70 + "\n")
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
    print(
        f"[ATTACK][SETUP] Initializing attack='{attack_name}' "
        f"eps={epsilon:.6f} steps={steps}"
    )
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
            attack = torchattacks.AutoAttack(model, norm='Linf', eps=epsilon, version='standard', verbose=False)
        elif attack_name == "CW":
            attack = torchattacks.CW(model, c=1, lr=0.01, steps=1000, kappa=0)
        elif attack_name == "DeepFool":
            attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        else:
            raise ValueError(f"Unsupported attack: {attack_name}")
        print(f"[ATTACK][SETUP] Ready: {attack.__class__.__name__}")
        return attack
    except Exception as e:
        print(f"[WARN] Failed to instantiate {attack_name}: {e}. Skipping.")
        print(f"[ATTACK][ERROR] setup traceback for '{attack_name}':\n{traceback.format_exc().rstrip()}")
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

    total_batches = len(loader) if hasattr(loader, "__len__") else -1
    total_examples = len(loader.dataset) if hasattr(loader, "dataset") else -1
    t0 = time.perf_counter()
    print(
        f"[ATTACK][START] attack='{attack_name}' batches={total_batches} "
        f"examples={total_examples} eps={epsilon:.6f} steps={steps}"
    )

    clean_images = []
    adv_images = []
    true_labels = []
    clean_predictions = []
    adv_predictions = []
    running_seen = 0
    running_clean_correct = 0
    running_adv_correct = 0
    try:
        for batch_idx, (imgs, lbls) in enumerate(loader, start=1):
            imgs, lbls = imgs.cuda(), lbls.cuda()
            with torch.no_grad():
                clean_preds = model(imgs).argmax(dim=1)
            adv = attack(imgs, lbls)
            with torch.no_grad():
                adv_preds = model(adv).argmax(dim=1)

            batch_size = lbls.size(0)
            clean_correct = (clean_preds == lbls).sum().item()
            adv_correct = (adv_preds == lbls).sum().item()
            running_seen += batch_size
            running_clean_correct += clean_correct
            running_adv_correct += adv_correct

            if batch_idx == 1 or batch_idx % 10 == 0 or (total_batches > 0 and batch_idx == total_batches):
                with torch.no_grad():
                    delta = (adv - imgs).detach()
                    linf = float(delta.abs().amax().item())
                    l2_mean = float(delta.view(delta.shape[0], -1).norm(p=2, dim=1).mean().item())
                clean_acc_running = running_clean_correct / max(1, running_seen)
                adv_acc_running = running_adv_correct / max(1, running_seen)
                asr_running = 1.0 - adv_acc_running
                print(
                    f"[ATTACK][BATCH] {attack_name} batch={batch_idx}/{total_batches if total_batches > 0 else '?'} "
                    f"seen={running_seen} clean_acc={clean_acc_running:.2%} adv_acc={adv_acc_running:.2%} "
                    f"asr={asr_running:.2%} linf={linf:.5f} l2_mean={l2_mean:.5f}"
                )

            clean_images.append(imgs.cpu())
            adv_images.append(adv.cpu())
            true_labels.append(lbls.cpu())
            clean_predictions.append(clean_preds.cpu())
            adv_predictions.append(adv_preds.cpu())

        elapsed = time.perf_counter() - t0
        clean_acc_final = running_clean_correct / max(1, running_seen)
        adv_acc_final = running_adv_correct / max(1, running_seen)
        print(
            f"[ATTACK][DONE] attack='{attack_name}' samples={running_seen} "
            f"clean_acc={clean_acc_final:.2%} adv_acc={adv_acc_final:.2%} "
            f"asr={(1.0 - adv_acc_final):.2%} runtime={elapsed:.2f}s"
        )

        return {
            "clean_images": torch.cat(clean_images),
            "adversarial_images": torch.cat(adv_images),
            "true_labels": torch.cat(true_labels),
            "source_clean_predictions": torch.cat(clean_predictions),
            "source_adversarial_predictions": torch.cat(adv_predictions),
        }
    except Exception as e:
        print(f"[ERROR] Attack generation failed for {attack_name}: {e}")
        print(
            f"[ATTACK][ERROR] attack='{attack_name}' failed after seen={running_seen} "
            f"elapsed={time.perf_counter() - t0:.2f}s"
        )
        print(f"[ATTACK][ERROR] traceback:\n{traceback.format_exc().rstrip()}")
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


def count_model_parameters(model: nn.Module) -> int:
    """Return total number of parameters in a model."""
    return int(sum(p.numel() for p in model.parameters()))


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
    print(
        f"[ATTACK][PLAN] filters model={model_filter} dataset={dataset_filter} "
        f"kind={kind_filter} attack={attack_filter}"
    )
    
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
    print(f"[ATTACK][PLAN] selected_attacks={attacks}")

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
        param_count = count_model_parameters(model)
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
            attack_t0 = time.perf_counter()
            print(
                f"[ATTACK][RUN] source={model_name} kind={kind} dataset={dataset_name} "
                f"attack={attack_name}"
            )
            adv_bundle = generate_adversarial_dataset(model, test_loader, attack_name)
            if adv_bundle is None:
                print(
                    f"[ATTACK][SKIP] source={model_name} kind={kind} dataset={dataset_name} "
                    f"attack={attack_name} reason=generation_failed"
                )
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
            print(
                f"[ATTACK][SAVE] path={adv_path} samples={adv_imgs.shape[0]} "
                f"elapsed={time.perf_counter() - attack_t0:.2f}s"
            )

            records.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "kind": kind,
                    "attack": attack_name,
                    "model_label": model_kind_label(model_name, kind),
                    "param_count": param_count,
                    "clean_acc": clean_acc,
                    "adv_acc": adv_acc,
                    **summarize_direct_metrics(clean_acc, adv_acc),
                }
            )
            print(f"[INFO] {model_name} ({kind}) {dataset_name} – {attack_name}: clean {clean_acc:.2%}, adv {adv_acc:.2%}")
            print(
                f"[ATTACK][SUMMARY] source={model_name} kind={kind} dataset={dataset_name} "
                f"attack={attack_name} asr={(1.0 - adv_acc):.2%}"
            )

    return records, model_cache, loader_cache, adv_datasets


def generate_comparison_tables(output_dir: str, records: List[Dict]):
    """Generate collapsed-vs-original comparison tables for explainability and model size.

    The produced tables are designed to be directly comparable to the journal
    summaries: they include clean accuracy and parameter counts for both model
    kinds, plus attack-driven explainability deltas.
    """
    if not records:
        return

    os.makedirs(output_dir, exist_ok=True)
    df = enrich_summary_dataframe(pd.DataFrame(records))
    if df.empty:
        return

    if "param_count" not in df.columns:
        df["param_count"] = np.nan

    # One-row-per-model-kind profile used for accuracy/parameter comparison.
    profile = (
        df.groupby(["model", "dataset", "kind"], as_index=False)
        .agg(
            clean_acc=("clean_acc", "mean"),
            robust_accuracy=("robust_accuracy", "mean"),
            attack_success_rate=("attack_success_rate", "mean"),
            param_count=("param_count", "mean"),
        )
    )

    orig_profile = profile[profile["kind"] == "Original"].rename(
        columns={
            "clean_acc": "original_clean_acc",
            "robust_accuracy": "original_robust_accuracy_mean",
            "attack_success_rate": "original_attack_success_rate_mean",
            "param_count": "original_param_count",
        }
    )[[
        "model",
        "dataset",
        "original_clean_acc",
        "original_robust_accuracy_mean",
        "original_attack_success_rate_mean",
        "original_param_count",
    ]]

    finetuned_profile = profile[profile["kind"] == "Finetuned"].rename(
        columns={
            "clean_acc": "collapsed_clean_acc",
            "robust_accuracy": "collapsed_robust_accuracy_mean",
            "attack_success_rate": "collapsed_attack_success_rate_mean",
            "param_count": "collapsed_param_count",
        }
    )[[
        "model",
        "dataset",
        "collapsed_clean_acc",
        "collapsed_robust_accuracy_mean",
        "collapsed_attack_success_rate_mean",
        "collapsed_param_count",
    ]]

    acc_param_df = orig_profile.merge(
        finetuned_profile,
        on=["model", "dataset"],
        how="outer",
    )
    acc_param_df["collapsed_minus_original_clean_acc"] = (
        acc_param_df["collapsed_clean_acc"] - acc_param_df["original_clean_acc"]
    )
    acc_param_df["params_reduction_percent"] = np.where(
        acc_param_df["original_param_count"] > 0,
        100.0
        * (
            1.0
            - acc_param_df["collapsed_param_count"]
            / acc_param_df["original_param_count"]
        ),
        np.nan,
    )
    acc_param_path = os.path.join(output_dir, "accuracy_parameter_comparison.csv")
    acc_param_df.to_csv(acc_param_path, index=False)
    print(f"[INFO] Saved: {acc_param_path}")

    # Per-attack explainability table (collapsed vs original).
    by_attack = (
        df.groupby(["model", "dataset", "attack", "kind"], as_index=False)
        .agg(
            clean_acc=("clean_acc", "mean"),
            robust_accuracy=("robust_accuracy", "mean"),
            attack_success_rate=("attack_success_rate", "mean"),
            relative_accuracy_drop=("relative_accuracy_drop", "mean"),
            robustness_ratio=("robustness_ratio", "mean"),
            param_count=("param_count", "mean"),
        )
    )

    orig_attack = by_attack[by_attack["kind"] == "Original"].rename(
        columns={
            "clean_acc": "original_clean_acc",
            "robust_accuracy": "original_robust_accuracy",
            "attack_success_rate": "original_attack_success_rate",
            "relative_accuracy_drop": "original_relative_accuracy_drop",
            "robustness_ratio": "original_robustness_ratio",
            "param_count": "original_param_count",
        }
    )[[
        "model",
        "dataset",
        "attack",
        "original_clean_acc",
        "original_robust_accuracy",
        "original_attack_success_rate",
        "original_relative_accuracy_drop",
        "original_robustness_ratio",
        "original_param_count",
    ]]

    finetuned_attack = by_attack[by_attack["kind"] == "Finetuned"].rename(
        columns={
            "clean_acc": "collapsed_clean_acc",
            "robust_accuracy": "collapsed_robust_accuracy",
            "attack_success_rate": "collapsed_attack_success_rate",
            "relative_accuracy_drop": "collapsed_relative_accuracy_drop",
            "robustness_ratio": "collapsed_robustness_ratio",
            "param_count": "collapsed_param_count",
        }
    )[[
        "model",
        "dataset",
        "attack",
        "collapsed_clean_acc",
        "collapsed_robust_accuracy",
        "collapsed_attack_success_rate",
        "collapsed_relative_accuracy_drop",
        "collapsed_robustness_ratio",
        "collapsed_param_count",
    ]]

    explainability_df = orig_attack.merge(
        finetuned_attack,
        on=["model", "dataset", "attack"],
        how="inner",
    )
    explainability_df["collapsed_minus_original_attack_success_rate"] = (
        explainability_df["collapsed_attack_success_rate"]
        - explainability_df["original_attack_success_rate"]
    )
    explainability_df["collapsed_minus_original_robust_accuracy"] = (
        explainability_df["collapsed_robust_accuracy"]
        - explainability_df["original_robust_accuracy"]
    )
    explainability_df["collapsed_minus_original_relative_accuracy_drop"] = (
        explainability_df["collapsed_relative_accuracy_drop"]
        - explainability_df["original_relative_accuracy_drop"]
    )
    explainability_df["collapsed_minus_original_robustness_ratio"] = (
        explainability_df["collapsed_robustness_ratio"]
        - explainability_df["original_robustness_ratio"]
    )
    explainability_df["params_reduction_percent"] = np.where(
        explainability_df["original_param_count"] > 0,
        100.0
        * (
            1.0
            - explainability_df["collapsed_param_count"]
            / explainability_df["original_param_count"]
        ),
        np.nan,
    )

    explainability_path = os.path.join(
        output_dir,
        "collapsed_vs_original_explainability_by_attack.csv",
    )
    explainability_df.to_csv(explainability_path, index=False)
    print(f"[INFO] Saved: {explainability_path}")

    explainability_summary_df = (
        explainability_df.groupby(["model", "dataset"], as_index=False)
        .agg(
            original_clean_acc=("original_clean_acc", "mean"),
            collapsed_clean_acc=("collapsed_clean_acc", "mean"),
            original_param_count=("original_param_count", "mean"),
            collapsed_param_count=("collapsed_param_count", "mean"),
            params_reduction_percent=("params_reduction_percent", "mean"),
            mean_original_attack_success_rate=("original_attack_success_rate", "mean"),
            mean_collapsed_attack_success_rate=("collapsed_attack_success_rate", "mean"),
            mean_delta_attack_success_rate=("collapsed_minus_original_attack_success_rate", "mean"),
            mean_delta_robust_accuracy=("collapsed_minus_original_robust_accuracy", "mean"),
            mean_delta_relative_accuracy_drop=("collapsed_minus_original_relative_accuracy_drop", "mean"),
            mean_delta_robustness_ratio=("collapsed_minus_original_robustness_ratio", "mean"),
        )
    )
    explainability_summary_path = os.path.join(
        output_dir,
        "collapsed_vs_original_explainability_summary.csv",
    )
    explainability_summary_df.to_csv(explainability_summary_path, index=False)
    print(f"[INFO] Saved: {explainability_summary_path}")


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


# =============================================================================
# EXPERIMENT 4 — Gradient Similarity Analysis
# =============================================================================

def compute_gradient_similarity(
    model_s: nn.Module,
    model_t: nn.Module,
    dataloader,
    device: str,
    n_samples: int = 1000,
) -> float:
    """Compute mean cosine similarity of input-space loss gradients between two models.

    Returns a scalar in [-1, 1].  Values near 1 predict high adversarial transfer;
    values near 0 predict low transfer.
    """
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

        grad_s_flat = grad_s.detach().view(images.size(0), -1)
        grad_t_flat = grad_t.detach().view(images.size(0), -1)

        cos_sim = torch.nn.functional.cosine_similarity(grad_s_flat, grad_t_flat, dim=1)
        similarities.extend(cos_sim.cpu().tolist())
        count += images.size(0)

    return float(torch.tensor(similarities).mean()) if similarities else float("nan")


def gradient_similarity_phase(
    output_dir: str,
    model_cache: dict,
    loader_cache: dict,
    n_samples: int = 1000,
) -> list[dict]:
    """Experiment 4: compute pairwise gradient similarity for all model-kind pairs."""
    os.makedirs(output_dir, exist_ok=True)
    records: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list(model_cache.keys())
    print(f"[EXP4] Computing gradient similarity for {len(pairs)} model variants "
          f"({len(pairs)**2} pairs)...")

    for dataset_name in set(k[1] for k in pairs):
        loader_key = dataset_name
        if loader_key not in loader_cache:
            print(f"[WARN] No loader cached for {dataset_name}; skipping gradient sim.")
            continue
        _, test_loader = loader_cache[loader_key]

        dataset_pairs = [k for k in pairs if k[1] == dataset_name]
        for src_key in dataset_pairs:
            for tgt_key in dataset_pairs:
                src_name, _, src_kind = src_key
                tgt_name, _, tgt_kind = tgt_key
                src_model = model_cache[src_key]
                tgt_model = model_cache[tgt_key]
                try:
                    sim = compute_gradient_similarity(
                        src_model, tgt_model, test_loader, device, n_samples
                    )
                except Exception as e:
                    print(f"[WARN] Gradient sim failed for {src_key}→{tgt_key}: {e}")
                    sim = float("nan")

                records.append({
                    "source_model": src_name,
                    "source_kind": src_kind,
                    "source_label": model_kind_label(src_name, src_kind),
                    "target_model": tgt_name,
                    "target_kind": tgt_kind,
                    "target_label": model_kind_label(tgt_name, tgt_kind),
                    "dataset": dataset_name,
                    "gradient_similarity": sim,
                    "same_architecture": src_name == tgt_name,
                    "same_kind": src_kind == tgt_kind,
                    "pair_type": classify_transfer_pair(src_name, src_kind, tgt_name, tgt_kind),
                })
                print(f"[EXP4] {src_name}({src_kind}) → {tgt_name}({tgt_kind}) "
                      f"[{dataset_name}]: GradSim={sim:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "gradient_similarity.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXP4] Saved: {csv_path}")

    # square matrix per dataset
    for dataset_name in df["dataset"].unique():
        sub = df[df["dataset"] == dataset_name]
        mat = sub.pivot_table(
            index="source_label", columns="target_label",
            values="gradient_similarity", aggfunc="mean",
        )
        mat_path = os.path.join(output_dir, f"gradient_similarity_matrix_{dataset_name}.csv")
        mat.to_csv(mat_path)
        print(f"[EXP4] Saved: {mat_path}")

        plt.figure(figsize=(12, 9))
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn", vmin=-1, vmax=1,
                    cbar_kws={"label": "Gradient Cosine Similarity"})
        plt.title(f"Input-Gradient Similarity – {dataset_name}", fontsize=14, fontweight="bold")
        plt.xlabel("Target Model Variant", fontweight="bold")
        plt.ylabel("Source Model Variant", fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"gradient_similarity_heatmap_{dataset_name}.png"), dpi=300)
        plt.close()
        print(f"[EXP4] Saved: gradient_similarity_heatmap_{dataset_name}.png")

    return records


# =============================================================================
# EXPERIMENT 7 — Epsilon Sensitivity Analysis
# =============================================================================

EPSILON_VALUES = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]


def epsilon_sensitivity_phase(
    output_dir: str,
    model_cache: dict,
    loader_cache: dict,
    attacks: list[str] | None = None,
) -> list[dict]:
    """Experiment 7: sweep epsilon values and record direct + transfer ASR."""
    os.makedirs(output_dir, exist_ok=True)
    if attacks is None:
        attacks = ["PGD", "FGSM", "BIM"]  # fast attacks only for the sweep

    records: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[EXP7] Epsilon sweep over {EPSILON_VALUES} for attacks {attacks}...")

    for (model_name, dataset_name, kind), model in model_cache.items():
        loader_key = dataset_name
        if loader_key not in loader_cache:
            continue
        _, test_loader = loader_cache[loader_key]

        for attack_name in attacks:
            for eps in EPSILON_VALUES:
                steps = max(10, int(eps * 255 * 2))  # scale steps with eps
                try:
                    attack_obj = instantiate_attack(attack_name, model, epsilon=eps, steps=steps)
                    if attack_obj is None:
                        continue
                    # Direct ASR on source model
                    model.eval()
                    correct = total = 0
                    for imgs, lbls in test_loader:
                        imgs, lbls = imgs.to(device), lbls.to(device)
                        adv = attack_obj(imgs, lbls)
                        with torch.no_grad():
                            preds = model(adv).argmax(dim=1)
                        correct += (preds == lbls).sum().item()
                        total += lbls.size(0)
                    asr = 1.0 - correct / total if total else float("nan")
                except Exception as e:
                    print(f"[WARN] EXP7 failed for {model_name}({kind}) {attack_name} eps={eps:.5f}: {e}")
                    asr = float("nan")

                records.append({
                    "model": model_name,
                    "kind": kind,
                    "model_label": model_kind_label(model_name, kind),
                    "dataset": dataset_name,
                    "attack": attack_name,
                    "epsilon": eps,
                    "epsilon_255": round(eps * 255, 2),
                    "attack_success_rate": asr,
                })
                print(f"[EXP7] {model_name}({kind}) {dataset_name} {attack_name} "
                      f"ε={eps*255:.1f}/255 → ASR={asr:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "epsilon_sensitivity.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXP7] Saved: {csv_path}")

    # delta (Finetuned - Original) per epsilon
    delta_rows: list[dict] = []
    for (model_name, dataset_name, attack_name, eps), grp in df.groupby(
        ["model", "dataset", "attack", "epsilon"]
    ):
        orig = grp[grp["kind"] == "Original"]["attack_success_rate"]
        fine = grp[grp["kind"] == "Finetuned"]["attack_success_rate"]
        if not orig.empty and not fine.empty:
            delta_rows.append({
                "model": model_name,
                "dataset": dataset_name,
                "attack": attack_name,
                "epsilon": eps,
                "epsilon_255": round(eps * 255, 2),
                "asr_original": float(orig.mean()),
                "asr_finetuned": float(fine.mean()),
                "delta_asr": float(fine.mean()) - float(orig.mean()),
            })

    if delta_rows:
        delta_df = pd.DataFrame(delta_rows)
        delta_csv = os.path.join(output_dir, "epsilon_sensitivity_delta.csv")
        delta_df.to_csv(delta_csv, index=False)
        print(f"[EXP7] Saved: {delta_csv}")

    # ── plots ─────────────────────────────────────────────────────────────────
    for dataset_name in df["dataset"].unique():
        for attack_name in df["attack"].unique():
            sub = df[(df["dataset"] == dataset_name) & (df["attack"] == attack_name)]
            if sub.empty:
                continue
            plt.figure(figsize=(9, 6))
            for label, grp in sub.groupby("model_label"):
                grp_sorted = grp.sort_values("epsilon_255")
                plt.plot(grp_sorted["epsilon_255"], grp_sorted["attack_success_rate"],
                         marker="o", label=label)
            plt.xlabel("Perturbation Budget ε (×1/255)", fontweight="bold")
            plt.ylabel("Attack Success Rate", fontweight="bold")
            plt.title(f"Epsilon Sensitivity – {dataset_name} ({attack_name})",
                      fontsize=13, fontweight="bold")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f"epsilon_sensitivity_{dataset_name}_{attack_name}.png"), dpi=300)
            plt.close()
            print(f"[EXP7] Saved: epsilon_sensitivity_{dataset_name}_{attack_name}.png")

    return records


# =============================================================================
# EXPERIMENT 9 — Statistical Significance Testing
# =============================================================================

def _load_multi_run_summaries(result_dirs: list[str]) -> pd.DataFrame:
    """Concatenate summary.csv files from multiple output directories."""
    frames = []
    for d in result_dirs:
        p = os.path.join(d, "summary.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df["run_dir"] = os.path.basename(d)
            frames.append(df)
        else:
            print(f"[WARN] EXP9: summary.csv not found in {d}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def statistical_significance_phase(
    output_dir: str,
    result_dirs: list[str] | None = None,
) -> list[dict]:
    """Experiment 9: paired t-test and Kruskal-Wallis across three run configurations.

    ``result_dirs`` should point to the three epoch-budget output directories,
    e.g. ["adversarial_results_ep100_pre300", "adversarial_results_ep200_pre200",
           "adversarial_results_ep300_pre100"].  Falls back to the single output_dir
    if not supplied.
    """
    try:
        from scipy import stats as scipy_stats  # soft dependency
    except ImportError:
        print("[WARN] EXP9: scipy not installed.  Run: pip install scipy")
        return []

    os.makedirs(output_dir, exist_ok=True)

    if result_dirs:
        df_all = _load_multi_run_summaries(result_dirs)
    else:
        p = os.path.join(output_dir, "summary.csv")
        if not os.path.exists(p):
            print(f"[WARN] EXP9: no summary.csv at {p}")
            return []
        df_all = pd.read_csv(p)
        df_all["run_dir"] = "single_run"

    if df_all.empty:
        print("[WARN] EXP9: no data to analyse.")
        return []

    records: list[dict] = []
    for (model_name, dataset_name, attack_name), grp in df_all.groupby(
        ["model", "dataset", "attack"]
    ):
        orig = grp[grp["kind"] == "Original"]["attack_success_rate"].dropna().values
        fine = grp[grp["kind"] == "Finetuned"]["attack_success_rate"].dropna().values

        if len(orig) == 0 or len(fine) == 0:
            continue

        mean_orig = float(np.mean(orig))
        mean_fine = float(np.mean(fine))
        std_orig = float(np.std(orig, ddof=1)) if len(orig) > 1 else float("nan")
        std_fine = float(np.std(fine, ddof=1)) if len(fine) > 1 else float("nan")
        delta_asr = mean_fine - mean_orig

        # Paired t-test (requires equal-length matched pairs)
        if len(orig) == len(fine) and len(orig) > 1:
            t_stat, p_ttest = scipy_stats.ttest_rel(fine, orig)
        else:
            t_stat, p_ttest = float("nan"), float("nan")

        # Kruskal-Wallis across run configurations
        if len(result_dirs or []) >= 2 and len(orig) >= 2:
            _, p_kw = scipy_stats.kruskal(orig, fine)
        else:
            p_kw = float("nan")

        significant = (not np.isnan(p_ttest)) and (p_ttest < 0.05)
        effect_size = abs(delta_asr)  # simple absolute difference

        records.append({
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
            "significant_at_0.05": significant,
            "effect_size_abs": effect_size,
        })

    # ── save ──────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "statistical_significance.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[EXP9] Saved: {csv_path}")

    # print summary table
    if not df_out.empty:
        print(f"\n[EXP9] Statistical summary ({len(df_out)} model-attack combinations):")
        n_sig = int(df_out["significant_at_0.05"].sum())
        print(f"       Significant differences (p<0.05): {n_sig} / {len(df_out)}")
        print(f"       Mean |ΔASR|: {df_out['effect_size_abs'].mean():.4f}  "
              f"(max: {df_out['effect_size_abs'].max():.4f})")

    # ── forest plot ───────────────────────────────────────────────────────────
    if not df_out.empty:
        for dataset_name in df_out["dataset"].unique():
            sub = df_out[df_out["dataset"] == dataset_name].copy()
            sub = sub.sort_values("delta_asr")
            fig, ax = plt.subplots(figsize=(10, max(4, len(sub) * 0.35)))
            colors = ["#e74c3c" if s else "#2ecc71"
                      for s in sub["significant_at_0.05"]]
            ax.barh(range(len(sub)), sub["delta_asr"], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_yticks(range(len(sub)))
            ax.set_yticklabels(sub["model"] + " / " + sub["attack"], fontsize=7)
            ax.set_xlabel("ΔASR (Finetuned − Original)", fontweight="bold")
            ax.set_title(f"Compression Effect on Attack Success Rate – {dataset_name}\n"
                         f"(red = p<0.05, green = not significant)", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f"statistical_significance_{dataset_name}.png"), dpi=300)
            plt.close()
            print(f"[EXP9] Saved: statistical_significance_{dataset_name}.png")

    return records


# =============================================================================
# EXPERIMENT 10 — CKA Feature Similarity Analysis
# =============================================================================

def _compute_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between representation matrices X (N×p) and Y (N×q)."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    dot_xx = (X @ X.T).norm(p="fro") ** 2
    dot_yy = (Y @ Y.T).norm(p="fro") ** 2
    dot_xy = (X @ Y.T).norm(p="fro") ** 2
    denom = (dot_xx.sqrt() * dot_yy.sqrt())
    return float(dot_xy / denom) if denom > 0 else float("nan")


def _extract_representations(
    model: nn.Module,
    dataloader,
    layer_name: str,
    device: str,
    max_samples: int = 512,
) -> torch.Tensor | None:
    """Extract and flatten feature maps at ``layer_name`` for up to max_samples images."""
    reps: list[torch.Tensor] = []
    count = 0

    def hook_fn(module, inp, out):
        reps.append(out.detach().cpu())

    target = dict(model.named_modules()).get(layer_name)
    if target is None:
        print(f"[WARN] EXP10: layer '{layer_name}' not found in model.")
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
    except Exception as e:
        print(f"[WARN] EXP10: forward pass failed at layer '{layer_name}': {e}")
    finally:
        handle.remove()

    if not reps:
        return None
    out = torch.cat(reps, dim=0)[:max_samples]
    return out.view(out.size(0), -1)  # flatten spatial dims


def _candidate_layers(model: nn.Module) -> list[str]:
    """Return names of Conv2d and Linear layers suitable for CKA probing."""
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]


def cka_similarity_phase(
    output_dir: str,
    model_cache: dict,
    loader_cache: dict,
    max_samples: int = 512,
    max_layers: int = 8,
) -> list[dict]:
    """Experiment 10: layer-wise CKA between all model-kind pairs.

    Probes up to ``max_layers`` evenly-spaced convolutional/linear layers in each model.
    """
    os.makedirs(output_dir, exist_ok=True)
    records: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs = list(model_cache.keys())
    print(f"[EXP10] Computing CKA for {len(pairs)} model variants...")

    for dataset_name in set(k[1] for k in pairs):
        if dataset_name not in loader_cache:
            continue
        _, test_loader = loader_cache[dataset_name]

        dataset_pairs = [k for k in pairs if k[1] == dataset_name]
        for src_key in dataset_pairs:
            src_name, _, src_kind = src_key
            src_model = model_cache[src_key]

            # Determine probe layers for source model (evenly spaced)
            all_layers = _candidate_layers(src_model.module if hasattr(src_model, "module") else src_model)
            if not all_layers:
                continue
            step = max(1, len(all_layers) // max_layers)
            probe_layers = all_layers[::step][:max_layers]

            for tgt_key in dataset_pairs:
                tgt_name, _, tgt_kind = tgt_key
                tgt_model = model_cache[tgt_key]

                for layer_name in probe_layers:
                    # resolve same-name layer in target (best-effort)
                    tgt_layers = _candidate_layers(
                        tgt_model.module if hasattr(tgt_model, "module") else tgt_model
                    )
                    # use same index position if name doesn't exist in target
                    tgt_layer = layer_name if layer_name in tgt_layers else (
                        tgt_layers[all_layers.index(layer_name)]
                        if layer_name in all_layers and all_layers.index(layer_name) < len(tgt_layers)
                        else None
                    )
                    if tgt_layer is None:
                        continue

                    X = _extract_representations(src_model, test_loader, layer_name, device, max_samples)
                    Y = _extract_representations(tgt_model, test_loader, tgt_layer, device, max_samples)
                    if X is None or Y is None or X.size(0) < 4 or Y.size(0) < 4:
                        continue

                    # truncate to same N
                    n = min(X.size(0), Y.size(0))
                    try:
                        cka_val = _compute_linear_cka(X[:n].float(), Y[:n].float())
                    except Exception as e:
                        print(f"[WARN] EXP10 CKA failed {src_key}→{tgt_key} at {layer_name}: {e}")
                        cka_val = float("nan")

                    records.append({
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
                    })
                    print(f"[EXP10] {src_name}({src_kind})→{tgt_name}({tgt_kind}) "
                          f"[{layer_name}] CKA={cka_val:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    csv_path = os.path.join(output_dir, "cka_similarity.csv")
    df.to_csv(csv_path, index=False)
    print(f"[EXP10] Saved: {csv_path}")

    # ── layer-curve plots ─────────────────────────────────────────────────────
    for dataset_name in df["dataset"].unique():
        for src_label in df["source_label"].unique():
            sub = df[(df["dataset"] == dataset_name) & (df["source_label"] == src_label)]
            if sub.empty:
                continue
            plt.figure(figsize=(11, 5))
            for tgt_label, grp in sub.groupby("target_label"):
                plt.plot(range(len(grp)), grp["cka"].values, marker="o",
                         label=f"→ {tgt_label}")
            plt.xlabel("Layer Index (shallow → deep)", fontweight="bold")
            plt.ylabel("CKA", fontweight="bold")
            plt.ylim(0, 1.05)
            plt.title(f"Layer-wise CKA from {src_label} – {dataset_name}",
                      fontsize=12, fontweight="bold")
            plt.legend(fontsize=7, ncol=2)
            plt.tight_layout()
            safe_label = src_label.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(output_dir,
                        f"cka_layerwise_{dataset_name}_{safe_label}.png"), dpi=300)
            plt.close()

    # ── pair-type summary heatmap ─────────────────────────────────────────────
    if not df.empty:
        for dataset_name in df["dataset"].unique():
            sub = df[df["dataset"] == dataset_name]
            # mean CKA across all layers per source-target pair
            mean_cka = sub.groupby(["source_label", "target_label"])["cka"].mean().reset_index()
            pivot = mean_cka.pivot(index="source_label", columns="target_label", values="cka")
            plt.figure(figsize=(12, 9))
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                        cbar_kws={"label": "Mean CKA (all layers)"})
            plt.title(f"Mean Layer-wise CKA – {dataset_name}",
                      fontsize=14, fontweight="bold")
            plt.xlabel("Target Model Variant", fontweight="bold")
            plt.ylabel("Source Model Variant", fontweight="bold")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,
                        f"cka_mean_heatmap_{dataset_name}.png"), dpi=300)
            plt.close()
            mat_path = os.path.join(output_dir, f"cka_mean_matrix_{dataset_name}.csv")
            pivot.to_csv(mat_path)
            print(f"[EXP10] Saved: {mat_path}")

    return records


def _rebuild_model_and_loader_cache(args) -> tuple[dict, dict]:
    """Re-load all checkpoints and data loaders when running a standalone experiment phase."""
    model_cache: dict = {}
    loader_cache: dict = {}
    checkpoints = discover_checkpoints()
    for model_name, dataset_name, kind, ckpt_path in checkpoints:
        if args.model and model_name != args.model:
            continue
        if args.dataset and dataset_name != args.dataset:
            continue
        if args.kind and kind != args.kind:
            continue
        if dataset_name == "Cifar10":
            if "Cifar10" not in loader_cache:
                loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
            train_loader, _ = loader_cache["Cifar10"]
            num_classes = 10
        elif dataset_name == "Cifar100":
            if "Cifar100" not in loader_cache:
                loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
            train_loader, _ = loader_cache["Cifar100"]
            num_classes = 100
        else:
            continue
        one_batch = next(iter(train_loader))[0]
        try:
            model = build_model_for_checkpoint(
                model_name=model_name, dataset_name=dataset_name, kind=kind,
                num_classes=num_classes, one_batch=one_batch,
                ckpt_path=ckpt_path, device="cuda",
            )
            robust_load_state_dict(model, ckpt_path)
        except Exception as e:
            print(f"[WARN] Could not load {model_name}({kind}) on {dataset_name}: {e}")
            continue
        model = torch.nn.DataParallel(model)
        model_cache[(model_name, dataset_name, kind)] = model
    return model_cache, loader_cache


def main():
    parser = argparse.ArgumentParser(description="Adversarial robustness analysis of pruned models.")
    parser.add_argument("--mode", choices=[
        "full", "generate", "analyze", "plot",
        "gradient_sim", "epsilon_sweep", "statistics", "cka",
    ], default="full",
                        help=(
                            "Execution mode: full (all phases), generate (attacks only), "
                            "analyze (transferability), plot (visualizations), "
                            "gradient_sim (Exp 4), epsilon_sweep (Exp 7), "
                            "statistics (Exp 9), cka (Exp 10)."
                        ))
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., VGG16).")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., Cifar10).")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack (e.g., PGD).")
    parser.add_argument("--kind", choices=["Original", "Finetuned"], default=None,
                        help="Filter the source checkpoint kind for attack generation and source dataset selection.")
    parser.add_argument("--output-dir", type=str, default="adversarial_results", help="Output directory for results.")
    # Experiment 7
    parser.add_argument("--epsilon-attacks", type=str, nargs="+",
                        default=["PGD", "FGSM", "BIM"],
                        help="Attack methods to use in the epsilon sensitivity sweep (Exp 7).")
    # Experiment 9
    parser.add_argument("--result-dirs", type=str, nargs="+", default=None,
                        help="List of output dirs from multiple runs for statistical testing (Exp 9). "
                             "E.g. adversarial_results_ep100_pre300 adversarial_results_ep200_pre200 "
                             "adversarial_results_ep300_pre100")
    # Experiment 10
    parser.add_argument("--cka-max-samples", type=int, default=512,
                        help="Max images to use per CKA layer probe (Exp 10).")
    parser.add_argument("--cka-max-layers", type=int, default=8,
                        help="Max layers to probe per model in CKA (Exp 10).")
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

            # --- NEW LOGIC: Unique filenames for parallel jobs ---
            if args.model and args.attack:
                job_label = f"_{args.model}_{args.dataset}_{args.attack}_{args.kind or 'ALL'}"
                csv_path = os.path.join(args.output_dir, f"summary{job_label}.csv")
            else:
                csv_path = os.path.join(args.output_dir, "summary.csv")
            # ----------------------------------------------------

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

    # Comparison tables for explainability + model profile alignment with journal tables.
    if records and args.mode != "generate":
        generate_comparison_tables(args.output_dir, records)

    # =========================================================================
    # EXPERIMENT 4 — Gradient Similarity
    # =========================================================================
    if args.mode in ["full", "gradient_sim"]:
        print(f"\n{'='*70}")
        print(f"[PHASE EXP4] Gradient Similarity Analysis")
        print(f"{'='*70}")

        if args.mode == "gradient_sim" or not model_cache:
            model_cache, loader_cache = _rebuild_model_and_loader_cache(args)

        gradient_similarity_phase(args.output_dir, model_cache, loader_cache)

    # =========================================================================
    # EXPERIMENT 7 — Epsilon Sensitivity
    # =========================================================================
    if args.mode in ["full", "epsilon_sweep"]:
        print(f"\n{'='*70}")
        print(f"[PHASE EXP7] Epsilon Sensitivity Analysis")
        print(f"{'='*70}")

        if args.mode == "epsilon_sweep" or not model_cache:
            model_cache, loader_cache = _rebuild_model_and_loader_cache(args)

        epsilon_sensitivity_phase(
            args.output_dir, model_cache, loader_cache,
            attacks=args.epsilon_attacks,
        )

    # =========================================================================
    # EXPERIMENT 9 — Statistical Significance
    # =========================================================================
    if args.mode in ["full", "statistics"]:
        print(f"\n{'='*70}")
        print(f"[PHASE EXP9] Statistical Significance Testing")
        print(f"{'='*70}")

        result_dirs = args.result_dirs or [args.output_dir]
        statistical_significance_phase(args.output_dir, result_dirs=result_dirs)

    # =========================================================================
    # EXPERIMENT 10 — CKA Feature Similarity
    # =========================================================================
    if args.mode in ["full", "cka"]:
        print(f"\n{'='*70}")
        print(f"[PHASE EXP10] CKA Feature Similarity Analysis")
        print(f"{'='*70}")

        if args.mode == "cka" or not model_cache:
            model_cache, loader_cache = _rebuild_model_and_loader_cache(args)

        cka_similarity_phase(
            args.output_dir, model_cache, loader_cache,
            max_samples=args.cka_max_samples,
            max_layers=args.cka_max_layers,
        )


if __name__ == "__main__":
    main()
