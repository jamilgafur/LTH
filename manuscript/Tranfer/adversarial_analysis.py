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
* ``summary.csv`` – Direct attack results (clean_acc, adv_acc per model/dataset/kind/attack)
* ``transferability.csv`` – Cross-model transferability rates
* Plots:
  - ``attack_success_rates.png`` – Accuracy drop by model, attack, and kind
  - ``transferability_heatmap_*.png`` – Heatmaps showing cross-architecture transferability
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

# =========================================================================
# Robust Checkpoint Loading Helper
# =========================================================================

def robust_load_state_dict(model: nn.Module, ckpt_path: str) -> None:
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
    model.load_state_dict(sd, strict=False)

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

def discover_checkpoints():
    entries = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for kind in ("Finetuned", "Original"):
                path = get_checkpoint_path(model, dataset, kind)
                if path:
                    entries.append((model, dataset, kind, path))
    return entries

def load_model(model_name: str, num_classes: int) -> nn.Module:
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
    return mapping[model_name](num_classes=num_classes)


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


def generate_adversarial_dataset(model: nn.Module, loader, attack_name: str, epsilon: float = 0.03, steps: int = 40) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate adversarial examples using the specified attack.

    Returns a tuple ``(adv_images, adv_labels)`` where ``adv_images`` is a tensor
    of shape ``(N, C, H, W)`` and ``adv_labels`` are the original labels.
    """
    model.eval()
    attack = instantiate_attack(attack_name, model, epsilon, steps)
    if attack is None:
        return None, None

    adv_images = []
    adv_labels = []
    try:
        for imgs, lbls in loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            adv = attack(imgs, lbls)
            adv_images.append(adv.cpu())
            adv_labels.append(lbls.cpu())
        return torch.cat(adv_images), torch.cat(adv_labels)
    except Exception as e:
        print(f"[ERROR] Attack generation failed for {attack_name}: {e}")
        return None, None


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


def generate_attacks_phase(output_dir: str, model_filter: str = None, dataset_filter: str = None, attack_filter: str = None):
    """Phase 1: Generate adversarial examples and save datasets.
    
    Args:
        output_dir: Output directory for results.
        model_filter: Filter by model name (e.g., 'VGG16'). None means all.
        dataset_filter: Filter by dataset name (e.g., 'Cifar10'). None means all.
        attack_filter: Filter by attack name (e.g., 'PGD'). None means all.
    
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
    # Filter by user preference if specified
    if attack_filter:
        attacks = [attack_filter] if attack_filter in available_attacks else []
        if not attacks:
            print(f"[WARN] Requested attack '{attack_filter}' not available. Available: {available_attacks}")
    else:
        attacks = available_attacks

    for model_name, dataset_name, kind, ckpt_path in checkpoints:
        # Apply filters
        if model_filter and model_name != model_filter:
            continue
        if dataset_filter and dataset_name != dataset_filter:
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
        
        model = load_model(model_name, num_classes).cuda()
        try:
            robust_load_state_dict(model, ckpt_path)
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
            adv_imgs, adv_lbls = generate_adversarial_dataset(model, test_loader, attack_name)
            if adv_imgs is None:
                continue

            adv_dataset = torch.utils.data.TensorDataset(adv_imgs, adv_lbls)
            adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=256, shuffle=False)
            adv_acc = evaluate_clean_accuracy(model, adv_loader)

            # Save adversarial dataset.
            adv_path = os.path.join(
                output_dir,
                f"{model_name}_{dataset_name}_{kind}_{attack_name}_adv.pt",
            )
            torch.save((adv_imgs, adv_lbls), adv_path)
            adv_datasets[(model_name, dataset_name, kind, attack_name)] = adv_path

            records.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "kind": kind,
                    "attack": attack_name,
                    "clean_acc": clean_acc,
                    "adv_acc": adv_acc,
                    "accuracy_drop": clean_acc - adv_acc,
                }
            )
            print(f"[INFO] {model_name} ({kind}) {dataset_name} – {attack_name}: clean {clean_acc:.2%}, adv {adv_acc:.2%}")

    return records, model_cache, loader_cache, adv_datasets


def analyze_transferability_phase(output_dir: str, model_cache: dict, loader_cache: dict, adv_datasets: dict):
    """Phase 2: Evaluate transferability of adversarial examples across models."""
    transfer_records = []

    for (src_model, src_dataset, src_kind, src_attack), adv_path in adv_datasets.items():
        # Load the pre-generated adversarial dataset.
        adv_imgs, adv_lbls = torch.load(adv_path)
        adv_dataset = torch.utils.data.TensorDataset(adv_imgs, adv_lbls)
        adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=256, shuffle=False)

        # Evaluate on all target models with the same dataset.
        for (tgt_model, tgt_dataset, tgt_kind), tgt_model_obj in model_cache.items():
            if tgt_dataset != src_dataset:
                continue
            
            tgt_acc = evaluate_clean_accuracy(tgt_model_obj, adv_loader)
            transfer_records.append(
                {
                    "source_model": src_model,
                    "source_kind": src_kind,
                    "source_attack": src_attack,
                    "target_model": tgt_model,
                    "target_kind": tgt_kind,
                    "dataset": src_dataset,
                    "transfer_acc": tgt_acc,
                }
            )

    return transfer_records


def generate_plots(output_dir: str, records: List[Dict], transfer_records: List[Dict]):
    """Phase 3: Generate comprehensive visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Attack Success Rates (Accuracy Drop)
    if records:
        df = pd.DataFrame(records)
        
        # Overall accuracy drop by model, dataset, and kind
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df, x="model", y="accuracy_drop", hue="attack", palette="Set2")
        plt.ylabel("Accuracy Drop (clean - adv)", fontweight='bold')
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
            sns.barplot(data=df_kind, x="model", y="accuracy_drop", hue="attack", palette="husl")
            plt.ylabel("Accuracy Drop", fontweight='bold')
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
            sns.boxplot(data=df_dataset, x="attack", y="accuracy_drop", hue="kind", palette="Set1")
            plt.ylabel("Accuracy Drop", fontweight='bold')
            plt.xlabel("Attack Method", fontweight='bold')
            plt.title(f"Attack Effectiveness Comparison – {dataset}", fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"attack_comparison_{dataset}.png"), dpi=300)
            plt.close()
            print(f"[INFO] Saved: attack_comparison_{dataset}.png")

    # Plot 2: Transferability Heatmaps
    if transfer_records:
        tf_df = pd.DataFrame(transfer_records)
        
        # Heatmap for each dataset and attack
        for dataset in tf_df["dataset"].unique():
            for attack in tf_df["source_attack"].unique():
                tf_subset = tf_df[(tf_df["dataset"] == dataset) & (tf_df["source_attack"] == attack)]
                
                # Create pivot table: rows = source models, cols = target models
                pivot_data = tf_subset.pivot_table(
                    index="source_model",
                    columns="target_model",
                    values="transfer_acc",
                    aggfunc="mean"
                )
                
                if pivot_data is not None and not pivot_data.empty:
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_data, annot=True, fmt=".2%", cmap="RdYlGn", vmin=0, vmax=1,
                                cbar_kws={"label": "Transfer Accuracy"})
                    plt.title(f"Adversarial Transferability – {dataset} ({attack})", fontsize=14, fontweight='bold')
                    plt.xlabel("Target Model", fontweight='bold')
                    plt.ylabel("Source Model", fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f"transferability_heatmap_{dataset}_{attack}.png"), dpi=300)
                    plt.close()
                    print(f"[INFO] Saved: transferability_heatmap_{dataset}_{attack}.png")


def main():
    parser = argparse.ArgumentParser(description="Adversarial robustness analysis of pruned models.")
    parser.add_argument("--mode", choices=["full", "generate", "analyze", "plot"], default="full",
                        help="Execution mode: full (all phases), generate (attacks only), analyze (transferability), plot (visualizations).")
    parser.add_argument("--model", type=str, default=None, help="Filter by model name (e.g., VGG16).")
    parser.add_argument("--dataset", type=str, default=None, help="Filter by dataset (e.g., Cifar10).")
    parser.add_argument("--attack", type=str, default=None, help="Filter by attack (e.g., PGD).")
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
            args.output_dir, args.model, args.dataset, args.attack
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
                if dataset_name == "Cifar10":
                    if "Cifar10" not in loader_cache:
                        loader_cache["Cifar10"] = load_cifar10(batch_size=256, num_workers=4)
                    _, _ = loader_cache["Cifar10"]
                    num_classes = 10
                elif dataset_name == "Cifar100":
                    if "Cifar100" not in loader_cache:
                        loader_cache["Cifar100"] = load_cifar100(batch_size=256, num_workers=4)
                    _, _ = loader_cache["Cifar100"]
                    num_classes = 100
                else:
                    continue
                
                model = load_model(model_name, num_classes).cuda()
                print(f"[DEBUG] Loading {model_name} ({kind}) on {dataset_name}")
                print(f"[DEBUG] Checkpoint path: {ckpt_path}")
                try:
                    robust_load_state_dict(model, ckpt_path)
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
