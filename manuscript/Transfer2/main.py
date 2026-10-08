import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd

# Import architectures and utilities from your framework
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from utils import *

from collapse import collapse_only
import argparse
from attacks import import fgsm_attack, pgd_attack

# =========================================================
# Adversarial Attack Implementations
# =========================================================

def evaluate_adversarial_robustness(models_dict, data_batch, device, epsilon=0.03):
    """
    Modular evaluation function for adversarial attacks.
    Easily expandable by adding new attack functions to the 'attacks' dictionary.
    """
    inputs, labels = data_batch
    inputs, labels = inputs.to(device), labels.to(device)

    # Modular registry of attacks. Add new attacks here.
    attacks = {
        "FGSM": lambda m, x, y: fgsm_attack(m, x, y, device, epsilon=epsilon),
        "PGD":  lambda m, x, y: pgd_attack(m, x, y, device, epsilon=epsilon, alpha=0.01, iters=10)
    }

    results = {}

    for attack_name, attack_fn in attacks.items():
        print(f"[INFO] Running {attack_name} attack (epsilon={epsilon})...")
        for model_name, model in models_dict.items():
            model.eval()
            
            # Generate perturbed batch
            perturbed_inputs = attack_fn(model, inputs, labels)
            
            # Evaluate model on the perturbed batch
            with torch.no_grad():
                outputs = model(perturbed_inputs)
                _, predicted = outputs.max(1)
                correct = predicted.eq(labels).sum().item()
                total = labels.size(0)
                acc = correct / total
                
            print(f"       -> {model_name} Accuracy: {acc:.4f}")
            results[f"adv_acc_{attack_name}_{model_name}"] = acc
            
    return results

# =========================================================
# Main Pipeline
# =========================================================

def process_checkpoint(model_before_path, checkpoints, args, device):
    """Executes the full loading, collapsing, pruning, and evaluation pipeline."""
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print("[INFO] Parsing experiment directory...")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path)
    
    print("[INFO] Initializing baseline model...")
    base_model, test_loader, dummy_input = initialize_architecture(model_name, dataset_name)
    model_before = load_weights(base_model, model_before_path, device)

    # 2. Load Original Post-Finetuning Model
    print("[INFO] Loading original post-finetuning model...")
    model_after_path = find_matching_checkpoint(model_before_path, checkpoints["after"])
    original_after_base, _, _ = initialize_architecture(model_name, dataset_name)
    original_after_model = load_weights(original_after_base, model_after_path, device)

    # 3. Execute Structural Collapse
    print("[INFO] Preparing and executing structural collapse...")
    full_collapsed_base, _, _ = initialize_architecture(model_name, dataset_name)
    compression_dict = load_collapse_regions(model_name, dataset_name, args.pre, args.post)
    
    collapse_start_time = time.time()
    full_collapsed_structure = collapse_only(
        model=full_collapsed_base,
        compression_set=compression_dict,
        input_shape=dummy_input.shape,
        device=device,
        dry_run=False,
        debug=False,
        handle_skips=True,
    )
    print(f"[INFO] Structural collapse completed in {time.time() - collapse_start_time:.2f}s")

    # 4. Load weights into collapsed structure
    collapsed_model_path = find_matching_checkpoint(model_before_path, checkpoints["collapsed"])
    full_collapsed_model_ready = load_weights(full_collapsed_structure, collapsed_model_path, device)

    # 5. Calculate Dynamic Sparsity & Apply Unstructured Pruning
    print("[INFO] Calculating dynamic sparsity to match collapse...")
    original_param_count = sum(p.numel() for p in model_before.parameters())
    collapsed_param_count = sum(p.numel() for p in full_collapsed_model_ready.parameters())
    
    target_sparsity = 1.0 - (collapsed_param_count / original_param_count)
    print(f"[INFO] Target unstructured sparsity calculated at: {target_sparsity * 100:.2f}%")

    unstructured_pruning_model = apply_unstructured_pruning(model_before, amount=float(target_sparsity))

    # 6. Evaluate Accuracies
    print("[INFO] Evaluating standard model accuracies...")
    acc_original_after = evaluate_accuracy(original_after_model, test_loader, device)
    acc_unstructured = evaluate_accuracy(unstructured_pruning_model, test_loader, device)
    acc_collapsed = evaluate_accuracy(full_collapsed_model_ready, test_loader, device)

    print(f"[RESULT] Standard Accuracy Original:     {acc_original_after:.4f}")
    print(f"[RESULT] Standard Accuracy Unstructured: {acc_unstructured:.4f}")
    print(f"[RESULT] Standard Accuracy Collapsed:    {acc_collapsed:.4f}")

    # 7. Extract Features & Compute CKA
    print("[INFO] Extracting features and computing CKA...")
    features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
    features_original = extract_features(original_after_model, test_loader, device)
    features_collapsed = extract_features(full_collapsed_model_ready, test_loader, device)

    print("[INFO] Computing CKA: original vs unstructured...")
    cka_unstructured = compare_CKA(features_original, features_unstructured)

    print("[INFO] Computing CKA: original vs collapsed...")
    cka_collapsed = compare_CKA(features_original, features_collapsed)

    print(f"[RESULT] CKA Unstructured: {cka_unstructured:.6f}")
    print(f"[RESULT] CKA Collapsed:    {cka_collapsed:.6f}")

    # 8. Evaluate Adversarial Robustness on First Batch
    print("[INFO] Extracting first batch for adversarial evaluation...")
    first_batch = next(iter(test_loader))
    
    models_to_test = {
        "original": original_after_model,
        "unstructured": unstructured_pruning_model,
        "collapsed": full_collapsed_model_ready
    }
    
    adv_metrics = evaluate_adversarial_robustness(models_to_test, first_batch, device)

    print(f"[INFO] Checkpoint completed in {time.time() - checkpoint_start_time:.2f}s")

    # Construct final results dictionary dynamically
    result_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "model_before": model_before_path,
        "intermediate_epoch": ckpt_epoch,
        "acc_original_after": acc_original_after,
        "acc_unstructured": acc_unstructured,
        "acc_collapsed": acc_collapsed,
        "cka_unstructured": cka_unstructured,
        "cka_collapsed": cka_collapsed,
    }
    
    # Merge the adversarial metrics into the final dictionary
    result_dict.update(adv_metrics)
    
    return result_dict


def main():
    parser = argparse.ArgumentParser(description="CKA and Adversarial comparison post-processing")
    parser.add_argument("--model", required=True, help="Model name, e.g. VGG16")
    parser.add_argument("--pre", type=int, default=300, help="pre collapse epochs")
    parser.add_argument("--post", type=int, default=100, help="post collapse epochs")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--output", default="cka_comparison_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Starting Comparison Post-Processing")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 1. Discover Checkpoints
    print("[INFO] Searching for model checkpoints...")
    checkpoints = find_experiment_checkpoints(args.model, args.dataset, args.pre, args.post)
    
    print(f"[INFO] Found {len(checkpoints['before'])} 'before' checkpoints.")
    print(f"[INFO] Found {len(checkpoints['collapsed'])} collapsed checkpoints.")
    print(f"[INFO] Found {len(checkpoints['after'])} 'after' checkpoints.")

    results = []

    # 2. Process Checkpoints
    for idx, model_before_path in enumerate(checkpoints['before'], start=1):
        print("\n" + "=" * 70)
        print(f"Processing checkpoint {idx}/{len(checkpoints['before'])}")
        print(f"Path: {model_before_path}")
        print("=" * 70)

        try:
            result_metrics = process_checkpoint(model_before_path, checkpoints, args, device)
            results.append(result_metrics)
        except Exception as e:
            print(f"[ERROR] Failed processing checkpoint: {model_before_path}")
            print(f"[ERROR] {type(e).__name__}: {e}")

    # 3. Save Results
    print("\n" + "=" * 70)
    print("[INFO] Saving final results...")
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Results successfully saved to '{args.output}'")
    print(f"[INFO] Total successful experiments: {len(results)}")
    print(f"[INFO] Total runtime: {time.time() - total_start_time:.2f}s")
    
    print("\n" + "=" * 70 + "\nFINAL RESULTS\n" + "=" * 70)
    print(df.to_string())

if __name__ == "__main__":
    main()