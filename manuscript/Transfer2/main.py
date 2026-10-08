import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
from copy import deepcopy

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
from attacks import fgsm_attack, pgd_attack


def evaluate_adversarial_robustness(models_dict, data_batch, device, epsilon=0.03):
    """Modular evaluation function for adversarial attacks."""
    inputs, labels = data_batch
    inputs, labels = inputs.to(device), labels.to(device)

    attacks = {
        "FGSM": lambda m, x, y: fgsm_attack(m, x, y, device, epsilon=epsilon),
        "PGD":  lambda m, x, y: pgd_attack(m, x, y, device, epsilon=epsilon, alpha=0.01, iters=10)
    }

    results = {}
    for attack_name, attack_fn in attacks.items():
        print(f"[INFO] Running {attack_name} attack (epsilon={epsilon})...")
        for model_name, model in models_dict.items():
            model.eval()
            perturbed_inputs = attack_fn(model, inputs, labels)
            
            with torch.no_grad():
                outputs = model(perturbed_inputs)
                _, predicted = outputs.max(1)
                acc = predicted.eq(labels).sum().item() / labels.size(0)
                
            print(f"       -> {model_name} Accuracy: {acc:.4f}")
            results[f"adv_acc_{attack_name}_{model_name}"] = acc
            
    return results

# =========================================================
# Iterative Magnitude Pruning (IMP) Training Loop
# =========================================================

def train_imp(model, train_loader, device, epochs, target_sparsity, save_path):
    """Trains a model on the fly using Iterative Magnitude Pruning to match target sparsity."""
    print(f"\n[INFO] Starting Iterative Magnitude Pruning (IMP) for {epochs} epochs...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    # Prune over the first 75% of epochs, finetune for the rest
    pruning_steps = max(1, int(epochs * 0.75))
    
    # Calculate incremental amount so that compounding hits exactly the target sparsity
    incremental_amount = 1.0 - (1.0 - target_sparsity) ** (1.0 / pruning_steps)
    print(f"[INFO] Calculated incremental prune rate: {incremental_amount*100:.2f}% per step over {pruning_steps} steps.")

    model.train()
    for epoch in range(epochs):
        if epoch < pruning_steps:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=incremental_amount
            )
            print(f"[INFO] Pruned epoch {epoch+1}: applied incremental sparsity step.")
            
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"[INFO] Epoch {epoch+1}/{epochs} - Train Loss: {running_loss/len(train_loader):.4f}")
        
    # Make pruning permanent by removing the parameter hooks
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    # Save the finetuned sparse weights
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[INFO] IMP model saved to {save_path}\n")
    
    model.eval()
    return model

# =========================================================
# Updated Framework Initialization
# =========================================================

def initialize_architecture(model_name: str, dataset_name: str):
    """Updated to return the train_loader needed for on-the-fly finetuning."""
    print(f"[INFO] Initializing architecture: model={model_name}, dataset={dataset_name}")
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    
    model_kwargs = {"num_classes": num_classes}
    dummy_input = next(iter(train_loader))[0][0:1]
    
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    model = model_class(**model_kwargs)
    
    return model, train_loader, test_loader, dummy_input

# =========================================================
# Main Pipeline
# =========================================================
def process_checkpoint(model_before_path, checkpoints, args, device):
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline (Pass args.model and args.dataset)
    print("[INFO] Parsing experiment directory...")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(
        model_before_path, args.model, args.dataset
    )
    
    print("[INFO] Initializing baseline model...")
    base_model, train_loader, test_loader, dummy_input = initialize_architecture(model_name, dataset_name)
    model_before = load_weights(base_model, model_before_path, device)

    # 2. Load Original Post-Finetuning Model
    print("[INFO] Loading original post-finetuning model...")
    model_after_path = find_matching_checkpoint(model_before_path, checkpoints["after"])
    original_after_base, _, _, _ = initialize_architecture(model_name, dataset_name)
    original_after_model = load_weights(original_after_base, model_after_path, device)

    # 3. Execute Structural Collapse
    print("[INFO] Preparing and executing structural collapse...")
    full_collapsed_base, _, _, _ = initialize_architecture(model_name, dataset_name)
    compression_dict = load_collapse_regions(model_name, dataset_name, args.pre, args.post)
    
    if compression_dict is None:
        print(f"[WARN] Skipping evaluation for {model_before_path} - No collapse regions available.")
        return None
    
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

    # 5. Calculate Dynamic Sparsity & Handle IMP Unstructured Pruning
    print("[INFO] Calculating dynamic sparsity to match collapse...")
    original_param_count = sum(p.numel() for p in model_before.parameters())
    collapsed_param_count = sum(p.numel() for p in full_collapsed_model_ready.parameters())
    target_sparsity = 1.0 - (collapsed_param_count / original_param_count)
    print(f"[INFO] Target unstructured sparsity calculated at: {target_sparsity * 100:.2f}%")

    checkpoint_dir = os.path.dirname(model_before_path)
    unstructured_ckpt_path = os.path.join(checkpoint_dir, f"final_JF_Unstructured_IMP_post{args.post}.pt")
    
    if os.path.exists(unstructured_ckpt_path):
        print(f"[INFO] Loading existing IMP unstructured checkpoint: {unstructured_ckpt_path}")
        unstructured_base, _, _, _ = initialize_architecture(model_name, dataset_name)
        unstructured_pruning_model = load_weights(unstructured_base, unstructured_ckpt_path, device)
    else:
        print("[INFO] IMP checkpoint not found. Starting on-the-fly finetuning...")
        unstructured_clone = deepcopy(model_before)
        unstructured_pruning_model = train_imp(
            model=unstructured_clone, 
            train_loader=train_loader, 
            device=device, 
            epochs=args.post, 
            target_sparsity=float(target_sparsity), 
            save_path=unstructured_ckpt_path,
        )

    # 6. Evaluate Accuracy, Power Draw, and Energy Consumption
    print("\n[INFO] Evaluating model accuracy, power draw, and energy consumption...")
    acc_original, power_original, energy_original = evaluate_accuracy(
        original_after_model, test_loader, device, power_interval=args.power_interval
    )
    acc_unstructured, power_unstructured, energy_unstructured = evaluate_accuracy(
        unstructured_pruning_model, test_loader, device, power_interval=args.power_interval
    )
    acc_collapsed, power_collapsed, energy_collapsed = evaluate_accuracy(
        full_collapsed_model_ready, test_loader, device, power_interval=args.power_interval
    )

    print(f"[RESULT] Accuracy Original:     {acc_original:.4f} | Power: {power_original:.2f}W | Energy: {energy_original:.2f}J")
    print(f"[RESULT] Accuracy Unstructured: {acc_unstructured:.4f} | Power: {power_unstructured:.2f}W | Energy: {energy_unstructured:.2f}J")
    print(f"[RESULT] Accuracy Collapsed:    {acc_collapsed:.4f} | Power: {power_collapsed:.2f}W | Energy: {energy_collapsed:.2f}J")

    # 7. Extract Features & Compute CKA
    print("\n[INFO] Extracting features and computing CKA...")
    features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
    features_original = extract_features(original_after_model, test_loader, device)
    features_collapsed = extract_features(full_collapsed_model_ready, test_loader, device)

    cka_unstructured = compare_CKA(features_original, features_unstructured)
    cka_collapsed = compare_CKA(features_original, features_collapsed)

    print(f"[RESULT] CKA Unstructured: {cka_unstructured:.6f}")
    print(f"[RESULT] CKA Collapsed:    {cka_collapsed:.6f}")

    # 8. Evaluate Adversarial Robustness on First Batch
    print("\n[INFO] Extracting first batch for adversarial evaluation...")
    first_batch = next(iter(test_loader))
    models_to_test = {
        "original": original_after_model,
        "unstructured": unstructured_pruning_model,
        "collapsed": full_collapsed_model_ready
    }
    adv_metrics = evaluate_adversarial_robustness(models_to_test, first_batch, device)

    print(f"[INFO] Checkpoint completed in {time.time() - checkpoint_start_time:.2f}s")

    result_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "pre_epochs": args.pre,
        "post_epochs": args.post,
        "model_before": model_before_path,
        "intermediate_epoch": ckpt_epoch,
        "acc_original_after": acc_original,
        "acc_unstructured": acc_unstructured,
        "acc_collapsed": acc_collapsed,
        "power_watts_original": round(power_original, 2),
        "energy_joules_original": round(energy_original, 2),
        "power_watts_unstructured": round(power_unstructured, 2),
        "energy_joules_unstructured": round(energy_unstructured, 2),
        "power_watts_collapsed": round(power_collapsed, 2),
        "energy_joules_collapsed": round(energy_collapsed, 2),
        "cka_unstructured": cka_unstructured,
        "cka_collapsed": cka_collapsed,
    }
    result_dict.update(adv_metrics)
    return result_dict

def main():
    parser = argparse.ArgumentParser(description="CKA, Energy, and Adversarial comparison post-processing")
    parser.add_argument("--model", required=True, help="Model name, e.g. VGG16")
    parser.add_argument("--pre", type=int, default=300, help="pre collapse epochs")
    parser.add_argument("--post", type=int, default=100, help="post collapse epochs")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--output", default="auto", help="Output CSV filename (set to 'auto' to avoid collisions)")
    parser.add_argument("--power-interval", type=int, default=1, help="Query interval in seconds for power monitoring (default: 1)")
    args = parser.parse_args()

    # Automatically format unique output path if 'auto' or default to prevent overwrite collisions
    if args.output == "auto" or args.output == "cka_comparison_results.csv":
        args.output = f"cka_results_{args.model}_{args.dataset}_pre{args.pre}_post{args.post}.csv"

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Starting Comparison Post-Processing Pipeline")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    print(f"Epochs Config: pre={args.pre}, post={args.post}")
    print(f"Output File: {args.output}")
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

        result_metrics = process_checkpoint(model_before_path, checkpoints, args, device)
        results.append(result_metrics)

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
    