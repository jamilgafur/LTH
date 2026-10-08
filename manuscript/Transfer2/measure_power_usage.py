import os
import glob
import re
import json
import time
import threading
import subprocess
import argparse
from copy import deepcopy

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
from attacks import *
import measure_power_usage

# =========================================================
# Power and Energy Monitoring Context Manager
# =========================================================

class PowerTracker:
    """
    Context manager to track average power (Watts) and total energy (Joules)
    using measure_power_usage.py parsers in a non-blocking background thread.
    """
    def __init__(self, device=None, interval_s: int = 1, max_watts: float = None):
        self.device = device
        self.interval_s = max(1, int(interval_s))
        self.max_watts = max_watts
        self.parser = None
        self.proc = None
        self.thread = None
        self.stop_event = threading.Event()
        self.start_time = None
        self.elapsed_time = 0.0

        # Select parser based on target compute architecture
        if torch.cuda.is_available() and (device is None or device.type == "cuda"):
            try:
                self.parser = measure_power_usage.NvidiaSmiParser(max_watts=self.max_watts)
            except Exception as e:
                print(f"[WARN] Failed to initialize NvidiaSmiParser: {e}")
                self.parser = None
        else:
            try:
                self.parser = measure_power_usage.TurbostatParser(max_watts=self.max_watts)
            except Exception as e:
                print(f"[WARN] Failed to initialize TurbostatParser: {e}")
                self.parser = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def _reader(self):
        while not self.stop_event.is_set():
            if self.proc and self.proc.stdout:
                line = self.proc.stdout.readline()
                if line:
                    self.parser.parse(line)
                else:
                    break

    def start(self):
        self.start_time = time.time()
        if self.parser:
            try:
                self.proc = self.parser.start_monitoring(self.interval_s)
                self.thread = threading.Thread(target=self._reader, daemon=True)
                self.thread.start()
            except Exception as e:
                print(f"[WARN] Could not spawn power monitoring process: {e}")
                self.proc = None

    def stop(self):
        self.elapsed_time = time.time() - (self.start_time or time.time())
        self.stop_event.set()
        if self.proc:
            try:
                self.proc.terminate()
                self.proc.kill()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def get_results(self):
        if not self.parser:
            return 0.0, 0.0, self.elapsed_time

        avg_watts, num_intervals = self.parser.get_results()

        # Fallback for short inference durations where no full interval elapsed
        if avg_watts == 0.0 and torch.cuda.is_available():
            try:
                res = subprocess.run(
                    "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits".split(),
                    capture_output=True, text=True, check=True
                )
                vals = [float(x.strip()) for x in res.stdout.strip().splitlines() if x.strip()]
                if vals:
                    avg_watts = sum(vals)
            except Exception:
                pass

        total_joules = avg_watts * self.elapsed_time
        return float(avg_watts), float(total_joules), float(self.elapsed_time)

# =========================================================
# Adversarial Attack Implementations
# =========================================================

def fgsm_attack(model, images, labels, device, epsilon=0.03):
    """Fast Gradient Sign Method (FGSM)"""
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    return images + epsilon * images.grad.sign()

def pgd_attack(model, images, labels, device, epsilon=0.03, alpha=0.01, iters=10):
    """Projected Gradient Descent (PGD)"""
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = (original_images + eta).detach_()
        
    return images

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
# Model Evaluation with Energy & Power Measurement
# =========================================================

def evaluate_accuracy(model: nn.Module, dataloader, device: torch.device, power_interval: int = 1):
    """Evaluates top-1 accuracy while profiling power (W) and energy (J)."""
    print("[INFO] Starting accuracy and energy evaluation...")
    model.eval()
    correct = 0
    total = 0

    with PowerTracker(device=device, interval_s=power_interval) as tracker:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 50 == 0:
                    print(f"[INFO] Accuracy evaluation: batch={batch_idx + 1}, samples={total}")

    avg_watts, total_joules, elapsed_sec = tracker.get_results()
    accuracy = correct / total if total > 0 else 0.0

    print(
        f"[INFO] Evaluation complete: acc={accuracy:.4f}, "
        f"power={avg_watts:.2f}W, energy={total_joules:.2f}J, time={elapsed_sec:.2f}s"
    )
    return accuracy, avg_watts, total_joules

# =========================================================
# Iterative Magnitude Pruning (IMP) Training Loop
# =========================================================

def train_imp(model, train_loader, device, epochs, target_sparsity, save_path, power_interval: int = 1):
    """Trains on-the-fly using Iterative Magnitude Pruning to match target sparsity."""
    print(f"\n[INFO] Starting Iterative Magnitude Pruning (IMP) for {epochs} epochs...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    pruning_steps = max(1, int(epochs * 0.75))
    incremental_amount = 1.0 - (1.0 - target_sparsity) ** (1.0 / pruning_steps)
    print(f"[INFO] Compounding prune rate: {incremental_amount*100:.2f}% per step over {pruning_steps} steps.")

    model.train()
    with PowerTracker(device=device, interval_s=power_interval) as tracker:
        for epoch in range(epochs):
            if epoch < pruning_steps:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=incremental_amount
                )
                
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

    train_watts, train_joules, train_time = tracker.get_results()
    print(f"[INFO] IMP training energy: {train_watts:.2f}W avg, {train_joules:.2f}J total ({train_time:.2f}s)")

    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[INFO] IMP model checkpoint saved to {save_path}\n")
    
    model.eval()
    return model

# =========================================================
# Architecture and Region Utilities
# =========================================================

def initialize_architecture(model_name: str, dataset_name: str):
    """Initializes architecture and returns data loaders and sample tensor."""
    print(f"[INFO] Initializing architecture: model={model_name}, dataset={dataset_name}")
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    
    model_kwargs = {"num_classes": num_classes}
    dummy_input = next(iter(train_loader))[0][0:1]
    
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    model = model_class(**model_kwargs)
    return model, train_loader, test_loader, dummy_input

def find_experiment_checkpoints(model_name, dataset_name, pre_epochs, post_epochs):
    base_pattern = f"../Tranfer/{model_name}_{dataset_name}_*epochs{post_epochs}_*pre*{pre_epochs}"
    return {
        "before": glob.glob(f"{base_pattern}/checkpoints/final_JF_Control.pt"),
        "collapsed": glob.glob(f"{base_pattern}/checkpoints/final_JF_Dynamic_Region_All_Combined.pt"),
        "after": glob.glob(f"{base_pattern}/checkpoints/final_JF_Control_Continuted.pt")
    }

def load_collapse_regions(model_name, dataset_name, pre_epochs, post_epochs):
    json_filename = (
        f"../Tranfer/{model_name}_{dataset_name}_"
        f"epochs{post_epochs}_pretrain{pre_epochs}_JF_discovered_regions.json"
    )
    if not os.path.isfile(json_filename):
        raise FileNotFoundError(f"Discovered regions file not found: {json_filename}")

    with open(json_filename, "r") as f:
        discovered_regions = json.load(f)

    json_to_collapse = discovered_regions.get("Dynamic_Region_All_Combined")
    if not json_to_collapse:
        raise ValueError(f"No combined collapse regions found in {json_filename}.")
        
    return {f"Region_{i}": pair for i, pair in enumerate(json_to_collapse)}

# =========================================================
# Pipeline Execution
# =========================================================

def process_checkpoint(model_before_path, checkpoints, args, device):
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print("[INFO] Parsing experiment directory...")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path)
    
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
    unstructured_ckpt_path = os.path.join(checkpoint_dir, "final_JF_Unstructured_IMP.pt")
    
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
            power_interval=args.power_interval
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
    parser.add_argument("--output", default="cka_comparison_results.csv", help="Output CSV filename")
    parser.add_argument("--power-interval", type=int, default=1, help="Query interval in seconds for power monitoring (default: 1)")
    args = parser.parse_args()

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Starting Comparison Post-Processing Pipeline")
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