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
from attacks import fgsm_attack, pgd_attack
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

        avg_watts, _ = self.parser.get_results()

        # Fallback for fast inference batches (isolated to assigned GPU)
        if avg_watts == 0.0 and torch.cuda.is_available():
            try:
                gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                res = subprocess.run(
                    f"nvidia-smi -i {gpu_id} --query-gpu=power.draw --format=csv,noheader,nounits".split(),
                    capture_output=True, text=True, check=True
                )
                vals = [float(x.strip()) for x in res.stdout.strip().splitlines() if x.strip()]
                if vals:
                    avg_watts = vals[0]
            except Exception:
                pass

        total_joules = avg_watts * self.elapsed_time
        return float(avg_watts), float(total_joules), float(self.elapsed_time)

# =========================================================
# Adversarial Attack Robustness
# =========================================================

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
# Main Processing Pipeline
# =========================================================

def process_checkpoint(model_before_path, checkpoints, args, device):
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print("[INFO] Parsing experiment directory...")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path, args.model, args.dataset)
    
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

    # Move models to CPU to free VRAM for IMP training and avoid OOM during evaluation.
    model_before.to("cpu")
    original_after_model.to("cpu")
    full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()
    
    print(f"[INFO] Target unstructured sparsity calculated at: {target_sparsity * 100:.2f}%")

    checkpoint_dir = os.path.dirname(model_before_path)
    unstructured_ckpt_path = os.path.join(checkpoint_dir, f"final_JF_Unstructured_IMP_post{args.post}.pt")
    
    if os.path.exists(unstructured_ckpt_path):
        print(f"[INFO] Loading existing IMP unstructured checkpoint: {unstructured_ckpt_path}")
        unstructured_base, _, _, _ = initialize_architecture(model_name, dataset_name)
        unstructured_pruning_model = load_weights(unstructured_base, unstructured_ckpt_path, device)
        unstructured_pruning_model.to("cpu")
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
        unstructured_pruning_model.to("cpu")
        torch.cuda.empty_cache()

    # 6. Evaluate Accuracy, Power Draw, and Energy Consumption (SEQUENTIAL GPU ALLOCATION)
    print("\n[INFO] Evaluating model accuracy, power draw, and energy consumption...")
    
    print("       -> Evaluating Baseline")
    model_before = model_before.to(device)
    acc_before, power_before, energy_before = evaluate_accuracy(
        model_before, test_loader, device, power_interval=args.power_interval
    )
    model_before = model_before.to("cpu")
    torch.cuda.empty_cache()
    
    print("       -> Evaluating Original After Finetuning")
    original_after_model = original_after_model.to(device)
    acc_original, power_original, energy_original = evaluate_accuracy(
        original_after_model, test_loader, device, power_interval=args.power_interval
    )
    if acc_before > (acc_original + 0.20):
        print(f"\n[CRITICAL WARNING] 'original_after_model' suffered catastrophic forgetting ({acc_original*100:.2f}%).")
        print(f"[CRITICAL WARNING] Re-evaluating healthy 'model_before' ({acc_before*100:.2f}%) on warm cache as the true baseline...")
        model_before = model_before.to(device)
        acc_original, power_original, energy_original = evaluate_accuracy(
            model_before, test_loader, device, power_interval=args.power_interval
        )
        model_before = model_before.to("cpu")
    original_after_model = original_after_model.to("cpu")
    torch.cuda.empty_cache()

    print("       -> Evaluating Unstructured Pruned Model")
    unstructured_pruning_model = unstructured_pruning_model.to(device)
    acc_unstructured, power_unstructured, energy_unstructured = evaluate_accuracy(
        unstructured_pruning_model, test_loader, device, power_interval=args.power_interval
    )
    unstructured_pruning_model = unstructured_pruning_model.to("cpu")
    torch.cuda.empty_cache()

    print("       -> Evaluating Collapsed Model")
    full_collapsed_model_ready = full_collapsed_model_ready.to(device)
    acc_collapsed, power_collapsed, energy_collapsed = evaluate_accuracy(
        full_collapsed_model_ready, test_loader, device, power_interval=args.power_interval
    )
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()

    print(f"[RESULT] Accuracy Original:     {acc_original:.4f} | Power: {power_original:.2f}W | Energy: {energy_original:.2f}J")
    print(f"[RESULT] Accuracy Unstructured: {acc_unstructured:.4f} | Power: {power_unstructured:.2f}W | Energy: {energy_unstructured:.2f}J")
    print(f"[RESULT] Accuracy Collapsed:    {acc_collapsed:.4f} | Power: {power_collapsed:.2f}W | Energy: {energy_collapsed:.2f}J")

    # 7. Benchmark Theoretical and Measured FLOPs (SEQUENTIAL GPU ALLOCATION)
    print("\n[INFO] Benchmarking theoretical and measured FLOPs...")
    
    original_after_model = original_after_model.to(device)
    _, flops_original, _ = benchmark_model(original_after_model, test_loader, device, num_batches=1)
    original_after_model = original_after_model.to("cpu")
    torch.cuda.empty_cache()
    
    full_collapsed_model_ready = full_collapsed_model_ready.to(device)
    _, flops_collapsed, _ = benchmark_model(full_collapsed_model_ready, test_loader, device, num_batches=1)
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()
    
    # Calculate Effective FLOPs for IMP
    flops_unstructured = flops_original * (1.0 - float(target_sparsity))
    
    print(f"[RESULT] FLOPs Original:     {flops_original:.2e}")
    print(f"[RESULT] FLOPs Unstructured: {flops_unstructured:.2e} (Theoretical/Effective)")
    print(f"[RESULT] FLOPs Collapsed:    {flops_collapsed:.2e} (Measured)")

    # 8. Extract Features & Compute CKA (SEQUENTIAL GPU ALLOCATION)
    print("\n[INFO] Extracting features and computing CKA...")
    
    original_after_model = original_after_model.to(device)
    features_original = extract_features(original_after_model, test_loader, device)
    original_after_model = original_after_model.to("cpu")
    torch.cuda.empty_cache()

    unstructured_pruning_model = unstructured_pruning_model.to(device)
    features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
    unstructured_pruning_model = unstructured_pruning_model.to("cpu")
    torch.cuda.empty_cache()

    full_collapsed_model_ready = full_collapsed_model_ready.to(device)
    features_collapsed = extract_features(full_collapsed_model_ready, test_loader, device)
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()
    
    cka_unstructured = compare_CKA(features_original, features_unstructured)
    cka_collapsed = compare_CKA(features_original, features_collapsed)

    # 9. Evaluate Adversarial Robustness
    print("\n[INFO] Evaluating Adversarial Robustness...")
    data_batch = next(iter(test_loader))
    
    original_after_model = original_after_model.to(device)
    unstructured_pruning_model = unstructured_pruning_model.to(device)
    full_collapsed_model_ready = full_collapsed_model_ready.to(device)
    
    models_dict = {
        "original": original_after_model,
        "unstructured": unstructured_pruning_model,
        "collapsed": full_collapsed_model_ready
    }
    
    adv_metrics = evaluate_adversarial_robustness(models_dict, data_batch, device)

    # Final cleanup to prevent cross-experiment bleed
    original_after_model = original_after_model.to("cpu")
    unstructured_pruning_model = unstructured_pruning_model.to("cpu")
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    del data_batch, models_dict
    torch.cuda.empty_cache()

    # 10. Compile Results
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    result_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "gpu_name": gpu_name,
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
        "flops_original": float(flops_original),
        "flops_unstructured": float(flops_unstructured),
        "flops_collapsed": float(flops_collapsed),
        "cka_unstructured": cka_unstructured,
        "cka_collapsed": cka_collapsed,
    }
    result_dict.update(adv_metrics)
    return result_dict

# =========================================================
# Entry Point
# =========================================================


def main():
    df, all_models = load_and_compute_deltas()
    if df is None:
        return

    grouped = df.groupby(['dataset', 'pre_epochs', 'post_epochs'])

    for (dataset, pre, post), group_df in grouped:
        print(f"\n--- Generating plots for {dataset} | Pre: {pre} | Post: {post} ---")
        
        file_suffix = f"{dataset}_pre{pre}_post{post}"
        title_suffix = f"\n({dataset} | Pre: {pre}, Post: {post})"

        # Plot 1: Change in Task Accuracy
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['delta_acc_imp', 'delta_acc_collapse'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Change in Task Accuracy (Compared to Original){title_suffix}',
            ylabel='Accuracy Delta\n(Higher is Better)',
            filename=f'plot_1_delta_accuracy_{file_suffix}.png',
            zero_line=True,
            df_annotations=['acc_unstructured', 'acc_collapsed'],
            anno_fmt="{val:.1%}"
        )

        # Plot 2: Normalized Energy Savings (%) across heterogeneous GPUs
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['energy_saved_imp_pct', 'energy_saved_collapse_pct'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Relative Energy Savings (% Reduced from Baseline){title_suffix}',
            ylabel='Energy Saved (%)\n(Higher is Better)',
            filename=f'plot_2_energy_savings_{file_suffix}.png',
            zero_line=True,
            df_annotations=['energy_saved_imp_pct', 'energy_saved_collapse_pct'],
            anno_fmt="{val:+.1f}%"
        )

        # Plot 3: CKA Scores
        plot_bar_chart(
            df=group_df,
            all_models=all_models,
            columns=['cka_unstructured', 'cka_collapsed'],
            labels=['IMP (Unstructured)', 'Collapsed'],
            title=f'Representational Similarity (CKA) to Original{title_suffix}',
            ylabel='CKA Score\n(Higher = More Similar to Original)',
            filename=f'plot_3_cka_scores_{file_suffix}.png',
            zero_line=False,
            df_annotations=['cka_unstructured', 'cka_collapsed'],
            anno_fmt="{val:.3f}"
        )

        # Plot 4: Adversarial Robustness Subplots
        plot_adversarial_subplots(
            df=group_df,
            all_models=all_models,
            title=f'Change in Adversarial Robustness{title_suffix}',
            ylabel='Accuracy Delta under Attack\n(Higher is Better)',
            filename=f'plot_4_delta_adversarial_{file_suffix}.png'
        )
        
        # Plot 5: Percentage Pruned (if present)
        if 'percent_pruned' in group_df.columns:
            plot_bar_chart(
                df=group_df,
                all_models=all_models,
                columns=['percent_pruned'],
                labels=['Target Sparsity (Same for IMP & Collapsed)'],
                title=f'Percentage of Weights Pruned from Original{title_suffix}',
                ylabel='Percentage Pruned\n(Higher = More Compressed)',
                filename=f'plot_5_percent_pruned_{file_suffix}.png',
                zero_line=False,
                df_annotations=['percent_pruned'],
                anno_fmt="{val:.1%}"
            )
    
    print("\nAll delta comparisons generated successfully!")
    
if __name__ == "__main__":
    main()