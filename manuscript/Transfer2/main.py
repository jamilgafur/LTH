import os
import glob
import csv
import io
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

def evaluate_adversarial_robustness(models_dict, data_batch, device, epsilons=(0.03,), curve_rows=None):
    """Modular evaluation function for adversarial attacks.

    Sweeps `epsilons`. epsilon=0.03 keeps the historical flat column names
    (adv_acc_FGSM_original etc.); other epsilons get `_eps<value>` suffixes.
    When `curve_rows` (a list) is provided, every (attack, epsilon, model)
    point is also appended for long-format curve output.
    """
    inputs, labels = data_batch
    inputs, labels = inputs.to(device), labels.to(device)

    results = {}
    for epsilon in epsilons:
        suffix = "" if abs(epsilon - 0.03) < 1e-12 else f"_eps{epsilon:g}"
        attacks = {
            "FGSM": lambda m, x, y: fgsm_attack(m, x, y, device, epsilon=epsilon),
            "PGD":  lambda m, x, y: pgd_attack(m, x, y, device, epsilon=epsilon, alpha=0.01, iters=20)
        }
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
                results[f"adv_acc_{attack_name}{suffix}_{model_name}"] = acc
                if curve_rows is not None:
                    curve_rows.append({"attack": attack_name, "epsilon": epsilon,
                                       "model": model_name, "adv_acc": acc})

    return results

# =========================================================
# Main Processing Pipeline
# =========================================================

def _append_long_csv(path, rows, fieldnames):
    """Append long-format (tidy) rows to a CSV, writing the header on first creation."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Appended {len(rows)} rows to {path}")


def process_checkpoint(model_before_path, checkpoints, args, device):
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print(f"[INFO] Parsing experiment directory... {model_before_path}")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path, args.model, args.dataset)
    
    print("[INFO] Initializing baseline model...")
    base_model, train_loader, test_loader, _ = initialize_architecture(model_name, dataset_name)
    model_before = load_weights(base_model, model_before_path, device)

    # 2. Load Original Post-Finetuning Model (optional). When Control_Continuted is
    #    still training, run a baseline-only pass: sparsity/IMP/structural and the
    #    collapsed-variant metrics don't need it; after-columns come out None and a
    #    later rerun (IMP/structural artifacts persist on disk) fills them in.
    print("[INFO] Loading original post-finetuning model...")
    try:
        model_after_path = find_matching_checkpoint(model_before_path, checkpoints["after"])
        original_after_base, _, _, _ = initialize_architecture(model_name, dataset_name)
        original_after_model = load_weights(original_after_base, model_after_path, device)
    except FileNotFoundError:
        print("[WARN] No Control_Continuted artifact for this dir yet — baseline-only pass.")
        original_after_model = None

    # 3. Load the collapsed model as a whole artifact (architecture + weights).
    #    The Tranfer experiments persist `*_whole.pt` files, so analysis no
    #    longer re-runs initialize -> collapse -> load per checkpoint.
    print("[INFO] Loading whole collapsed model artifact...")
    try:
        collapsed_model_path = find_matching_checkpoint(model_before_path, checkpoints["collapsed"])
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{e}\n[HINT] No `*_whole.pt` collapsed artifact found for this experiment. "
            f"Re-run the Tranfer experiments to generate `final_*_whole.pt` artifacts."
        ) from e

    load_start_time = time.time()
    full_collapsed_model_ready = torch.load(collapsed_model_path, map_location=device, weights_only=False)
    full_collapsed_model_ready.to(device)
    full_collapsed_model_ready.eval()
    print(f"[INFO] Whole-model load completed in {time.time() - load_start_time:.2f}s")

    # 5. Calculate Dynamic Sparsity & Handle IMP Unstructured Pruning
    print("[INFO] Calculating dynamic sparsity to match collapse...")
    original_param_count = sum(p.numel() for p in model_before.parameters())
    collapsed_param_count = sum(p.numel() for p in full_collapsed_model_ready.parameters())
    target_sparsity = 1.0 - (collapsed_param_count / original_param_count)

    # Move models to CPU to free VRAM for IMP training and avoid OOM during evaluation.
    model_before.to("cpu")
    if original_after_model is not None:
        original_after_model.to("cpu")
    full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()
    
    print(f"[INFO] Target unstructured sparsity calculated at: {target_sparsity * 100:.2f}%")

    # IMP epochs derive from the checkpoint's own directory naming (epochs token);
    # --post acts only as an explicit override.
    parsed_post, _ = parse_epoch_tokens(model_before_path)
    imp_epochs = args.post if args.post is not None else parsed_post
    if imp_epochs is None:
        raise ValueError(
            f"Could not derive post-collapse epochs from '{model_before_path}'. "
            f"Pass --post explicitly."
        )

    checkpoint_dir = os.path.dirname(model_before_path)
    if "_scorerall" in checkpoint_dir and os.path.isdir(checkpoint_dir.replace("_scorerall", "")):
        # borrowed scorerall before: keep derived baselines (IMP/structural) in the default universe
        checkpoint_dir = checkpoint_dir.replace("_scorerall", "")
    qs = ""  # PTQ: IMP/structural baselines are shared across fp32/fp16 — quant converts at measurement time
    unstructured_ckpt_path = os.path.join(checkpoint_dir, f"final_JF_Unstructured_IMP{qs}_post{imp_epochs}.pt")

    # IMP retrains from the (inherited) 'before' model -> scorer-independent:
    # copy a finished artifact from the default universe instead of retraining.
    inherit_from_default_universe(unstructured_ckpt_path)

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
            epochs=imp_epochs,
            target_sparsity=float(target_sparsity),
            save_path=unstructured_ckpt_path,
            power_interval=args.power_interval
        )
        unstructured_pruning_model.to("cpu")
        torch.cuda.empty_cache()

    # 5b. Structural pruning baseline (L1 filter importance, matched param count).
    #     Unlike IMP, this actually shrinks the compute graph, so its measured
    #     FLOPs/energy are the fair comparator for the collapse claim.
    structural_model = None
    structural_ckpt_path = os.path.join(checkpoint_dir, f"final_JF_Structural_L1{qs}_post{imp_epochs}.pt")
    # Same scorer-independence argument as IMP: inherit from the default universe.
    inherit_from_default_universe(structural_ckpt_path)
    if os.path.exists(structural_ckpt_path):
        try:
            print(f"[INFO] Loading existing structural checkpoint: {structural_ckpt_path}")
            ckpt = torch.load(structural_ckpt_path, map_location="cpu", weights_only=False)
            structural_model = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            structural_model.eval()
        except Exception as e:
            print(f"[WARN] Failed to load structural checkpoint ({e}). Will attempt re-training.")
            structural_model = None
    if structural_model is None:
        try:
            structural_clone = deepcopy(model_before)
            structural_model = train_structural(
                model=structural_clone,
                train_loader=train_loader,
                device=device,
                epochs=imp_epochs,
                target_sparsity=float(target_sparsity),
                save_path=structural_ckpt_path,
            )
        except ImportError as e:
            print(f"[WARN] {e} Skipping structural baseline.")
            structural_model = None
        except Exception as e:
            print(f"[WARN] Structural baseline failed for {model_name}: {e}. Continuing without it.")
            structural_model = None
    structural_param_count = (sum(p.numel() for p in structural_model.parameters())
                              if structural_model is not None else None)

    # 6. Evaluate Accuracy, Power Draw, and Energy Consumption (SEQUENTIAL GPU ALLOCATION)
    print("\n[INFO] Evaluating model accuracy, power draw, and energy consumption...")
    
    print("       -> Evaluating Baseline")
    model_before = model_before.to(device)
    acc_before, power_before, energy_before = evaluate_accuracy(
        model_before, test_loader, device, power_interval=args.power_interval, quant=args.quant
    )
    model_before = model_before.to("cpu")
    torch.cuda.empty_cache()
    
    if original_after_model is not None:
        print("       -> Evaluating Original After Finetuning")
        original_after_model = original_after_model.to(device)
        acc_original, power_original, energy_original = evaluate_accuracy(
            original_after_model, test_loader, device, power_interval=args.power_interval, quant=args.quant
        )
        if acc_before > (acc_original + 0.20):
            print(f"\n[CRITICAL WARNING] 'original_after_model' suffered catastrophic forgetting ({acc_original*100:.2f}%).")
            print(f"[CRITICAL WARNING] Re-evaluating healthy 'model_before' ({acc_before*100:.2f}%) on warm cache as the true baseline...")
            model_before = model_before.to(device)
            acc_original, power_original, energy_original = evaluate_accuracy(
                model_before, test_loader, device, power_interval=args.power_interval, quant=args.quant
            )
            model_before = model_before.to("cpu")
        original_after_model = original_after_model.to("cpu")
        torch.cuda.empty_cache()
    else:
        acc_original = power_original = energy_original = None

    print("       -> Evaluating Unstructured Pruned Model")
    unstructured_pruning_model = unstructured_pruning_model.to(device)
    acc_unstructured, power_unstructured, energy_unstructured = evaluate_accuracy(
        unstructured_pruning_model, test_loader, device, power_interval=args.power_interval, quant=args.quant
    )
    unstructured_pruning_model = unstructured_pruning_model.to("cpu")
    torch.cuda.empty_cache()

    print("       -> Evaluating Collapsed Model")
    full_collapsed_model_ready = full_collapsed_model_ready.to(device)
    acc_collapsed, power_collapsed, energy_collapsed = evaluate_accuracy(
        full_collapsed_model_ready, test_loader, device, power_interval=args.power_interval, quant=args.quant
    )
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    torch.cuda.empty_cache()

    acc_structural = power_structural = energy_structural = float("nan")
    if structural_model is not None:
        print("       -> Evaluating Structural L1 Pruned Model")
        structural_model = structural_model.to(device)
        acc_structural, power_structural, energy_structural = evaluate_accuracy(
            structural_model, test_loader, device, power_interval=args.power_interval, quant=args.quant
        )
        structural_model = structural_model.to("cpu")
        torch.cuda.empty_cache()

    if acc_original is not None:
        print(f"[RESULT] Accuracy Original:     {acc_original:.4f} | Power: {power_original:.2f}W | Energy: {energy_original:.2f}J")
    print(f"[RESULT] Accuracy Unstructured: {acc_unstructured:.4f} | Power: {power_unstructured:.2f}W | Energy: {energy_unstructured:.2f}J")
    print(f"[RESULT] Accuracy Collapsed:    {acc_collapsed:.4f} | Power: {power_collapsed:.2f}W | Energy: {energy_collapsed:.2f}J")
    if structural_model is not None:
        print(f"[RESULT] Accuracy Structural:   {acc_structural:.4f} | Power: {power_structural:.2f}W | Energy: {energy_structural:.2f}J")

    # 7. Benchmark Efficiency: FLOPs, Latency, Throughput, Memory (SEQUENTIAL GPU ALLOCATION)
    print("\n[INFO] Benchmarking efficiency (FLOPs, latency, throughput, peak memory)...")

    bench_variants = []
    if original_after_model is not None:
        bench_variants.append(("original", original_after_model))
    bench_variants += [
        ("unstructured", unstructured_pruning_model),
        ("collapsed", full_collapsed_model_ready),
    ]
    if structural_model is not None:
        bench_variants.append(("structural", structural_model))

    benchmarks = {}
    for bench_name, bench_model in bench_variants:
        bench_model = bench_model.to(device)
        benchmarks[bench_name] = benchmark_model(bench_model, test_loader, device, num_batches=20, quant=args.quant)
        bench_model = bench_model.to("cpu")
        torch.cuda.empty_cache()

    bench_original = benchmarks.get("original")
    bench_unstructured = benchmarks["unstructured"]
    bench_collapsed = benchmarks["collapsed"]

    flops_original = bench_original["flops"] if bench_original is not None else None
    flops_collapsed = bench_collapsed["flops"]
    # IMP zeroes weights but keeps the dense graph: measured FLOPs should ~match the
    # original. flops_unstructured keeps the old theoretical column for back-compat.
    flops_unstructured = (flops_original * (1.0 - float(target_sparsity))
                          if flops_original is not None else None)

    for bench_name, bench in benchmarks.items():
        mem_str = f"{bench['peak_mem_mb']:.1f} MiB" if bench["peak_mem_mb"] is not None else "n/a (CPU)"
        print(
            f"[RESULT] {bench_name.capitalize():<12} "
            f"FLOPs={bench['flops']:.3e} | latency={bench['latency_ms']:.2f} ms/batch | "
            f"throughput={bench['throughput_imgs']:.1f} img/s | peak_mem={mem_str} | "
            f"feature_maps={bench['feature_map_mb']:.1f} MiB"
        )
    if flops_unstructured is not None:
        print(f"[RESULT] FLOPs Unstructured (theoretical/effective, kept for back-compat): {flops_unstructured:.2e}")

    def _state_dict_mb(m):
        sd = m.state_dict()
        if args.quant:
            # PTQ storage: the fp16 artifact is what would be shipped/measured
            sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        buf = io.BytesIO()
        torch.save(sd, buf)
        return buf.getbuffer().nbytes / (1024 ** 2)

    storage_mb = {}
    if original_after_model is not None:
        storage_mb["original"] = _state_dict_mb(original_after_model)
    storage_mb.update({
        "unstructured": _state_dict_mb(unstructured_pruning_model),
        "collapsed": _state_dict_mb(full_collapsed_model_ready),
    })
    if structural_model is not None:
        storage_mb["structural"] = _state_dict_mb(structural_model)
    storage_str = ", ".join(f"{k}={v:.1f}" for k, v in storage_mb.items())
    print(f"[RESULT] Storage (state_dict MiB): {storage_str}")

    # 8. Extract Features & Compute CKA (SEQUENTIAL GPU ALLOCATION)
    cka_unstructured = cka_collapsed = cka_structural = None
    if original_after_model is not None:
        print("\n[INFO] Extracting features and computing CKA...")

        original_after_model = original_after_model.to(device)
        features_original = extract_features(original_after_model, test_loader, device)
        original_after_model = original_after_model.to("cpu")
        torch.cuda.empty_cache()

        # Sanity: CKA(f, f) must be ~1.0; if not, the feature pipeline is broken
        cka_self = compare_CKA(features_original, features_original)
        print(f"[INFO] CKA self-check (expect ~1.0): {cka_self:.6f}")
        assert 0.99 <= cka_self <= 1.01, f"CKA self-check failed: {cka_self}"

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

        if structural_model is not None:
            structural_model = structural_model.to(device)
            features_structural = extract_features(structural_model, test_loader, device)
            structural_model = structural_model.to("cpu")
            torch.cuda.empty_cache()
            cka_structural = compare_CKA(features_original, features_structural)
    else:
        print("\n[INFO] Skipping CKA (no 'original' reference until Control_Continuted finishes).")

    # 9. Evaluate Adversarial Robustness (+ full-analysis extras when enabled)
    print("\n[INFO] Evaluating Adversarial Robustness...")
    data_batch = next(iter(test_loader))

    unstructured_pruning_model = unstructured_pruning_model.to(device)
    full_collapsed_model_ready = full_collapsed_model_ready.to(device)

    models_dict = {}
    if original_after_model is not None:
        original_after_model = original_after_model.to(device)
        models_dict["original"] = original_after_model
    models_dict.update({
        "unstructured": unstructured_pruning_model,
        "collapsed": full_collapsed_model_ready
    })
    if structural_model is not None:
        structural_model = structural_model.to(device)
        models_dict["structural"] = structural_model

    full_analysis = getattr(args, "analysis", "standard") == "full"
    adv_epsilons = (0.005, 0.01, 0.02, 0.03, 0.05) if full_analysis else (0.03,)
    curve_rows = [] if full_analysis else None
    # shortcut: adversarial eval stays fp32 even under --quant — fp16 gradients
    # underflow at eps~0.03 and poison the attack; upgrade if quant adv numbers
    # are ever needed for the paper.
    adv_metrics = evaluate_adversarial_robustness(
        models_dict, data_batch, device, epsilons=adv_epsilons, curve_rows=curve_rows
    )

    analysis_metrics = {}
    calibration_rows = []
    corruption_rows = []
    if full_analysis:
        print("\n[INFO] Full analysis: calibration (ECE/MCE), OOD AUROC, corruptions...")
        for mname, m in models_dict.items():
            ece, mce, bin_rows = evaluate_calibration(m, test_loader, device)
            analysis_metrics[f"ece_{mname}"] = ece
            analysis_metrics[f"mce_{mname}"] = mce
            for b in bin_rows:
                b["model"] = mname
                calibration_rows.append(b)

        ood_loader = load_ood_test_loader(dataset_name)
        if ood_loader is not None:
            for mname, m in models_dict.items():
                analysis_metrics[f"ood_auroc_{mname}"] = evaluate_ood(m, test_loader, ood_loader, device)
                print(f"       -> OOD AUROC {mname}: {analysis_metrics[f'ood_auroc_{mname}']:.4f}")

        corruptions_dir = getattr(args, "corruptions_dir", None)
        if corruptions_dir:
            if not os.path.isdir(corruptions_dir):
                raise FileNotFoundError(
                    f"Corruptions directory not found: {corruptions_dir}\n"
                    f"[HINT] Stage CIFAR-10-C / CIFAR-100-C on the login node with "
                    f"download_corruptions.sh (compute nodes have no internet)."
                )
            mean_corrupt_acc, corruption_rows = evaluate_corruptions(
                models_dict, corruptions_dir, dataset_name, device
            )
            for mname, mca in mean_corrupt_acc.items():
                analysis_metrics[f"corrupt_mca_{mname}"] = mca
                print(f"       -> Mean corruption accuracy {mname}: {mca:.4f}")

        # Long-format tidy CSVs for plotting (accumulate across checkpoints)
        stem = os.path.splitext(os.path.basename(args.output))[0]
        if curve_rows:
            _append_long_csv(f"robustness_curves_{stem}.csv", curve_rows,
                             ["attack", "epsilon", "model", "adv_acc"])
        if calibration_rows:
            _append_long_csv(f"calibration_{stem}.csv", calibration_rows,
                             ["model", "bin_lo", "bin_hi", "n", "acc", "conf"])
        if corruption_rows:
            _append_long_csv(f"corruption_{stem}.csv", corruption_rows,
                             ["model", "corruption", "severity", "acc"])

    # Final cleanup to prevent cross-experiment bleed
    if original_after_model is not None:
        original_after_model = original_after_model.to("cpu")
    unstructured_pruning_model = unstructured_pruning_model.to("cpu")
    full_collapsed_model_ready = full_collapsed_model_ready.to("cpu")
    if structural_model is not None:
        structural_model = structural_model.to("cpu")
    del data_batch, models_dict
    torch.cuda.empty_cache()

    # 10. Compile Results
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    def _bench_field(name, field):
        b = benchmarks.get(name)
        return b[field] if b is not None else None

    result_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "gpu_name": gpu_name,
        "pre_epochs": parse_epoch_tokens(model_before_path)[1],
        "post_epochs": imp_epochs,
        "model_before": model_before_path,
        "intermediate_epoch": ckpt_epoch,
        "acc_original_after": acc_original,
        "acc_unstructured": acc_unstructured,
        "acc_collapsed": acc_collapsed,
        "power_watts_original": round(power_original, 2) if power_original is not None else None,
        "energy_joules_original": round(energy_original, 2) if energy_original is not None else None,
        "power_watts_unstructured": round(power_unstructured, 2),
        "energy_joules_unstructured": round(energy_unstructured, 2),
        "power_watts_collapsed": round(power_collapsed, 2),
        "energy_joules_collapsed": round(energy_collapsed, 2),
        "flops_original": float(flops_original) if flops_original is not None else None,
        "flops_unstructured": float(flops_unstructured) if flops_unstructured is not None else None,
        "flops_collapsed": float(flops_collapsed),
        "flops_unstructured_measured": float(bench_unstructured["flops"]),
        "latency_ms_original": _bench_field("original", "latency_ms"),
        "latency_ms_unstructured": bench_unstructured["latency_ms"],
        "latency_ms_collapsed": bench_collapsed["latency_ms"],
        "throughput_imgs_original": _bench_field("original", "throughput_imgs"),
        "throughput_imgs_unstructured": bench_unstructured["throughput_imgs"],
        "throughput_imgs_collapsed": bench_collapsed["throughput_imgs"],
        "peak_mem_mb_original": _bench_field("original", "peak_mem_mb"),
        "peak_mem_mb_unstructured": bench_unstructured["peak_mem_mb"],
        "peak_mem_mb_collapsed": bench_collapsed["peak_mem_mb"],
        "feature_maps_mb_original": _bench_field("original", "feature_map_mb"),
        "feature_maps_mb_unstructured": bench_unstructured["feature_map_mb"],
        "feature_maps_mb_collapsed": bench_collapsed["feature_map_mb"],
        "storage_mb_original": storage_mb.get("original"),
        "storage_mb_unstructured": storage_mb["unstructured"],
        "storage_mb_collapsed": storage_mb["collapsed"],
        "cka_unstructured": cka_unstructured,
        "cka_collapsed": cka_collapsed,
        # Structural L1 baseline (None/NaN when torch-pruning is unavailable)
        "acc_structural": acc_structural,
        "power_watts_structural": (None if power_structural != power_structural else round(power_structural, 2)),
        "energy_joules_structural": (None if energy_structural != energy_structural else round(energy_structural, 2)),
        "flops_structural": _bench_field("structural", "flops"),
        "latency_ms_structural": _bench_field("structural", "latency_ms"),
        "throughput_imgs_structural": _bench_field("structural", "throughput_imgs"),
        "peak_mem_mb_structural": _bench_field("structural", "peak_mem_mb"),
        "feature_maps_mb_structural": _bench_field("structural", "feature_map_mb"),
        "storage_mb_structural": storage_mb.get("structural"),
        "cka_structural": cka_structural,
        "params_structural": structural_param_count,
        "sparsity_structural": (1.0 - structural_param_count / original_param_count)
                               if structural_param_count is not None else None,
    }
    result_dict.update(adv_metrics)
    result_dict.update(analysis_metrics)
    return result_dict

# =========================================================
# Entry Point
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="CKA, Energy, and Adversarial comparison post-processing")
    parser.add_argument("--model", required=True, help="Model name, e.g. VGG16")
    parser.add_argument("--pre", type=int, default=None, help="pre-collapse (pretrain) epochs; derived from the checkpoint path when omitted")
    parser.add_argument("--post", type=int, default=None, help="post-collapse (finetune) epochs; derived from the checkpoint path when omitted")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--output", default="auto", help="Output CSV filename (set to 'auto' to avoid collisions)")
    parser.add_argument("--power-interval", type=int, default=1, help="Query interval in seconds for power monitoring (default: 1)")
    parser.add_argument("--analysis", choices=["standard", "full"], default="standard",
                        help="'full' adds adversarial epsilon sweeps, calibration (ECE/MCE), OOD AUROC, "
                             "and corruption robustness (with --corruptions-dir). Default 'standard' keeps "
                             "the historical runtime and CSV columns.")
    parser.add_argument("--corruptions-dir", default=None,
                        help="Directory containing CIFAR-10-C / CIFAR-100-C (see download_corruptions.sh). "
                             "Enables corruption evaluation under --analysis full.")
    parser.add_argument("--quant", action="store_true",
                        help="Analyze the fp16-quantized (PTQ) universe: loads the SAME fp32 "
                             "artifacts as the default run and measures them converted to fp16 "
                             "(autocast inference, fp16 storage estimate). Results are written "
                             "to *_quant_*.csv. No retraining is involved — *_quant.pt files "
                             "(legacy QAT) are not read.")
    args = parser.parse_args()

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Discover Checkpoints (epoch config is derived from directory names;
    #    --pre/--post act only as explicit filters)
    print("[INFO] Searching for model checkpoints...")
    checkpoints = find_experiment_checkpoints(args.model, args.dataset, args.pre, args.post, quant=args.quant)
    print(f"[INFO] Found {len(checkpoints['before'])} 'before' checkpoints.")
    print(f"[INFO] Found {len(checkpoints['collapsed'])} collapsed checkpoints.")
    print(f"[INFO] Found {len(checkpoints['after'])} 'after' checkpoints.")

    # Automatically format unique output path if 'auto' or default to prevent overwrite collisions
    quant_tag = "_quant" if args.quant else ""
    if args.output == "auto" or args.output == "cka_comparison_results.csv":
        if args.pre is not None or args.post is not None:
            args.output = f"cka_results_{args.model}_{args.dataset}{quant_tag}_pre{args.pre}_post{args.post}.csv"
        elif checkpoints["before"]:
            post, pre = parse_epoch_tokens(checkpoints["before"][0])
            args.output = f"cka_results_{args.model}_{args.dataset}{quant_tag}_pre{pre}_post{post}.csv"
        else:
            args.output = f"cka_results_{args.model}_{args.dataset}{quant_tag}.csv"

    print("=" * 70)
    print("Starting Comparison Post-Processing Pipeline")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    if args.pre is not None or args.post is not None:
        print(f"Epochs Config (override): pre={args.pre}, post={args.post}")
    else:
        print("Epochs Config: derived per checkpoint directory")
    print(f"Output File: {args.output}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

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