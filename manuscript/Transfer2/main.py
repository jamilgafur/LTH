import os
import glob
import csv
import io
import json
import time
import traceback
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
from pyPrune.power import PowerTracker

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


def _load_region_map_keys(args, model_before_path):
    """Key set of the combo's discovery map (legacy name first, then
    scorerlocal_dev-suffixed ablation map — temp2_run.sh's lookup order).
    Returns None when no map file exists."""
    post, pre = parse_epoch_tokens(model_before_path)
    if post is None or pre is None:
        return None
    stem = os.path.join("..", "Tranfer",
                        f"{args.model}_{args.dataset}_epochs{post}_pretrain{pre}")
    for suffix in ("_JF_discovered_regions.json", "_scorerlocal_dev_JF_discovered_regions.json"):
        path = stem + suffix
        if os.path.exists(path):
            try:
                with open(path) as f:
                    return set(json.load(f))
            except Exception:
                return None  # unreadable map; behave as if absent
    return None


def no_collapsed_artifact_planned(args, model_before_path):
    """True when the combo's discovery map contains no collapsible regions at
    all (no Dynamic_Region_All_Combined AND no Set_* spans): discovery found
    nothing to collapse, so no collapsed artifact can ever exist for this
    combo. Single-set maps (Set_0 only) are NOT verdicts — their collapse is
    merely untrained (temp2 subset's Set_0 stand-in)."""
    keys = _load_region_map_keys(args, model_before_path)
    if keys is None:
        return False  # no map at all: discovery has not run yet, artifact may still be coming
    return "Dynamic_Region_All_Combined" not in keys and not any(k.startswith("Set_") for k in keys)


def no_regions_row(args, model_before_path):
    """Verdict row for a combo whose discovery found no collapsible regions."""
    post, pre = parse_epoch_tokens(model_before_path)
    return {
        "model": args.model,
        "dataset": args.dataset,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "pre_epochs": pre,
        "post_epochs": post,
        "model_before": model_before_path,
        "intermediate_epoch": 0,
        "status": "no_collapsed_regions",
    }


def process_checkpoint(model_before_path, checkpoints, args, device):
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print(f"[INFO] Parsing experiment directory... {model_before_path}")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path, args.model, args.dataset)
    
    print("[INFO] Initializing baseline model...")
    base_model, train_loader, test_loader, _ = initialize_architecture(model_name, dataset_name)
    model_before = load_weights(base_model, model_before_path, device)
    recalibrate_bn(model_before, train_loader, device)

    # 2. Load Original Post-Finetuning Model (optional). When Control_Continuted is
    #    still training, run a baseline-only pass: sparsity/IMP/structural and the
    #    collapsed-variant metrics don't need it; after-columns come out None and a
    #    later rerun (IMP/structural artifacts persist on disk) fills them in.
    print("[INFO] Loading original post-finetuning model...")
    try:
        model_after_path = find_matching_checkpoint(model_before_path, checkpoints["after"])
        original_after_base, _, _, _ = initialize_architecture(model_name, dataset_name)
        original_after_model = load_weights(original_after_base, model_after_path, device)
        recalibrate_bn(original_after_model, train_loader, device)
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
        keys = _load_region_map_keys(args, model_before_path)
        sets = sorted(k for k in (keys or ()) if k.startswith("Set_"))
        if sets and "Dynamic_Region_All_Combined" not in (keys or ()):
            hint = (f"This combo's regions map has no Dynamic_Region_All_Combined key; "
                    f"train the single span {sets[0]} (temp2_run.sh subset submits it "
                    f"automatically) to produce `final_JF_{sets[0]}_whole.pt`.")
        else:
            hint = "Re-run the Tranfer experiments to generate `final_*_whole.pt` artifacts."
        raise FileNotFoundError(
            f"{e}\n[HINT] No `*_whole.pt` collapsed artifact found for this experiment. {hint}"
        ) from e

    load_start_time = time.time()
    full_collapsed_model_ready = torch.load(collapsed_model_path, map_location=device, weights_only=False)
    full_collapsed_model_ready.to(device)
    full_collapsed_model_ready.eval()
    print(f"[INFO] Whole-model load completed in {time.time() - load_start_time:.2f}s")
    recalibrate_bn(full_collapsed_model_ready, train_loader, device)

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

    # 5b. Structural (L1 filter pruning) comparator retired: its artifacts measured
    #     at chance on both splits across configs (comparator training never learned),
    #     and the study design dropped the comparator entirely.
    structural_model = None
    structural_param_count = None

    # 6-7. Measure accuracy/power/energy + benchmark efficiency, `args.reps` times.
    # Acc/CKA/adversarial are deterministic; the reps give the noisy power/latency
    # numbers enough depth for the plot-side median to reject bad samples.
    reps = max(1, int(args.reps))
    print(f"\n[INFO] Evaluating model accuracy, power draw, and energy consumption ({reps} rep(s))...")

    def _eval_one(model, label):
        print(f"       -> Evaluating {label}")
        model = model.to(device)
        res = evaluate_accuracy(model, test_loader, device, power_interval=args.power_interval, quant=args.quant, acc_loader=train_loader)
        model = model.to("cpu")
        torch.cuda.empty_cache()
        return res

    bench_variants = []
    if original_after_model is not None:
        bench_variants.append(("original", original_after_model))
    bench_variants += [
        ("unstructured", unstructured_pruning_model),
        ("collapsed", full_collapsed_model_ready),
    ]
    if structural_model is not None:
        bench_variants.append(("structural", structural_model))

    eval_reps, bench_reps = [], []
    for rep in range(reps):
        if reps > 1:
            print(f"\n[INFO] === Measurement rep {rep + 1}/{reps} ===")

        m = {"before": _eval_one(model_before, "Baseline")}
        if original_after_model is not None:
            m["original"] = _eval_one(original_after_model, "Original After Finetuning")
            if rep == 0 and m["before"][0] > (m["original"][0] + 0.20):
                print(f"\n[CRITICAL WARNING] 'original_after_model' suffered catastrophic forgetting ({m['original'][0]*100:.2f}%).")
                print(f"[CRITICAL WARNING] Re-evaluating healthy 'model_before' ({m['before'][0]*100:.2f}%) on warm cache as the true baseline...")
                m["original"] = _eval_one(model_before, "healthy 'model_before' (fallback baseline)")
        else:
            m["original"] = (None,) * 8
        m["unstructured"] = _eval_one(unstructured_pruning_model, "Unstructured Pruned Model")
        m["collapsed"] = _eval_one(full_collapsed_model_ready, "Collapsed Model")
        if structural_model is not None:
            m["structural"] = _eval_one(structural_model, "Structural L1 Pruned Model")
        else:
            m["structural"] = (float("nan"),) * 4 + (None,) * 4
        eval_reps.append(m)

        b = {}
        for bench_name, bench_model in bench_variants:
            bench_model = bench_model.to(device)
            b[bench_name] = benchmark_model(bench_model, test_loader, device, num_batches=20, quant=args.quant,
                                            measure_energy=True, power_interval=args.power_interval)
            bench_model = bench_model.to("cpu")
            torch.cuda.empty_cache()
        bench_reps.append(b)

    # Data-pipeline floor, once per config: the time/joules every variant's eval
    # window shares. Loader state is warm by now (evals already ran).
    loader_floor = measure_loader_floor(test_loader, device, power_interval=args.power_interval)

    # Console summary from rep 0 (per-rep numbers land as rows in the CSV)
    m0, b0 = eval_reps[0], bench_reps[0]
    acc_original, att_original, power_original, energy_original = m0["original"][:4]
    acc_unstructured, att_unstructured, power_unstructured, energy_unstructured = m0["unstructured"][:4]
    acc_collapsed, att_collapsed, power_collapsed, energy_collapsed = m0["collapsed"][:4]
    acc_structural, att_structural, power_structural, energy_structural = m0["structural"][:4]

    def _fmt_power(value, unit):
        return f"{value:.2f}{unit}" if value is not None else "N/A"

    def _fmt_acc(train_acc, test_acc):
        return f"{train_acc:.4f} (test {test_acc:.4f})" if test_acc is not None else f"{train_acc:.4f}"

    if acc_original is not None:
        print(f"[RESULT] Accuracy Original:     {_fmt_acc(acc_original, att_original)} | Power: {_fmt_power(power_original, 'W')} | Energy: {_fmt_power(energy_original, 'J')}")
    print(f"[RESULT] Accuracy Before (anchor): {_fmt_acc(m0['before'][0], m0['before'][1]) if m0['before'][0] is not None else 'N/A'}")
    print(f"[RESULT] Accuracy Unstructured: {_fmt_acc(acc_unstructured, att_unstructured)} | Power: {_fmt_power(power_unstructured, 'W')} | Energy: {_fmt_power(energy_unstructured, 'J')}")
    print(f"[RESULT] Accuracy Collapsed:    {_fmt_acc(acc_collapsed, att_collapsed)} | Power: {_fmt_power(power_collapsed, 'W')} | Energy: {_fmt_power(energy_collapsed, 'J')}")
    if structural_model is not None:
        print(f"[RESULT] Accuracy Structural:   {_fmt_acc(acc_structural, att_structural)} | Power: {_fmt_power(power_structural, 'W')} | Energy: {_fmt_power(energy_structural, 'J')}")
    if reps > 1:
        print(f"[INFO] {reps} measurement reps completed; full per-rep values are rows in the CSV.")

    # FLOPs/latency/throughput/memory prints from rep 0 (deterministic parts)
    bench_original = b0.get("original")
    flops_original = bench_original["flops"] if bench_original is not None else None
    flops_collapsed = b0["collapsed"]["flops"]
    # IMP zeroes weights but keeps the dense graph: measured FLOPs should ~match the
    # original. flops_unstructured keeps the old theoretical column for back-compat.
    flops_unstructured = (flops_original * (1.0 - float(target_sparsity))
                          if flops_original is not None else None)

    for bench_name, bench in b0.items():
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

        ood_loader, ood_source = load_ood_test_loader(dataset_name)
        if ood_loader is not None:
            analysis_metrics["ood_source"] = ood_source
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

    def _bench_field(b, name, field):
        v = b.get(name)
        return v[field] if v is not None else None

    def _bench_opt(b, name, field):
        v = b.get(name)
        return v.get(field) if v is not None else None

    result_rows = []
    for rep, (m, b) in enumerate(zip(eval_reps, bench_reps)):
        acc_b, att_b = m["before"][0], m["before"][1]
        acc_o, att_o, pow_o, eng_o, util_o, clk_o, ela_o, npw_o = m["original"]
        acc_u, att_u, pow_u, eng_u, util_u, clk_u, ela_u, npw_u = m["unstructured"]
        acc_c, att_c, pow_c, eng_c, util_c, clk_c, ela_c, npw_c = m["collapsed"]
        acc_s, att_s, pow_s, eng_s, util_s, clk_s, ela_s, npw_s = m["structural"]

        result_dict = {
            "model": model_name,
            "dataset": dataset_name,
            "gpu_name": gpu_name,
            "pre_epochs": parse_epoch_tokens(model_before_path)[1],
            "post_epochs": imp_epochs,
            "model_before": model_before_path,
            "intermediate_epoch": ckpt_epoch,
            "rep": rep,
            "acc_before": acc_b,
            "acc_test_before": att_b,
            "acc_original_after": acc_o,
            "acc_test_original_after": att_o,
            "acc_unstructured": acc_u,
            "acc_test_unstructured": att_u,
            "acc_collapsed": acc_c,
            "acc_test_collapsed": att_c,
            "power_watts_original": round(pow_o, 2) if pow_o is not None else None,
            "energy_joules_original": round(eng_o, 2) if eng_o is not None else None,
            "power_watts_unstructured": round(pow_u, 2) if pow_u is not None else None,
            "energy_joules_unstructured": round(eng_u, 2) if eng_u is not None else None,
            "power_watts_collapsed": round(pow_c, 2) if pow_c is not None else None,
            "energy_joules_collapsed": round(eng_c, 2) if eng_c is not None else None,
            "flops_original": float(b["original"]["flops"]) if b.get("original") is not None else None,
            "flops_unstructured": float(flops_unstructured) if flops_unstructured is not None else None,
            "flops_collapsed": float(b["collapsed"]["flops"]),
            "flops_unstructured_measured": float(b["unstructured"]["flops"]),
            "latency_ms_original": _bench_field(b, "original", "latency_ms"),
            "latency_ms_unstructured": b["unstructured"]["latency_ms"],
            "latency_ms_collapsed": b["collapsed"]["latency_ms"],
            "throughput_imgs_original": _bench_field(b, "original", "throughput_imgs"),
            "throughput_imgs_unstructured": b["unstructured"]["throughput_imgs"],
            "throughput_imgs_collapsed": b["collapsed"]["throughput_imgs"],
            "peak_mem_mb_original": _bench_field(b, "original", "peak_mem_mb"),
            "peak_mem_mb_unstructured": b["unstructured"]["peak_mem_mb"],
            "peak_mem_mb_collapsed": b["collapsed"]["peak_mem_mb"],
            "feature_maps_mb_original": _bench_field(b, "original", "feature_map_mb"),
            "feature_maps_mb_unstructured": b["unstructured"]["feature_map_mb"],
            "feature_maps_mb_collapsed": b["collapsed"]["feature_map_mb"],
            "storage_mb_original": storage_mb.get("original"),
            "storage_mb_unstructured": storage_mb["unstructured"],
            "storage_mb_collapsed": storage_mb["collapsed"],
            "cka_unstructured": cka_unstructured,
            "cka_collapsed": cka_collapsed,
            # Structural L1 baseline (None/NaN when torch-pruning is unavailable)
            "acc_structural": acc_s,
            "acc_test_structural": att_s,
            "power_watts_structural": (None if pow_s is None or pow_s != pow_s else round(pow_s, 2)),
            "energy_joules_structural": (None if eng_s is None or eng_s != eng_s else round(eng_s, 2)),
            "flops_structural": _bench_field(b, "structural", "flops"),
            "latency_ms_structural": _bench_field(b, "structural", "latency_ms"),
            "throughput_imgs_structural": _bench_field(b, "structural", "throughput_imgs"),
            "peak_mem_mb_structural": _bench_field(b, "structural", "peak_mem_mb"),
            "feature_maps_mb_structural": _bench_field(b, "structural", "feature_map_mb"),
            "storage_mb_structural": storage_mb.get("structural"),
            "cka_structural": cka_structural,
            "params_structural": structural_param_count,
            "sparsity_structural": (1.0 - structural_param_count / original_param_count)
                                   if structural_param_count is not None else None,
            # SM utilization / clocks during the tracked eval — diagnostic for the
            # "collapsed draws more power" effect; None on CPU.
            "gpu_util_original": util_o,
            "gpu_clock_mhz_original": clk_o,
            "gpu_util_unstructured": util_u,
            "gpu_clock_mhz_unstructured": clk_u,
            "gpu_util_collapsed": util_c,
            "gpu_clock_mhz_collapsed": clk_c,
            "gpu_util_structural": util_s,
            "gpu_clock_mhz_structural": clk_s,
            # Window audit columns: eval duration + power-sample depth per variant,
            # so every energy row carries the evidence of what it measured.
            "eval_elapsed_s_original": round(ela_o, 3) if ela_o is not None else None,
            "eval_elapsed_s_unstructured": round(ela_u, 3) if ela_u is not None else None,
            "eval_elapsed_s_collapsed": round(ela_c, 3) if ela_c is not None else None,
            "eval_elapsed_s_structural": round(ela_s, 3) if ela_s is not None else None,
            "n_power_samples_original": npw_o,
            "n_power_samples_unstructured": npw_u,
            "n_power_samples_collapsed": npw_c,
            "n_power_samples_structural": npw_s,
            # Data-pipeline floor (config-level; identical across reps/variants).
            "loader_time_s": round(loader_floor["loader_time_s"], 3),
            "loader_joules": (round(loader_floor["loader_joules"], 2)
                              if loader_floor["loader_joules"] is not None else None),
            "loader_n_samples": loader_floor["n_power_samples"],
            # Saturated compute-only energy: J/img with zero data pipeline in the
            # window — the FLOPs->joules transferability claim's numerator.
            "power_watts_saturated_original": _bench_opt(b, "original", "power_watts_saturated"),
            "power_watts_saturated_unstructured": _bench_opt(b, "unstructured", "power_watts_saturated"),
            "power_watts_saturated_collapsed": _bench_opt(b, "collapsed", "power_watts_saturated"),
            "power_watts_saturated_structural": _bench_opt(b, "structural", "power_watts_saturated"),
            "j_per_img_saturated_original": _bench_opt(b, "original", "j_per_img_saturated"),
            "j_per_img_saturated_unstructured": _bench_opt(b, "unstructured", "j_per_img_saturated"),
            "j_per_img_saturated_collapsed": _bench_opt(b, "collapsed", "j_per_img_saturated"),
            "j_per_img_saturated_structural": _bench_opt(b, "structural", "j_per_img_saturated"),
            "saturated_imgs_original": _bench_opt(b, "original", "saturated_imgs"),
            "saturated_imgs_unstructured": _bench_opt(b, "unstructured", "saturated_imgs"),
            "saturated_imgs_collapsed": _bench_opt(b, "collapsed", "saturated_imgs"),
            "saturated_imgs_structural": _bench_opt(b, "structural", "saturated_imgs"),
            "n_power_samples_saturated_original": _bench_opt(b, "original", "n_power_samples_saturated"),
            "n_power_samples_saturated_unstructured": _bench_opt(b, "unstructured", "n_power_samples_saturated"),
            "n_power_samples_saturated_collapsed": _bench_opt(b, "collapsed", "n_power_samples_saturated"),
            "n_power_samples_saturated_structural": _bench_opt(b, "structural", "n_power_samples_saturated"),
        }
        result_dict.update(adv_metrics)
        result_dict.update(analysis_metrics)
        result_rows.append(result_dict)
    return result_rows

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
    parser.add_argument("--power-interval", type=float, default=1, help="Query interval in seconds for power monitoring (default: 1; sub-second values down to 0.1 supported on NVIDIA)")
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
    parser.add_argument("--reps", type=int, default=1,
                        help="Measurement repetitions for the accuracy/power/energy eval and the "
                             "efficiency benchmark (sections 6-7). The CSV gets one row per rep "
                             "(see the 'rep' column); deterministic steps (CKA, adversarial, "
                             "calibration) run once. Use ~10 so plot-side medians can reject "
                             "noisy power samples. Default 1 keeps historical behavior.")
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
            result_rows = process_checkpoint(model_before_path, checkpoints, args, device)
            results.extend(result_rows)
        except FileNotFoundError as e:
            if no_collapsed_artifact_planned(args, model_before_path):
                # Discovery found no collapsible regions for this combo: record an
                # explicit verdict row so the grid reads "no regions" (and
                # run_postprocess.sh stops resubmitting it) instead of an empty CSV.
                print(f"[INFO] No collapsible regions for this combo — writing status row. ({e})")
                results.append(no_regions_row(args, model_before_path))
            else:
                print(f"[ERROR] Failed processing checkpoint: {model_before_path}")
                print(f"[ERROR] {type(e).__name__}: {e}")
                traceback.print_exc()
        except Exception as e:
            print(f"[ERROR] Failed processing checkpoint: {model_before_path}")
            print(f"[ERROR] {type(e).__name__}: {e}")
            traceback.print_exc()

    # 3. Save Results — never overwrite an existing CSV with an empty frame; a
    #    fully-failed run leaves a previous good file intact (run_postprocess.sh
    #    still resubmits 1-byte placeholders/missing files).
    print("\n" + "=" * 70)
    print("[INFO] Saving final results...")
    if not results:
        print(f"[ERROR] 0 successful experiments — NOT overwriting '{args.output}'. "
              f"Fix the errors above and rerun.")
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(results)
        df.to_csv(args.output, index=False)
        print(f"[INFO] Results successfully saved to '{args.output}'")
    print(f"[INFO] Total successful experiments: {len(results)}")
    print(f"[INFO] Total runtime: {time.time() - total_start_time:.2f}s")
    
    print("\n" + "=" * 70 + "\nFINAL RESULTS\n" + "=" * 70)
    print(df.to_string())

if __name__ == "__main__":
    main()