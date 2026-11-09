"""Recompute test-set accuracy for finished runs WITHOUT retraining.

Why: the training loop historically recorded training accuracy as
best_acc/final_accuracy (train_one_epoch measures the train batches in train
mode), so the manuscript tables showed ~99.9% values. This script walks run
directories, evaluates each saved whole-model checkpoint on the test set with
trainer.evaluate(), and backfills the corrected values into
metrics/merged_metrics.json (original saved once as merged_metrics.json.bak).
Old train-based values are preserved as train_best_acc/train_final_accuracy.

Note: only the last per-epoch checkpoint is kept by the training loop, so for
already-finished runs this evaluates the final model (a single accuracy).
Future runs record per-epoch test accuracy via run_experiment.

Usage:
    python recompute_test_acc.py [base_path] [--device cuda]

where base_path contains the run directories (default: current directory).
"""

import argparse
import glob
import json
import os
import shutil

import torch


def dataset_of(folder_name):
    if "_Cifar100_" in folder_name:
        return "Cifar100"
    if "_Cifar10_" in folder_name:
        return "Cifar10"
    if "_tinyimagenet_" in folder_name:
        return "tinyimagenet"
    return None


def arch_of(folder_name):
    return "RegNetX_400MF" if folder_name.startswith("RegNetX_400MF") else folder_name.split("_")[0]


def metrics_key_for(stem, metrics):
    """Map a checkpoint stem to its metrics key.

    Checkpoints are named 'final_{workflow}_{exp_name}[_quant]_whole.pt' while
    metrics keys are '{exp_name}[_quant]_{workflow}' (experiments.py); try all
    underscore splits and keep the one that exists in the metrics dict.
    """
    parts = stem.split("_")
    for i in range(1, len(parts)):
        key = f"{'_'.join(parts[i:])}_{'_'.join(parts[:i])}"
        if key in metrics:
            return key
    return None


def fix_legacy_units(metrics):
    """auto_recover_metrics used to write a 0-1 fraction; everything else is percent."""
    changed = False
    for rec in metrics.values():
        if not isinstance(rec, dict):
            continue
        if rec.get("timestamp_recovered") == "auto_recovered_live" and 0 < rec.get("final_accuracy", 0) <= 1.0:
            rec["final_accuracy"] *= 100.0
            changed = True
    return changed


def recompute_run(metrics_path, device):
    from trainer import evaluate
    from utils import load_dataset

    run_dir = os.path.dirname(os.path.dirname(metrics_path))
    folder = os.path.basename(run_dir)
    dataset = dataset_of(folder)
    if dataset is None:
        print(f"[!] {folder}: cannot infer dataset from folder name, skipping")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    changed = fix_legacy_units(metrics)
    test_loader = None

    for ckpt_path in sorted(glob.glob(os.path.join(run_dir, "checkpoints", "final_*_whole.pt"))):
        stem = os.path.basename(ckpt_path)[len("final_"):-len("_whole.pt")]
        key = metrics_key_for(stem, metrics)
        if key is None:
            print(f"[!] {folder}: no metrics entry for checkpoint '{stem}', skipping")
            continue

        if test_loader is None:
            print(f"[•] {folder}: loading {dataset} test set...")
            test_loader = load_dataset(dataset, arch_of(folder))[1]

        print(f"[•] {folder}/{stem}: evaluating on the {dataset} test set...")
        model = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.to(device)
        acc = evaluate(model, test_loader, device, quant=stem.endswith("_quant"))
        del model

        rec = metrics[key]
        rec.setdefault("train_best_acc", rec.get("best_acc", 0.0))
        rec.setdefault("train_final_accuracy", rec.get("final_accuracy", 0.0))
        rec["best_acc"] = acc
        rec["final_accuracy"] = acc
        rec["test_accuracy"] = acc
        rec["test_eval_source"] = "recomputed_from_final_checkpoint"
        changed = True
        print(f"    {key}: test acc {acc:.2f}% (train-based value kept as train_best_acc)")

    if changed:
        backup = metrics_path + ".bak"
        if not os.path.exists(backup):
            shutil.copy2(metrics_path, backup)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"[✓] {metrics_path} updated (original: {os.path.basename(backup)})")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("base_path", nargs="?", default=".", help="directory containing the run folders")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    runs = sorted(glob.glob(os.path.join(args.base_path, "*", "metrics", "merged_metrics.json")))
    if not runs:
        print(f"[!] No */metrics/merged_metrics.json found under {args.base_path}")
        return

    for metrics_path in runs:
        recompute_run(metrics_path, device)


if __name__ == "__main__":
    main()
