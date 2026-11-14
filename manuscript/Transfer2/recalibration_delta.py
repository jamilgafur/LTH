#!/usr/bin/env python3
"""Recalibration-delta exhibit (dissertation measurement-foundation chapter).

Measures every artifact of a config BEFORE and AFTER batch-normalization
running-statistics recalibration — same evaluator as the main grid
(`evaluate_accuracy`, same splits, same footing) — one row per artifact.
This is the standing evidence that recalibration changes measured accuracy
without touching weights, and by how much per artifact.

Artifacts: `control_before` (pre-continuation anchor), `control_continued`,
`collapsed`, `imp`. Missing artifacts are skipped.

Usage (one GPU):
    python recalibration_delta.py --model VGG16 --dataset tinyimagenet --pre 300 --post 100
Rows append to recalibration_delta.csv (shared across configs).
"""
import argparse
import csv
import os

import torch

from utils import (evaluate_accuracy, find_experiment_checkpoints,
                   find_matching_checkpoint, inherit_from_default_universe,
                   initialize_architecture, load_weights, parse_epoch_tokens,
                   recalibrate_bn)

FIELDS = ["model", "dataset", "quant", "pre_epochs", "post_epochs", "artifact",
          "acc_train_pre", "acc_test_pre", "acc_train_post", "acc_test_post",
          "d_train_pp", "d_test_pp", "path"]


def measure(model, train_loader, test_loader, device, quant):
    """(train, test) accuracy via the main grid's evaluator; res[0]/res[1]."""
    model = model.to(device)
    res = evaluate_accuracy(model, test_loader, device, quant=quant, acc_loader=train_loader)
    model.to("cpu")
    torch.cuda.empty_cache()
    acc_tr, acc_te = res[0], res[1]
    assert 0.0 <= acc_tr <= 1.0 and 0.0 <= acc_te <= 1.0, f"accuracy out of range: {acc_tr}, {acc_te}"
    return acc_tr, acc_te


def process(before_path, checkpoints, args, device):
    model_name, dataset_name, *_ = parse_directory_context(before_path, args.model, args.dataset)
    _, train_loader, test_loader, _ = initialize_architecture(model_name, dataset_name)
    post, pre = parse_epoch_tokens(before_path)

    artifacts = [("control_before", before_path, "state")]
    for name, key, kind in [("control_continued", "after", "state"),
                            ("collapsed", "collapsed", "whole")]:
        try:
            artifacts.append((name, find_matching_checkpoint(before_path, checkpoints[key]), kind))
        except FileNotFoundError:
            pass

    checkpoint_dir = os.path.dirname(before_path)
    if "_scorerall" in checkpoint_dir and os.path.isdir(checkpoint_dir.replace("_scorerall", "")):
        checkpoint_dir = checkpoint_dir.replace("_scorerall", "")
    imp_epochs = args.post if args.post is not None else post
    imp_path = os.path.join(checkpoint_dir, f"final_JF_Unstructured_IMP_post{imp_epochs}.pt")
    inherit_from_default_universe(imp_path)
    if os.path.exists(imp_path):
        artifacts.append(("imp", imp_path, "state"))
    # shortcut: imp delta measures only a saved artifact (no on-the-fly IMP
    # training here), run main.py once for this config first to produce it

    rows = []
    for name, path, kind in artifacts:
        if kind == "whole":
            model = torch.load(path, map_location=device, weights_only=False)
            model.to(device)
            model.eval()
        else:
            base, _, _, _ = initialize_architecture(model_name, dataset_name)
            model = load_weights(base, path, device)

        tr_pre, te_pre = measure(model, train_loader, test_loader, device, args.quant)
        recalibrate_bn(model, train_loader, device)
        tr_post, te_post = measure(model, train_loader, test_loader, device, args.quant)

        row = {"model": model_name, "dataset": dataset_name, "quant": args.quant,
               "pre_epochs": pre, "post_epochs": post, "artifact": name,
               "acc_train_pre": tr_pre, "acc_test_pre": te_pre,
               "acc_train_post": tr_post, "acc_test_post": te_post,
               "d_train_pp": (tr_post - tr_pre) * 100, "d_test_pp": (te_post - te_pre) * 100,
               "path": path}
        print(f"[DELTA] {name:17s} train {tr_pre:.4f} -> {tr_post:.4f} ({row['d_train_pp']:+.2f} pp) | "
              f"test {te_pre:.4f} -> {te_post:.4f} ({row['d_test_pp']:+.2f} pp)")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="BN recalibration-delta exhibit: same artifacts, measured pre/post recalibration")
    parser.add_argument("--model", required=True, help="Model name, e.g. VGG16")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--pre", type=int, default=None, help="pre-collapse (pretrain) epochs; derived from the checkpoint path when omitted")
    parser.add_argument("--post", type=int, default=None, help="post-collapse (finetune) epochs; derived from the checkpoint path when omitted")
    parser.add_argument("--quant", action="store_true", help="measure the fp16 universe (fp32 artifacts under autocast), matching the *_quant_* grid rows")
    parser.add_argument("--out", default="recalibration_delta.csv", help="output CSV; rows append across invocations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = find_experiment_checkpoints(args.model, args.dataset, args.pre, args.post, quant=args.quant)
    if not checkpoints["before"]:
        raise SystemExit(f"No 'before' checkpoints for {args.model}/{args.dataset} (pre={args.pre}, post={args.post}).")

    rows = []
    for before_path in checkpoints["before"]:
        print(f"\n[INFO] {before_path}")
        rows.extend(process(before_path, checkpoints, args, device))

    write_header = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} artifact rows appended to {args.out}")


if __name__ == "__main__":
    main()
