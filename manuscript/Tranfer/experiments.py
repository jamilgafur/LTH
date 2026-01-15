# experiments.py (updated)

# Standard libraries
import os
import glob
import json
import time
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict
import tempfile

# Third-party libraries
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fvcore.nn import FlopCountAnalysis


# Local modules
from pyPrune.models.Vgg16 import VGG16
from pyPrune.utils import *
from plots import *
from diagnostic import *
from utils import *
from filemanager import *
from collapse import collapse_only
from trainer import train_and_evaluate
import os
import glob
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# -------------------------
# Safe JSON Write (Per-job unique, no lock, no SLURM ID)
# -------------------------
def safe_update_metrics_json(model_root, exp_name, new_data, base_dir="./runs/metrics"):
    """
    Writes metrics to a per-job JSON file with a unique timestamp.
    Fully preserves raw NumPy arrays by converting them to lists.
    """
    ensure_dir(base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    json_path = os.path.join(
        base_dir, f"{model_root}_metrics_{timestamp}_{pid}.json"
    )

    try:
        safe_data = convert_ndarrays_to_lists(new_data)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=base_dir, prefix="tmp_metrics_", suffix=".json"
        )

        with os.fdopen(tmp_fd, "w") as f:
            json.dump({exp_name: safe_data}, f, indent=4)

        # Atomic replace (prevents corrupted files)
        os.replace(tmp_path, json_path)

        print(f"[✓] Saved metrics for '{exp_name}' → {json_path}")
        return json_path

    except Exception as e:
        print(f"[!] Failed to save metrics JSON: {e}")
        return None


def convert_ndarrays_to_lists(obj):
    """
    Recursively convert NumPy arrays to Python lists so JSON can serialize them.
    """
    if isinstance(obj, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [convert_ndarrays_to_lists(v) for v in obj]

    if isinstance(obj, tuple):
        return [convert_ndarrays_to_lists(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    return obj
# -------------------------
# Merge All Metrics (Hybrid mode)
# -------------------------
def merge_all_metrics(base_dir="./runs/metrics", merged_name="merged_metrics.json"):
    """
    Safely merges all metrics JSON files into one consolidated file.
    Uses temp files and avoids concurrent write collisions.
    """
    ensure_dir(base_dir)
    json_files = glob.glob(os.path.join(base_dir, "*_metrics_*.json"))
    merged_data = {}

    for jf in json_files:
        try:
            if os.path.getsize(jf) == 0:
                print(f"[!] Skipping empty file: {jf}")
                continue
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_data.update(data)
        except Exception as e:
            print(f"[!] Skipping {jf}: {e}")

    # Write to a temp file first
    tmp_fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix="tmp_merge_", suffix=".json")
    with os.fdopen(tmp_fd, "w") as tmp_file:
        json.dump(merged_data, tmp_file, indent=4)

    merged_path = os.path.join(base_dir, merged_name)

    # Atomic replace
    os.replace(tmp_path, merged_path)

    print(f"[✓] Merged {len(json_files)} metrics files → {merged_path}")
    return merged_path


# -------------------------
# Helper: normalize collapse_range -> list of 2-tuples
# -------------------------
def _make_compression_set(collapse_range):
    """
    Normalize collapse_range into a flat list of (start_name, end_name) tuples.
    Acceptable inputs:
      - None
      - ("a","b")
      - ["a","b"]  (2-element list)
      - [("a","b"), ("c","d")]
    Returns None when collapse_range is falsy.
    """
    if not collapse_range:
        return None

    # Single pair (tuple or 2-element list of strings)
    if isinstance(collapse_range, (tuple, list)) and len(collapse_range) == 2 and all(isinstance(x, str) for x in collapse_range):
        return [(collapse_range[0], collapse_range[1])]

    # Already a list of pairs
    if isinstance(collapse_range, list):
        compression_set = []
        for idx, item in enumerate(collapse_range):
            if not (isinstance(item, (tuple, list)) and len(item) == 2):
                raise ValueError(f"collapse_range list element #{idx} must be a 2-tuple/list of strings, got: {item!r}")
            if not all(isinstance(x, str) for x in item):
                raise ValueError(f"collapse_range list element #{idx} must contain strings, got: {item!r}")
            compression_set.append((item[0], item[1]))
        return compression_set

    raise ValueError("collapse_range must be None, a 2-tuple, or a list of 2-tuples")


# -------------------------
# Modified run_experiment (calls merge_all_metrics at the end)
# -------------------------

def run_experiment(
    model,
    model_kwargs=None,
    train_loader=None,
    test_loader=None,
    device="cuda",
    epochs=10,
    workflow="default",
    exp_name="experiment",
    collapse_range=None,
    data_shape=(1, 3, 32, 32),
    save_path="./runs",
    post_compress_epochs=False,
    quant=False,
):
    import os, glob, torch

    if quant:
        exp_name += "_quant"

    print(f"[•] Starting experiment '{exp_name}' in workflow '{workflow}'")

    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")

    ensure_dir(ckpt_dir)
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[!] Warning: CUDA not available.")
        quit()

    model.to(device)

    # ----------------------------------------
    # Checkpoint discovery
    # ----------------------------------------
    base_ckpt_name = f"{workflow}_{exp_name}"
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")

    existing_ckpts = sorted(
        glob.glob(ckpt_pattern),
        key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0]),
    )

    start_epoch = 0
    all_data = {"accuracies": [], "losses": []}

    # ----------------------------------------
    # Resume logic (FULL MODEL LOAD)
    # ----------------------------------------
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        ckpt = torch.load(last_ckpt, map_location=device)

        model = ckpt["model"].to(device)
        start_epoch = ckpt["epoch"]
        all_data = ckpt["data"]

        print(f"[✓] Resuming from checkpoint: epoch {start_epoch}")
    else:
        print("[•] No checkpoint found — starting fresh")

    # ----------------------------------------
    # Training loop
    # ----------------------------------------
    for epoch in range(start_epoch + 1, epochs + 1):
        print(f"[•] Epoch {epoch}/{epochs}")

        data = train_and_evaluate(
            model,
            train_loader,
            test_loader,
            device,
            epochs=1,
            post_compress_epochs=post_compress_epochs,
            quant=quant,
        )

        all_data["accuracies"].extend(data.get("accuracies", []))
        all_data["losses"].extend(data.get("losses", []))

        ckpt_path = os.path.join(
            ckpt_dir, f"{base_ckpt_name}_epoch{epoch}.pt"
        )
        tmp_path = ckpt_path + ".tmp"

        torch.save(
            {
                "epoch": epoch,
                "model": model,  # FULL MODEL OBJECT
                "data": all_data,
            },
            tmp_path,
        )
        os.replace(tmp_path, ckpt_path)

        print(f"[✓] Checkpoint saved → {ckpt_path}")

        if epoch > 1:
            old_ckpt = os.path.join(
                ckpt_dir, f"{base_ckpt_name}_epoch{epoch - 1}.pt"
            )
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
                print(f"[•] Removed old checkpoint → {old_ckpt}")

    data = all_data

    # ----------------------------------------
    # Final checkpoint
    # ----------------------------------------
    final_path = os.path.join(ckpt_dir, f"final_{base_ckpt_name}.pt")
    torch.save(
        {
            "model": model,
            "data": data,
        },
        final_path,
    )

    # ----------------------------------------
    # Metrics & diagnostics (unchanged)
    # ----------------------------------------
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(
        model, test_loader, device, quant=quant
    )

    data.update(
        {
            "param_count": param_count,
            "inference_time": infer_time,
            "flops": flops,
            "total_size_mb": total_size_mb,
            "final_accuracy": data.get("accuracies", [0])[-1]
            if data.get("accuracies")
            else 0,
        }
    )

    diagnostics = run_full_diagnostics(
        model,
        data_shape,
        {exp_name: data},
        plots_dir,
        exp_name,
        test_dataloader=test_loader,
        collapse_range=collapse_range,
        device=device,
        quant=quant,
    )
    data["diagnostics"] = diagnostics

    plot_accuracy_loss_curve(
        acc_list=data.get("accuracies", []),
        loss_list=data.get("losses", []),
        workflow=workflow,
        experiment=exp_name,
        save_dir=plots_dir,
    )

    model_root = (
        f"{model.__class__.__name__}_{train_loader.dataset.__class__.__name__}"
    )
    safe_update_metrics_json(
        model_root, f"{exp_name}_{workflow}", data, base_dir=metrics_dir
    )
    merged_path = merge_all_metrics(base_dir=metrics_dir)

    print(f"[✓] Metrics merged successfully → {merged_path}")
    print(f"[✓] Experiment '{exp_name}' completed successfully.")

    return data



# =====================================================
# === Experiment Entry Points (JF & Kevin) ===
# =====================================================


def run_jf_experiment(
    experiments,
    model_path_097,
    train_loader,
    test_loader,
    device,
    epochs,
    pretrain,
    model_class=VGG16,
    model_kwargs=None,
    data_shape=None,
    save_path="./runs",
    post_compress_epochs=False,
    quant=False
):

    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]

    # Load pretrained model
    base_model = model_class(**model_kwargs)
    if not "None" in model_path_097:
        ckpt = torch.load(model_path_097, map_location='cpu')
        base_model.load_state_dict(ckpt['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    # Collapse if requested
    compression_set = _make_compression_set(collapse_range)
    if compression_set:
        print(f"[•] Collapsing ranges {compression_set} for {exp_name}")
        base_model = collapse_only(
            model=base_model,
            compression_set=compression_set,
            input_shape=model_kwargs['one_batch'].shape,
            device=device,
            dry_run=False,
            debug=True,
            handle_skips=True
        )

    print(f"[INFO] Collabsed Model: {describe_model(base_model, train_loader)}")

    # Run training & diagnostics
    data = run_experiment(
        model=base_model,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        workflow="JF",
        exp_name=exp_name,
        data_shape=data_shape,
        save_path=save_path,
        post_compress_epochs=post_compress_epochs, 
        quant=quant
    )
    return base_model


def run_kevin_experiment(
    experiments,
    model_path_000,
    train_loader,
    test_loader,
    device,
    epochs,
    model_class=VGG16,
    model_kwargs=None,
    data_shape=None,
    save_path="./runs",
    post_compress_epochs=False, 
    quant=False
):
    model_kwargs = model_kwargs or {}
    print("\n=== Running Kevin experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]

    # Initialize and load pretrained model
    base_model = model_class(**model_kwargs)
    if not "None" in model_path_000:
        ckpt = torch.load(model_path_000, map_location='cpu')
        base_model.load_state_dict(ckpt['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    # Collapse if requested
    compression_set = _make_compression_set(collapse_range)
    if compression_set:
        print(f"[•] Collapsing ranges {compression_set} for {exp_name}")
        base_model = collapse_only(
            model=base_model,
            compression_set=compression_set,
            input_shape=model_kwargs['one_batch'].shape,
            device=device,
            dry_run=False,
            debug=True,
            handle_skips=True
        )
    print(f"[INFO] Collabsed Model: {describe_model(base_model, train_loader)}")

    # Run training & diagnostics
    data = run_experiment(
        model=base_model,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        epochs=epochs,
        workflow="Kevin",
        exp_name=exp_name,
        data_shape=data_shape,
        save_path=save_path,
        post_compress_epochs=post_compress_epochs,
        quant=quant
    )
    return base_model

