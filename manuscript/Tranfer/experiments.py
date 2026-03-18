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
from collapse import collapse_only, _wrap_pools_safe

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
import os
import glob
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim

from trainer import train_one_epoch

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
    import os, glob, torch, torch.optim as optim
    from torch.cuda.amp import GradScaler

    # Adjust name for quantization if needed
    if quant:
        exp_name += "_quant"

    print(f"[•] Starting experiment '{exp_name}' in workflow '{workflow}'")

    # 1. Directory Setup
    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    ensure_dir(ckpt_dir); ensure_dir(metrics_dir); ensure_dir(plots_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the base name used for all checkpoints
    base_ckpt_name = f"{workflow}_{exp_name}"

    # ---------------------------------------------------------
    # 2. Check for "Final" Completion (NEW ADDITION)
    # ---------------------------------------------------------
    final_ckpt_path = os.path.join(ckpt_dir, f"final_{base_ckpt_name}.pt")
    
    if os.path.exists(final_ckpt_path):
        print(f"[✓] Experiment '{exp_name}' already completed. Found: {os.path.basename(final_ckpt_path)}")
        print(f"    Skipping training and reloading metrics...")
        
        # Load the final data to return it (so downstream code doesn't break)
        try:
            checkpoint = torch.load(final_ckpt_path, map_location=device)
            return checkpoint.get("data", {})
        except Exception as e:
            print(f"[!] Corrupt final checkpoint found ({e}). Restarting experiment...")
            # If load fails, we fall through and restart/resume as normal
            pass

    # 3. Initialize Training Components
    # These MUST be initialized before loading an epoch checkpoint
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler(enabled=quant)
    
    # 4. Checkpoint Discovery (Intermediate Epochs)
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")
    existing_ckpts = sorted(
        glob.glob(ckpt_pattern),
        key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0]),
    )

    start_epoch = 0
    # Store training state inside all_data to persist across restarts
    all_data = {
        "accuracies": [], 
        "losses": [], 
        "best_acc": 0.0, 
        "patience_counter": 0
    }

    # 5. Resume Logic (State Dict approach)
    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        print(f"[•] Loading intermediate checkpoint: {last_ckpt}")
        checkpoint = torch.load(last_ckpt, map_location=device)

        # Restore structural states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if quant and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # Restore RNG states (ensures reproducibility on resume)
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())

        start_epoch = checkpoint['epoch']
        all_data = checkpoint['data']
        print(f"[✓] Resumed at epoch {start_epoch}")
    else:
        print("[•] No intermediate checkpoint found — starting fresh")

    # 6. Training Loop
    for epoch in range(start_epoch + 1, epochs + 101 if post_compress_epochs else epochs + 1):
        print(f"[•] Epoch {epoch}")

        # Execute one epoch of training
        avg_loss, acc = train_one_epoch(
            model, train_loader, optimizer, device, 
            scaler=scaler, use_autocast=quant
        )

        all_data["accuracies"].append(acc)
        all_data["losses"].append(avg_loss)

        # Logic for Early Stopping / Best Acc tracking
        if acc > all_data["best_acc"] + 0.05: # 0.05 threshold
            all_data["best_acc"] = acc
            all_data["patience_counter"] = 0
        else:
            all_data["patience_counter"] += 1

        # Save State-Aware Checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch}.pt")
        tmp_path = ckpt_path + ".tmp"

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if quant else None,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "data": all_data,
        }, tmp_path)
        
        os.replace(tmp_path, ckpt_path) # Atomic operation
        
        # Cleanup old checkpoints to save HPC quota
        if epoch > 1:
            old_ckpt = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch - 1}.pt")
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)

        # Early Stopping Logic (Post-compression phase)
        if post_compress_epochs and epoch > epochs:
            if all_data["patience_counter"] >= 5:
                print(f"[!] Early stopping triggered at epoch {epoch}")
                break

    # 7. Finalization & Diagnostics
    # We re-use final_ckpt_path defined in Step 2.5
    torch.save({"model_state_dict": model.state_dict(), "data": all_data}, final_ckpt_path)

    # Performance Benchmarking
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device, quant=quant)

    all_data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": all_data["accuracies"][-1] if all_data["accuracies"] else 0,
    })

    # Run remaining diagnostics/plotting
    run_full_diagnostics(model, data_shape, {exp_name: all_data}, plots_dir, exp_name, 
                         test_dataloader=test_loader, collapse_range=collapse_range, 
                         device=device, quant=quant)

    plot_accuracy_loss_curve(all_data["accuracies"], all_data["losses"], workflow, exp_name, plots_dir)

    model_root = f"{model.__class__.__name__}_{train_loader.dataset.__class__.__name__}"
    safe_update_metrics_json(model_root, f"{exp_name}_{workflow}", all_data, metrics_dir)
    merge_all_metrics(metrics_dir)

    print(f"[✓] Experiment '{exp_name}' completed.")
    return all_data


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
    
    # 1. Check for existing checkpoint before structural changes
    ckpt_dir = os.path.join(save_path, "checkpoints")
    base_ckpt_name = f"JF_{exp_name}"
    if quant: base_ckpt_name += "_quant"
    
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")
    existing_ckpts = glob.glob(ckpt_pattern)

    # 2. Initialize the Base Architecture
    base_model = model_class(**model_kwargs)
    _wrap_pools_safe(base_model)
    # 3. Handle Initialization vs. Resumption
    if not existing_ckpts:
        # FRESH START: Load initial pretrained weights if they exist
        if model_path_097 and "None" not in model_path_097:
            print(f"[•] Loading initial weights from {model_path_097}")
            ckpt = torch.load(model_path_097, map_location='cpu', weights_only=True)
            # Use strict=False if the pretrained model has extra/missing layers
            base_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['model'], strict=False)
    else:
        print(f"[•] Resumption detected. Architecture will be prepared for checkpoint loading.")

    # 4. Apply Collapse Logic 
    # (Must be done every time to ensure base_model keys match the checkpoint keys)
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

    # 5. Hand off to the state-aware runner
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

    # 1. Check for existing checkpoint
    ckpt_dir = os.path.join(save_path, "checkpoints")
    base_ckpt_name = f"Kevin_{exp_name}"
    if quant: base_ckpt_name += "_quant"
    
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")
    existing_ckpts = glob.glob(ckpt_pattern)

    # 2. Initialize
    base_model = model_class(**model_kwargs)
    _wrap_pools_safe(base_model)

    # 3. Handle Initial Weights
    if not existing_ckpts:
        if model_path_000 and "None" not in model_path_000:
            print(f"[•] Loading initial weights from {model_path_000}")
            ckpt = torch.load(model_path_000, map_location='cpu', weights_only=True)
            base_model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['model'], strict=False)

    # 4. Apply Collapse
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

    # 5. Run
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
