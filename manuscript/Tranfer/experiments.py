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



def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


# -------------------------
# Safe JSON Write (Per-job unique, no lock, no SLURM ID)
# -------------------------
def safe_update_metrics_json(model_root, exp_name, new_data, base_dir="./runs/metrics"):
    """
    Writes metrics to a per-job JSON file with a unique timestamp.
    Later, all these per-job JSONs can be merged using merge_all_metrics().
    """
    ensure_dir(base_dir)

    # Create a unique filename per process based on timestamp and PID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    json_path = os.path.join(base_dir, f"{model_root}_metrics_{timestamp}_{pid}.json")

    try:
        with open(json_path, "w") as f:
            json.dump({exp_name: new_data}, f, indent=4)
        print(f"[✓] Saved metrics for '{exp_name}' → {json_path}")
        return json_path
    except Exception as e:
        print(f"[!] Failed to save metrics JSON: {e}")
        return None


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
def run_experiment(model, model_kwargs=None, train_loader=None, test_loader=None, device='cuda',
                   epochs=10, workflow='default', exp_name='experiment', collapse_range=None,
                   data_shape=(1, 3, 32, 32), save_path="./runs", post_compress_epochs=False):

    print(f"[•] Starting experiment '{exp_name}' in workflow '{workflow}'")
    ckpt_dir = os.path.join(save_path, "checkpoints")
    metrics_dir = os.path.join(save_path, "metrics")
    plots_dir = os.path.join(save_path, "plots")
    ensure_dir(ckpt_dir)
    ensure_dir(metrics_dir)
    ensure_dir(plots_dir)

    ckpt_path = os.path.join(
        ckpt_dir, get_checkpoint_filename(workflow, exp_name, model.__class__.__name__, epochs)
    )
    model.to(device)

    model_root = f"{model.__class__.__name__}_{train_loader.dataset.__class__.__name__}"

    # --- Run training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if not torch.cuda.is_available():
        print("[!] Warning: CUDA not available.")
        quit()

    data = train_and_evaluate(
        model, train_loader, test_loader, device, epochs, post_compress_epochs=post_compress_epochs
    )

    torch.save({'model': model.state_dict()}, ckpt_path)

    # --- Compute metrics and diagnostics ---
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)
    data.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": data.get("accuracies", [0])[-1] if data.get("accuracies") else 0,
    })

    diagnostics = run_full_diagnostics(
        model, data_shape, {exp_name: data}, plots_dir, exp_name,
        collapse_range=collapse_range, device=device
    )
    data["diagnostics"] = diagnostics

    plot_accuracy_loss_curve(data, plots_dir, exp_name, workflow)
    # --- Save per-job metrics ---
    safe_update_metrics_json(model_root, f"{exp_name}_{workflow}", data, base_dir=metrics_dir)

    merged_path = merge_all_metrics(base_dir=metrics_dir)
    print(f"[✓] Metrics merged successfully → {merged_path}")

    # --- Save final checkpoint ---
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)
    with open(merged_path, "r") as f:
        for attempt in range(100):
            try:
                with open(merged_path, "r") as f:
                    all_metrics = json.load(f)
                break
            except json.JSONDecodeError:
                print(f"[!] JSON not ready yet, retrying ({attempt+1}/3)...")
                time.sleep(1)

        params = []
        accs = []
        names = []
        infer_times = []
        mem_usages = []
        flops = []
        total_sizes = []  # List to store total size for plotting

        # Iterate through each model's metrics to prepare data for plotting
        for name, metrics in all_metrics.items():
            names.append(name)
            params.append(metrics.get("param_count", 0))
            accs.append(metrics.get("final_accuracy", 0))
            infer_times.append(metrics.get("inference_time", 0))
            mem_usages.append(metrics.get("total_size_mb", 0))
            flops.append(metrics.get("flops", 0))  # Collect FLOPs

        # Save comparison plot
        save_path = merged_path.replace("metrics", "plots").replace("json", "svg")
        
        plot_results(params, accs, names, f"{workflow} Experiments", save_path,
                    dataset=workflow, infer_times=infer_times, mem_usages=mem_usages, flops=flops, total_sizes=total_sizes)
    norm_metrics = normalize_metrics(all_metrics)
    # Plots (each function is robust to input)
    for func in [plot_flops_vs_latency, analyze_collapse_effects, plot_delta_accuracy_vs_params,
                 plot_flops_vs_memory, plot_accuracy_vs_memory, plot_heatmap]:
        try:
            if func.__name__ == "analyze_collapse_effects":
                try:
                    func(model, collapse_range, plots_dir, exp_name)
                except TypeError:
                    func(norm_metrics, plots_dir, exp_name)
            else:
                func(norm_metrics, plots_dir, exp_name)
        except Exception as e:
            print(f"[!] {func.__name__} error: {e}")

    # Cross-experiment plots
    plot_memory_per_layer_across_experiments(glob.glob(os.path.join(metrics_dir, "*.json")), plots_dir, workflow)
    # Final checkpoint
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)
    print(f"[✓] Experiment '{exp_name}' completed. Checkpoints and metrics saved.")
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
    post_compress_epochs=False
):
    model_kwargs = model_kwargs or {}
    print("\n=== Running JF experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]

    # Load pretrained model
    base_model = model_class(**model_kwargs)
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
        post_compress_epochs=post_compress_epochs
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
    post_compress_epochs=False
):
    model_kwargs = model_kwargs or {}
    print("\n=== Running Kevin experiment ===")
    exp_name, collapse_range = list(experiments.items())[0]

    # Initialize and load pretrained model
    base_model = model_class(**model_kwargs)
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
        post_compress_epochs=post_compress_epochs
    )
    return base_model

# -------------------------
