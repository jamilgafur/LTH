# experiment.py

# Standard libraries
import os
import glob
import json
from datetime import datetime
from copy import deepcopy
from collections import OrderedDict

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
from plots import plot_accuracy_loss_curve, plot_results


import os
import json
import glob
from datetime import datetime


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
    Merge all per-job metrics JSONs into one consolidated JSON file.
    Skips malformed or unreadable files.
    """
    ensure_dir(base_dir)
    json_files = glob.glob(os.path.join(base_dir, "*_metrics_*.json"))
    merged_data = {}

    for jf in json_files:
        try:
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_data.update(data)
        except Exception as e:
            print(f"[!] Skipping {jf}: {e}")

    merged_path = os.path.join(base_dir, merged_name)
    with open(merged_path, "w") as f:
        json.dump(merged_data, f, indent=4)

    print(f"[✓] Merged {len(json_files)} metrics files → {merged_path}")
    return merged_path


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

    # --- Save per-job metrics ---
    safe_update_metrics_json(model_root, f"{exp_name}_{workflow}", data, base_dir=metrics_dir)

    merged_path = merge_all_metrics(base_dir=metrics_dir)
    print(f"[✓] Metrics merged successfully → {merged_path}")

    # --- Save final checkpoint ---
    final_path = os.path.join(ckpt_dir, f"final_{os.path.basename(ckpt_path)}")
    torch.save({'model': model.state_dict()}, final_path)
    # def plot_results(params, accs, names, title, filename, dataset=None, infer_times=None, mem_usages=None, flops=None, total_sizes=None):
    # load the merged metrics to plot final results
    with open(merged_path, "r") as f:
        merged_data = json.load(f)
    print(f"[✓] Loaded merged metrics from {merged_path} for plotting final results for workflow '{workflow}' merged_data keys: {list(merged_data.keys())} with values: {merged_data}")
    plot_results(
        params=[d.get("param_count", 0) for d in merged_data.values()],
        accs=[d.get("final_accuracy", 0) for d in merged_data.values()],
        names=[k for k in merged_data.keys()],
        title=f"Final Accuracy vs. Parameter Count for '{workflow}'",
        filename=os.path.join(plots_dir, f"final_results_{workflow}.png"),
        dataset=train_loader.dataset.__class__.__name__,
        infer_times=[d.get("inference_time", 0) for d in merged_data.values()],
        mem_usages=[d.get("total_size_mb", 0) for d in merged_data.values()],
        flops=[d.get("flops", 0) for d in merged_data.values()],
        total_sizes=[d.get("total_size_mb", 0) for d in merged_data.values()]
    )
    print(f"[✓] Saved final checkpoint: {final_path}")
    print(f"[✓] Saved final metrics and diagnostics for '{exp_name}' in workflow '{workflow}'")
    print(f"[✓] Checkpoints and metrics saved for '{exp_name}'.")

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

    # load pretrained model
    base_model = model_class(**model_kwargs)
    ckpt = torch.load(model_path_097, map_location='cpu')
    base_model.load_state_dict(ckpt['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    # collapse if requested
    if collapse_range:
        print(f"[•] Collapsing range {collapse_range} for {exp_name}")
        base_model = collapse_only(
            model=base_model,
            compression_set=[collapse_range],                   # <- changed
            input_shape=model_kwargs['one_batch'].shape,
            device=device,
            dry_run=False,
            debug=True,
            handle_skips=True
        )

    # run training & diagnostics
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

    base_model = model_class(**model_kwargs)
    ckpt = torch.load(model_path_000, map_location='cpu')
    base_model.load_state_dict(ckpt['model'])
    print(f"[INFO] Initialized Model: {describe_model(base_model, train_loader)}")

    if collapse_range:
        print(f"[•] Collapsing range {collapse_range} for {exp_name}")
        base_model = collapse_only(
            model=base_model,
            compression_set=[collapse_range],                   # <- changed
            input_shape=model_kwargs['one_batch'].shape,
            device=device,
            dry_run=False,
            debug=True,
            handle_skips=True
        )

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
