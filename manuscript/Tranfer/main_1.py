import os
import glob
import torch
import random
import numpy as np
import argparse
import json
import tempfile
import gc
from datetime import datetime
from torch.backends import cudnn

from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.strategies.collapse import *
from utils import *
from trainer import *
from pyPrune.utils import *
from config import *

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

set_seed(42)

def convert_ndarrays_to_lists(data):
    """Recursively converts numpy arrays and torch tensors to lists for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_ndarrays_to_lists(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_ndarrays_to_lists(v) for v in data]
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        return data.tolist()
    elif isinstance(data, (np.float32, np.float64)):
        return float(data)
    elif isinstance(data, (np.int32, np.int64)):
        return int(data)
    else:
        return data

def initialize_model_and_data(args):
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

def get_latest_checkpoint(ckpt_dir):
    """Finds the checkpoint file with the highest epoch number."""
    ckpts = glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pt"))
    if not ckpts:
        return None
    # Extract epoch numbers and find the max
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in ckpts]
    latest_idx = np.argmax(epochs)
    return ckpts[latest_idx]

def train_model(model, train_loader, test_loader, epochs, device, 
                optimizer, scheduler, start_epoch=0, checkpoint_dir=None, history=None):
    
    if history is None:
        history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)

        model.eval()
        test_loss, test_accuracy = evaluate_model(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} | Train Acc: {train_accuracy:.4f} | Test Acc: {test_accuracy:.4f}")
        scheduler.step()

        # --- Rotating HPC Checkpoint Logic ---
        if checkpoint_dir:
            current_ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            
            # 1. Save to a temporary file first (Atomic Write to prevent corruption)
            temp_path = os.path.join(checkpoint_dir, "temp.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
            }, temp_path)
            
            # 2. Rename temp to the formal epoch name
            os.replace(temp_path, current_ckpt)
            
            # 3. Space Management: Delete the checkpoint from TWO epochs ago
            # This keeps 'current' and 'previous' only.
            old_ckpt = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch - 1}.pt")
            if os.path.exists(old_ckpt):
                os.remove(old_ckpt)
                print(f"Removed old checkpoint: {old_ckpt}")

    return history

def create_optimizer_scheduler(model, learning_rate=1e-3):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="VGG16")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--pretrain", type=int, default=0)
    parser.add_argument("--break_group", type=int, default=3)
    args = parser.parse_args()

    # 1. Environment Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{args.model}_{args.dataset}_pretrain{args.pretrain}"
    
    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    # 2. Path Logic
    base_path = CHECKPOINT_BASES[args.model][args.dataset]
    model_path_pruned = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][0])
    model_path_initalized = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][1])
    baseline_model_dir = os.path.join("baseline_models", f"{args.model}_{args.dataset}_pretrain{args.pretrain}_break{args.break_group}")

    # 3. Directory Setup
    metrics_dir = os.path.join(baseline_model_dir, "metrics")
    ckpt_dir = os.path.join(baseline_model_dir, "checkpoints")
    for d in [metrics_dir, ckpt_dir]: os.makedirs(d, exist_ok=True)

    # 4. Model & Optimizer Initialization
    model = eval(model_class)(**model_kwargs).to(device)
    optimizer, scheduler = create_optimizer_scheduler(model)
    
    # 5. HPC Resume Logic
    latest_ckpt = get_latest_checkpoint(ckpt_dir)
    start_epoch = 0
    history = None

    if latest_ckpt:
        print(f"[!] Found existing checkpoint: {latest_ckpt}. Resuming...")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        history = checkpoint['history']
        print(f"[✓] Resumed from epoch {start_epoch}")
    else:
        # If no checkpoint exists, use your original path-loading logic
        if "None" not in model_path_initalized and os.path.isfile(model_path_initalized):
            model.load_state_dict(torch.load(model_path_initalized, map_location=device)['model'])
            print(f"Loaded initialized weights from: {model_path_initalized}")

        if "None" not in model_path_pruned and os.path.isfile(model_path_pruned):
            model.load_state_dict(torch.load(model_path_pruned, map_location=device)['model'])
            print(f"Loaded pruned weights from: {model_path_pruned}")

    # 6. Training (Skips or finishes based on start_epoch)
    if start_epoch < args.pretrain:
        history = train_model(
            model, train_loader, test_loader, 
            epochs=args.pretrain, 
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            start_epoch=start_epoch,
            checkpoint_dir=ckpt_dir,
            history=history
        )
    else:
        print("[•] Training already completed for target epochs.")

    # 7. Diagnostics (Only run after training loop is done)
    print("[•] Running diagnostics and CKA...")
    cka_scores, layer_names = calculate_central_kernel_alignment(model, test_loader, baseline_model_dir, 3, device)
    
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)

    # 8. Structured Metadata Assembly
    diagnostic = {
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": history["test_accuracy"][-1] if history and history["test_accuracy"] else 0,
        "history": history,
        "cka": {"scores": cka_scores, "layers": layer_names},
        "dataset": args.dataset,
        "architecture": args.model,
        "pretrain_epochs": args.pretrain,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # 9. Final Save (Final Model & JSON)
    torch.save({
        'model': model.state_dict(),
        'history': history,
        'args': vars(args)
    }, os.path.join(ckpt_dir, "final_pretrained_model.pt"))

    master_path = os.path.join(metrics_dir, "master_metrics.json")
    master_data = {}
    if os.path.exists(master_path):
        with open(master_path, "r") as f:
            try: master_data = json.load(f)
            except: pass

    master_data[exp_name] = convert_ndarrays_to_lists(diagnostic)
    with open(master_path, "w") as f:
        json.dump(master_data, f, indent=4)

    # 10. Generate Bash Script (unchanged)
    PBS_DIR = "/Users/jgafur/LTH/manuscript/Tranfer/"
    bash_script = f"""#!/bin/bash
# Auto-generated script to submit collapse jobs
set -e
MODEL={args.model}
DATASET={args.dataset}
EPOCHS={args.epochs}
cd {PBS_DIR} || exit 1
"""
    for layer_name in layer_names:
        collapse_start, collapse_end = layer_name[1], layer_name[2]
        bash_script += f"""
command="qsub -q all.q -l ngpus=1 \\
  -v MODEL=${{MODEL}},DATASET=${{DATASET}},EPOCHS=${{EPOCHS}},COLLAPSE_START={collapse_start},COLLAPSE_END={collapse_end} \\
  main_2.pbs"
echo "Submitting: $command"
eval "$command"
"""

    script_path = os.path.join(baseline_model_dir, "submit_collapse_jobs.sh")
    with open(script_path, "w") as f:
        f.write(bash_script)
    os.chmod(script_path, 0o755)

    print(f"[✓] Baseline complete. Results: {baseline_model_dir}")

if __name__ == "__main__":
    main()