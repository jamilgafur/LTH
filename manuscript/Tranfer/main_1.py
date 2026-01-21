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

def train_model(model, train_loader, test_loader, epochs, device):
    optimizer, scheduler = create_optimizer_scheduler(model)
    history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(epochs):
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
    args = parser.parse_args()

    # 1. Environment Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_name = f"{args.model}_{args.dataset}_pretrain{args.pretrain}"
    
    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    # 2. Path Logic
    base_path = CHECKPOINT_BASES[args.model][args.dataset]
    model_path_pruned = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][0])
    model_path_initalized = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][1])

    break_group = 6
    baseline_model_dir = os.path.join("baseline_models", f"{args.model}_{args.dataset}_pretrain{args.pretrain}_break{break_group}")
    
    # 3. Model Loading
    model = eval(model_class)(**model_kwargs).to(device)
    
    if "None" not in model_path_initalized and os.path.isfile(model_path_initalized):
        model.load_state_dict(torch.load(model_path_initalized, map_location=device)['model'])
        baseline_model_dir += "_initialized"        

    if "None" not in model_path_pruned and os.path.isfile(model_path_pruned):
        model.load_state_dict(torch.load(model_path_pruned, map_location=device)['model'])
        baseline_model_dir += "_pruned"

    # 4. Directory & Sub-dir Setup
    metrics_dir = os.path.join(baseline_model_dir, "metrics")
    ckpt_dir = os.path.join(baseline_model_dir, "checkpoints")
    for d in [metrics_dir, ckpt_dir]: os.makedirs(d, exist_ok=True)

    # 5. Pretraining
    history = train_model(model, train_loader, test_loader, epochs=args.pretrain, device=device)       

    # 6. Diagnostics
    print("[•] Running diagnostics and CKA...")
    cka_scores, layer_names = calculate_central_kernel_alignment(model, test_loader, baseline_model_dir, 3, device)
    
    # Benchmark metrics
    param_count = count_trainable_params(model)
    infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device)

    # 7. Structured Metadata Assembly
    diagnostic = {
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": history["test_accuracy"][-1] if history["test_accuracy"] else 0,
        "history": history,
        "cka": {"scores": cka_scores, "layers": layer_names},
        "metadata": {
            "dataset": args.dataset,
            "architecture": args.model,
            "pretrain_epochs": args.pretrain,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # 8. Safe Saving
    # Save Model Checkpoint
    torch.save({
        'model': model.state_dict(),
        'history': history,
        'args': vars(args)
    }, os.path.join(ckpt_dir, "pretrained_model.pt"))

    # Save Master JSON
    master_path = os.path.join(metrics_dir, "master_metrics.json")
    master_data = {}
    if os.path.exists(master_path):
        with open(master_path, "r") as f:
            try: master_data = json.load(f)
            except: pass

    master_data[exp_name] = convert_ndarrays_to_lists(diagnostic)

    with open(master_path, "w") as f:
        json.dump(master_data, f, indent=4)

    # 9. Generate Bash Script
    PBS_DIR = "/Users/jgafur/LTH/manuscript/Tranfer/"
    PBS_SCRIPT = os.path.join(PBS_DIR, "main_2.pbs")

    bash_script = f"""#!/bin/bash
# Auto-generated script to submit collapse jobs

set -e

MODEL={args.model}
DATASET={args.dataset}
EPOCHS=5

cd {PBS_DIR} || exit 1

"""

    for layer_name in layer_names:
        collapse_start = layer_name[1]
        collapse_end = layer_name[2]

        print(f"[•] Generating collapse job for {collapse_start} → {collapse_end}")

        bash_script += f"""
command="qsub -q all.q -l ngpus=1 \\
  -v MODEL=${{MODEL}},DATASET=${{DATASET}},EPOCHS=${{EPOCHS}},COLLAPSE_START={collapse_start},COLLAPSE_END={collapse_end} \\
  main_2.pbs"
output_file="{baseline_model_dir}/collapse_{collapse_start}_to_{collapse_end}.out"
echo "Logging output to: $output_file"
echo "Submitting job with command:"
echo "$command"
eval "$command > $output_file 2>&1 &"
"""

    script_path = os.path.join(baseline_model_dir, "submit_collapse_jobs.sh")
    with open(script_path, "w") as f:
        f.write(bash_script)

    os.chmod(script_path, 0o755)
    print(f"[✓] Collapse job submission script saved to: {script_path}")

    print(f"[✓] Baseline complete. Results saved in: {baseline_model_dir}")

if __name__ == "__main__":
    main()