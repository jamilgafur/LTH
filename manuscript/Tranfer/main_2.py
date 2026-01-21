import glob
import os
import torch
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.strategies.collapse import *
from diagnostic import *
from utils import *
from trainer import *
from pyPrune.utils import *
import tempfile

from datetime import datetime
from torch.backends import cudnn
import random
import math 
import numpy as np
import argparse
import json
from config import *
# Set seed for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility across different modules."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
# python main_2.py --model VGG16 --dataset Cifar10 --epochs 6 --epochs 1 --collapse_start features.conv_1 --collapse_end features.bn_4

set_seed(42)


def initialize_model_and_data(args):
    """Initialize the model and dataset based on the provided arguments."""
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}

    # Ensure InceptionNet is not used with JF experiments
    if model_class == InceptionNet and args.JF:
        raise ValueError("JF experiments are not supported for InceptionNet.")
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

def train_model(model, train_loader, test_loader, epochs, device):
    """Train the model and return losses and accuracies."""
    optimizer, scheduler = create_optimizer_scheduler(model)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, device
        )

        model.eval()
        test_loss, test_accuracy = evaluate_model(
            model, test_loader, device
        )

        # Store metrics
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}"
        )

        scheduler.step()

    return history

def create_optimizer_scheduler(model, learning_rate=1e-3):
    """Creates optimizer and scheduler for training."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

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

def safe_update_metrics_json(model_root, exp_name, new_data, base_dir="./runs/metrics"):
    """Writes metrics to a per-job JSON file with a unique timestamp."""
    os.makedirs(base_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    pid = os.getpid()
    json_path = os.path.join(base_dir, f"{model_root}_metrics_{timestamp}_{pid}.json")

    try:
        # Crucial: Deep clean data for JSON compatibility
        safe_data = convert_ndarrays_to_lists(new_data)

        # Write to temp file first for atomic operations
        tmp_fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix="tmp_metrics_", suffix=".json")
        with os.fdopen(tmp_fd, "w") as f:
            json.dump({exp_name: safe_data}, f, indent=4)

        os.replace(tmp_path, json_path)
        print(f"[✓] Saved metrics for '{exp_name}' → {json_path}")
        return json_path
    except Exception as e:
        print(f"[!] Failed to save metrics JSON: {e}")
        return None

def merge_all_metrics(base_dir="./runs/metrics", merged_name="merged_metrics.json"):
    """Safely merges all metrics JSON files into one consolidated file."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Improved glob: Look for any .json file that contains 'metrics' but isn't the merged file itself
    json_files = [
        f for f in glob.glob(os.path.join(base_dir, "*.json")) 
        if "_metrics_" in os.path.basename(f) and merged_name not in os.path.basename(f)
    ]
    
    merged_data = {}
    print(f"[•] Found {len(json_files)} component files in {base_dir}")

    for jf in json_files:
        try:
            if os.path.getsize(jf) == 0: continue
            with open(jf, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_data.update(data)
        except Exception as e:
            print(f"[!] Skipping {jf}: {e}")

    merged_path = os.path.join(base_dir, merged_name)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix="tmp_merge_", suffix=".json")
    
    with os.fdopen(tmp_fd, "w") as tmp_file:
        json.dump(merged_data, tmp_file, indent=4)

    os.replace(tmp_path, merged_path)
    print(f"[✓] Successfully merged all metrics → {merged_path}")
    return merged_path

import os
import glob
import torch
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="XceptionNet")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--collapse_start", type=str, default=None)
    parser.add_argument("--collapse_end", type=str, default=None)
    parser.add_argument("--quant", action="store_true")
    args = parser.parse_args()

    # 1. Environment & Shared Paths
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"{args.model}_{args.dataset}"
    if args.quant: exp_name += "_quant"
    
    # 2. Model & Data Initialization
    model_files = glob.glob(f"baseline*/*/check*/*pt")
    if not model_files:
        raise ValueError(f"No baseline model found for {args.model} on {args.dataset}.")
    baseline_model_file = model_files[0]
    
    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)

    # 3. Experiment Directory Setup
    post_collapse_dir = os.path.join(os.path.dirname(os.path.dirname(model_files[0])), f"post_collapse_{args.collapse_start}_{args.collapse_end}_epochs{args.epochs}")
    shared_metrics_dir = f"./{post_collapse_dir}/metrics" 
    
    ckpt_dir = os.path.join(post_collapse_dir, "checkpoints")
    for d in [ckpt_dir, shared_metrics_dir]: os.makedirs(d, exist_ok=True)

    # 4. Resume Logic - Efficiency: Loading state_dict instead of full objects
    base_ckpt_name = f"{args.collapse_start+args.collapse_end}_{exp_name}"
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")
    existing_ckpts = sorted(glob.glob(ckpt_pattern), key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0]))

    start_epoch = 0
    all_data = {"accuracies": [], "losses": []}
    model = eval(model_class)(**model_kwargs).to(device)

    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        print(f"[✓] Resuming from: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"]) # Memory efficient loading
        start_epoch = ckpt["epoch"]
        all_data = ckpt["data"]
        del ckpt # Free memory
    else:
        checkpoint = torch.load(baseline_model_file, map_location=device)
        model.load_state_dict(checkpoint['model'])
        if args.collapse_start and args.collapse_end:
            model = collapse_only(model=model, model_weights_1=None, compression_set={(args.collapse_start, args.collapse_end)}, 
                                 model_class=model_class, input_shape=input_size, device=device)
        del checkpoint

    # 5. Training Loop - Efficiency: Periodically clearing CUDA cache
    for epoch in range(start_epoch + 1, args.epochs + 1):
        step_data = train_model(model, train_loader, test_loader, epochs=1, device=device)
        all_data["accuracies"].extend(step_data.get("test_accuracy", []))
        all_data["losses"].extend(step_data.get("test_loss", []))

        # Save state_dict only (much smaller files and lower RAM usage)
        ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch, 
            "model_state_dict": model.state_dict(), 
            "data": all_data
        }, ckpt_path)

        if epoch > 1:
            old_ckpt = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch-1}.pt")
            if os.path.exists(old_ckpt): os.remove(old_ckpt)
        
        torch.cuda.empty_cache()
        gc.collect()

    # 6. Final Evaluation & Single JSON Merge
    print("[•] Running final diagnostics...")
    model.eval()
    with torch.no_grad(): # Ensure no gradients are stored during evaluation
        cka_scores, layer_names = calculate_central_kernel_alignment(model, test_loader, post_collapse_dir, 1, device)
        param_count = count_trainable_params(model)
        infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device, quant=args.quant)
        
        diagnostic = run_full_diagnostics(model, input_size, {exp_name: all_data}, post_collapse_dir, exp_name, 
                                          test_loader, (args.collapse_start, args.collapse_end), device, args.quant)

    # Consolidate everything into diagnostic via .update
    diagnostic.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": all_data["accuracies"][-1] if all_data["accuracies"] else 0,
        "metadata": {
            "dataset": args.dataset,
            "architecture": args.model,
            "collapse": [args.collapse_start, args.collapse_end]
        },
        "cka": {"scores": cka_scores, "layers": layer_names},
        "history": all_data
    })

    # Memory efficient JSON writing
    master_path = os.path.join(shared_metrics_dir, "master_metrics.json")
    master_data = {}
    if os.path.exists(master_path):
        with open(master_path, "r") as f:
            try: master_data = json.load(f)
            except: pass

    master_data[f"{exp_name}_{args.collapse_start+args.collapse_end}"] = convert_ndarrays_to_lists(diagnostic)

    with open(master_path, "w") as f:
        json.dump(master_data, f, indent=4)

    print(f"[✓] Experiment complete. Merged metrics saved to {master_path}")

if __name__ == "__main__":
    main()