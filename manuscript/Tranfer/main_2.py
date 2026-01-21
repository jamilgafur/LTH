import glob
import os
import torch
import gc
import time
import random
import tempfile
import numpy as np
import json
import argparse
from datetime import datetime
from torch.backends import cudnn

# Import custom modules
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
from config import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def initialize_model_and_data(args):
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}

    if model_class == "InceptionNet" and hasattr(args, 'JF') and args.JF:
        raise ValueError("JF experiments are not supported for InceptionNet.")
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

def train_one_step(model, train_loader, test_loader, device, optimizer, scheduler):
    """Wrapper to train for exactly one epoch and return metrics."""
    model.train()
    train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, device)
    model.eval()
    test_loss, test_accuracy = evaluate_model(model, test_loader, device)
    scheduler.step()
    return {"train_loss": train_loss, "train_acc": train_accuracy, 
            "test_loss": test_loss, "test_acc": test_accuracy}

def convert_ndarrays_to_lists(data):
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

def safe_update_metrics_json(exp_name, new_data, base_dir):
    """HPC-Safe: Writes metrics using directory-based locking."""
    os.makedirs(base_dir, exist_ok=True)
    master_path = os.path.join(base_dir, "master_metrics.json")
    lock_dir = master_path + ".lock"
    
    safe_data = convert_ndarrays_to_lists(new_data)
    
    max_retries = 100
    acquired = False
    for _ in range(max_retries):
        try:
            os.makedirs(lock_dir)
            acquired = True
            break
        except FileExistsError:
            time.sleep(random.uniform(0.5, 2.0))

    if not acquired:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_path = os.path.join(base_dir, f"backup_{timestamp}_{exp_name}.json")
        with open(emergency_path, "w") as f:
            json.dump({exp_name: safe_data}, f, indent=4)
        return

    try:
        master_data = {}
        if os.path.exists(master_path):
            with open(master_path, "r") as f:
                try: master_data = json.load(f)
                except: pass
        
        master_data[exp_name] = safe_data
        
        tmp_fd, tmp_path = tempfile.mkstemp(dir=base_dir, prefix="tmp_master_", suffix=".json")
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(master_data, f, indent=4)
            f.flush()
            os.fsync(f.fileno())
        
        os.replace(tmp_path, master_path)
    finally:
        try: os.rmdir(lock_dir)
        except: pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="XceptionNet")
    parser.add_argument("--dataset", type=str, default="Cifar10")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--collapse_start", type=str, default=None)
    parser.add_argument("--collapse_end", type=str, default=None)
    parser.add_argument("--quant", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_name = f"{args.model}_{args.dataset}"
    if args.quant: exp_name += "_quant"
    
    # 1. Locate Baseline Model
    search_pattern = f"baseline*/*{args.model}*{args.dataset}*/**/*.pt"
    # Note: This pattern might still find Cifar100 checkpoints if the folder structure implies it.
    # The fix below handles the file mismatch regardless of how it was found.
    model_files = glob.glob(search_pattern, recursive=True)
    model_files = [f for f in model_files if "checkpoint" in f.lower() or "pretrained" in f.lower() or "final" in f.lower()]

    if not model_files:
        # Fallback: Try finding *any* baseline for this model if the specific dataset one is missing (Transfer Learning case)
        search_pattern_fallback = f"baseline*/*{args.model}*/**/*.pt"
        model_files = glob.glob(search_pattern_fallback, recursive=True)
        model_files = [f for f in model_files if "checkpoint" in f.lower() or "pretrained" in f.lower() or "final" in f.lower()]
        
        if not model_files:
            raise ValueError(f"No baseline model found for {args.model}")

    baseline_model_file = model_files[0]
    
    # 2. Initialization
    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    project_root = os.path.dirname(os.path.dirname(model_files[0]))
    post_collapse_dir = os.path.join(project_root, f"post_collapse_{args.collapse_start}_{args.collapse_end}_epochs{args.epochs}")
    central_metrics_dir = os.path.join(project_root, "metrics_consolidated")
    ckpt_dir = os.path.join(post_collapse_dir, "checkpoints")
    
    for d in [ckpt_dir, central_metrics_dir]: os.makedirs(d, exist_ok=True)

    # 3. Resume Logic
    base_ckpt_name = f"{args.collapse_start}_{args.collapse_end}_{exp_name}"
    ckpt_pattern = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch*.pt")
    existing_ckpts = sorted(glob.glob(ckpt_pattern), key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0]))

    model = eval(model_class)(**model_kwargs).to(device)
    optimizer, scheduler = create_optimizer_scheduler(model)
    start_epoch = 0
    all_data = {"accuracies": [], "losses": []}

    if existing_ckpts:
        last_ckpt = existing_ckpts[-1]
        print(f"[✓] Resuming from checkpoint: {last_ckpt}")
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"]
        all_data = ckpt["data"]
        del ckpt
    else:
        # Initial Load & Collapse
        print(f"Loading baseline from: {baseline_model_file}")
        checkpoint = torch.load(baseline_model_file, map_location=device)
        state_dict = checkpoint.get('model', checkpoint.get('model_state_dict', checkpoint))
        
        ### --- MODIFIED SECTION START ---
        # Robust loading: Filter out layers with shape mismatches (e.g., Cifar100 FC -> Cifar10 FC)
        model_dict = model.state_dict()
        pretrained_dict = {}
        mismatched_keys = []

        for k, v in state_dict.items():
            if k in model_dict:
                if v.size() == model_dict[k].size():
                    pretrained_dict[k] = v
                else:
                    mismatched_keys.append(k)
            # Handle DataParallel 'module.' prefix if necessary
            elif k.startswith("module.") and k[7:] in model_dict:
                k_clean = k[7:]
                if v.size() == model_dict[k_clean].size():
                    pretrained_dict[k_clean] = v
                else:
                    mismatched_keys.append(k_clean)

        if mismatched_keys:
            print(f"[!] Warning: The following layers were skipped due to shape mismatch (expected for transfer learning): {mismatched_keys}")
        
        # strict=False allows us to load the weights even if the Final Layer (FC) is missing from pretrained_dict
        model.load_state_dict(pretrained_dict, strict=False)
        ### --- MODIFIED SECTION END ---
        
        if args.collapse_start and args.collapse_end:
            model = collapse_only(model=model, model_weights_1=None, 
                                  compression_set={(args.collapse_start, args.collapse_end)}, 
                                  model_class=model_class, input_shape=input_size, device=device)
        del checkpoint

    # 4. Training Loop (Rotating Checkpoints)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        step_metrics = train_one_step(model, train_loader, test_loader, device, optimizer, scheduler)
        
        all_data["accuracies"].append(step_metrics["test_acc"])
        all_data["losses"].append(step_metrics["test_loss"])
        print(f"Epoch {epoch}/{args.epochs} | Loss: {step_metrics['test_loss']:.4f} | Acc: {step_metrics['test_acc']:.4f}")

        # Atomic Save
        ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch}.pt")
        temp_ckpt = ckpt_path + ".tmp"
        torch.save({
            "epoch": epoch, 
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "data": all_data
        }, temp_ckpt)
        os.replace(temp_ckpt, ckpt_path)

        # Space Management: Keep only last 2 epochs
        old_ckpt = os.path.join(ckpt_dir, f"{base_ckpt_name}_epoch{epoch-2}.pt")
        if os.path.exists(old_ckpt):
            try: os.remove(old_ckpt)
            except: pass
        
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Final Diagnostics
    model.eval()
    with torch.no_grad():
        cka_scores, layer_names = calculate_central_kernel_alignment(model, test_loader, post_collapse_dir, 1, device)
        param_count = count_trainable_params(model)
        infer_time, flops, total_size_mb = benchmark_model(model, test_loader, device, quant=args.quant)
        
        diagnostic = run_full_diagnostics(model, input_size, {exp_name: all_data}, post_collapse_dir, exp_name, 
                                          test_loader, (args.collapse_start, args.collapse_end), device, args.quant)

    diagnostic.update({
        "param_count": param_count,
        "inference_time": infer_time,
        "flops": flops,
        "total_size_mb": total_size_mb,
        "final_accuracy": all_data["accuracies"][-1] if all_data["accuracies"] else 0,
        "dataset": args.dataset, 
        "architecture": args.model, 
        "collapse": [args.collapse_start, args.collapse_end],
        "cka": {"scores": cka_scores, "layers": layer_names},
        "history": all_data
    })

    # 6. Final Save to CENTRAL directory
    entry_key = f"{exp_name}_{args.collapse_start}_{args.collapse_end}"
    safe_update_metrics_json(entry_key, diagnostic, base_dir=central_metrics_dir)
    print(f"[✓] Post-collapse complete for {args.collapse_start} -> {args.collapse_end}")

if __name__ == "__main__":
    main()