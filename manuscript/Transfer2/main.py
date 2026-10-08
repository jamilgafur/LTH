import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd

# Import architectures and utilities from your framework
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from utils import load_dataset
from collapse import collapse_only
import argparse

# =========================================================
# Utility Functions
# =========================================================


def parse_directory_context(filepath: str):
    """
    Extracts the model, dataset, epoch budget, and pretrain budget from the directory name.
    Example: ../Tranfer/XceptionNet_tinyimagenet_None_epochs100_pretrain300/checkpoints/...
    """
    base_dir = filepath.split("/checkpoints/")[0].split("/")[-1]
    parts = base_dir.split("_")

    model_name = parts[0]
    dataset_name = parts[1]

    epochs_str = next(p for p in parts if p.startswith("epochs"))
    pretrain_str = next(p for p in parts if p.startswith("pretrain"))

    match = re.search(r"epoch(\d+)\.pt", filepath)
    ckpt_epoch = int(match.group(1)) if match else 0

    return model_name, dataset_name, epochs_str, pretrain_str, ckpt_epoch, base_dir


def initialize_architecture(model_name: str, dataset_name: str):
    """Dynamically loads the dataset and initializes the correct model architecture."""
    print(
        f"[INFO] Initializing architecture: model={model_name}, dataset={dataset_name}"
    )

    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(
        dataset_name, model_name
    )

    print(
        f"[INFO] Dataset loaded: input_size={input_size}, "
        f"input_channels={input_channels}, num_classes={num_classes}"
    )

    model_kwargs = {"num_classes": num_classes}

    # Capture a dummy batch for collapse input shape inference
    dummy_input = next(iter(train_loader))[0][0:1]

    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False

    model_class = eval(model_name)
    model = model_class(**model_kwargs)

    print(
        f"[INFO] Model initialized: {model_name} "
        f"({sum(p.numel() for p in model.parameters()):,} parameters)"
    )

    return model, test_loader, dummy_input


def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    """Matches the corresponding checkpoint within the same base experiment directory."""
    base_dir = before_path.split("/checkpoints/")[0]

    print(f"[INFO] Searching for checkpoint matching: {base_dir}")

    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            print(f"[INFO] Found matching checkpoint: {candidate}")
            return candidate

    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")


def load_weights(model: nn.Module, filepath: str, device: torch.device):
    """Loads a PyTorch model checkpoint state_dict into the instantiated model."""
    print(f"[INFO] Loading weights: {filepath}")
    start_time = time.time()

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    print(f"[INFO] Checkpoint loaded. State dict keys: {len(state_dict)}")

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"[INFO] Weights loaded in {time.time() - start_time:.2f}s")

    return model


def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    """Applies L1 Unstructured Pruning globally to all convolutional and linear layers."""
    print(f"[INFO] Applying {amount * 100:.1f}% global unstructured pruning...")
    start_time = time.time()

    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, "weight"))

    print(f"[INFO] Found {len(parameters_to_prune)} Conv/Linear layers to prune.")

    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )

    for module, name in parameters_to_prune:
        prune.remove(module, name)

    print(f"[INFO] Pruning completed in {time.time() - start_time:.2f}s")

    return model


def evaluate_accuracy(model: nn.Module, dataloader, device: torch.device):
    """Evaluates the top-1 accuracy of the model over the full dataset."""
    print("[INFO] Starting accuracy evaluation...")
    start_time = time.time()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"[INFO] Accuracy evaluation: "
                    f"batch={batch_idx + 1}, samples={total}"
                )

    accuracy = correct / total if total > 0 else 0.0

    print(
        f"[INFO] Accuracy evaluation complete: "
        f"correct={correct}, total={total}, accuracy={accuracy:.4f}, "
        f"time={time.time() - start_time:.2f}s"
    )

    return accuracy


def extract_features(
    model: nn.Module, dataloader, device: torch.device, max_batches: int = 10
):
    """Extracts flattened activations from the penultimate layer for CKA."""
    print(f"[INFO] Extracting features from {max_batches} batches...")
    start_time = time.time()

    features = []

    def hook(module, input, output):
        features.append(output.flatten(start_dim=1).detach())

    target_layer = list(model.modules())[-2]
    print(f"[INFO] Feature extraction target layer: {target_layer.__class__.__name__}")

    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            inputs = inputs.to(device)
            model(inputs)

            print(f"[INFO] Feature extraction batch {i + 1}/{max_batches}")

    handle.remove()

    features = torch.cat(features, dim=0)

    print(
        f"[INFO] Feature extraction complete: "
        f"shape={tuple(features.shape)}, "
        f"time={time.time() - start_time:.2f}s"
    )

    return features


def compare_CKA(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between two feature matrices."""
    print(
        f"[INFO] Computing CKA: "
        f"features_x={tuple(features_x.shape)}, "
        f"features_y={tuple(features_y.shape)}"
    )

    start_time = time.time()

    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)

    dot_prod_xx = torch.norm(features_x.T @ features_x, p="fro")
    dot_prod_yy = torch.norm(features_y.T @ features_y, p="fro")
    dot_prod_xy = torch.norm(features_y.T @ features_x, p="fro")

    cka_score = (dot_prod_xy**2) / (dot_prod_xx * dot_prod_yy)
    cka_score = cka_score.item()

    print(
        f"[INFO] CKA complete: score={cka_score:.6f}, "
        f"time={time.time() - start_time:.2f}s"
    )

    return cka_score


# =========================================================
# Main Post-Processing Routine
# =========================================================
def find_experiment_checkpoints(model_name, dataset_name, pre_epochs, post_epochs):
    """Finds all relevant checkpoints for the given experiment configuration."""
    base_pattern = f"../Tranfer/{model_name}_{dataset_name}_*epochs{post_epochs}_*pre*{pre_epochs}"
    
    return {
        "before": glob.glob(f"{base_pattern}/checkpoints/final_JF_Control.pt"),
        "collapsed": glob.glob(f"{base_pattern}/checkpoints/final_JF_Dynamic_Region_All_Combined.pt"),
        "after": glob.glob(f"{base_pattern}/checkpoints/final_JF_Control_Continuted.pt")
    }

def load_collapse_regions(model_name, dataset_name, pre_epochs, post_epochs):
    """Loads and formats the discovered structural regions from the JSON map."""
    json_filename = (
        f"../Tranfer/{model_name}_{dataset_name}_"
        f"epochs{post_epochs}_pretrain{pre_epochs}_JF_discovered_regions.json"
    )
    
    print(f"[INFO] Loading discovered regions: {json_filename}")
    if not os.path.isfile(json_filename):
        raise FileNotFoundError(f"Discovered regions file not found: {json_filename}")

    with open(json_filename, "r") as f:
        discovered_regions = json.load(f)

    json_to_collapse = discovered_regions.get("Dynamic_Region_All_Combined")
    if not json_to_collapse:
        raise ValueError(f"No combined collapse regions found in {json_filename}.")
        
    print(f"[INFO] Found {len(json_to_collapse)} collapse regions.")
    
    return {f"Region_{i}": pair for i, pair in enumerate(json_to_collapse)}

def process_checkpoint(model_before_path, checkpoints, args, device):
    """Executes the full loading, collapsing, pruning, and evaluation pipeline."""
    checkpoint_start_time = time.time()
    
    # 1. Parse Context & Load Baseline
    print("[INFO] Parsing experiment directory...")
    model_name, dataset_name, epochs_str, pre_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path)
    
    print("[INFO] Initializing baseline model...")
    base_model, test_loader, dummy_input = initialize_architecture(model_name, dataset_name)
    model_before = load_weights(base_model, model_before_path, device)

    # 2. Load Original Post-Finetuning Model
    print("[INFO] Loading original post-finetuning model...")
    model_after_path = find_matching_checkpoint(model_before_path, checkpoints["after"])
    original_after_base, _, _ = initialize_architecture(model_name, dataset_name)
    original_after_model = load_weights(original_after_base, model_after_path, device)

    # 3. Execute Structural Collapse
    print("[INFO] Preparing and executing structural collapse...")
    full_collapsed_base, _, _ = initialize_architecture(model_name, dataset_name)
    compression_dict = load_collapse_regions(model_name, dataset_name, args.pre, args.post)
    
    collapse_start_time = time.time()
    full_collapsed_structure = collapse_only(
        model=full_collapsed_base,
        compression_set=compression_dict,
        input_shape=dummy_input.shape,
        device=device,
        dry_run=False,
        debug=False,
        handle_skips=True,
    )
    print(f"[INFO] Structural collapse completed in {time.time() - collapse_start_time:.2f}s")

    # 4. Load weights into collapsed structure
    collapsed_model_path = find_matching_checkpoint(model_before_path, checkpoints["collapsed"])
    full_collapsed_model_ready = load_weights(full_collapsed_structure, collapsed_model_path, device)

    # 5. Calculate Dynamic Sparsity & Apply Unstructured Pruning
    print("[INFO] Calculating dynamic sparsity to match collapse...")
    original_param_count = sum(p.numel() for p in model_before.parameters())
    collapsed_param_count = sum(p.numel() for p in full_collapsed_model_ready.parameters())
    
    target_sparsity = 1.0 - (collapsed_param_count / original_param_count)
    print(f"[INFO] Target unstructured sparsity calculated at: {target_sparsity * 100:.2f}%")

    unstructured_pruning_model = apply_unstructured_pruning(model_before, amount=float(target_sparsity))

    # 6. Evaluate Accuracies
    print("[INFO] Evaluating model accuracies...")
    acc_original_after = evaluate_accuracy(original_after_model, test_loader, device)
    acc_unstructured = evaluate_accuracy(unstructured_pruning_model, test_loader, device)
    acc_collapsed = evaluate_accuracy(full_collapsed_model_ready, test_loader, device)

    print(f"[RESULT] Accuracy Original:     {acc_original_after:.4f}")
    print(f"[RESULT] Accuracy Unstructured: {acc_unstructured:.4f}")
    print(f"[RESULT] Accuracy Collapsed:    {acc_collapsed:.4f}")

    # 7. Extract Features & Compute CKA
    print("[INFO] Extracting features and computing CKA...")
    features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
    features_original = extract_features(original_after_model, test_loader, device)
    features_collapsed = extract_features(full_collapsed_model_ready, test_loader, device)

    print("[INFO] Computing CKA: original vs unstructured...")
    cka_unstructured = compare_CKA(features_original, features_unstructured)

    print("[INFO] Computing CKA: original vs collapsed...")
    cka_collapsed = compare_CKA(features_original, features_collapsed)

    print(f"[RESULT] CKA Unstructured: {cka_unstructured:.6f}")
    print(f"[RESULT] CKA Collapsed:    {cka_collapsed:.6f}")
    print(f"[INFO] Checkpoint completed in {time.time() - checkpoint_start_time:.2f}s")

    return {
        "model": model_name,
        "dataset": dataset_name,
        "model_before": model_before_path,
        "intermediate_epoch": ckpt_epoch,
        "acc_original_after": acc_original_after,
        "acc_unstructured": acc_unstructured,
        "acc_collapsed": acc_collapsed,
        "cka_unstructured": cka_unstructured,
        "cka_collapsed": cka_collapsed,
    }

def main():
    parser = argparse.ArgumentParser(description="CKA comparison post-processing")
    parser.add_argument("--model", required=True, help="Model name, e.g. VGG16")
    parser.add_argument("--pre", type=int, default=300, help="pre collapse epochs")
    parser.add_argument("--post", type=int, default=100, help="post collapse epochs")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--output", default="cka_comparison_results.csv", help="Output CSV filename")
    args = parser.parse_args()

    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 70)
    print("Starting CKA comparison post-processing")
    print(f"Model: {args.model} | Dataset: {args.dataset} | Device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # 1. Discover Checkpoints
    print("[INFO] Searching for model checkpoints...")
    checkpoints = find_experiment_checkpoints(args.model, args.dataset, args.pre, args.post)
    
    print(f"[INFO] Found {len(checkpoints['before'])} 'before' checkpoints.")
    print(f"[INFO] Found {len(checkpoints['collapsed'])} collapsed checkpoints.")
    print(f"[INFO] Found {len(checkpoints['after'])} 'after' checkpoints.")

    results = []

    # 2. Process Checkpoints
    for idx, model_before_path in enumerate(checkpoints['before'], start=1):
        print("\n" + "=" * 70)
        print(f"Processing checkpoint {idx}/{len(checkpoints['before'])}")
        print(f"Path: {model_before_path}")
        print("=" * 70)

        try:
            result_metrics = process_checkpoint(model_before_path, checkpoints, args, device)
            results.append(result_metrics)
        except Exception as e:
            print(f"[ERROR] Failed processing checkpoint: {model_before_path}")
            print(f"[ERROR] {type(e).__name__}: {e}")

    # 3. Save Results
    print("\n" + "=" * 70)
    print("[INFO] Saving final results...")
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Results successfully saved to '{args.output}'")
    print(f"[INFO] Total successful experiments: {len(results)}")
    print(f"[INFO] Total runtime: {time.time() - total_start_time:.2f}s")
    
    print("\n" + "=" * 70 + "\nFINAL RESULTS\n" + "=" * 70)
    print(df.to_string())

if __name__ == "__main__":
    main()
