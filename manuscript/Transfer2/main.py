import os
import glob
import re
import json
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
from .collapse import collapse_only

# =========================================================
# Utility Functions
# =========================================================

def parse_directory_context(filepath: str):
    """
    Extracts the model, dataset, epoch budget, and pretrain budget from the directory name.
    Example: ../Transfer/XceptionNet_tinyimagenet_None_epochs100_pretrain300/checkpoints/...
    """
    base_dir = filepath.split("/checkpoints/")[0].split("/")[-1]
    parts = base_dir.split("_")
    
    model_name = parts[0]
    dataset_name = parts[1]
    
    epochs_str = next(p for p in parts if p.startswith("epochs"))
    pretrain_str = next(p for p in parts if p.startswith("pretrain"))
    
    match = re.search(r'epoch(\d+)\.pt', filepath)
    ckpt_epoch = int(match.group(1)) if match else 0
    
    return model_name, dataset_name, epochs_str, pretrain_str, ckpt_epoch, base_dir

def initialize_architecture(model_name: str, dataset_name: str):
    """Dynamically loads the dataset and initializes the correct model architecture."""
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    
    model_kwargs = {"num_classes": num_classes}
    # Capture a dummy batch for collapse input shape inference
    dummy_input = next(iter(train_loader))[0][0:1] 
    
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    model = model_class(**model_kwargs)
    
    return model, test_loader, dummy_input

def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    """Matches the corresponding checkpoint within the same base experiment directory."""
    base_dir = before_path.split("/checkpoints/")[0]
    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            return candidate
    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")

def load_weights(model: nn.Module, filepath: str, device: torch.device):
    """Loads a PyTorch model checkpoint state_dict into the instantiated model."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    """Applies L1 Unstructured Pruning globally to all convolutional and linear layers."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    return model

def extract_features(model: nn.Module, dataloader, device: torch.device, max_batches: int = 10):
    """Extracts flattened activations from the penultimate layer for CKA."""
    features = []
    def hook(module, input, output):
        features.append(output.flatten(start_dim=1).detach())
        
    target_layer = list(model.modules())[-2] 
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= max_batches: break
            inputs = inputs.to(device)
            model(inputs)
            
    handle.remove()
    return torch.cat(features, dim=0)

def compare_CKA(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """Computes Linear Centered Kernel Alignment (CKA) between two feature matrices."""
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)
    
    dot_prod_xx = torch.norm(features_x.T @ features_x, p='fro')
    dot_prod_yy = torch.norm(features_y.T @ features_y, p='fro')
    dot_prod_xy = torch.norm(features_y.T @ features_x, p='fro')
    
    cka_score = (dot_prod_xy ** 2) / (dot_prod_xx * dot_prod_yy)
    return cka_score.item()

# =========================================================
# Main Post-Processing Routine
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # 1. Find all model checkpoints
    original_model_before = glob.glob("../Transfer/*/checkpoints/JF_Control_Continuted_epoch*.pt")
    full_collapsed_model = glob.glob("../Transfer/*/checkpoints/final_JF_Dynamic_Region_All_Combined_quant.pt")
    original_model_after = glob.glob("../Transfer/*/checkpoints/final_JF_Control_Continuted.pt")

    print(f"Found {len(original_model_before)} 'before' checkpoints to process.")

    # 2. For each "before" checkpoint
    for model_before_path in original_model_before:
        print(f"\nProcessing: {model_before_path}")
        
        # 2a. Parse directory string to identify architecture and budgets
        model_name, dataset_name, epochs_str, pretrain_str, ckpt_epoch, base_dir = parse_directory_context(model_before_path)
        
        # 2b. Apply unstructured pruning to the intermediate baseline
        base_model, test_loader, dummy_input = initialize_architecture(model_name, dataset_name)
        model_before = load_weights(base_model, model_before_path, device)
        unstructured_pruning_model = apply_unstructured_pruning(model_before, amount=0.2)
        
        # 2c. Find/load the corresponding original model after finetuning
        model_after_path = find_matching_checkpoint(model_before_path, original_model_after)
        original_after_base, _, _ = initialize_architecture(model_name, dataset_name)
        original_after_model = load_weights(original_after_base, model_after_path, device)

        # 2d. Find/load the corresponding fully collapsed model
        collapsed_model_path = find_matching_checkpoint(model_before_path, full_collapsed_model)
        full_collapsed_base, _, _ = initialize_architecture(model_name, dataset_name)
        
        # Locate the JSON map generated during the discovery phase
        # Format: <Model>_<dataset>_epochs<X>_pretrain<Y>_JF_discovered_regions.json
        json_filename = f"{model_name}_{dataset_name}_{epochs_str}_{pretrain_str}_JF_discovered_regions.json"
        
        with open(json_filename, "r") as f:
            discovered_regions = json.load(f)
            
        json_to_collapse = discovered_regions.get("Dynamic_Region_All_Combined")
        
        if not json_to_collapse:
            print(f"[WARN] No combined collapse regions found in {json_filename}. Skipping CKA collapse comparison.")
            continue

        # Execute structural collapse BEFORE loading the finetuned collapsed weights
        full_collapsed_structure = collapse_only(
            model=full_collapsed_base,
            compression_set={"Dynamic_Region_All_Combined": json_to_collapse},
            input_shape=dummy_input.shape,
            device=device,
            dry_run=False,
            debug=False,
            handle_skips=True
        )
        
        full_collapsed_model_ready = load_weights(full_collapsed_structure, collapsed_model_path, device)

        # 2e. Compare all models against the original model after training using CKA
        print("Extracting features and computing CKA...")
        features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
        features_original = extract_features(original_after_model, test_loader, device)
        features_collapsed = extract_features(full_collapsed_model_ready, test_loader, device)

        cka_unstructured = compare_CKA(features_original, features_unstructured)
        cka_collapsed = compare_CKA(features_original, features_collapsed)

        # 2f. Store results
        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "model_before": model_before_path,
            "intermediate_epoch": ckpt_epoch,
            "cka_unstructured": cka_unstructured,
            "cka_collapsed": cka_collapsed,
        })

    # 3. Analyze / save results
    df = pd.DataFrame(results)
    df.to_csv("cka_comparison_results.csv", index=False)
    print("\nResults successfully saved to 'cka_comparison_results.csv'")
    print(df.to_string())

if __name__ == "__main__":
    main()