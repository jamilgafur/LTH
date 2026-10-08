import os
import glob
import re
import argparse
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd

# Replicating imports from your pipeline to instantiate dynamically
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from utils import load_dataset 

# =========================================================
# Utility Functions
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Standalone CKA Post-Processing")
    parser.add_argument("--model", type=str, required=True, help="Model architecture")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    return parser.parse_args()

def parse_directory_info(filepath: str):
    base_dir = filepath.split("/checkpoints/")[0].split("/")[-1]
    match = re.search(r'epoch(\d+)\.pt', filepath)
    epochs = int(match.group(1)) if match else 0
    return epochs, base_dir

def initialize_architecture(model_name: str, dataset_name: str):
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    model_kwargs = {"num_classes": num_classes}
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    model = model_class(**model_kwargs)
    return model, test_loader

def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    base_dir = before_path.split("/checkpoints/")[0]
    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            return candidate
    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")

def load_weights(model: nn.Module, filepath: str, device: torch.device):
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
    )
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    return model

def extract_features(model: nn.Module, dataloader, device: torch.device, num_batches: int = 10):
    features = []
    def hook(module, input, output):
        features.append(output.flatten(start_dim=1).detach())
        
    target_layer = list(model.modules())[-2] 
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= num_batches: break
            inputs = inputs.to(device)
            model(inputs)
            
    handle.remove()
    return torch.cat(features, dim=0)

def compare_CKA(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
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
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    print(f"--- Starting CKA Post-Processing for {args.model} on {args.dataset} ---")

    # 1. Find all model checkpoints specific to this job's model and dataset
    search_pattern = f"{args.model}_{args.dataset}_*"
    original_model_before = glob.glob(f"{search_pattern}/checkpoints/JF_Control_Continuted_epoch*.pt")
    full_collapsed_model = glob.glob(f"{search_pattern}/checkpoints/final_JF_Dynamic_Region_All_Combined_quant.pt")
    original_model_after = glob.glob(f"{search_pattern}/checkpoints/final_JF_Control_Continuted.pt")

    print(f"Found {len(original_model_before)} 'before' checkpoints to process.")

    # 2. Process each checkpoint
    for model_before_path in original_model_before:
        print(f"\nProcessing: {model_before_path}")
        epochs, _ = parse_directory_info(model_before_path)
        
        base_model, test_loader = initialize_architecture(args.model, args.dataset)
        
        model_before = load_weights(base_model, model_before_path, device)
        unstructured_pruning_model = apply_unstructured_pruning(model_before, amount=0.2)
        
        model_after_path = find_matching_checkpoint(model_before_path, original_model_after)
        model_after_base, _ = initialize_architecture(args.model, args.dataset)
        original_after_model = load_weights(model_after_base, model_after_path, device)

        collapsed_model_path = find_matching_checkpoint(model_before_path, full_collapsed_model)
        model_collapsed_base, _ = initialize_architecture(args.model, args.dataset)
        full_collapsed = load_weights(model_collapsed_base, collapsed_model_path, device)

        print("Extracting feature representations...")
        features_unstructured = extract_features(unstructured_pruning_model, test_loader, device)
        features_original = extract_features(original_after_model, test_loader, device)
        features_collapsed = extract_features(full_collapsed, test_loader, device)

        print("Computing CKA similarities...")
        cka_unstructured = compare_CKA(features_original, features_unstructured)
        cka_collapsed = compare_CKA(features_original, features_collapsed)

        results.append({
            "model": args.model,
            "dataset": args.dataset,
            "model_before": model_before_path,
            "epochs": epochs,
            "cka_unstructured": cka_unstructured,
            "cka_collapsed": cka_collapsed,
        })

    # 3. Analyze / save results locally
    df = pd.DataFrame(results)
    output_csv = f"cka_results_{args.model}_{args.dataset}.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nResults successfully saved to '{output_csv}'")

if __name__ == "__main__":
    main()