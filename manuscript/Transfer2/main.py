import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd

# =========================================================
# Utility Functions
# =========================================================

def parse_epoch_number(filepath: str) -> int:
    """Extracts the epoch number from the checkpoint filename."""
    match = re.search(r'epoch(\d+)\.pt', filepath)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse epoch number from {filepath}")

def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    """
    Matches the 'after' or 'collapsed' checkpoint to the 'before' checkpoint
    by ensuring they belong to the same base experiment directory.
    """
    # Assumes structure: <base_experiment_dir>/checkpoints/<model_file>.pt
    base_dir = before_path.split("/checkpoints/")[0]
    
    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            return candidate
            
    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")

def load_model(filepath: str, device: torch.device):
    """
    Loads a PyTorch model checkpoint. 
    Note: You must instantiate your base architecture (e.g., from transfer.py) 
    before loading the state_dict. This returns the loaded state dict for now.
    """
    checkpoint = torch.load(filepath, map_location=device)
    # Handle both wrapped dictionaries and raw state dicts
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint

def apply_unstructured_pruning(model: nn.Module, amount: float = 0.2) -> nn.Module:
    """
    Applies L1 Unstructured Pruning globally to all convolutional and linear layers.
    Note: The pseudo-code mentioned training for `epochs`. If active finetuning 
    is required post-pruning, integrate your training loop from transfer.py here.
    """
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    return model

def extract_features(model: nn.Module, dataloader, device: torch.device):
    """Extracts flattened activations from the penultimate layer for CKA."""
    model.eval()
    features = []
    
    # Hook to capture features before the final classifier
    def hook(module, input, output):
        features.append(output.flatten(start_dim=1).detach())
        
    # Assuming standard architectures, attach hook to average pooling or final block
    # Modify the target module based on your specific architecture from "collapse.py"
    target_layer = list(model.modules())[-2] 
    handle = target_layer.register_forward_hook(hook)
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            model(inputs)
            
    handle.remove()
    return torch.cat(features, dim=0)

def compare_CKA(features_x: torch.Tensor, features_y: torch.Tensor) -> float:
    """
    Computes Linear Centered Kernel Alignment (CKA) between two feature matrices.
    """
    # Center the features
    features_x = features_x - features_x.mean(dim=0, keepdim=True)
    features_y = features_y - features_y.mean(dim=0, keepdim=True)
    
    # Compute dot products
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
    
    # Dummy dataloader placeholder - replace with load_dataset() from transfer.py
    dummy_dataloader = [(torch.randn(16, 3, 32, 32), torch.randint(0, 10, (16,)))]
    results = []

    # 1. Find all model checkpoints
    original_model_before = glob.glob("../Transfer/*/checkpoints/JF_Control_Continuted_epoch*.pt")
    full_collapsed_model = glob.glob("../Transfer/*/checkpoints/final_JF_Dynamic_Region_All_Combined_quant.pt")
    original_model_after = glob.glob("../Transfer/*/checkpoints/final_JF_Control_Continuted.pt")

    print(f"Found {len(original_model_before)} 'before' checkpoints to process.")

    # 2. For each "before" checkpoint
    for model_before_path in original_model_before:
        print(f"Processing: {model_before_path}")
        
        # 2a. Get the number of epochs from the checkpoint name
        epochs = parse_epoch_number(model_before_path)

        # 2b. Apply unstructured pruning
        # NOTE: You must instantiate your base model architecture here before loading state
        # base_model = MyModelClass().to(device)
        # base_model.load_state_dict(load_model(model_before_path, device))
        base_model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(device) # Placeholder
        
        unstructured_pruning_model = apply_unstructured_pruning(base_model, amount=0.2)
        
        # 2c. Find/load the corresponding original model after
        model_after_path = find_matching_checkpoint(model_before_path, original_model_after)
        # original_after_model = MyModelClass().to(device)
        # original_after_model.load_state_dict(load_model(model_after_path, device))
        original_after_model = nn.Sequential(nn.Conv2d(3, 64, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(device) # Placeholder

        # 2d. Find/load the corresponding fully collapsed model
        collapsed_model_path = find_matching_checkpoint(model_before_path, full_collapsed_model)
        # full_collapsed = MyModelClass().to(device)
        # full_collapsed.load_state_dict(load_model(collapsed_model_path, device))
        full_collapsed = nn.Sequential(nn.Conv2d(3, 64, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)).to(device) # Placeholder

        # Extract features for CKA
        features_unstructured = extract_features(unstructured_pruning_model, dummy_dataloader, device)
        features_original = extract_features(original_after_model, dummy_dataloader, device)
        features_collapsed = extract_features(full_collapsed, dummy_dataloader, device)

        # 2e. Compare all models against the original model after training using CKA
        cka_unstructured = compare_CKA(features_original, features_unstructured)
        cka_collapsed = compare_CKA(features_original, features_collapsed)

        # 2f. Store results
        results.append({
            "model_before": model_before_path,
            "epochs": epochs,
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