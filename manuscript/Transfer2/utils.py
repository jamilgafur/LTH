# utils.pt
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import shutil
import sys
import json
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis
import time
from torchinfo import summary
import numpy as np
from pyPrune.utils import load_cifar10, load_cifar100, load_tiny_imagenet, load_imagenet
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.utils.prune as prune
import torch
import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
import torch.nn as nn

from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet

import argparse
import glob
import re
from main import PowerTracker
# =========================================================
# Utility Functions
# =========================================================

def parse_directory_context(filepath: str, expected_model: str, expected_dataset: str):
    """Safely extracts context without relying on naive underscore splitting."""
    base_dir = filepath.split("/checkpoints/")[0].split("/")[-1]
    parts = base_dir.split("_")
    
    # Safely find epochs and pretrain tokens
    epochs_str = next((p for p in parts if p.startswith("epochs")), "epochs100")
    pretrain_str = next((p for p in parts if p.startswith("pretrain")), "pretrain300")
    
    match = re.search(r"epoch(\d+)\.pt", filepath)
    ckpt_epoch = int(match.group(1)) if match else 0
    
    # Return the explicitly provided model and dataset names
    return expected_model, expected_dataset, epochs_str, pretrain_str, ckpt_epoch, base_dir

def find_matching_checkpoint(before_path: str, candidate_paths: list) -> str:
    """Matches the corresponding checkpoint within the same base experiment directory."""
    base_dir = before_path.split("/checkpoints/")[0]

    print(f"[INFO] Searching for checkpoint matching: {base_dir}")

    for candidate in candidate_paths:
        if candidate.startswith(base_dir):
            print(f"[INFO] Found matching checkpoint: {candidate}")
            return candidate

    raise FileNotFoundError(f"No matching checkpoint found for {base_dir}")

import re




def load_weights(model: nn.Module, filepath: str, device: torch.device):
    """Loads a PyTorch model checkpoint state_dict into the instantiated model with extreme logging."""
    print(f"[INFO] Loading weights: {filepath}")
    start_time = time.time()

    try:
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        raw_state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to open checkpoint file {filepath}. Error: {e}")
        raise e

    # --- KEY MAPPING INTERCEPT ---
    state_dict = {}
    for key, value in raw_state_dict.items():
        # Fix ConvNeXt collapse mismatch & XceptionNet DataParallel artifacts
        new_key = key.replace("conv_dw", "conv_g1x1").replace("module.", "")
        
        # --- MOBILENET FIX ---
        # Flatten unescalated nested keys from pyPrune (e.g., features.12.0.conv_g1x1 -> features.12.conv_g1x1)
        if model.__class__.__name__ == "MobileNet":
            new_key = re.sub(r'(features\.\d+)\.(?:depthwise|pointwise|0|1)\.(conv_1x1|conv_g1x1|bn|adaptive_pool|relu)', r'\1.\2', new_key)
        state_dict[new_key] = value

    print(f"[INFO] Checkpoint extracted. Candidate state dict keys: {len(state_dict)}")

    # --- SHAPE MISMATCH DIAGNOSTICS & DYNAMIC MODULE UPGRADE ---
    model_state = model.state_dict()
    mismatched_keys = []
    
    for k, v in state_dict.items():
        if k in model_state:
            expected_shape = model_state[k].shape
            checkpoint_shape = v.shape
            if expected_shape != checkpoint_shape:
                print(f"[SHAPE MISMATCH FATAL] Layer: '{k}'")
                print(f"   -> Checkpoint holds : {checkpoint_shape}")
                print(f"   -> Model expects    : {expected_shape}")
                
                # --- DYNAMIC MODULE HOT-SWAPPING ---
                if len(checkpoint_shape) == 4 and len(expected_shape) == 4 and checkpoint_shape[:2] == expected_shape[:2] and k.endswith(".weight"):
                    layer_path = k[:-7]  # strip '.weight'
                    try:
                        # Traverse model hierarchy to find the parent module and target Conv2d
                        parts = layer_path.split('.')
                        parent_mod = model
                        for part in parts[:-1]:
                            parent_mod = parent_mod[int(part)] if part.isdigit() else getattr(parent_mod, part)
                        attr_name = parts[-1]
                        old_conv = parent_mod[int(attr_name)] if attr_name.isdigit() else getattr(parent_mod, attr_name)
                        
                        if isinstance(old_conv, nn.Conv2d):
                            new_k = (checkpoint_shape[2], checkpoint_shape[3])
                            new_pad = (new_k[0] // 2, new_k[1] // 2)
                            print(f"[INFO] Auto-resolving mismatch: Upgrading '{layer_path}' kernel from {old_conv.kernel_size} to {new_k} (padding={new_pad})")
                            
                            new_conv = nn.Conv2d(
                                in_channels=old_conv.in_channels,
                                out_channels=old_conv.out_channels,
                                kernel_size=new_k,
                                stride=old_conv.stride,
                                padding=new_pad,
                                groups=old_conv.groups,
                                bias=(old_conv.bias is not None)
                            ).to(device)
                            
                            if attr_name.isdigit():
                                parent_mod[int(attr_name)] = new_conv
                            else:
                                setattr(parent_mod, attr_name, new_conv)
                            continue
                    except Exception as e:
                        print(f"[WARN] Dynamic module upgrade failed for '{layer_path}': {e}")
                
                mismatched_keys.append(k)

    if mismatched_keys:
        print(f"\n[CRITICAL WARNING] Dropping {len(mismatched_keys)} keys due to shape mismatches to prevent hard crash!")
        for k in mismatched_keys:
            del state_dict[k]

    try:
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        
        if incompatible_keys.missing_keys:
            print(f"\n[CRITICAL WARNING] {len(incompatible_keys.missing_keys)} Missing keys (Expected by Model, missing in Checkpoint File):")
            for key in incompatible_keys.missing_keys:
                print(f"   -> {key}")
                
        if incompatible_keys.unexpected_keys:
            print(f"\n[CRITICAL WARNING] {len(incompatible_keys.unexpected_keys)} Unexpected keys (Found in Checkpoint File, not in Model):")
            for key in incompatible_keys.unexpected_keys:
                print(f"   -> {key}")
                
    except RuntimeError as e:
        print(f"\n[CRITICAL ERROR] PyTorch RuntimeError during load_state_dict: {e}")
        raise e

    model.to(device)
    model.eval()
    print(f"[INFO] Weights loaded in {time.time() - start_time:.2f}s\n")
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


# =========================================================
# Model Evaluation with Energy Tracking
# =========================================================

def evaluate_accuracy(model: nn.Module, dataloader, device: torch.device, power_interval: int = 1):
    """Evaluates top-1 accuracy while profiling power draw (W) and energy (J)."""
    print("[INFO] Starting accuracy and energy evaluation...")
    model.to(device)
    model.eval()
    correct = 0
    total = 0

    with PowerTracker(device=device, interval_s=power_interval) as tracker:
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (batch_idx + 1) % 50 == 0:
                    print(f"[INFO] Accuracy evaluation: batch={batch_idx + 1}, samples={total}")

    avg_watts, total_joules, elapsed_sec = tracker.get_results()
    accuracy = correct / total if total > 0 else 0.0

    print(
        f"[INFO] Evaluation complete: acc={accuracy:.4f}, "
        f"power={avg_watts:.2f}W, energy={total_joules:.2f}J, time={elapsed_sec:.2f}s"
    )
    return accuracy, avg_watts, total_joules

# =========================================================
# Iterative Magnitude Pruning (IMP) Training Loop
# =========================================================

def train_imp(model, train_loader, device, epochs, target_sparsity, save_path, checkpoint_path=None, power_interval: int = 1):
    """Trains on-the-fly using Iterative Magnitude Pruning with SLURM-safe checkpointing."""
    print(f"\n[INFO] Starting Iterative Magnitude Pruning (IMP) for {epochs} epochs...")
    
    # Default checkpoint path based on save_path if none is provided
    if checkpoint_path is None:
        base, ext = os.path.splitext(save_path)
        checkpoint_path = f"{base}_ resume_ckpt{ext}"

    # 1. Move model to device BEFORE initializing pruning and optimizer
    model.to(device)
    
    # 2. If final model already exists, load and return immediately
    if os.path.exists(save_path):
        print(f"[INFO] Final trained model already exists at {save_path}. Loading and skipping training...")
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return model

    # 3. Collect parameters to prune
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    pruning_steps = max(1, int(epochs * 0.75))
    incremental_amount = 1.0 - (1.0 - target_sparsity) ** (1.0 / pruning_steps)
    print(f"[INFO] Compounding prune rate: {incremental_amount * 100:.2f}% per step over {pruning_steps} steps.")

    # 4. Initialize pruning structures (`weight_orig` and `weight_mask`) BEFORE creating the optimizer.
    # This ensures the optimizer tracks the trainable `weight_orig` leaf tensors and allows 
    # load_state_dict() to match the pruned keys when resuming from a checkpoint.
    for module, name in parameters_to_prune:
        if not prune.is_pruned(module):
            prune.identity(module, name)

    # 5. Initialize optimizer and loss AFTER pruning hooks are attached
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 6. Check for existing in-progress checkpoint
    start_epoch = 0
    prev_joules = 0.0
    prev_time = 0.0

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Found existing checkpoint at {checkpoint_path}. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        prev_joules = checkpoint.get('total_joules', 0.0)
        prev_time = checkpoint.get('total_time', 0.0)
        print(f"[INFO] Resuming from Epoch {start_epoch + 1}/{epochs} (Previous time: {prev_time:.2f}s)")

    # 7. Main Training Loop
    model.train()
    interactive = sys.stderr.isatty()
    if start_epoch < epochs:
        with PowerTracker(device=device, interval_s=power_interval) as tracker:
            epoch_pbar = tqdm(
                range(start_epoch, epochs),
                desc="IMP",
                disable=not interactive,
            )
            for epoch in epoch_pbar:
                if epoch < pruning_steps:
                    prune.global_unstructured(
                        parameters_to_prune,
                        pruning_method=prune.L1Unstructured,
                        amount=incremental_amount
                    )

                loss_sum = torch.zeros((), device=device)
                batch_pbar = tqdm(
                    train_loader,
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    leave=False,
                    disable=not interactive,
                )
                for step, (inputs, targets) in enumerate(batch_pbar):
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    loss_sum += loss.detach()

                    # Throttled postfix refresh (bounded host syncs)
                    if not batch_pbar.disable and (step + 1) % 20 == 0:
                        batch_pbar.set_postfix({"loss": f"{loss_sum.item() / (step + 1):.4f}"})

                    # Aggressively delete tensors to free memory
                    del inputs, targets, outputs, loss

                epoch_loss = loss_sum.item() / len(train_loader)
                if not epoch_pbar.disable:
                    epoch_pbar.set_postfix({"loss": f"{epoch_loss:.4f}"})
                else:
                    print(f"[INFO] Epoch {epoch + 1}/{epochs} | Train Loss: {epoch_loss:.4f}")
                
                # Query current energy statistics so far in this session
                _, session_joules, session_time = tracker.get_results()
                
                # Save atomic checkpoint at the end of every epoch
                # Writing to a temp file first prevents corrupt checkpoints if SLURM kills the job mid-save
                temp_ckpt_path = checkpoint_path + ".tmp"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'total_joules': prev_joules + session_joules,
                    'total_time': prev_time + session_time,
                }, temp_ckpt_path)
                os.replace(temp_ckpt_path, checkpoint_path)
                
                # Force CUDA cache clear at the end of every epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        _, session_joules, session_time = tracker.get_results()
        total_joules = prev_joules + session_joules
        total_time = prev_time + session_time
        avg_watts = total_joules / total_time if total_time > 0 else 0.0
    else:
        total_joules = prev_joules
        total_time = prev_time
        avg_watts = total_joules / total_time if total_time > 0 else 0.0

    print(f"[INFO] IMP total training energy: {avg_watts:.2f}W avg, {total_joules:.2f}J total ({total_time:.2f}s)")

    # 8. Make pruning permanent (`weight_orig` * `weight_mask` -> `weight`)
    for module, name in parameters_to_prune:
        if prune.is_pruned(module):
            prune.remove(module, name)
        
    # 9. Save final clean weights and remove temporary resume checkpoint
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"[INFO] Final IMP model saved to {save_path}\n")
    
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return model
# =========================================================
# Architecture and Region Utilities
# =========================================================

def initialize_architecture(model_name: str, dataset_name: str):
    """Initializes architecture and returns data loaders and sample tensor."""
    print(f"[INFO] Initializing architecture: model={model_name}, dataset={dataset_name}")
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset_name, model_name)
    
    # Grab a batch of 2 to ensure BatchNorm doesn't crash on 1x1 spatial dimensions
    dummy_input = next(iter(train_loader))[0][0:2]
    print(f"[DEBUG] dummy_input shape generated for initialization: {dummy_input.shape}")
    
    # Pass 'one_batch' so custom architectures can dynamically calculate their FC layer sizes
    model_kwargs = {
        "num_classes": num_classes,
        "one_batch": dummy_input
    }
    
    if model_name == "InceptionNet":
        model_kwargs["aux_logits"] = False
        
    model_class = eval(model_name)
    
    try:
        print(f"[DEBUG] Attempting to initialize {model_name} with kwargs: {model_kwargs.keys()}")
        model = model_class(**model_kwargs)
        print(f"[DEBUG] {model_name} successfully initialized WITH one_batch.")
    except TypeError as e:
        print(f"[CRITICAL WARNING] {model_name} rejected 'one_batch' kwarg! Error: {e}")
        print(f"[DEBUG] Falling back to default initialization without 'one_batch'.")
        model_kwargs.pop("one_batch", None)
        model = model_class(**model_kwargs)

    # --- BRUTE-FORCE SHAPE FIX FOR VGG16 TINYIMAGENET ---
    # If the above failed and the classifier is still 512, we forcefully overwrite it.
    if model_name == "VGG16" and dataset_name.lower() in ["tinyimagenet", "tiny_imagenet"]:
        # Check if the first layer in the classifier (fc_1) has 512 in_features
        if hasattr(model, 'classifier') and isinstance(model.classifier[0], torch.nn.Linear):
            if model.classifier[0].in_features == 512:
                print("[CRITICAL WARNING] VGG16 initialized with 512 dims for TinyImageNet. Forcefully hot-swapping to 2048 dims!")
                model.classifier[0] = torch.nn.Linear(2048, 4096)
    # ----------------------------------------------------

    return model, train_loader, test_loader, dummy_input

def parse_epoch_tokens(path):
    """Extract (post_epochs, pre_epochs) from an experiment path's
    'epochs{N}_pretrain{M}' tokens. Returns (None, None) when absent.

    post_epochs  = the 'epochs' token   (post-collapse / finetune training length)
    pre_epochs   = the 'pretrain' token (pre-collapse / baseline training length)
    """
    base = path.split("/checkpoints/")[0]
    m_post = re.search(r"epochs(\d+)", base)
    m_pre = re.search(r"pretrain(\d+)", base)
    return (
        int(m_post.group(1)) if m_post else None,
        int(m_pre.group(1)) if m_pre else None,
    )


def _default_universe_donor_path(ckpt_path):
    """Map a scorer-universe checkpoint path to its default-universe donor.

    Scorer universes are named '{model}_{dataset}_{tag}_scorer{S}_epochs{E}_pretrain{P}';
    the default universe drops the '_scorer{S}' token. Returns the donor path if
    the input is inside a scorer universe, else None.
    """
    universe_dir = os.path.dirname(ckpt_path)
    ckpt_name = os.path.basename(ckpt_path)
    if os.path.basename(universe_dir) != "checkpoints":
        return None
    scorer_dir = os.path.dirname(universe_dir)
    donor_dir_name = re.sub(
        r"_scorer[^_]+(?=_epochs\d+_pretrain\d+)", "", os.path.basename(scorer_dir)
    )
    if donor_dir_name == os.path.basename(scorer_dir):
        return None  # not a scorer universe
    return os.path.join(os.path.dirname(scorer_dir), donor_dir_name, "checkpoints", ckpt_name)


def inherit_from_default_universe(ckpt_path):
    """If ckpt_path is missing but the default-universe sibling has it, copy it.

    IMP / Structural baselines train from the universe's 'before' model — which
    scorer universes inherit from the default universe — so these artifacts are
    scorer-independent. Copying keeps baselines byte-identical across universes
    and avoids retraining IMP/Structural per scorer.
    """
    if os.path.exists(ckpt_path):
        return ckpt_path
    donor = _default_universe_donor_path(ckpt_path)
    if donor and os.path.exists(donor):
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        shutil.copy2(donor, ckpt_path)
        print(f"[INFO] Inherited scorer-independent artifact from {donor}")
    return ckpt_path


def find_experiment_checkpoints(model_name, dataset_name, pre_epochs=None, post_epochs=None):
    """
    Finds model checkpoints across varied directory conventions (handles optional _None_ tags
    and case-insensitive dataset matching). Automatically falls back to Set_0 per base directory
    if Dynamic_Region_All_Combined is not present.

    Epoch configuration (epochsN / pretrainM tokens) is derived from the directory names
    themselves, so multiple training configurations can coexist: each before-checkpoint is
    matched to artifacts in its own base directory by find_matching_checkpoint(). Pass
    pre_epochs / post_epochs only as an explicit filter when a specific configuration
    should be selected.

    The collapsed bucket only matches whole-model artifacts (`*_whole.pt`) written by
    the Tranfer experiments: these contain the collapsed architecture plus trained
    weights and are loaded directly (no re-collapse step in analysis).
    """
    dataset_patterns = [dataset_name, dataset_name.lower(), dataset_name.capitalize()]

    def allowed(path):
        if pre_epochs is None and post_epochs is None:
            return True
        post, pre = parse_epoch_tokens(path)
        if post_epochs is not None and post != post_epochs:
            return False
        if pre_epochs is not None and pre != pre_epochs:
            return False
        return True

    before_ckpts = []
    after_ckpts = []
    collapsed_by_dir = {}

    for d_name in set(dataset_patterns):
        base_pattern = f"../Tranfer/{model_name}_{d_name}_*"

        # Original and Control check
        before_ckpts.extend(p for p in glob.glob(f"{base_pattern}/checkpoints/final_JF_Control.pt") if allowed(p))
        after_ckpts.extend(p for p in glob.glob(f"{base_pattern}/checkpoints/final_JF_Control_Continuted.pt") if allowed(p))

        # Collapsed whole-model artifacts; Combined preferred over Set_0 per base dir.
        for p in glob.glob(f"{base_pattern}/checkpoints/final_JF_Dynamic_Region_All_Combined_whole.pt"):
            if allowed(p):
                collapsed_by_dir.setdefault(p.split("/checkpoints/")[0], {})["combined"] = p
        for p in glob.glob(f"{base_pattern}/checkpoints/final_JF_Set_0_whole.pt"):
            if allowed(p):
                collapsed_by_dir.setdefault(p.split("/checkpoints/")[0], {})["set0"] = p

    collapsed_ckpts = [
        entry.get("combined") or entry["set0"]
        for _, entry in sorted(collapsed_by_dir.items())
        if entry.get("combined") or entry.get("set0")
    ]

    return {
        "before": sorted(list(set(before_ckpts))),
        "collapsed": sorted(list(set(collapsed_ckpts))),
        "after": sorted(list(set(after_ckpts)))
    }



def extract_features(
    model: nn.Module, dataloader, device: torch.device, max_batches: int = 10
):
    """Extracts features robustly by stripping the final classifier layer, avoiding broken hook bindings."""
    print(f"[INFO] Extracting features from {max_batches} batches (Classifier-free extraction)...")
    start_time = time.time()

    model.eval()
    model.to(device)

    # Clone and strip the final classifier/linear layer to get raw features directly
    feature_extractor = deepcopy(model)
    
    # Identify and remove the final classifier layer safely
    if hasattr(feature_extractor, 'fc') and isinstance(feature_extractor.fc, nn.Linear):
        feature_extractor.fc = nn.Identity()
    elif hasattr(feature_extractor, 'classifier'):
        if isinstance(feature_extractor.classifier, nn.Sequential):
            feature_extractor.classifier[-1] = nn.Identity()
        elif isinstance(feature_extractor.classifier, nn.Linear):
            feature_extractor.classifier = nn.Identity()
    else:
        # Fallback: remove the absolute last module if it's a Linear layer
        modules = list(feature_extractor.named_modules())
        for name, mod in reversed(modules):
            if isinstance(mod, nn.Linear):
                parent_name, subname = name.rsplit('.', 1) if '.' in name else ('', name)
                parent = feature_extractor if parent_name == '' else dict(feature_extractor.named_modules())[parent_name]
                if isinstance(parent, nn.Sequential):
                    parent[int(subname)] = nn.Identity()
                else:
                    setattr(parent, subname, nn.Identity())
                break

    features = []
    with torch.no_grad():
        for i, (inputs, _) in enumerate(dataloader):
            if i >= max_batches:
                break

            inputs = inputs.to(device)
            out = feature_extractor(inputs)
            
            # Flatten spatial dimensions if output is 4D (e.g., [Batch, Channels, H, W])
            if out.ndim > 2:
                out = out.flatten(start_dim=1)
                
            features.append(out.detach().cpu())
            print(f"[INFO] Feature extraction batch {i + 1}/{max_batches}")

    if not features:
        raise RuntimeError("[ERROR] Feature extraction captured zero batches!")

    features = torch.cat(features, dim=0).to(device)

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

# -======================
def plot_experiment_heuristics(model_name, dataset_name, stats_csv_path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # [✓] MOVED HERE: This breaks the circular import!
    from transfer import EXPERIMENTS 

    # Load the raw layer stats
    df_layers = pd.read_csv(stats_csv_path)
    layer_names = df_layers['Layer'].tolist()
    variances = dict(zip(df_layers['Layer'], df_layers['Variance']))
    activations = dict(zip(df_layers['Layer'], df_layers['Mean Activation']))

    exp_dict = EXPERIMENTS[model_name][dataset_name]
    
    exp_names, total_vars, avg_acts = [], [], []

    # Calculate Total Variance and Average Activation per experiment
    for exp_name, layer_range in exp_dict.items():
        if layer_range is None or exp_name == "Original Model":
            continue
            
        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        b_vars, b_acts = [], []
        
        for start_layer, end_layer in ranges:
            in_range = False
            for name in layer_names:
                if start_layer in name: in_range = True
                if in_range:
                    if name in variances: b_vars.append(variances[name])
                    if name in activations: b_acts.append(activations[name])
                if end_layer in name: break
                
        if b_vars and b_acts:
            exp_names.append(exp_name)
            total_vars.append(np.sum(b_vars)) # SUM of variance (Total Information)
            avg_acts.append(np.mean(b_acts))  # MEAN of activation (Average Volume)

    # Generate Plot
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    df_plot = pd.DataFrame({"Experiment": exp_names, "Total Variance": total_vars, "Mean Activation": avg_acts})

    # Top Plot: Mean Activation
    sns.barplot(data=df_plot, x="Experiment", y="Mean Activation", color="#4C72B0", edgecolor="black", ax=ax1)
    ax1.set_title(f"Heuristic Profiling by Target Region: {model_name}", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Avg Mean Activation", fontweight='bold')
    ax1.axhline(0, color='black', linewidth=1.5)

    # Bottom Plot: Total Variance
    sns.barplot(data=df_plot, x="Experiment", y="Total Variance", color="#C44E52", edgecolor="black", ax=ax2)
    ax2.set_ylabel("Total Sum of Variance", fontweight='bold')
    ax2.set_xlabel("Targeted Collapse Region", fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"runs/plots/{model_name}_heuristic_target_summary.png", dpi=300)
    print(f"Saved runs/plots/{model_name}_heuristic_target_summary.png")



# -------------------------
# Helper utilities
# -------------------------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def is_dict_like(x):
    return isinstance(x, dict)

def normalize_metrics(metrics):
    """
    Normalize incoming metrics into a dict[str -> dict] mapping for plotting functions.
    Accepts:
      - dict mapping experiment_name -> metrics (ideal)
      - list of dicts (will pick 'name'/'experiment' if present, else index-based)
      - single dict that might contain nested dicts
    Returns dict.
    """
    if is_dict_like(metrics):
        # If it looks like {exp_name: { ... }}, keep only dict values
        # If metrics itself is single experiment (contains final_accuracy etc), wrap it
        contains_nested = any(isinstance(v, dict) for v in metrics.values())
        if contains_nested:
            result = {k: v for k, v in metrics.items() if isinstance(v, dict)}
            # If result empty but metrics seems like one experiment record, wrap it
            if not result and metrics and all(k in metrics for k in ("accuracies", "losses", "param_count")):
                return {"metric_record": metrics}
            return result
        # fallback: treat as single experiment
        if all(k in metrics for k in ("accuracies", "losses", "param_count")):
            return {"metric_record": metrics}
        return {}
    elif isinstance(metrics, list):
        out = {}
        for i, entry in enumerate(metrics):
            if not is_dict_like(entry):
                continue
            name = entry.get("name") or entry.get("experiment") or f"exp_{i}"
            out[name] = entry
        return out
    else:
        return {}

def safe_get(d, key, default=None):
    if not is_dict_like(d):
        return default
    return d.get(key, default)

def timestamped_filename(base):
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    name, ext = os.path.splitext(base)
    return f"{name}_{t}{ext}" if ext else f"{base}_{t}"

def load_dataset(dataset_name, model_name="VGG16"):
    if model_name == "VGG16":
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

    elif model_name == "RegNetX_400MF":
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for RegNetX_400MF...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for RegNetX_400MF...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for RegNetX_400MF...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for RegNetX_400MF...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset for {model_name}: {dataset_name}")
    
    elif model_name == "InceptionNet":
        if dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for InceptionNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for InceptionNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100

        elif dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for InceptionNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10

        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for InceptionNet...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes

        else:
            raise ValueError(f"Unsupported dataset for {model_name}: {dataset_name}")
    
    elif model_name == "XceptionNet":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for XceptionNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for XceptionNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for XceptionNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200

    elif model_name == "MobileNet":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for MobileNet...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for MobileNet...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for MobileNet...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200
    elif model_name == "ConvNeXt":
        if dataset_name == "Cifar10":
            print("Loading CIFAR-10 data for ConvNeXt...")
            train_loader, test_loader = load_cifar10()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 10
        elif dataset_name == "Cifar100":
            print("Loading CIFAR-100 data for ConvNeXt...")
            train_loader, test_loader = load_cifar100()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 100
        elif dataset_name == "TinyImageNet" or dataset_name == "tinyimagenet":
            print("Loading Tiny ImageNet data for ConvNeXt...")
            train_loader, test_loader = load_tiny_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 200
        elif dataset_name == "ImageNet" or dataset_name == "imagenet":
            print("Loading ImageNet data for ConvNeXt...")
            train_loader, test_loader = load_imagenet()
            sample_input = next(iter(train_loader))[0]
            input_size = sample_input.shape[-2:]
            input_channels = sample_input.shape[1]
            num_classes = 1000  # ImageNet has 1000 classes
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return train_loader, test_loader, input_size, input_channels, num_classes
 
# -------------------------
# Benchmark Inference
# -------------------------
import torch
import time
from copy import deepcopy
from fvcore.nn import FlopCountAnalysis
from torch.utils.data import DataLoader

def benchmark_model(model, loader, device, num_batches=20, warmup_batches=5, quant=False):
    """
    Returns a dict of efficiency metrics measured over `num_batches` forwards:

        {
            "avg_time_s":      mean forward time per batch (seconds),
            "latency_ms":      avg_time_s * 1000,
            "throughput_imgs": batch_size / avg_time_s,
            "batch_size":      loader batch size used,
            "flops":           total FLOPs for one batch (fvcore, first batch),
            "feature_map_mb":  total activation bytes for one batch (MiB),
            "peak_mem_mb":     peak CUDA memory allocated during measurement (MiB),
                               None when CUDA is unavailable,
        }

    Notes:
    - Uses a local DataLoader with num_workers=0 to ensure forward runs in the main process
      (avoids worker deaths hiding OOMs).
    - Hooks only accumulate the number of bytes of feature maps (do NOT keep tensors).
    - If quant=True and CUDA is available, uses mixed precision (fp16) for forward pass.
    """
    from copy import deepcopy
    import torch
    import time
    from torch.utils.data import DataLoader
    from fvcore.nn import FlopCountAnalysis

    # clone model to avoid modifying original
    tempmodel = deepcopy(model)
    tempmodel.eval()
    tempmodel.to(device)

    times = []
    flops = 0
    total_feature_map_size_mb = 0.0

    # Build a single-process DataLoader
    dataset = getattr(loader, "dataset", None)
    batch_size = getattr(loader, "batch_size", 1)
    if dataset is None:
        data_iterable = loader
        def make_iterable():
            return iter(data_iterable)
    else:
        safe_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)
        def make_iterable():
            return iter(safe_loader)

    # Helper to register lightweight hooks that accumulate bytes
    def register_size_hooks(mod):
        acc = {"bytes": 0}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                try:
                    if isinstance(output, torch.Tensor):
                        acc["bytes"] += output.numel() * output.element_size()
                    elif isinstance(output, (list, tuple)):
                        for o in output:
                            if isinstance(o, torch.Tensor):
                                acc["bytes"] += o.numel() * o.element_size()
                except Exception:
                    pass
            return hook

        for _, m in mod.named_modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.AdaptiveAvgPool2d,
                              torch.nn.MaxPool2d, torch.nn.BatchNorm2d,
                              torch.nn.ReLU, torch.nn.Linear)):
                hooks.append(m.register_forward_hook(make_hook(None)))
        return hooks, acc

    # Warmup passes
    it = make_iterable()
    use_autocast = quant and device.type == 'cuda'
    for _ in range(warmup_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        xb = xb.to(device)
        with torch.no_grad():
            if use_autocast:
                with torch.cuda.amp.autocast():
                    _ = tempmodel(xb)
            else:
                _ = tempmodel(xb)

    # Reset peak stats if using CUDA
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    # Measurement passes
    it = make_iterable()
    for i in range(num_batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        xb = xb.to(device)

        # Attach hooks on first batch
        size_hooks = []
        size_acc = None
        if i == 0:
            size_hooks, size_acc = register_size_hooks(tempmodel)

        # Forward timing
        with torch.no_grad():
            if torch.cuda.is_available():
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                starter.record()
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        _ = tempmodel(xb)
                else:
                    _ = tempmodel(xb)
                ender.record()
                torch.cuda.synchronize()
                times.append(starter.elapsed_time(ender) / 1000.0)  # ms -> s
            else:
                start = time.time()
                if use_autocast:
                    with torch.cuda.amp.autocast():
                        _ = tempmodel(xb)
                else:
                    _ = tempmodel(xb)
                times.append(time.time() - start)

        # Capture total bytes for first batch
        if i == 0 and size_acc is not None:
            total_bytes = size_acc.get("bytes", 0)
            total_feature_map_size_mb = total_bytes / (1024 ** 2)
            try:
                flop_analyzer = FlopCountAnalysis(tempmodel, xb)
                flop_analyzer.unsupported_ops_warnings(False)  # Suppress aten::gelu/add noise
                flop_analyzer.uncalled_modules_warnings(False) # Suppress bypassed SmartIdentity warnings
                flops = flop_analyzer.total()
            except Exception as e:
                print(f"[DEBUG] FlopCountAnalysis failed on {device}: {e}")
                flops = 0

        # Remove hooks
        if size_hooks:
            for h in size_hooks:
                h.remove()

    # Read back peak CUDA memory accumulated over the measurement loop
    peak_mem_mb = None
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        except Exception:
            pass

    del tempmodel
    avg_time = sum(times) / len(times) if times else 0.0
    return {
        "avg_time_s": avg_time,
        "latency_ms": avg_time * 1000.0,
        "throughput_imgs": (batch_size / avg_time) if avg_time > 0 else 0.0,
        "batch_size": batch_size,
        "flops": flops,
        "feature_map_mb": total_feature_map_size_mb,
        "peak_mem_mb": peak_mem_mb,
    }

def describe_model(model, loader, device='cpu'):
    print("=" * 60)
    print("🔍 Model Summary (via torchinfo)")
    print("=" * 60)
    # summary(model, input_size=next(iter(loader))[0].shape, device=device)
    # layer_stats(model)
    print("=" * 60)


def calibrate_hyperparameters(df):
    """
    Analyzes the heuristic DataFrame to find optimal scaling factors.
    Returns a dict of tuned parameters: {'lambda_v', 'd_0'}
    """
    # 1. Calibrate Lambda (Variance Sensitivity)
    # We want the bottom 20% of layers (by variance) to have a Silence Score > 0.5
    # Formula: exp(-lambda * var_20th) = 0.5
    # Solve for lambda: lambda = -ln(0.5) / var_20th
    
    # Filter for valid variances (conv/linear layers only)
    variances = df[df['act_var'] > 0]['act_var']
    
    if variances.empty:
        return {'lambda_v': 10.0, 'd_0': 0.15} # Fallback defaults
        
    var_20th_percentile = np.percentile(variances, 20)
    
    # Avoid division by zero if variance is extremely small
    var_threshold = max(var_20th_percentile, 1e-6)
    
    lambda_v = -np.log(0.5) / var_threshold
    
    # 2. Calibrate Depth Gate (d_0)
    # We assume the "Stem" is roughly the first 10% of layers, 
    # but at least the first 5 layers.
    total_layers = len(df)
    stem_layers = max(5, int(total_layers * 0.10))
    d_0 = stem_layers / total_layers
    
    print(f"[Auto-Calibrate] Tuned lambda_v: {lambda_v:.4f} (based on p20 var: {var_threshold:.4e})")
    print(f"[Auto-Calibrate] Tuned d_0: {d_0:.4f} (Protecting first {stem_layers} layers)")
    
    return {'lambda_v': lambda_v, 'd_0': d_0}

def calculate_adaptive_score(row, total_layers, tuned_params):
    """
    Calculates CS using the auto-calibrated parameters.
    """
    # Unpack tuned params
    lambda_v = tuned_params['lambda_v']
    d_0 = tuned_params['d_0']
    
    # Fixed params (these are generally robust)
    k = 20.0       # Steepness of depth gate (20 makes it a sharp wall)
    gamma = 0.2    # Residual bonus
    
    # --- Metrics from Dataframe ---
    variance = row['act_var']
    identity = row['identity_score']
    # We approximate 'has_residual' by checking layer name or using a passed flag
    # For now, we'll assume False or you can map it from your model graph
    has_residual = False 
    
    # Calculate Relative Depth (0.0 to 1.0)
    # Assuming the dataframe index corresponds to depth
    relative_depth = (row.name + 1) / total_layers
    
    # 1. Depth Gating
    depth_gate = 1 / (1 + np.exp(-k * (relative_depth - d_0)))
    
    # 2. Functional Score
    silence_score = np.exp(-lambda_v * variance)
    redundancy_score = identity
    
    # Weighted average (favoring silence slightly as it's a stronger signal)
    functional_score = 0.6 * silence_score + 0.4 * redundancy_score
    
    # 3. Residual Bonus
    residual_multiplier = 1.0 + (gamma if has_residual else 0.0)
    
    final_score = depth_gate * functional_score * residual_multiplier
    
    return final_score
# ===============================
# Basic Counting Utilities
# ===============================

def count_zeros(tensor): 
    return torch.sum(tensor == 0).item()

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===============================
# Model Statistics
# ===============================

def layer_stats(model):
    print("\nLayer-wise zero parameter stats:\n")
    for name, param in model.named_parameters():
        if param.requires_grad:
            zeros = count_zeros(param)
            total = param.numel()
            # print(f"{name}: {zeros}/{total} zeros ({100 * zeros/total:.2f}%)")



# ===============================
# Cloning Utility
# ===============================

def clone_model(model, model_class):
    """Utility to clone a model and load weights to keep experiments isolated."""
    new_model = model_class()
    new_model.load_state_dict(model.state_dict())
    return new_model



# =========================================================
# Structural Pruning Baseline (torch-pruning, optional dep)
# =========================================================

def train_structural(model, train_loader, device, epochs, target_sparsity, save_path,
                     max_prune_fraction: float = 0.25):
    """
    Structurally prune with L1 filter importance (torch-pruning DependencyGraph,
    correct across residual/inception topologies) to `target_sparsity` parameter
    reduction, interleaved with SGD finetuning.

    Saves/loads a WHOLE-model checkpoint at `save_path` (the architecture changes,
    so a plain state_dict is not enough). Resumable: if `save_path` exists the
    finished artifact is loaded and training is skipped.

    Raises ImportError if torch-pruning is not installed; the caller decides policy.
    """
    if os.path.exists(save_path):
        # Resume path first: does not require torch-pruning.
        print(f"[INFO] Structural checkpoint exists: {save_path}. Loading and skipping training.")
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        loaded = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        loaded.eval()
        return loaded

    try:
        import torch_pruning as tp
    except ImportError as e:
        raise ImportError(
            "torch-pruning is required for the structural baseline "
            "(pip install torch-pruning)."
        ) from e

    print(f"\n[INFO] Starting structural L1 filter pruning for {epochs} epochs "
          f"(target param reduction {target_sparsity * 100:.2f}%)...")

    model.to(device)
    model.eval()  # eval mode for DependencyGraph/example forward (BN batch-stat safety)
    example_inputs = next(iter(train_loader))[0][:1].to(device)

    # Keep the final classifier (last Linear) intact
    last_linear = None
    for _, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    ignored_layers = [last_linear] if last_linear is not None else []

    max_prune_rounds = max(1, int(epochs * max_prune_fraction))
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=tp.importance.MagnitudeImportance(p=1),
        iterative_steps=max_prune_rounds,
        ch_sparsity=0.10,
        ignored_layers=ignored_layers,
        global_pruning=True,
    )

    original_params = sum(p.numel() for p in model.parameters())
    target_params = original_params * (1.0 - float(target_sparsity))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    interactive = sys.stderr.isatty()
    epoch_pbar = tqdm(range(epochs), desc="StructuralL1", disable=not interactive)
    prune_rounds = 0
    for epoch in epoch_pbar:
        current_params = sum(p.numel() for p in model.parameters())
        if current_params > target_params and prune_rounds < max_prune_rounds:
            try:
                pruner.step()
                prune_rounds += 1
                current_params = sum(p.numel() for p in model.parameters())
                print(f"[INFO] Prune round {prune_rounds}: params {current_params:,} "
                      f"(target <= {int(target_params):,})")
            except Exception as e:
                print(f"[WARN] Pruner step failed: {e}. Continuing finetuning with current structure.")
                prune_rounds = max_prune_rounds  # stop trying to prune

        model.train()
        loss_sum = torch.zeros((), device=device)
        batch_pbar = tqdm(
            train_loader,
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{epochs}",
            leave=False,
            disable=not interactive,
        )
        for step, (inputs, targets) in enumerate(batch_pbar):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()
            if not batch_pbar.disable and (step + 1) % 20 == 0:
                batch_pbar.set_postfix({"loss": f"{loss_sum.item() / (step + 1):.4f}"})
            del inputs, targets, outputs, loss
        scheduler.step()

        torch.save({"model": model.to("cpu")}, save_path)
        model.to(device)

    model.eval()
    model.to("cpu")
    final_params = sum(p.numel() for p in model.parameters())
    achieved = 1.0 - (final_params / original_params)
    print(f"[RESULT] Structural sparsity achieved: {achieved * 100:.2f}% "
          f"(target {target_sparsity * 100:.2f}%; {final_params:,} / {original_params:,} params)")
    torch.cuda.empty_cache()
    return model


# =========================================================
# Full-Analysis Evaluation Suite (gated by --analysis full)
# =========================================================

# Normalization constants used by the training pipeline (pyPrune.utils loaders)
DATASET_NORMS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}
OOD_DATASETS = {"cifar10": "SVHN", "cifar100": "SVHN"}

CIFAR_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


@torch.no_grad()
def collect_confidences(model, dataloader, device, max_batches=None):
    """Run `model` over `dataloader`; return (max_softmax_conf, correct) CPU tensors."""
    model.to(device)
    model.eval()
    confs, corrects = [], []
    for i, (inputs, targets) in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        inputs = inputs.to(device)
        targets = targets.to(device)
        probs = F.softmax(model(inputs), dim=1)
        conf, pred = probs.max(dim=1)
        confs.append(conf.detach().cpu())
        corrects.append(pred.eq(targets).detach().cpu())
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(confs), torch.cat(corrects).float()


def evaluate_calibration(model, dataloader, device, n_bins: int = 15):
    """Expected/Maximum Calibration Error over max-softmax confidences.

    Returns (ece, mce, bin_rows) where bin_rows carry per-bin
    {bin_lo, bin_hi, n, acc, conf} for reliability diagrams.
    """
    confs, corrects = collect_confidences(model, dataloader, device)
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, rows = 0.0, 0.0, []
    n_total = confs.numel()
    for i in range(n_bins):
        lo, hi = edges[i].item(), edges[i + 1].item()
        mask = (confs >= lo) & (confs <= hi) if i == 0 else (confs > lo) & (confs <= hi)
        n = int(mask.sum().item())
        if n == 0:
            rows.append({"bin_lo": lo, "bin_hi": hi, "n": 0,
                         "acc": float("nan"), "conf": float("nan")})
            continue
        acc = corrects[mask].mean().item()
        conf = confs[mask].mean().item()
        gap = abs(acc - conf)
        ece += (n / n_total) * gap
        mce = max(mce, gap)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": n, "acc": acc, "conf": conf})
    return ece, mce, rows


def auroc(pos_scores, neg_scores):
    """Rank-based AUROC (Mann-Whitney U): P(pos > neg) + 0.5 * P(pos == neg).

    Rank-based implementation with average ranks for ties; no sklearn dependency.
    """
    pos = np.asarray(list(pos_scores), dtype=np.float64)
    neg = np.asarray(list(neg_scores), dtype=np.float64)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    _, first_idx, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for start, c in zip(first_idx, counts):
        if c > 1:
            # average ranks of the tie group occupying sorted positions [start, start+c)
            ranks[order[start:start + c]] = (2.0 * (start + 1) + c - 1) / 2.0
    sum_pos = ranks[:n_pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def load_ood_test_loader(dataset_name, batch_size: int = 512, data_root: str = "data",
                         num_workers: int = 2):
    """OOD test loader for MSP-based OOD detection. CIFAR-10/100 -> SVHN (32x32).
    Returns None (with a warning) when unsupported or the data is not staged."""
    key = dataset_name.lower()
    if OOD_DATASETS.get(key) != "SVHN":
        print(f"[WARN] No OOD dataset configured for '{dataset_name}'. Skipping OOD evaluation.")
        return None
    mean, std = DATASET_NORMS[key]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    try:
        ds = datasets.SVHN(data_root, split="test", download=True, transform=tf)
    except Exception as e:
        print(f"[WARN] SVHN unavailable ({e}). Skipping OOD evaluation.")
        return None
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=(num_workers > 0))


def evaluate_ood(model, in_loader, ood_loader, device):
    """Max-softmax-probability AUROC: in-distribution vs OOD."""
    confs_in, _ = collect_confidences(model, in_loader, device)
    confs_ood, _ = collect_confidences(model, ood_loader, device)
    return auroc(confs_in.numpy(), confs_ood.numpy())


def load_cifar_c(corruptions_dir, dataset_name, corruption, severity,
                 batch_size: int = 512, num_workers: int = 2):
    """One (corruption, severity) slice of CIFAR-10-C / CIFAR-100-C as a DataLoader.

    Expects <corruptions_dir>/CIFAR-{10,100}-C/<corruption>.npy and labels.npy,
    each holding 5 severities x 10k samples. severity is 1-based (1..5).
    """
    key = dataset_name.lower()
    if key not in DATASET_NORMS:
        raise ValueError(f"Corruptions not supported for dataset '{dataset_name}' "
                         f"(supported: {sorted(DATASET_NORMS)})")
    if not (1 <= int(severity) <= 5):
        raise ValueError(f"severity must be in 1..5, got {severity}")
    subdir = "CIFAR-10-C" if key == "cifar10" else "CIFAR-100-C"
    base = os.path.join(corruptions_dir, subdir)
    data = np.load(os.path.join(base, f"{corruption}.npy"))
    labels = np.load(os.path.join(base, "labels.npy"))
    s = int(severity) - 1
    data = data[s * 10000:(s + 1) * 10000]
    labels = labels[s * 10000:(s + 1) * 10000]
    mean, std = DATASET_NORMS[key]
    x = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
    x = transforms.functional.normalize(
        x, torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1))
    ds = torch.utils.data.TensorDataset(x, torch.from_numpy(labels).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, persistent_workers=(num_workers > 0))


@torch.no_grad()
def _accuracy_only(model, dataloader, device):
    """Top-1 accuracy without power/energy tracking."""
    model.to(device)
    model.eval()
    correct, total = 0, 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        correct += model(inputs).argmax(dim=1).eq(targets).sum().item()
        total += targets.size(0)
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return correct / total if total > 0 else float("nan")


def evaluate_corruptions(models_dict, corruptions_dir, dataset_name, device,
                         severities=(1, 2, 3, 4, 5)):
    """Evaluate every model over all corruptions x severities.

    Returns (mean_corruption_acc_by_model, rows) where rows are long-format
    {model, corruption, severity, acc} dicts for tidy CSV output.
    """
    rows = []
    n_total = len(CIFAR_C_CORRUPTIONS) * len(severities)
    done = 0
    for corruption in CIFAR_C_CORRUPTIONS:
        for sev in severities:
            loader = load_cifar_c(corruptions_dir, dataset_name, corruption, sev)
            for model_name, model in models_dict.items():
                acc = _accuracy_only(model, loader, device)
                rows.append({"model": model_name, "corruption": corruption,
                             "severity": sev, "acc": acc})
            done += 1
            print(f"[INFO] Corruptions: {done}/{n_total} ({corruption} s{sev})")
    mean_by_model = {}
    for model_name in models_dict:
        accs = [r["acc"] for r in rows if r["model"] == model_name]
        mean_by_model[model_name] = sum(accs) / len(accs) if accs else float("nan")
    return mean_by_model, rows
