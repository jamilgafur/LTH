# Transfer.py
import os
import torch
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.ConvNetX import ConvNeXt
from pyPrune.models.InceptionNet import InceptionNet
from pyPrune.models.XceptionNet import XceptionNet
from pyPrune.models.MobileNet import MobileNet
from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies import MagnitudePruningStrategy
from experiments import *
from utils import *
from plots import *
from pyPrune.utils import *
from torch.backends import cudnn
import random
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# set seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
cudnn.benchmark = False
import glob
import argparse
import os


def safe_glob(path_pattern):
    matches = glob.glob(path_pattern)
    return matches[0] + "/" if matches else "None"   # or "" if you prefer empty string

CHECKPOINT_BASES = {
    "VGG16": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*Vgg16*datasettinyimagenet_*"),
    },
    "RegNetX_400MF": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*RegNetX*datasettinyimagenet_*"),
    },
    "InceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*InceptionNet*datasettinyimagenet_*"),
    },
    "MobileNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*MobileNet*datasettinyimagenet_*"),
    },
    "XceptionNet": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*XceptionNet*datasettinyimagenet_*"),
    },
    "ConvNeXt": {
        "Cifar10": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar10_*"),
        "Cifar100": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*cifar100_*"),
        "imagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasetimagenet_*"),
        "tinyimagenet": safe_glob("../structured_study/pruning_checkpoints/*ConvNeXt*datasettinyimagenet_*"),
    },
    
}

CHECKPOINT_FILES = {
    "VGG16": {
        "Cifar10": (
            "checkpoint_Finetuned_0.914101.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "Cifar100": (
            "checkpoint_Finetuned_0.981986.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "imagenet": (
            "checkpoint_Finetuned_0.790285.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "tinyimagenet": (
            "checkpoint_Finetuned_0.000000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
    },
    "RegNetX_400MF": {
        "Cifar10": (
            "checkpoint_Finetuned_0.945024.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "Cifar100": (
            "checkpoint_Finetuned_0.488000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "imagenet": (
            "checkpoint_Finetuned_0.914101.pth",
            "checkpoint_Original_0.000000.pth",
        ),
        "tinyimagenet": (
            "checkpoint_Finetuned_0.000000.pth",
            "checkpoint_Original_0.000000.pth",
        ),
    },
    "InceptionNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "MobileNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "XceptionNet": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
    "ConvNeXt": {
        "Cifar10": (
            "None",
            "None",
        ),
        "Cifar100": (
            "None",
            "None",
        ),
        "imagenet": (
            "None",
            "None",
        ),
        "tinyimagenet": (
            "None",
            "None",
        ),
    },
}

# ==============================================================================
# 1. VGG16 Common (The "V-Shape" Crash Probe)
# Updated Strategy: 
# - Test Stage 3 (High Variance) -> Expect Failure
# - Test Stage 1 (Low Variance)  -> Expect Success (Control Group)
# ==============================================================================
Vgg_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Stage 5 (Full)": ("features.conv_11", "features.conv_13"),
    "Stage 4 (Full)": ("features.conv_8", "features.conv_10"),
    "Stage 3 (Full)": ("features.conv_5", "features.conv_7"), # High Variance (Crash Site)
    "Stage 2 (Full)": ("features.conv_3", "features.conv_4"),
    "Stage 1 (Full)": ("features.conv_1", "features.conv_2"), # Low Variance (Safety Check)
    
    # --- 2. Granular Sensitivity Checks (High Variance Probes) ---
    # Isolating the Stage 3 crash
    "Stage 3 Conv 1 Only": ("features.conv_5", "features.conv_5"),
    "Stage 3 Conv 2 Only": ("features.conv_6", "features.conv_6"),
    "Stage 3 Conv 3 Only": ("features.conv_7", "features.conv_7"),

    # --- 3. Granular Sensitivity Checks (Low Variance Probes) ---
    # These layers have minimal variance; collapsing them should be safe.
    "Stage 1 Conv 1 Only": ("features.conv_1", "features.conv_1"),
    "Stage 1 Conv 2 Only": ("features.conv_2", "features.conv_2"),

    # --- 4. Multi-Stage Combinations ---
    "Last 2": ("features.conv_12", "features.conv_13"),
    "Stage 4-5": ("features.conv_8", "features.conv_13"),
    "Stage 3-5": ("features.conv_5", "features.conv_13"),
    "Stage 2-5": ("features.conv_3", "features.conv_13"),
    "Stage 1-5": ("features.conv_1", "features.conv_13"), # Aggressive full-network collapse
}

# ==============================================================================
# 2. RegNetX Common (The "Efficiency Wall")
# Updated Strategy:
# - Test Stage 4 (High Variance) -> Expect Failure
# - Test Stage 1 (Lower Variance) -> Expect Less Failure
# ==============================================================================
RegNetX_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Last 2": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block6.block.conv3"),
    "Stage 4 (Full)": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),
    "Stage 3 (Full)": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),
    "Stage 2 (Full)": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
    "Stage 1 (Full)": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

    # --- 2. Granular Stage 3 Decomposition (High Variance) ---
    "Stage 3 Block 0 Only": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block0.block.conv3"),
    "Stage 3 Block 1 Only": ("stage3.stage3_block1.block.conv1", "stage3.stage3_block1.block.conv3"),
    "Stage 3 Block 2 Only": ("stage3.stage3_block2.block.conv1", "stage3.stage3_block2.block.conv3"),
    "Stage 3 Block 3 Only": ("stage3.stage3_block3.block.conv1", "stage3.stage3_block3.block.conv3"),

    # --- 3. Granular Stage 4 Decomposition (High Variance) ---
    "Stage 4 Block 0 Only": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block0.block.conv3"),
    "Stage 4 Block 1 Only": ("stage4.stage4_block1.block.conv1", "stage4.stage4_block1.block.conv3"),
    "Stage 4 Block 2 Only": ("stage4.stage4_block2.block.conv1", "stage4.stage4_block2.block.conv3"),
    "Stage 4 Block 3 Only": ("stage4.stage4_block3.block.conv1", "stage4.stage4_block3.block.conv3"),
    "Stage 4 Block 4 Only": ("stage4.stage4_block4.block.conv1", "stage4.stage4_block4.block.conv3"),
    "Stage 4 Block 5 Only": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block5.block.conv3"),
    "Stage 4 Block 6 Only": ("stage4.stage4_block6.block.conv1", "stage4.stage4_block6.block.conv3"),

    # --- 4. Low Variance Control Group (Stage 1) ---
    "Stage 1 Block 0 Only": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

    # --- 5. Multi-Stage Combinations ---
    "Stage 3-4": [
        ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),
        ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),
    ],
    "Stage 2-4": [
        ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
        ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),
        ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),
    ],
    "Stage 1-4": [
        ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),
        ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
        ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),
        ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),
    ],

    # --- 6. Intra-Block Partial Collapses ---
    "Stage 1 first 2 conv": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv2"),
    "Stage 2 first 2 conv": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv2"),
    "Stage 3 first 2 conv": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block1.block.conv1"),
    "Stage 4 first 2 conv": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block1.block.conv1"),
    "Stage 1 last 2 conv": ("stage1.stage1_block0.block.conv2", "stage1.stage1_block0.block.conv3"),
    "Stage 2 last 2 conv": ("stage2.stage2_block0.block.conv2", "stage2.stage2_block0.block.conv3"),
    "Stage 3 last 2 conv": ("stage3.stage3_block2.block.conv3", "stage3.stage3_block3.block.conv3"),
    "Stage 4 last 2 conv": ("stage4.stage4_block4.block.conv3", "stage4.stage4_block5.block.conv3"),
}

# ==============================================================================
# 3. MobileNet Common (The "Safe" Baseline)
# Updated Strategy: 
# - Block 1 has lowest variance. Block 7 has slightly higher.
# - Verify safety across the spectrum.
# ==============================================================================
mobileNet_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Stage 7": ("block7.depthwise", "block7.bn2"),
    "Stage 6": ("block6.depthwise", "block7.depthwise"),
    "Stage 5": ("block5.depthwise", "block6.depthwise"),
    "Stage 4": ("block4.depthwise", "block5.depthwise"),
    "Stage 3": ("block3.depthwise", "block4.depthwise"),
    "Stage 2": ("block2.depthwise", "block3.depthwise"),
    "Stage 1": ("block1.depthwise", "block2.depthwise"),

    # --- 2. Granular Block Checks ---
    "Block 7 Only": ("block7.depthwise", "block7.bn2"),
    "Block 6 Only": ("block6.depthwise", "block6.pointwise"),
    "Block 5 Only": ("block5.depthwise", "block5.pointwise"),
    "Block 4 Only": ("block4.depthwise", "block4.pointwise"),
    "Block 3 Only": ("block3.depthwise", "block3.pointwise"),
    "Block 2 Only": ("block2.depthwise", "block2.pointwise"),
    "Block 1 Only": ("block1.depthwise", "block1.pointwise"), # Lowest Variance

    # --- 3. Multi-Stage Combinations ---
    "Stage 5-7": ("block5.depthwise", "block7.depthwise"),
    "Stage 4-7": ("block4.depthwise", "block7.depthwise"),
    "Stage 6-7": ("block6.depthwise", "block7.depthwise"),
    "Stage 3-7": ("block3.depthwise", "block7.depthwise"),
    "Stage 2-7": ("block2.depthwise", "block7.depthwise"),
    "Stage 1-7": ("block1.depthwise", "block7.depthwise"),
    "Last 2": ("block6.depthwise", "block7.depthwise"),
}

# ==============================================================================
# 4. XceptionNet Common (Low Variance)
# ==============================================================================
XceptionNet_common = {
    "Original Model": None,

    "Stage 5 (Full)": ("block5.depthwise", "block5.bn2"),
    "Stage 4 (Full)": ("block4.depthwise", "block5.depthwise"),
    "Stage 3 (Full)": ("block3.depthwise", "block4.depthwise"),
    "Stage 2 (Full)": ("block2.depthwise", "block3.depthwise"),
    "Stage 1 (Full)": ("block1.depthwise", "block2.depthwise"),

    "Block 5 Only": ("block5.depthwise", "block5.bn2"),
    "Block 4 Only": ("block4.depthwise", "block4.pointwise"),
    "Block 3 Only": ("block3.depthwise", "block3.pointwise"),
    "Block 2 Only": ("block2.depthwise", "block2.pointwise"),
    "Block 1 Only": ("block1.depthwise", "block1.pointwise"),

    "Stage 3-5": ("block3.depthwise", "block5.depthwise"),
    "Stage 2-5": ("block2.depthwise", "block5.depthwise"),
    "Stage 1-5": ("block1.depthwise", "block5.bn2"), 
}

# ==============================================================================
# 5. InceptionNet Common (The "Stage 3" Anomaly)
# Updated Strategy:
# - Stage 3a has MASSIVE variance (The "Trap").
# - Stem / Stage 1 has lower variance (Safety Check).
# ==============================================================================
InceptionNet_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Stage 5 (Full)": ("stage5.inception_5a", "stage5.inception_5b"),
    # Corrected: Stage 4 has blocks 4a through 4e
    "Stage 4 (Full)": ("stage4.inception_4a", "stage4.inception_4e"), 
    "Stage 3 (Full)": ("stage3.inception_3a", "stage3.inception_3b"),
    # Corrected: Stage 2 has conv2/conv3, not Inception blocks
    "Stage 2 (Full)": ("stage2.conv2", "stage2.conv3"), 
    
    # --- 5. Multi-Stage Combinations ---
    "Stage 2-5": [
        ("stage2.conv2", "stage2.conv3"),
        ("stage3.inception_3a", "stage3.inception_3b"),
        ("stage4.inception_4a", "stage4.inception_4e"),
        ("stage5.inception_5a", "stage5.inception_5b"),
    ],
    "Stage 3-5": [
        ("stage3.inception_3a", "stage3.inception_3b"),
        ("stage4.inception_4a", "stage4.inception_4e"),
        ("stage5.inception_5a", "stage5.inception_5b"),
    ],
    "Stage 4-5": [
        ("stage4.inception_4a", "stage4.inception_4e"),
        ("stage5.inception_5a", "stage5.inception_5b"),
    ],  
    "Last 2": ("stage5.inception_5a", "stage5.inception_5b"),
}

# ==============================================================================
# 6. ConvNeXt Common (The "Deep" Probe)
# Updated Strategy: 
# - Test Stage 1 (Low Variance) vs Stage 3 (High Redundancy)
# ==============================================================================
ConvNeXt_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Stage 4 (Full)": ("stage4.block4_1", "stage4.block4_2"),
    "Stage 3 (Full)": ("stage3.block3_1", "stage3.block3_3"),
    "Stage 2 (Full)": ("stage2.block2_1", "stage2.block2_2"),
    "Stage 1 (Full)": ("stage1.block1_1", "stage1.block1_2"),

    # --- 2. Granular Sensitivity Checks ---
    "Stage 3 Block 1 Only": ("stage3.block3_1", "stage3.block3_1"),
    "Stage 3 Block 2 Only": ("stage3.block3_2", "stage3.block3_2"), 
    "Stage 3 Block 3 Only": ("stage3.block3_3", "stage3.block3_3"),
    
    # --- 3. Low Variance / Stem Probe ---
    "Stage 1 Block 1 Only": ("stage1.block1_1", "stage1.block1_1"),

    # --- 4. Aggressive "Inner" Collapse ---
    "Stage 3 Inner (Block 2)": ("stage3.block3_2", "stage3.block3_2"), 
}

EXPERIMENTS = {
    "VGG16": {
        "Cifar10": Vgg_common,
        "Cifar100": Vgg_common,
        "tinyimagenet": Vgg_common,
        "imagenet": Vgg_common,
    },
    "RegNetX_400MF": {
        "Cifar10": RegNetX_common,
        "Cifar100": RegNetX_common,
        "tinyimagenet": RegNetX_common,
        "imagenet": RegNetX_common,
    },
    "XceptionNet": {
        "Cifar10": XceptionNet_common,
        "Cifar100": XceptionNet_common,
        "tinyimagenet": XceptionNet_common,
        "imagenet": XceptionNet_common,
    },
    "MobileNet": {
        "Cifar10": mobileNet_common,
        "Cifar100": mobileNet_common,
        "tinyimagenet": mobileNet_common,
        "imagenet": mobileNet_common,
    },
    "InceptionNet": {
        "Cifar10": InceptionNet_common,
        "Cifar100": InceptionNet_common,
        "tinyimagenet": InceptionNet_common,
        "imagenet":InceptionNet_common,
    },
    "ConvNeXt": {
        "Cifar10": ConvNeXt_common,
        "Cifar100": ConvNeXt_common,
        "tinyimagenet": ConvNeXt_common,
        "imagenet": ConvNeXt_common,
    }
}




# Helper functions
def create_optimizer_scheduler(model, learning_rate=1e-3):
    """Creates optimizer and scheduler for training."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler

def initialize_model_and_data(args):
    """Initialize the model and dataset based on the provided arguments."""
    model_class = args.model
    dataset = args.dataset
    model_kwargs = {}

    if model_class == InceptionNet and args.JF:
        raise ValueError("JF experiments are not supported for InceptionNet.")
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes


# -------------------------------------------------------------
# HPC Safe Heuristic Profiling (Single Job Only)
# -------------------------------------------------------------
def run_heuristic_profiling_safely(model_class, model_kwargs, train_loader, epochs, device, dataset, model_name):
    """
    Safely runs the pre-experiment training and heuristic analysis with Checkpointing.
    
    Updates:
    - If done_marker exists: Loads the last checkpoint and proceeds to analysis (skips training).
    - Resumes from the last saved epoch if interrupted.
    - Saves checkpoints to 'runs/checkpoints/heuristics'.
    """
    import os
    import glob
    import torch
    import time
    
    # --- 1. Define Paths ---
    lock_dir_path = f"{model_name}_{dataset}_heuristics.lock_dir"
    done_marker_path = f"{model_name}_{dataset}_heuristics_done.marker"
    
    plots_root_dir = os.path.join("runs", "plots")
    ckpt_root_dir = os.path.join("runs", "checkpoints", "heuristics")
    
    ensure_dir(plots_root_dir)
    ensure_dir(ckpt_root_dir)

    # --- 2. Check Completion Status (Flag only) ---
    # We do NOT return here anymore. We just note if it's done.
    is_already_done = os.path.exists(done_marker_path)
    if is_already_done:
        print(f"[INFO] Done marker found for {model_name}. Will load checkpoint and skip to analysis.")

    # --- 3. Acquire Lock ---
    print(f"[INFO] Attempting to acquire lock: {lock_dir_path}")
    try:
        os.mkdir(lock_dir_path)
        print("[INFO] Lock acquired! Starting/Resuming process...")
    except FileExistsError:
        print("[INFO] Lock busy. Another job is running this. Skipping.")
        return
    except OSError as e:
        print(f"[WARN] OS Error acquiring lock: {e}")
        return

    try:
        # --- Critical Section ---
        
        # A. Initialize Model & Optimizer
        if isinstance(model_class, str):
            model_class_obj = eval(model_class)
        else:
            model_class_obj = model_class

        model = model_class_obj(**model_kwargs).to(device)
        
        # Initialize Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # B. Checkpoint Discovery & Resumption
        ckpt_prefix = f"{model_name}_{dataset}_heuristic"
        ckpt_pattern = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch*.pt")
        
        # Find all existing checkpoints
        existing_ckpts = sorted(
            glob.glob(ckpt_pattern),
            key=lambda x: int(os.path.basename(x).split("epoch")[-1].split(".")[0])
        )

        start_epoch = 0
        
        if existing_ckpts:
            last_ckpt = existing_ckpts[-1]
            print(f"[INFO] Found checkpoint: {last_ckpt}. Loading state...")
            checkpoint = torch.load(last_ckpt, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"[✓] Model loaded. State is at Epoch {start_epoch}")
        else:
            if is_already_done:
                print("[WARN] Done marker exists but NO checkpoints found. Model will be random!")
            else:
                print(f"[INFO] No checkpoint found. Starting from Epoch 1.")

        # C. Training Loop
        # We only train if the marker does NOT exist AND we haven't reached epoch count
        if not is_already_done and start_epoch < epochs:
            print(f"[INFO] Training model for {epochs} epochs...")
            
            for epoch in range(start_epoch + 1, epochs + 1):
                # Train one epoch
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
                print(f"    [Epoch {epoch}/{epochs}] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
                
                # Save Checkpoint
                ckpt_path = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss
                }, ckpt_path)
                
                # Cleanup previous epoch to save space
                prev_ckpt = os.path.join(ckpt_root_dir, f"{ckpt_prefix}_epoch{epoch-1}.pt")
                if os.path.exists(prev_ckpt):
                    os.remove(prev_ckpt)
        else:
            reason = "Done marker exists" if is_already_done else "Epochs completed"
            print(f"[INFO] Skipping training loop ({reason}). Proceeding to analysis.")

        # D. Run Heuristic Analysis
        print("[INFO] Running analysis...")
        input_sample = model_kwargs["one_batch"].to(device)
        
        analyze_collapse_heuristics(
            model=model, 
            input_tensor=input_sample, 
            save_root_dir=plots_root_dir, 
            model_name=model_name,
            dataset_name=dataset
        )
        
        # E. Update marker (refresh timestamp)
        with open(done_marker_path, 'w') as f:
            f.write(f"Completed/Verified at {time.ctime()}")
            
        print("[INFO] Heuristic profiling complete. Plots saved.")

    except Exception as e:
        print(f"[ERROR] An error occurred during heuristic profiling: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # F. Release lock
        if os.path.exists(lock_dir_path):
            try:
                os.rmdir(lock_dir_path)
                print("[INFO] Lock released.")
            except OSError:
                print("[WARN] Could not remove lock directory.")
                
def get_experiment_config(model_name):
    """Matches model string to experiment config."""
    mn = model_name.lower()
    if "vgg" in mn: return Vgg_common
    if "regnet" in mn: return RegNetX_common
    if "mobile" in mn: return mobileNet_common
    if "xception" in mn: return XceptionNet_common
    if "convnext" in mn: return ConvNeXt_common
    return {}

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
#     """
#     Analyzes Conv2d and Linear layers, calculates Adaptive Collapse Scores (ACS),
#     generates LaTeX tables, and plots scores PER EXPERIMENT (aggregated layers).
#     """
#     print(f"[•] Running Extended Collapse Heuristics for {model_name} on {dataset_name}...")
    
#     model.eval()
#     if len(input_tensor.shape) == 3:
#         input_tensor = input_tensor.unsqueeze(0)

#     # 1. Get FLOPs
#     flops_dict = {}
#     try:
#         from fvcore.nn import FlopCountAnalysis
#         flops_counter = FlopCountAnalysis(model, input_tensor)
#         flops_dict = flops_counter.by_module()
#     except Exception as e:
#         print(f"[!] FLOPs count failed: {e}")

#     layer_stats = {}

#     # 2. Hook for metrics
#     def heuristic_hook(name, layer_type):
#         def fn(module, inp, out):
#             if not isinstance(out, torch.Tensor) or not isinstance(inp[0], torch.Tensor):
#                 return
            
#             x = inp[0].detach()
#             y = out.detach()
            
#             # Metric A: Identity Score
#             identity_score = 0.0
#             if x.shape == y.shape:
#                 x_flat = x.flatten(start_dim=1)
#                 y_flat = y.flatten(start_dim=1)
#                 try:
#                     identity_score = F.cosine_similarity(x_flat, y_flat, dim=1).mean().item()
#                 except:
#                     identity_score = 0.0
            
#             # Metric B: Memory & Bytes
#             dtype_size = x.element_size()
#             weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
#             total_bytes = (x.numel() * dtype_size) + (y.numel() * dtype_size) + weight_bytes

#             # Metric C: Weight Magnitude
#             weight_l1 = 0.0
#             if hasattr(module, 'weight') and module.weight is not None:
#                 weight_l1 = module.weight.norm(p=1).item() / module.weight.numel()

#             # Metric D: Activation Variance
#             if y.ndim == 4:
#                 act_var = y.var(dim=[2, 3]).mean().item()
#             else:
#                 act_var = y.var().item()
            
#             layer_stats[name] = {
#                 "layer_type": layer_type,
#                 "identity_score": identity_score,
#                 "memory_bytes": total_bytes,
#                 "weight_l1": weight_l1,
#                 "act_var": act_var
#             }
#         return fn

#     # Register hooks
#     hooks = []
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             hooks.append(module.register_forward_hook(heuristic_hook(name, type(module).__name__)))

#     # Run forward pass
#     with torch.no_grad():
#         model(input_tensor)

#     for h in hooks: h.remove()

#     # 3. Aggregate Data
#     results = []
#     for name, stats in layer_stats.items():
#         flops_val = flops_dict.get(name, 0)
#         mem_bytes = stats["memory_bytes"]
#         ai = flops_val / mem_bytes if mem_bytes > 0 else 0.0
        
#         results.append({
#             "layer": name,
#             "layer_type": stats["layer_type"],
#             "identity_score": stats["identity_score"],
#             "arithmetic_intensity": ai,
#             "weight_l1": stats["weight_l1"],
#             "act_var": stats["act_var"],
#             "flops": flops_val
#         })

#     df = pd.DataFrame(results)
    
#     if df.empty:
#         print("[WARN] No layers found for analysis.")
#         return df

#     # --- STEP 4: Save Raw Heuristics (ACS Dropped) ---
#     print("[•] Saving raw heuristics...")
    
#     # Save Enhanced CSV
#     csv_name = f"{model_name}_{dataset_name}_heuristics.csv"
#     df.to_csv(os.path.join(save_root_dir, csv_name), index=False)

#     # --- STEP 4.5: Aggregate Experiment Data (Median of Collapsed Blocks) ---
#     plot_data = [] 

#     try:
#         exp_config = get_experiment_config(model_name)
        
#         if exp_config:
#             summary_data_latex = []
#             layer_names = df['layer'].tolist()
            
#             for exp_name, layer_range in exp_config.items():
#                 # 1. Handle Original Model (No layers removed)
#                 # Compute the median variance across the ENTIRE network as a baseline.
#                 if layer_range is None: 
#                     global_median_var = float(df['act_var'].median())
#                     plot_data.append({
#                         "Experiment": exp_name.replace("_", " "), 
#                         "Median Variance": global_median_var
#                     })
                    
#                     var_str = f"{global_median_var:.2e}" if (global_median_var < 0.01 or global_median_var > 1000) else f"{global_median_var:.4f}"
#                     summary_data_latex.append({
#                         "Experiment Name": exp_name.replace("_", "\\_"),
#                         "Layer Range": "All Layers (Global Baseline)",
#                         "Median Var ($\\sigma^2$)": var_str
#                     })
#                     continue
                    
#                 # 2. Handle both single tuple and list of tuples for collapsed ranges
#                 ranges = layer_range if isinstance(layer_range, list) else [layer_range]
#                 subset_indices = []

#                 for start_layer, end_layer in ranges:
#                     start_idx = -1
#                     end_idx = -1
                    
#                     for i, lname in enumerate(layer_names):
#                         if start_layer in lname: start_idx = i
#                         if end_layer in lname: end_idx = i
                    
#                     if start_idx != -1 and end_idx != -1:
#                         if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
#                         subset_indices.extend(range(start_idx, end_idx + 1))
                
#                 # 3. Calculate median across all collapsed layers
#                 if subset_indices:
#                     subset_indices = list(set(subset_indices)) # Remove duplicates if ranges overlap
#                     subset = df.iloc[subset_indices]
                    
#                     median_var = float(subset['act_var'].median())
                    
#                     # Collect Raw Data for Plotting
#                     plot_data.append({
#                         "Experiment": exp_name.replace("_", " "), 
#                         "Median Variance": median_var
#                     })

#                     # Collect Formatted Data for Summary Table
#                     if isinstance(layer_range, list):
#                         range_str = "Multiple Ranges"
#                     else:
#                         range_str = f"{layer_range[0]} $\\to$ {layer_range[1]}".replace("_", "\\_")
                        
#                     exp_str = exp_name.replace("_", "\\_")
#                     var_str = f"{median_var:.2e}" if (median_var < 0.01 or median_var > 1000) else f"{median_var:.4f}"
                    
#                     summary_data_latex.append({
#                         "Experiment Name": exp_str,
#                         "Layer Range": range_str,
#                         "Median Var ($\\sigma^2$)": var_str
#                     })

#             # Save Summary LaTeX Table
#             if summary_data_latex:
#                 summary_df = pd.DataFrame(summary_data_latex)
#                 tex_filename = f"{model_name}_{dataset_name}_experiment_summary.tex"
#                 save_path = os.path.join(save_root_dir, tex_filename)
#                 summary_df.to_latex(
#                     buf=save_path, index=False, escape=False, 
#                     column_format="llc",
#                     caption=f"Predicted Collapse Variance Summary for {model_name}. Original Model represents the global network median.",
#                     label=f"tab:var_summary_{model_name.lower()}",
#                     position="h", header=True
#                 )
#                 print(f"[Saved] Experiment Summary Table -> {tex_filename}")

#     except Exception as e:
#         print(f"[!] Failed to process experiments: {e}")

#     # --- STEP 5: Generate Experiment-Wise Plots & Save Data ---
#     if plot_data:
#         exp_df = pd.DataFrame(plot_data)
        
#         # Save the Raw Plot Data to LaTeX
#         plot_data_tex = f"{model_name}_{dataset_name}_experiment_plot_data.tex"
#         exp_df.to_latex(
#             os.path.join(save_root_dir, plot_data_tex),
#             index=False,
#             float_format="%.4f",
#             caption=f"Aggregated Experiment Metrics for {model_name} Plots",
#             label=f"tab:exp_plot_data_{model_name.lower()}"
#         )
#         print(f"[Saved] Experiment Plot Data (TeX) -> {plot_data_tex}")

#         # [NEW] Save Variance Data to CSV for the Scatter Plot later
#         var_csv_path = os.path.join(save_root_dir, f"{model_name}_{dataset_name}_experiment_variance.csv")
#         exp_df.to_csv(var_csv_path, index=False)

#         # Plot: Median Variance per Experiment
#         plt.figure(figsize=(10, 6))
        
#         # Highlight Original Model with a different color
#         colors = ['crimson' if exp == 'Original Model' else 'steelblue' for exp in exp_df['Experiment']]
        
#         ax = sns.barplot(x="Experiment", y="Median Variance", data=exp_df, palette=colors)
        
#         # Add a horizontal dashed line across the chart at the Original Model's baseline
#         global_median = exp_df[exp_df['Experiment'] == 'Original Model']['Median Variance'].values[0]
#         plt.axhline(global_median, color='crimson', linestyle='--', alpha=0.7, label="Network Global Median")
        
#         plt.title(f"Median Activation Variance of Collapsed Blocks\n{model_name} | {dataset_name}", fontsize=14)
#         plt.ylabel("Median Variance", fontsize=12)
#         plt.xticks(rotation=45, ha='right')
        
#         plt.yscale("log")
#         plt.legend()
            
#         plt.grid(axis='y', linestyle='--', alpha=0.3, which='both')
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_root_dir, f"{model_name}_experiment_Variance.png"), dpi=300)
#         plt.close()
        
#     else:
#         print("[WARN] No experiment data available for plotting.")

#     # --- STEP 6: Generate Original Layer-wise Plots & Save Data ---
#     metrics_config = [
#         {"col": "identity_score", "title": "Identity Score (High = Redundant)", "folder": "identity_score", "color": "mediumpurple", "log_scale": False},
#         {"col": "act_var", "title": "Activation Variance", "folder": "activation_variance", "color": "goldenrod", "log_scale": True}
#     ]

#     # Save Layer-wise Plot Data to LaTeX (Long Table)
#     layer_tex_name = f"{model_name}_{dataset_name}_layerwise_plot_data.tex"
#     plot_cols = ["layer"] + [c["col"] for c in metrics_config]
    
#     # We use a subset of the main df for this export
#     df[plot_cols].to_latex(
#         os.path.join(save_root_dir, layer_tex_name),
#         index=False,
#         float_format="%.2e", # Scientific notation for compactness
#         longtable=True,      # Important for layer-wise tables
#         caption=f"Detailed Layer-wise Metrics for {model_name}",
#         label=f"tab:layer_metrics_{model_name.lower()}"
#     )
#     print(f"[Saved] Layer-wise Plot Data (TeX) -> {layer_tex_name}")

#     for config in metrics_config:
#         metric_dir = os.path.join(save_root_dir, config["folder"])
#         os.makedirs(metric_dir, exist_ok=True)
        
#         plt.figure(figsize=(max(12, len(df)*0.25), 6))
#         ax = sns.barplot(x="layer", y=config["col"], data=df, color=config["color"])
        
#         full_title = f"{config['title']}\nModel: {model_name} | Dataset: {dataset_name}"
#         ax.set_title(full_title, fontsize=12, fontweight='bold')
#         ax.grid(axis='y', linestyle='--', alpha=0.5)
        
#         if config["log_scale"]:
#             ax.set_yscale("log")
            
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=8)
#         ax.set_xlabel("Layer Name", fontsize=10)
#         ax.set_ylabel(config["col"], fontsize=10)

#         plt.tight_layout()
#         filename = f"{model_name}_{dataset_name}.png"
#         save_path = os.path.join(metric_dir, filename)
#         plt.savefig(save_path, dpi=150)
#         plt.close()
        
#     return df

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
    """
    Computes all three major pruning heuristics:
    1. Relative Activation Variance
    2. End-to-End Feature Redundancy (Cosine Similarity)
    3. Virtual Bypass Prediction Shift (KL Divergence)
    Saves outputs (CSV, TeX, PNG) into separate folders.
    """
    print(f"[•] Running Comprehensive Heuristic Analysis for {model_name} on {dataset_name}...")
    
    model.eval()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    module_dict = dict(model.named_modules())
    layer_names = list(module_dict.keys())

    # Create subdirectories for each heuristic
    dir_var = os.path.join(save_root_dir, "Heuristic_Variance")
    dir_sim = os.path.join(save_root_dir, "Heuristic_Redundancy")
    dir_kl = os.path.join(save_root_dir, "Heuristic_Bypass_KL")
    for d in [dir_var, dir_sim, dir_kl]:
        os.makedirs(d, exist_ok=True)

    # =========================================================================
    # PHASE 1: UNBROKEN NETWORK PASS (Variance & Cosine Similarity)
    # =========================================================================
    saved_tensors = {}
    layer_variances = {}

    def unbroken_hook(name):
        def fn(module, inp, out):
            if not isinstance(out, torch.Tensor) or not isinstance(inp[0], torch.Tensor):
                return
            x = inp[0].detach().cpu()
            y = out.detach().cpu()
            
            # Save for block-level Cosine Similarity
            saved_tensors[name] = {"in": x, "out": y}
            
            # Save for layer-level Variance
            if y.ndim == 4:
                act_var = y.var(dim=[2, 3]).mean().item()
            else:
                act_var = y.var().item()
            layer_variances[name] = act_var
        return fn

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(unbroken_hook(name)))

    # Run unbroken forward pass
    with torch.no_grad():
        baseline_logits = model(input_tensor)
        baseline_probs = F.softmax(baseline_logits, dim=1)

    for h in hooks:
        h.remove()

    # Calculate Global Baseline Variance
    if layer_variances:
        global_median_var = float(np.median(list(layer_variances.values())))
    else:
        global_median_var = 1.0

    # =========================================================================
    # PHASE 2: ITERATE EXPERIMENTS & CALCULATE METRICS
    # =========================================================================
    plot_data_var = []
    plot_data_sim = []
    plot_data_kl = []

    try:
        exp_config = get_experiment_config(model_name)
        if not exp_config:
            print("[WARN] No experiment config found.")
            return pd.DataFrame()

        for exp_name, layer_range in exp_config.items():
            exp_display = exp_name.replace("_", " ")

            # --- Handle Original Model (Baselines) ---
            if layer_range is None:
                # Variance (Baseline is 1.0x)
                plot_data_var.append({"Experiment": exp_display, "Relative Variance": 1.0})
                
                # Cosine Similarity (Global Median Redundancy of all layers)
                sim_scores = []
                for t_data in saved_tensors.values():
                    t_in, t_out = t_data["in"], t_data["out"]
                    if t_in.shape == t_out.shape:
                        in_flat, out_flat = t_in.flatten(start_dim=1), t_out.flatten(start_dim=1)
                        try:
                            sim_scores.append(F.cosine_similarity(in_flat, out_flat, dim=1).mean().item())
                        except: pass
                global_sim = float(np.median(sim_scores)) if sim_scores else 0.0
                plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": global_sim})
                
                # KL Divergence (Baseline is 0.0)
                plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": 0.0})
                continue

            ranges = layer_range if isinstance(layer_range, list) else [layer_range]
            
            # Variables for aggregation
            block_vars = []
            block_sims = []
            
            # Variables for Virtual Bypass
            bypass_handles = []
            bypass_cache = {}
            valid_bypass = True

            # Hook factories for Virtual Bypass
            def get_start_hook(idx):
                def hook(module, inp, out):
                    bypass_cache[idx] = inp[0]
                return hook

            def get_end_hook(idx):
                def hook(module, inp, out):
                    if idx in bypass_cache:
                        cached_inp = bypass_cache[idx]
                        if cached_inp.shape == out.shape:
                            return cached_inp
                        else:
                            return torch.zeros_like(out)
                    return out
                return hook

            # Process each range in the experiment
            for idx, (start_layer, end_layer) in enumerate(ranges):
                start_name = next((n for n in layer_names if start_layer in n), None)
                end_name = next((n for n in reversed(layer_names) if end_layer in n), None)

                # --- 1. Variance & Cosine Similarity Calculations ---
                if start_name and end_name:
                    # Collect variances for all layers in this range
                    in_range = False
                    for name in layer_names:
                        if name == start_name: in_range = True
                        if in_range and name in layer_variances:
                            block_vars.append(layer_variances[name])
                        if name == end_name: break
                    
                    # Calculate Block Cosine Similarity
                    if start_name in saved_tensors and end_name in saved_tensors:
                        block_in = saved_tensors[start_name]["in"]
                        block_out = saved_tensors[end_name]["out"]
                        
                        if block_in.shape == block_out.shape:
                            in_flat = block_in.flatten(start_dim=1)
                            out_flat = block_out.flatten(start_dim=1)
                            try:
                                sim = F.cosine_similarity(in_flat, out_flat, dim=1).mean().item()
                            except:
                                sim = 0.0
                        else:
                            sim = 0.0 # Shape mismatch = not an identity
                        block_sims.append(sim)

                    # --- 2. Setup Virtual Bypass Hooks ---
                    start_mod = module_dict[start_name]
                    end_mod = module_dict[end_name]
                    bypass_handles.append(start_mod.register_forward_hook(get_start_hook(idx)))
                    bypass_handles.append(end_mod.register_forward_hook(get_end_hook(idx)))
                else:
                    valid_bypass = False

            # --- Aggregate Variance & Sim for the Experiment ---
            exp_rel_var = (float(np.median(block_vars)) / global_median_var) if block_vars and global_median_var > 0 else 1.0
            plot_data_var.append({"Experiment": exp_display, "Relative Variance": exp_rel_var})

            exp_sim = float(np.median(block_sims)) if block_sims else 0.0
            plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": exp_sim})

            # --- Execute Virtual Bypass for the Experiment ---
            if not valid_bypass:
                kl_div = 50.0 # Cap failure
            else:
                try:
                    with torch.no_grad():
                        bypass_logits = model(input_tensor)
                        bypass_log_probs = F.log_softmax(bypass_logits, dim=1)
                    kl_div = F.kl_div(bypass_log_probs, baseline_probs, reduction='batchmean').item()
                except Exception:
                    kl_div = 50.0 # Cap failure
            
            # Cleanup bypass hooks
            for h in bypass_handles: h.remove()
            bypass_cache.clear()

            display_kl = kl_div if kl_div < 50.0 else 50.0
            plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": display_kl})

    except Exception as e:
        print(f"[!] Failed to process experiments: {e}")

    # =========================================================================
    # PHASE 3: PLOT AND SAVE DATA
    # =========================================================================
    def save_and_plot(data, y_col, directory, title_prefix, ylabel, hline_val, hline_label, color_base, color_alt, invert_safe_zone=False):
        if not data: return
        df = pd.DataFrame(data)
        
        # Save Data
        df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_{y_col.replace(' ', '_')}.csv"), index=False)
        df.to_latex(os.path.join(directory, f"{model_name}_{dataset_name}.tex"), index=False, float_format="%.4f")

        # Plot
        plt.figure(figsize=(12, 7))
        colors = [color_base if exp == 'Original Model' else color_alt for exp in df['Experiment']]
        sns.barplot(x="Experiment", y=y_col, data=df, palette=colors)
        
        plt.axhline(hline_val, color='crimson', linestyle='--', linewidth=2, label=hline_label)
        
        max_y = df[y_col].max() * 1.05 if df[y_col].max() > hline_val else hline_val * 1.5
        
        if invert_safe_zone:
            plt.axhspan(hline_val, max_y, color='green', alpha=0.05, label='Safe (High Redundancy)')
            plt.axhspan(0.0, hline_val, color='red', alpha=0.05, label='Dangerous')
        else:
            plt.axhspan(0.0, hline_val, color='green', alpha=0.05, label='Safe')
            plt.axhspan(hline_val, max_y, color='red', alpha=0.05, label='Dangerous')

        plt.title(f"{title_prefix}\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(directory, f"{model_name}_experiment_{y_col.split(' ')[0]}.png"), dpi=300)
        plt.close()

    # 1. Variance
    save_and_plot(
        plot_data_var, "Relative Variance", dir_var, 
        "Relative Activation Variance", "Relative Variance (Multiplier)", 1.0, "1.0x Baseline", 'crimson', 'steelblue'
    )

    # 2. Cosine Similarity
    global_sim_val = plot_data_sim[0]["Block Redundancy"] if plot_data_sim else 0.0
    save_and_plot(
        plot_data_sim, "Block Redundancy", dir_sim, 
        "Feature Redundancy (Cosine Similarity)", "Cosine Similarity (1.0 = Identity)", global_sim_val, "Global Median", 'crimson', 'mediumseagreen', invert_safe_zone=True
    )

    # 3. KL Divergence
    save_and_plot(
        plot_data_kl, "Prediction Shift (KL)", dir_kl, 
        "Virtual Bypass Prediction Damage", "KL Divergence (0.0 = Safe | 50.0 = Failed)", 1.0, "Critical Threshold (Approx)", 'crimson', 'teal'
    )

    return pd.DataFrame()

# def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
#     """
#     Analyzes Conv2d and Linear layers, calculates Taylor First-Order Gradient Sensitivity,
#     generates LaTeX tables, and plots readable metrics relative to a 1.0x baseline.
#     """
#     print(f"[•] Running Gradient-Based Sensitivity Analysis for {model_name} on {dataset_name}...")
    
#     # Must be in train mode (or at least require grad) to backpropagate
#     model.eval() 
#     if len(input_tensor.shape) == 3:
#         input_tensor = input_tensor.unsqueeze(0)
#     input_tensor.requires_grad_(True)

#     # 1. Get FLOPs
#     flops_dict = {}
#     try:
#         from fvcore.nn import FlopCountAnalysis
#         flops_counter = FlopCountAnalysis(model, input_tensor)
#         flops_dict = flops_counter.by_module()
#     except Exception as e:
#         print(f"[!] FLOPs count failed: {e}")

#     layer_stats = {}
#     activations = {}

#     # 2. Hooks for Taylor First-Order Sensitivity (|Activation * Gradient|)
#     def fwd_hook(name, layer_type, module):
#         def fn(module, inp, out):
#             if isinstance(out, torch.Tensor):
#                 activations[name] = out
                
#                 # Pre-calculate memory bytes while we are here
#                 x = inp[0]
#                 dtype_size = x.element_size()
#                 weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
#                 total_bytes = (x.numel() * dtype_size) + (out.numel() * dtype_size) + weight_bytes
                
#                 layer_stats[name] = {
#                     "layer_type": layer_type,
#                     "memory_bytes": total_bytes,
#                     "sensitivity": 0.0 # Placeholder until backward pass
#                 }
#         return fn

#     def bwd_hook(name):
#         def fn(module, grad_inp, grad_out):
#             if name in activations and grad_out[0] is not None:
#                 act = activations[name].detach()
#                 grad = grad_out[0].detach()
                
#                 # Taylor First-Order Expansion: | A * dL/dA |
#                 # Taking the mean computes the average structural importance of the layer's features
#                 sensitivity = (act * grad).abs().mean().item()
                
#                 if name in layer_stats:
#                     layer_stats[name]["sensitivity"] = sensitivity
#         return fn

#     # Register hooks
#     fwd_hooks = []
#     bwd_hooks = []
#     for name, module in model.named_modules():
#         if isinstance(module, (nn.Conv2d, nn.Linear)):
#             fwd_hooks.append(module.register_forward_hook(fwd_hook(name, type(module).__name__, module)))
#             bwd_hooks.append(module.register_full_backward_hook(bwd_hook(name)))

#     # 3. Run Forward and Backward Pass
#     model.zero_grad()
#     outputs = model(input_tensor)
    
#     # We use the mean of the outputs as a pseudo-loss to flow gradients evenly 
#     # through the entire network without needing actual labels.
#     pseudo_loss = outputs.mean()
#     pseudo_loss.backward()

#     # Clean up hooks
#     for h in fwd_hooks + bwd_hooks: 
#         h.remove()
#     model.zero_grad()

#     # 4. Aggregate Data
#     results = []
#     layer_names = []
#     for name, stats in layer_stats.items():
#         layer_names.append(name)
#         flops_val = flops_dict.get(name, 0)
#         mem_bytes = stats["memory_bytes"]
#         ai = flops_val / mem_bytes if mem_bytes > 0 else 0.0
        
#         results.append({
#             "layer": name,
#             "layer_type": stats["layer_type"],
#             "arithmetic_intensity": ai,
#             "sensitivity": stats["sensitivity"],
#             "flops": flops_val
#         })

#     df = pd.DataFrame(results)
    
#     if df.empty:
#         print("[WARN] No layers found for analysis.")
#         return df

#     # --- STEP 5: Save Raw Heuristics ---
#     csv_name = f"{model_name}_{dataset_name}_heuristics.csv"
#     df.to_csv(os.path.join(save_root_dir, csv_name), index=False)

#     # Compute Global Baseline for Normalization
#     global_median_sens = float(df['sensitivity'].median())

#     # --- STEP 6: Aggregate Experiment Data (Relative Sensitivity) ---
#     plot_data = [] 

#     try:
#         exp_config = get_experiment_config(model_name)
        
#         if exp_config:
#             summary_data_latex = []
            
#             for exp_name, layer_range in exp_config.items():
#                 # 1. Handle Original Model (Global Baseline)
#                 if layer_range is None: 
#                     plot_data.append({
#                         "Experiment": exp_name.replace("_", " "), 
#                         "Relative Sensitivity": 1.0 # Baseline is strictly 1.0x
#                     })
                    
#                     summary_data_latex.append({
#                         "Experiment Name": exp_name.replace("_", "\\_"),
#                         "Layer Range": "All Layers (Global Baseline)",
#                         "Relative Sensitivity": "1.00x"
#                     })
#                     continue
                    
#                 # 2. Handle both single tuple and list of tuples for collapsed ranges
#                 ranges = layer_range if isinstance(layer_range, list) else [layer_range]
#                 subset_indices = []

#                 for start_layer, end_layer in ranges:
#                     start_idx = -1
#                     end_idx = -1
                    
#                     for i, lname in enumerate(layer_names):
#                         if start_layer in lname: start_idx = i
#                         if end_layer in lname: end_idx = i
                    
#                     if start_idx != -1 and end_idx != -1:
#                         if start_idx > end_idx: start_idx, end_idx = end_idx, start_idx
#                         subset_indices.extend(range(start_idx, end_idx + 1))
                
#                 # 3. Calculate Relative Sensitivity
#                 if subset_indices:
#                     subset_indices = list(set(subset_indices)) 
#                     subset = df.iloc[subset_indices]
                    
#                     # Median sensitivity of the block divided by the global median
#                     median_sens = float(subset['sensitivity'].median())
#                     relative_sens = median_sens / global_median_sens if global_median_sens > 0 else 1.0
                    
#                     plot_data.append({
#                         "Experiment": exp_name.replace("_", " "), 
#                         "Relative Sensitivity": relative_sens
#                     })

#                     if isinstance(layer_range, list):
#                         range_str = "Multiple Ranges"
#                     else:
#                         range_str = f"{layer_range[0]} $\\to$ {layer_range[1]}".replace("_", "\\_")
                        
#                     exp_str = exp_name.replace("_", "\\_")
                    
#                     summary_data_latex.append({
#                         "Experiment Name": exp_str,
#                         "Layer Range": range_str,
#                         "Relative Sensitivity": f"{relative_sens:.2f}x"
#                     })

#             # Save Summary LaTeX Table
#             if summary_data_latex:
#                 summary_df = pd.DataFrame(summary_data_latex)
#                 tex_filename = f"{model_name}_{dataset_name}_experiment_summary.tex"
#                 save_path = os.path.join(save_root_dir, tex_filename)
#                 summary_df.to_latex(
#                     buf=save_path, index=False, escape=False, 
#                     column_format="llc",
#                     caption=f"Predicted Block Sensitivity Summary for {model_name}. 1.00x represents the global network median.",
#                     label=f"tab:rel_sens_summary_{model_name.lower()}",
#                     position="h", header=True
#                 )
#                 print(f"[Saved] Experiment Summary Table -> {tex_filename}")

#     except Exception as e:
#         print(f"[!] Failed to process experiments: {e}")

#     # --- STEP 7: Generate Experiment-Wise Plots ---
#     if plot_data:
#         exp_df = pd.DataFrame(plot_data)
        
#         plot_data_tex = f"{model_name}_{dataset_name}_experiment_plot_data.tex"
#         exp_df.to_latex(
#             os.path.join(save_root_dir, plot_data_tex),
#             index=False,
#             float_format="%.2f",
#             caption=f"Aggregated Experiment Metrics for {model_name} Plots",
#             label=f"tab:exp_plot_data_{model_name.lower()}"
#         )

#         var_csv_path = os.path.join(save_root_dir, f"{model_name}_{dataset_name}_experiment_rel_sensitivity.csv")
#         exp_df.to_csv(var_csv_path, index=False)

#         # Plot: Relative Sensitivity per Experiment
#         plt.figure(figsize=(12, 7))
        
#         colors = ['crimson' if exp == 'Original Model' else 'rebeccapurple' for exp in exp_df['Experiment']]
#         ax = sns.barplot(x="Experiment", y="Relative Sensitivity", data=exp_df, palette=colors)
        
#         # Add visual interpretation baseline
#         plt.axhline(1.0, color='crimson', linestyle='--', linewidth=2, label="1.0x Baseline (Network Average)")
        
#         # Add visual zones (Lower sensitivity = safer to prune)
#         plt.axhspan(0.0, 1.0, color='green', alpha=0.05, label='Safe to Collapse (Low Sensitivity)')
#         plt.axhspan(1.0, exp_df['Relative Sensitivity'].max() * 1.05, color='red', alpha=0.05, label='Dangerous (High Sensitivity)')

#         plt.title(f"Gradient-Based Sensitivity of Collapsed Blocks\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold')
#         plt.ylabel("Relative Sensitivity (Taylor First-Order)\n(<1.0x = Safe | >1.0x = Critical)", fontsize=12)
#         plt.xticks(rotation=45, ha='right')
        
#         plt.legend(loc='upper right')
#         plt.grid(axis='y', linestyle='--', alpha=0.4)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_root_dir, f"{model_name}_experiment_Sensitivity.png"), dpi=300)
#         plt.close()

#     return df

def run_jf_or_kevin_experiment(experiment_name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path, post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args):
    """Runs the appropriate experiment based on the arguments (JF or Kevin)."""
    model_class = eval(model_class)
    if args.JF:
        return run_jf_experiment(
            {experiment_name: layers},
            model_path_097,
            train_loader,
            test_loader,
            device,
            epochs,
            pretrain,
            model_class=model_class,
            model_kwargs=model_kwargs,
            data_shape=input_size,
            save_path=save_path,
            post_compress_epochs=post_compress_epochs,
            quant=quant
        )
    elif args.Kevin:
        if experiment_name == "Original Model":
            epochs = pretrain + epochs
        
        return run_kevin_experiment(
            {experiment_name: layers},
            model_path_000,
            train_loader,
            test_loader,
            device,
            epochs,
            model_class=model_class,
            model_kwargs=model_kwargs,
            data_shape=input_size,
            save_path=save_path,
            post_compress_epochs=post_compress_epochs,
            quant=quant
        )
    else:
        raise ValueError("You must specify either --JF or --Kevin to run the corresponding experiment.")

def run_experiments_for_dataset(
    experiments,
    dataset,
    model_path_097,
    model_path_000,
    train_loader,
    test_loader,
    device,
    epochs,
    pretrain,
    model_class,
    model_kwargs,
    post_compress_epochs,
    experiment_func,
    quant=False,
    args=None
):
    """Run specified experiments for a given dataset."""
    save_path = f"{model_class}_{dataset}_{CHECKPOINT_FILES[model_class][dataset][0]}_epochs{epochs}_pretrain{pretrain}_postcompress{post_compress_epochs}"

    if model_class in [InceptionNet, XceptionNet, MobileNet]:
        steps = [0]
        epochs = pretrain
        pretrain = 0
    else:
        steps = exponential_decay_list(steps=21)
    print(f"Pruning steps: {steps}")

    # Initialize the dataset
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)

    # ------------------------------------------------------------------------
    # STEP 1: Pre-Experiment Heuristic Profiling (HPC Safe)
    # ------------------------------------------------------------------------
    # This runs ONCE across all jobs to identify Identify Score & Arithmetic Intensity
    run_heuristic_profiling_safely(
        model_class=model_class,
        model_kwargs=model_kwargs,
        train_loader=train_loader,
        epochs=epochs, # Uses the main epoch count for the profiling training
        device=device,
        dataset=dataset,
        model_name=args.model
    )
    
    # ------------------------------------------------------------------------
    # STEP 2: Main Experiments
    # ------------------------------------------------------------------------
    for name, layers in experiments.items():
        print(f"\n--- Running experiment: {name} ---")
        model = run_jf_or_kevin_experiment(
            name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path,
            post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args
        )
# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RegNetX_400MF", choices=["VGG16", "RegNetX_400MF", "InceptionNet", "XceptionNet", "MobileNet", "ConvNeXt"], help="Model architecture to use")
    parser.add_argument("--dataset", type=str, default="Cifar10", help="Dataset to use (Cifar10, Cifar100, ImageNet, TinyImageNet)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--pretrain", type=int, default=10, help="Number of pretraining epochs")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment to run")
    parser.add_argument("--post_compress_epochs", type=int, default=0, help="Number of post-pruning compression epochs")
    parser.add_argument("--imp", action="store_false", help="Apply Iterative Magnitude Pruning")
    parser.add_argument("--JF", action="store_true", help="Run JF experiments")
    parser.add_argument("--Kevin", action="store_true", help="Run Kevin experiments")
    parser.add_argument("--quant", action="store_true", help="Apply Quantization Aware Training")
    args = parser.parse_args()
    print(args)
    print(f"has GPU: {torch.cuda.is_available()}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes = initialize_model_and_data(args)
    
    base_path = CHECKPOINT_BASES[args.model][args.dataset]
    print(f"Base path for checkpoints: {base_path}")
    model_path_097 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][0])
    model_path_000 = os.path.join(base_path, CHECKPOINT_FILES[args.model][args.dataset][1])

    if args.experiment not in EXPERIMENTS[args.model][args.dataset]:
        raise ValueError(f"Experiment '{args.experiment}' not found for model '{args.model}' and dataset '{args.dataset}'.")

    experiment_dict = {args.experiment: EXPERIMENTS[args.model][args.dataset][args.experiment]}

    run_experiments_for_dataset(
        experiment_dict,
        args.dataset,
        model_path_097,
        model_path_000,
        None, None, device, args.epochs, args.pretrain, model_class,
        model_kwargs, args.post_compress_epochs, None, args.quant, args
    )

# Entry point
if __name__ == "__main__":
    main()
