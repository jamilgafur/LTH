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
from trainer import train_one_epoch
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
            "None",
            "None",
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
            "None",
            "None",
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
    "Stage 1 (Full)": ("features.conv_1", "features.conv_2"), # Low Variance (Safe)
    "Stage 2 (Full)": ("features.conv_3", "features.conv_4"), # Low Variance (Safe)
    "Stage 3 (Full)": ("features.conv_5", "features.conv_7"), # Moderate Transition
    "Stage 4 (Full)": ("features.conv_8", "features.conv_10"), # High Variance (Danger)
    "Stage 5 (Full)": ("features.conv_11", "features.conv_13"), # Extreme Variance Peak

    # --- 2. Granular Probes: The "Safe" Zones (Low Variance) ---
    "Conv 1 Only": ("features.conv_1", "features.conv_1"), # Lowest variance in the network
    "Conv 2 Only": ("features.conv_2", "features.conv_2"),
    "Conv 3 Only": ("features.conv_3", "features.conv_3"),

    # --- 3. Granular Probes: The "Danger" Zones (High Variance Spikes) ---
    # We expect these to fail catastrophically based on the massive variance spikes
    "Conv 9 Only": ("features.conv_9", "features.conv_9"), # 13k variance spike
    "Conv 10 Only": ("features.conv_10", "features.conv_10"), # 16k variance spike
    "Conv 13 Only": ("features.conv_13", "features.conv_13"), # 6M peak before FC

    # --- 4. Multi-Stage Combinations ---
    "Stages 1 and 2": ("features.conv_1", "features.conv_4"),
    "Stages 3 and 4": ("features.conv_5", "features.conv_10"),
    "Stages 4 and 5": ("features.conv_8", "features.conv_13"),
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
    "Early Features (Full)": ("features.0.depthwise.0", "features.4.pointwise.0"),
    "Middle Features (Full)": ("features.5.depthwise.0", "features.8.pointwise.0"),
    "Late Features (Full)": ("features.9.depthwise.0", "features.12.pointwise.0"),

    # --- 2. Granular Probes: The "Safe" Zones (Low Variance) ---
    # These early blocks are relatively flat and should collapse easily
    "Block 0 Only": ("features.0.depthwise.0", "features.0.pointwise.0"),
    "Block 1 Only": ("features.1.depthwise.0", "features.1.pointwise.0"),
    "Block 2 Only": ("features.2.depthwise.0", "features.2.pointwise.0"),

    # --- 3. Granular Probes: The "Danger" Zones (High Variance) ---
    # We expect these to fail based on the massive pointwise variance spikes
    "Block 8 Only": ("features.8.depthwise.0", "features.8.pointwise.0"),
    "Block 10 Only": ("features.10.depthwise.0", "features.10.pointwise.0"),
    "Block 11 Only": ("features.11.depthwise.0", "features.11.pointwise.0"), # The biggest spike

    # --- 4. Multi-Stage Combinations ---
    "Early and Middle": ("features.0.depthwise.0", "features.8.pointwise.0"),
    "Middle and Late": ("features.5.depthwise.0", "features.12.pointwise.0"),
    "Almost All (1-11)": ("features.1.depthwise.0", "features.11.pointwise.0"),
}

# ==============================================================================
# 4. XceptionNet Common (Low Variance)
# ==============================================================================
XceptionNet_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Flow Collapses ---
    "Entry Flow (Full)": ("block1.rep.0.depthwise", "block3.rep.4.pointwise"),
    "Middle Flow (Full)": ("middle_flow.0.rep.1.depthwise", "middle_flow.7.rep.7.pointwise"),
    "Exit Flow (Full)": ("block4.rep.1.depthwise", "conv4.pointwise"),

    # --- 2. Granular Probes: The "Safe" Zones (Low Variance) ---
    "Block 1 Only": ("block1.rep.0.depthwise", "block1.rep.3.pointwise"),
    "Block 2 Only": ("block2.rep.1.depthwise", "block2.rep.4.pointwise"),
    "Block 3 Only": ("block3.rep.1.depthwise", "block3.rep.4.pointwise"),
    "Conv 3 and 4 Only": ("conv3.depthwise", "conv4.pointwise"),

    # --- 3. Granular Probes: The "Danger" Zones (High Variance) ---
    # We expect these to crash the model based on the massive variance spikes
    "Middle Flow Block 4 Only": ("middle_flow.4.rep.1.depthwise", "middle_flow.4.rep.7.pointwise"),
    "Middle Flow Block 7 Only": ("middle_flow.7.rep.1.depthwise", "middle_flow.7.rep.7.pointwise"),
    "Block 4 Only": ("block4.rep.1.depthwise", "block4.rep.4.pointwise"),

    # --- 4. Multi-Stage Combinations ---
    "Entry and Middle Flow": ("block1.rep.0.depthwise", "middle_flow.7.rep.7.pointwise"),
    "Middle and Exit Flow": ("middle_flow.0.rep.1.depthwise", "conv4.pointwise"),
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
    "Stage 2 (Full)": ("stage2", "stage2"), # Flatline (Safe)
    "Stage 3 (Full)": ("stage3a", "stage3b"), # Flatline (Safe)
    "Stage 4 (Full)": ("stage4a", "stage4e"), # Flatline (Safe)
    "Stage 5 (Full)": ("stage5a", "stage5b"), # Extreme Variance Peak (Danger)

    # --- 2. Granular Probes: The "Safe" Zones (Low Variance) ---
    "Stage 3a Only": ("stage3a", "stage3a"),
    "Stage 4a Only": ("stage4a", "stage4a"),

    # --- 3. Granular Probes: The "Danger" Zones (High Variance Spikes) ---
    # We expect these to fail catastrophically based on the massive variance spikes
    "Stage 5a Only": ("stage5a", "stage5a"), # 20k variance spike
    "Stage 5b Only": ("stage5b", "stage5b"), # 40k+ variance spike (Peak)

    # --- 4. Multi-Stage Combinations ---
    "Stages 2 and 3": ("stage2", "stage3b"),
    "Stages 3 and 4": ("stage3a", "stage4e"),
}
# ==============================================================================
# 6. ConvNeXt Common (The "Deep" Probe)
# Updated Strategy: 
# - Test Stage 1 (Low Variance) vs Stage 3 (High Redundancy)
# ==============================================================================
ConvNeXt_common = {
    "Original Model": None,

    # --- 1. Coarse-Grained Stage Collapses ---
    "Stage 1 (Full)": ("stages.0.0.dwconv", "stages.0.2.pwconv2"), # Highly redundant
    "Stage 2 (Full)": ("stages.1.0.dwconv", "stages.1.2.pwconv2"),
    "Stage 3 (Full)": ("stages.2.0.dwconv", "stages.2.8.pwconv2"),
    "Stage 4 (Full)": ("stages.3.0.dwconv", "stages.3.2.pwconv2"), # Highly volatile

    # --- 2. Granular Probes: The "Safe" Zones (Low Variance) ---
    "Stage 1 Block 0 Only": ("stages.0.0.dwconv", "stages.0.0.pwconv2"),
    "Stage 1 Block 2 Only": ("stages.0.2.dwconv", "stages.0.2.pwconv2"),

    # --- 3. Granular Probes: The "Danger" Zones (High Variance) ---
    # We expect these to fail catastrophically based on the massive variance spikes
    "Stage 3 Block 8 Only": ("stages.2.8.dwconv", "stages.2.8.pwconv2"), # Peak of stage 3
    "Stage 4 Block 0 Only": ("stages.3.0.dwconv", "stages.3.0.pwconv2"), # Absolute highest spike
    "Stage 4 Block 2 Only": ("stages.3.2.dwconv", "stages.3.2.pwconv2"),

    # --- 4. Multi-Stage Combinations ---
    "Stages 1 and 2": ("stages.0.0.dwconv", "stages.1.2.pwconv2"),
    "Stages 3 and 4": ("stages.2.0.dwconv", "stages.3.2.pwconv2"),
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
    if "inception" in mn: return InceptionNet_common
    if "regnet" in mn: return RegNetX_common
    if "mobile" in mn: return mobileNet_common
    if "xception" in mn: return XceptionNet_common
    if "convnext" in mn: return ConvNeXt_common
    return {}


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

def setup_directories(save_root_dir):
    """Creates and returns paths for all heuristic output directories."""
    dirs = {
        "var": os.path.join(save_root_dir, "Heuristic_Variance"),
        "sim": os.path.join(save_root_dir, "Heuristic_Redundancy"),
        "kl": os.path.join(save_root_dir, "Heuristic_Bypass_KL"),
        "cscore": os.path.join(save_root_dir, "Heuristic_Collapse_Score"),
        "layer_stats": os.path.join(save_root_dir, "Layer_Statistics")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def run_baseline_pass(model, input_tensor):
    """Executes the unbroken network pass to gather baseline variances, activations, and saved tensors."""
    saved_tensors = {}
    layer_variances = {}
    layer_activations = {}

    def unbroken_hook(name):
        def fn(module, inp, out):
            if not isinstance(out, torch.Tensor) or not isinstance(inp[0], torch.Tensor):
                return
            x = inp[0].detach().cpu()
            y = out.detach().cpu()
            
            # Save for block-level Cosine Similarity
            saved_tensors[name] = {"in": x, "out": y}
            
            # Save for layer-level Variance and Activation
            if y.ndim == 4:
                act_var = y.var(dim=[2, 3]).mean().item()
                act_mean = y.mean(dim=[2, 3]).mean().item()
            else:
                act_var = y.var().item()
                act_mean = y.mean().item()
                
            layer_variances[name] = act_var
            layer_activations[name] = act_mean
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
    global_median_var = float(np.median(list(layer_variances.values()))) if layer_variances else 1.0

    return saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs

def plot_individual_layers(layer_activations, layer_variances, directory, model_name, dataset_name, exp_config=None):
    """Plots the raw individual mean activation and variance for each tracked layer, 
    AND generates a second figure aggregating these metrics per experiment block."""
    if not layer_activations:
        return
    
    layers = list(layer_activations.keys())
    activations = list(layer_activations.values())
    variances = list(layer_variances.values())

    df = pd.DataFrame({
        "Layer": layers,
        "Mean Activation": activations,
        "Variance": variances
    })

    # Save raw stats to CSV
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats.csv"), index=False)

    # =========================================================================
    # FIGURE 1: INDIVIDUAL LAYER STATS (Annotated)
    # =========================================================================
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ---- Determine Stage Boundaries for Shading ----
    regions = {}
    if exp_config:
        for key, val in exp_config.items():
            if "(Full)" in key and isinstance(val, tuple):
                start_layer, end_layer = val
                start_idx = next((i for i, n in enumerate(layers) if start_layer in n), None)
                end_idx = next((i for i, n in reversed(list(enumerate(layers))) if end_layer in n), None)
                
                if start_idx is not None and end_idx is not None:
                    clean_name = key.replace(" (Full)", "")
                    regions[clean_name] = (start_idx, end_idx)

    # ---- Draw Shaded Regions & Labels ----
    bg_colors = ['#eaf2f8', '#fdf2e9', '#e8f8f5', '#f5eef8', '#f4f6f7']
    for i, (region_name, (start, end)) in enumerate(regions.items()):
        color = bg_colors[i % len(bg_colors)]
        ax1.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        ax2.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        
        y_max = max(variances) if variances else 1
        ax2.text(
            (start + end) / 2, y_max * 0.95, region_name, 
            ha='center', va='top', fontsize=11, fontweight='bold', 
            color='#555555', alpha=0.8,
            bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')
        )

    # ---- Mean Activation Plot ----
    sns.lineplot(data=df, x="Layer", y="Mean Activation", marker="o", color="steelblue", linewidth=2, ax=ax1, zorder=3)
    ax1.set_ylabel("Mean Activation", fontweight='bold', labelpad=10)
    ax1.set_title(f"Layer-wise Activation & Structural Stages\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=12)

    # ---- Variance Plot ----
    sns.lineplot(data=df, x="Layer", y="Variance", marker="s", color="crimson", linewidth=2, linestyle="--", ax=ax2, zorder=3)
    ax2.set_ylabel("Variance", fontweight='bold', labelpad=10)
    ax2.set_xlabel("Network Layer", fontweight='bold', labelpad=10)
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers, rotation=90, fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_{dataset_name}_layer_stats_annotated.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # FIGURE 2: AGGREGATED EXPERIMENT BLOCK STATS
    # =========================================================================
    if not exp_config:
        return
        
    exp_names, exp_vars, exp_acts = [], [], []
    
    # Calculate the block-level stats for each experiment
    for exp_name, layer_range in exp_config.items():
        if layer_range is None: # Skip the "Original Model" baseline
            continue
            
        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        block_vars, block_acts = [], []
        
        for start_layer, end_layer in ranges:
            start_name = next((n for n in layers if start_layer in n), None)
            end_name = next((n for n in reversed(layers) if end_layer in n), None)
            
            if start_name and end_name:
                in_range = False
                for name in layers:
                    if name == start_name: in_range = True
                    if in_range:
                        if name in layer_variances: block_vars.append(layer_variances[name])
                        if name in layer_activations: block_acts.append(layer_activations[name])
                    if name == end_name: break
                    
        # If we successfully captured data for this experiment block
        if block_vars and block_acts:
            exp_names.append(exp_name)
            exp_vars.append(float(np.median(block_vars))) # Median variance representing the block
            exp_acts.append(float(np.mean(block_acts)))   # Mean activation of the block
            
    if not exp_names:
        return
        
    df_exp = pd.DataFrame({
        "Experiment": exp_names,
        "Median Variance": exp_vars,
        "Mean Activation": exp_acts
    })
    
    # Save experiment stats to CSV
    df_exp.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_experiment_block_stats.csv"), index=False)
    
    # Plotting
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Mean Activation Barplot
    sns.barplot(data=df_exp, x="Experiment", y="Mean Activation", color="steelblue", ax=ax3, edgecolor="black")
    ax3.set_ylabel("Block Mean Activation", fontweight='bold', labelpad=10)
    ax3.set_title(f"Aggregated Heuristics per Target Experiment\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=12)
    ax3.axhline(0, color='black', linewidth=1)
    
    # Median Variance Barplot
    sns.barplot(data=df_exp, x="Experiment", y="Median Variance", color="crimson", ax=ax4, edgecolor="black")
    ax4.set_ylabel("Block Median Variance", fontweight='bold', labelpad=10)
    ax4.set_xlabel("Experiment Target", fontweight='bold', labelpad=10)
    ax4.axhline(0, color='black', linewidth=1)
    
    # Format X-axis
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(
        os.path.join(directory, f"{model_name}_{dataset_name}_experiment_block_stats.png"), 
        dpi=300, bbox_inches='tight'
    )
    plt.close()

def plot_normalized_metrics(layer_activations, layer_variances, directory, model_name, dataset_name):
    """NEW: Plots Normalized Variance and Normalized CV across layers."""
    if not layer_activations: return
    
    layers = list(layer_activations.keys())
    means = np.array(list(layer_activations.values()))
    vars_arr = np.array(list(layer_variances.values()))
    
    # 1. Normalized Variance
    avg_var = np.mean(vars_arr)
    norm_vars = vars_arr / (avg_var + 1e-12)
    
    # 2. Normalized CV (defined per Kevin's notes: Variance / Mean)
    # Using 1e-12 epsilon to prevent division by zero for dead layers
    cvs = vars_arr / (np.abs(means) + 1e-12) 
    avg_cv = np.mean(cvs)
    norm_cvs = cvs / (avg_cv + 1e-12)
    
    df = pd.DataFrame({
        "Layer": layers,
        "Normalized Variance": norm_vars,
        "Normalized CV": norm_cvs
    })
    
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_normalized_layer_stats.csv"), index=False)
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    
    # --- Plot 1: Normalized Variance ---
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized Variance", color="coral", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-Variance (1.0)")
    ax.set_title(f"Normalized Layer Variance\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Variance / Avg Variance", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_variance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Normalized CV ---
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=df, x="Layer", y="Normalized CV", color="mediumpurple", ax=ax)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=2, label="Average Layer-CV (1.0)")
    ax.set_title(f"Normalized Coefficient of Variation (CV)\n{model_name} | {dataset_name}", fontsize=16, fontweight='bold', pad=15)
    ax.set_ylabel("Layer CV / Avg CV", fontweight='bold')
    ax.set_xlabel("Network Layer", fontweight='bold')
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=9)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_normalized_cv.png"), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_experiments(model, input_tensor, exp_config, layer_names, module_dict, saved_tensors, layer_variances, global_median_var, baseline_probs):
    """Iterates through experiment configurations and calculates the four core metrics."""
    plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore = [], [], [], []

    for exp_name, layer_range in exp_config.items():
        exp_display = exp_name.replace("_", " ")

        # --- Handle Original Model (Baselines) ---
        if layer_range is None:
            plot_data_var.append({"Experiment": exp_display, "Relative Variance": 1.0})
            
            sim_scores = []
            for t_data in saved_tensors.values():
                t_in, t_out = t_data["in"], t_data["out"]
                if t_in.shape == t_out.shape:
                    in_flat = t_in.flatten(start_dim=1)
                    out_flat = t_out.flatten(start_dim=1)
                    try:
                        sim_scores.append(F.cosine_similarity(in_flat, out_flat, dim=1).mean().item())
                    except: pass
            
            global_sim = float(np.median(sim_scores)) if sim_scores else 0.0
            plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": global_sim})
            plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": 0.0})
            
            baseline_cscore = global_sim / (1.0 * (1.0 + 0.0))
            plot_data_cscore.append({"Experiment": exp_display, "Collapse Score": baseline_cscore})
            continue

        ranges = layer_range if isinstance(layer_range, list) else [layer_range]
        block_vars, block_sims = [], []
        bypass_handles, bypass_cache = [], {}
        valid_bypass = True

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

            if start_name and end_name:
                in_range = False
                for name in layer_names:
                    if name == start_name: in_range = True
                    if in_range and name in layer_variances:
                        block_vars.append(layer_variances[name])
                    if name == end_name: break
                
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
                        sim = 0.0 
                    block_sims.append(sim)

                start_mod = module_dict[start_name]
                end_mod = module_dict[end_name]
                bypass_handles.append(start_mod.register_forward_hook(get_start_hook(idx)))
                bypass_handles.append(end_mod.register_forward_hook(get_end_hook(idx)))
            else:
                valid_bypass = False

        # --- Aggregate Metrics ---
        exp_rel_var = (float(np.median(block_vars)) / global_median_var) if block_vars and global_median_var > 0 else 1.0
        plot_data_var.append({"Experiment": exp_display, "Relative Variance": exp_rel_var})

        exp_sim = float(np.median(block_sims)) if block_sims else 0.0
        plot_data_sim.append({"Experiment": exp_display, "Block Redundancy": exp_sim})

        # --- Execute Virtual Bypass ---
        if not valid_bypass:
            kl_div = 50.0 
        else:
            try:
                with torch.no_grad():
                    bypass_logits = model(input_tensor)
                    bypass_log_probs = F.log_softmax(bypass_logits, dim=1)
                kl_div = F.kl_div(bypass_log_probs, baseline_probs, reduction='batchmean').item()
            except Exception:
                kl_div = 50.0 
        
        for h in bypass_handles: h.remove()
        bypass_cache.clear()

        display_kl = kl_div if kl_div < 50.0 else 50.0
        plot_data_kl.append({"Experiment": exp_display, "Prediction Shift (KL)": display_kl})

        # --- Calculate Composite Collapse Score ---
        safe_rel_var = max(exp_rel_var, 1e-8)
        c_score = exp_sim / (safe_rel_var * (1.0 + display_kl))
        plot_data_cscore.append({"Experiment": exp_display, "Collapse Score": c_score})

    return plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore

def save_and_plot_metric(data, y_col, directory, title_prefix, ylabel, hline_val, hline_label, color_base, color_alt, model_name, dataset_name, invert_safe_zone=False):
    """Handles the saving of CSV/TeX files and the generation of publication-ready Seaborn plots."""
    if not data: return
    df = pd.DataFrame(data)
    
    # Save raw data
    df.to_csv(os.path.join(directory, f"{model_name}_{dataset_name}_{y_col.replace(' ', '_')}.csv"), index=False)
    df.to_latex(os.path.join(directory, f"{model_name}_{dataset_name}.tex"), index=False, float_format="%.4f")

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(14, 7))

    df['Color_Group'] = ['Baseline' if exp == 'Original Model' else 'Experiment' for exp in df['Experiment']]
    palette = {'Baseline': color_base, 'Experiment': color_alt}

    sns.barplot(
        data=df, x="Experiment", y=y_col, hue="Color_Group", palette=palette,
        dodge=False, edgecolor="black", linewidth=0.8, zorder=3, ax=ax
    )
    
    ax.legend_.remove()

    ymin, ymax = ax.get_ylim()
    
    if df[y_col].min() >= 0:
        ymin = 0.0  
    else:
        ymin = min(df[y_col].min() * 1.05, ymin)
        
    if hline_val > ymax:
        ymax = hline_val * 1.15
        
    ax.set_ylim(ymin, ymax)

    ax.axhline(0, color='black', linewidth=1.5, zorder=4) 
    ax.axhline(hline_val, color='crimson', linestyle='--', linewidth=2.5, zorder=4, label=hline_label)

    if invert_safe_zone:
        ax.axhspan(hline_val, ymax, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe (High Redundancy)')
        ax.axhspan(ymin, hline_val, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')
    else:
        ax.axhspan(ymin, hline_val, color='#e6f4ea', alpha=0.6, zorder=1, label='Safe')
        ax.axhspan(hline_val, ymax, color='#fce8e6', alpha=0.6, zorder=1, label='Dangerous')

    ax.set_title(f"{title_prefix}\n{model_name} | {dataset_name}", fontsize=18, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', labelpad=10)
    ax.set_xlabel("Structural Modification", fontsize=14, fontweight='bold', labelpad=10)
    
    plt.xticks(rotation=45, ha='right', fontsize=11)
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray', zorder=0)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor='gray', fontsize=12)
    sns.despine(bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"{model_name}_experiment_{y_col.split(' ')[0]}.png"), dpi=300, bbox_inches='tight')
    plt.close()

# def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
#     """
#     Main orchestrator: Computes all four heuristic targets and saves outputs.
#     """
#     print(f"[•] Running Comprehensive Heuristic Analysis for {model_name} on {dataset_name}...")
    
#     model.eval()
#     if len(input_tensor.shape) == 3:
#         input_tensor = input_tensor.unsqueeze(0)

#     module_dict = dict(model.named_modules())
#     layer_names = list(module_dict.keys())

#     # 1. Setup
#     dirs = setup_directories(save_root_dir)

#     # 2. Baseline Pass
#     saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs = run_baseline_pass(model, input_tensor)

#     # 3. Plot Individual & Normalized Layer Stats
#     plot_individual_layers(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name)
    
#     # NEW: Call the normalized plots
#     plot_normalized_metrics(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name)

#     # 4. Process Experiments
#     try:
#         exp_config = get_experiment_config(model_name)
#         if not exp_config:
#             print("[WARN] No experiment config found.")
#             return pd.DataFrame()

#         plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore = evaluate_experiments(
#             model, input_tensor, exp_config, layer_names, module_dict, 
#             saved_tensors, layer_variances, global_median_var, baseline_probs
#         )
#     except Exception as e:
#         print(f"[!] Failed to process experiments: {e}")
#         return pd.DataFrame()

#     # 5. Save and Plot Core Metrics
#     save_and_plot_metric(plot_data_var, "Relative Variance", dirs["var"], "Relative Activation Variance", "Relative Variance (Multiplier)", 1.0, "1.0x Baseline", 'crimson', 'steelblue', model_name, dataset_name)
    
#     global_sim_val = plot_data_sim[0]["Block Redundancy"] if plot_data_sim else 0.0
#     save_and_plot_metric(plot_data_sim, "Block Redundancy", dirs["sim"], "Feature Redundancy (Cosine Similarity)", "Cosine Similarity (1.0 = Identity)", global_sim_val, "Global Median", 'crimson', 'mediumseagreen', model_name, dataset_name, invert_safe_zone=True)

#     save_and_plot_metric(plot_data_kl, "Prediction Shift (KL)", dirs["kl"], "Virtual Bypass Prediction Damage", "KL Divergence (0.0 = Safe | 50.0 = Failed)", 1.0, "Critical Threshold (Approx)", 'crimson', 'teal', model_name, dataset_name)

#     global_cscore_val = plot_data_cscore[0]["Collapse Score"] if plot_data_cscore else 1.0
#     save_and_plot_metric(plot_data_cscore, "Collapse Score", dirs["cscore"], "Composite Activational Collapse Score", "C_Score (Higher = Safer to Collapse)", global_cscore_val, "Baseline Architecture Score", 'crimson', 'purple', model_name, dataset_name, invert_safe_zone=True)

#     return pd.DataFrame(plot_data_cscore)

def analyze_collapse_heuristics(model, input_tensor, save_root_dir, model_name, dataset_name):
    """
    Main orchestrator: Computes all four heuristic targets and saves outputs.
    """
    print(f"[•] Running Comprehensive Heuristic Analysis for {model_name} on {dataset_name}...")
    
    model.eval()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    module_dict = dict(model.named_modules())
    layer_names = list(module_dict.keys())

    # 1. Setup
    dirs = setup_directories(save_root_dir)

    # 2. Baseline Pass
    saved_tensors, layer_variances, layer_activations, global_median_var, baseline_probs = run_baseline_pass(model, input_tensor)

    # 3. Fetch Experiment Config EARLY so the plotter can use it
    exp_config = get_experiment_config(model_name)
    if not exp_config:
        print("[WARN] No experiment config found.")
        # Proceed with empty config so it at least plots the raw data
        exp_config = {}

    # 4. Plot Individual & Normalized Layer Stats (Now passing exp_config!)
    plot_individual_layers(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name, exp_config)
    
    # NEW: Call the normalized plots
    plot_normalized_metrics(layer_activations, layer_variances, dirs["layer_stats"], model_name, dataset_name)

    # 5. Process Experiments
    try:
        plot_data_var, plot_data_sim, plot_data_kl, plot_data_cscore = evaluate_experiments(
            model, input_tensor, exp_config, layer_names, module_dict, 
            saved_tensors, layer_variances, global_median_var, baseline_probs
        )
    except Exception as e:
        print(f"[!] Failed to process experiments: {e}")
        return pd.DataFrame()

    # 6. Save and Plot Core Metrics
    save_and_plot_metric(plot_data_var, "Relative Variance", dirs["var"], "Relative Activation Variance", "Relative Variance (Multiplier)", 1.0, "1.0x Baseline", 'crimson', 'steelblue', model_name, dataset_name)
    
    global_sim_val = plot_data_sim[0]["Block Redundancy"] if plot_data_sim else 0.0
    save_and_plot_metric(plot_data_sim, "Block Redundancy", dirs["sim"], "Feature Redundancy (Cosine Similarity)", "Cosine Similarity (1.0 = Identity)", global_sim_val, "Global Median", 'crimson', 'mediumseagreen', model_name, dataset_name, invert_safe_zone=True)

    save_and_plot_metric(plot_data_kl, "Prediction Shift (KL)", dirs["kl"], "Virtual Bypass Prediction Damage", "KL Divergence (0.0 = Safe | 50.0 = Failed)", 1.0, "Critical Threshold (Approx)", 'crimson', 'teal', model_name, dataset_name)

    global_cscore_val = plot_data_cscore[0]["Collapse Score"] if plot_data_cscore else 1.0
    save_and_plot_metric(plot_data_cscore, "Collapse Score", dirs["cscore"], "Composite Activational Collapse Score", "C_Score (Higher = Safer to Collapse)", global_cscore_val, "Baseline Architecture Score", 'crimson', 'purple', model_name, dataset_name, invert_safe_zone=True)

    return pd.DataFrame(plot_data_cscore)

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
