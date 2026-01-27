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
from pyPrune.utils import *
from torch.backends import cudnn
import random
import numpy as np
import seaborn as sns
import pandas as pd

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

Vgg_common = {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 2": ("features.conv_3", "features.conv_4"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        }
RegNetX_common =  {
            "Original Model": None,
            # Single-stage collapses (single tuples)
            "Last 2": (
                "stage4.stage4_block5.block.conv1",
                "stage4.stage4_block6.block.conv3",
            ),
            "Stage 4": (
                "stage4.stage4_block0.block.conv1",
                "stage4.stage4_block6.block.conv3",
            ),
            "Stage 3": (
                "stage3.stage3_block0.block.conv1",
                "stage3.stage3_block3.block.conv3",
            ),
            "Stage 2": (
                "stage2.stage2_block0.block.conv1",
                "stage2.stage2_block0.block.conv3",
            ),
            "Stage 1": (
                "stage1.stage1_block0.block.conv1",
                "stage1.stage1_block0.block.conv3",
            ),
            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            "Stage 2-4": [
                (
                    "stage2.stage2_block0.block.conv1",
                    "stage2.stage2_block0.block.conv3",
                ),  # Stage 2
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            "Stage 1-4": [
                (
                    "stage1.stage1_block0.block.conv1",
                    "stage1.stage1_block0.block.conv3",
                ),  # Stage 1
                (
                    "stage2.stage2_block0.block.conv1",
                    "stage2.stage2_block0.block.conv3",
                ),  # Stage 2
                (
                    "stage3.stage3_block0.block.conv1",
                    "stage3.stage3_block3.block.conv3",
                ),  # Stage 3
                (
                    "stage4.stage4_block0.block.conv1",
                    "stage4.stage4_block6.block.conv3",
                ),  # Stage 4
            ],
            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": (
                "stage1.stage1_block0.block.conv1",
                "stage1.stage1_block0.block.conv2",
            ),
            "Stage 2 first 2 conv": (
                "stage2.stage2_block0.block.conv1",
                "stage2.stage2_block0.block.conv2",
            ),
            "Stage 3 first 2 conv": (
                "stage3.stage3_block0.block.conv1",
                "stage3.stage3_block1.block.conv1",
            ),
            "Stage 4 first 2 conv": (
                "stage4.stage4_block0.block.conv1",
                "stage4.stage4_block1.block.conv1",
            ),
            "Stage 1 last 2 conv": (
                "stage1.stage1_block0.block.conv2",
                "stage1.stage1_block0.block.conv3",
            ),
            "Stage 2 last 2 conv": (
                "stage2.stage2_block0.block.conv2",
                "stage2.stage2_block0.block.conv3",
            ),
            "Stage 3 last 2 conv": (
                "stage3.stage3_block2.block.conv3",
                "stage3.stage3_block3.block.conv3",
            ),
            "Stage 4 last 2 conv": (
                "stage4.stage4_block4.block.conv3",
                "stage4.stage4_block5.block.conv3",
            ),
        }
XceptionNet_common =  {
            "Original Model": None,
            "Stage 5": ("block5.depthwise", "block5.bn2"),
            "Stage 4": ("block4.depthwise", "block5.depthwise"),
            "Stage 3": ("block3.depthwise", "block4.depthwise"),
            "Stage 2": ("block2.depthwise", "block3.depthwise"),
            "Stage 1": ("block1.depthwise", "block2.depthwise"),
            "Stage 3-5": ("block3.depthwise", "block5.depthwise"),
            "Stage 2-5": ("block2.depthwise", "block5.depthwise"),
            "Stage 1-5": ("block1.depthwise", "block5"),
        }
mobileNet_common =  {
            "Original Model": None,
            "Stage 7": ("block7.depthwise", "block7.bn2"),
            "Stage 6": ("block6.depthwise", "block7.depthwise"),
            "Stage 5": ("block5.depthwise", "block6.depthwise"),
            "Stage 4": ("block4.depthwise", "block5.depthwise"),
            "Stage 3": ("block3.depthwise", "block4.depthwise"),
            "Stage 2": ("block2.depthwise", "block3.depthwise"),
            "Stage 1": ("block1.depthwise", "block2.depthwise"),
            "Stage 5-7": ("block5.depthwise", "block7.depthwise"),
            "Stage 4-7": ("block4.depthwise", "block7.depthwise"),
            "Stage 6-7": ("block6.depthwise", "block7.depthwise"),
            "Stage 3-7": ("block3.depthwise", "block7.depthwise"),
            "Stage 2-7": ("block2.depthwise", "block7.depthwise"),
            "Stage 1-7": ("block1.depthwise", "block7.depthwise"),
            "Last 2": ("block6.depthwise", "block7.depthwise"),
        }
InceptionNet_common = {
            "Original Model": None,
            # Single-stage collapses
            "Stage 5": (
                "stage5.inception_5a",
                "stage5.inception_5b",
            ),
            "Stage 4": (
                "stage4.inception_4a",
                "stage4.inception_4b",
            ),
            "Stage 3": (
                "stage3.inception_3a",
                "stage3.inception_3b",
            ),
            "Stage 2": (
                "stage2.inception_2a",
                "stage2.inception_2b",
            ),
            "Stage 2-5": [
                (
                    "stage2.inception_2a",
                    "stage2.inception_2b",
                ),  # Stage 2
                (
                    "stage3.inception_3a",
                    "stage3.inception_3b",
                ),  # Stage 3
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],
            "Stage 3-5": [
                (
                    "stage3.inception_3a",
                    "stage3.inception_3b",
                ),  # Stage 3
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],
            "Stage 4-5": [
                (
                    "stage4.inception_4a",
                    "stage4.inception_4b",
                ),  # Stage 4
                (
                    "stage5.inception_5a",
                    "stage5.inception_5b",
                ),  # Stage 5
            ],  
            "Last 2": (
                "stage5.inception_5a",
                "stage5.inception_5b",
            ),
        }
ConvNeXt_common = {
    "Original Model": None,
    # Stage 1
    "Stage 1": ("stage1.block1_1", "stage1.block1_2"),

    # Stage 2
    "Stage 2": ("stage2.block2_1", "stage2.block2_2"),

    # Stage 3 (strong redundancy)
    "Stage 3": ("stage3.block3_1", "stage3.block3_3"),

    # Stage 4
    "Stage 4": ("stage4.block4_1", "stage4.block4_2"),
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
    Safely runs the pre-experiment training and heuristic analysis (Identity Score, Arithmetic Intensity).
    Uses atomic directory creation (os.mkdir) to ensure only ONE job in an HPC array performs this task.
    """
    # Define paths
    lock_dir_path = f"{model_name}_{dataset}_heuristics.lock_dir"
    done_marker_path = f"{model_name}_{dataset}_heuristics_done.marker"
    plots_dir = os.path.join("runs", "plots")
    
    ensure_dir(plots_dir)

    # 1. Check if already done
    if os.path.exists(done_marker_path):
        print("[INFO] Heuristic profiling already completed. Skipping.")
        return

    # 2. Attempt to acquire lock via Atomic Directory Creation
    # os.mkdir is atomic on POSIX systems (including NFS), making it a safe lock mechanism.
    print(f"[INFO] Attempting to acquire lock for heuristic profiling: {lock_dir_path}")
    try:
        os.mkdir(lock_dir_path)
        print("[INFO] Lock acquired! Starting training for heuristic analysis...")
    except FileExistsError:
        print("[INFO] Lock busy. Another job is performing the heuristic profiling. Skipping.")
        return
    except OSError as e:
        print(f"[WARN] Failed to acquire lock due to unexpected OS error: {e}")
        return

    try:
        # --- Critical Section (Only one job runs this) ---
        
        # A. Resolve class from string if necessary
        if isinstance(model_class, str):
            model_class_obj = eval(model_class) #
        else:
            model_class_obj = model_class

        # B. Initialize fresh model
        model = model_class_obj(**model_kwargs).to(device)
        optimizer, scheduler = create_optimizer_scheduler(model, learning_rate=0.001) 
        
        # C. Train for specified epochs to stabilize weights
        print(f"[INFO] Training model for {epochs} epochs to analyze functional redundancy...")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device) #
            print(f"    [Epoch {epoch}] Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
            if scheduler: scheduler.step()
            
        # D. Run Heuristic Analysis (Identity Score & Arithmetic Intensity)
        input_sample = model_kwargs["one_batch"].to(device)
        exp_name = f"{model_name}_{dataset}_Heuristics"
        
        # Call the updated diagnostic function
        analyze_collapse_heuristics(model, input_sample, plots_dir, exp_name) #
        
        # E. Mark as done
        with open(done_marker_path, 'w') as f:
            f.write(f"Completed at {time.ctime()}")
            
        print("[INFO] Heuristic profiling complete. Plots saved.")

    except Exception as e:
        print(f"[ERROR] An error occurred during heuristic profiling: {e}")
        # Optional: You might want to remove the lock dir here if you want to allow retries,
        # but usually on HPC errors you want to inspect manually.

    finally:
        # F. Release lock
        try:
            os.rmdir(lock_dir_path)
            print("[INFO] Lock released.")
        except OSError:
            print("[WARN] Could not remove lock directory (it may have been removed already).")

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

def analyze_collapse_heuristics(model, input_tensor, save_dir, exp_name):
    """
    Analyzes layers using 4 metrics to identify collapse candidates:
    1. Identity Score (Conv Only): Is input ≈ output? (High = Redundant)
    2. Arithmetic Intensity: FLOPs / Byte (Low = Memory Bound/Latency Bottleneck)
    3. Weight Magnitude (L1): Are weights near zero? (Low = Insignificant)
    4. Activation Variance: Is output static? (Low = Dead/Constant features)
    """
    print(f"[•] Running Extended Collapse Heuristics...")
    
    model.eval()
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # 1. Get FLOPs first
    try:
        from fvcore.nn import FlopCountAnalysis
        flops_counter = FlopCountAnalysis(model, input_tensor)
        flops_dict = flops_counter.by_module()
    except Exception as e:
        print(f"[!] FLOPs count failed: {e}")
        flops_dict = {}

    layer_stats = {}

    # 2. Hook for metrics
    def heuristic_hook(name, layer_type):
        def fn(module, inp, out):
            if not isinstance(out, torch.Tensor) or not isinstance(inp[0], torch.Tensor):
                return
            
            x = inp[0].detach()
            y = out.detach()
            
            # --- Metric A: Identity Score (Cosine Sim) ---
            identity_score = 0.0
            if x.shape == y.shape:
                x_flat = x.flatten(start_dim=1)
                y_flat = y.flatten(start_dim=1)
                try:
                    identity_score = F.cosine_similarity(x_flat, y_flat, dim=1).mean().item()
                except:
                    identity_score = 0.0
            
            # --- Metric B: Memory Bytes (for Intensity) ---
            dtype_size = x.element_size()
            weight_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            total_bytes = (x.numel() * dtype_size) + (y.numel() * dtype_size) + weight_bytes

            # --- Metric C: Weight Magnitude (Normalized L1) ---
            # Measures if the layer is "doing work" or just effectively zero
            weight_l1 = 0.0
            if hasattr(module, 'weight') and module.weight is not None:
                # Normalize by number of elements so large layers don't dominate
                weight_l1 = module.weight.norm(p=1).item() / module.weight.numel()

            # --- Metric D: Activation Variance ---
            # Measures if the feature map contains diverse information
            act_var = y.var().item()
            
            layer_stats[name] = {
                "layer_type": layer_type,
                "identity_score": identity_score,
                "memory_bytes": total_bytes,
                "weight_l1": weight_l1,
                "act_var": act_var
            }
        return fn

    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)) or "block" in name.lower():
            hooks.append(module.register_forward_hook(heuristic_hook(name, type(module).__name__)))

    # Run forward pass
    with torch.no_grad():
        model(input_tensor)

    for h in hooks: h.remove()

    # 3. Aggregate Data
    results = []
    for name, stats in layer_stats.items():
        flops_val = flops_dict.get(name, 0)
        mem_bytes = stats["memory_bytes"]
        ai = flops_val / mem_bytes if mem_bytes > 0 else 0.0
        
        results.append({
            "layer": name,
            "layer_type": stats["layer_type"],
            "identity_score": stats["identity_score"],
            "arithmetic_intensity": ai,
            "weight_l1": stats["weight_l1"],
            "act_var": stats["act_var"],
            "flops": flops_val
        })

    df = pd.DataFrame(results)
    
    # Save CSV
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f"{exp_name}_extended_heuristics.csv"), index=False)

    # 4. Plotting (4 Rows)
    if not df.empty:
        fig, axes = plt.subplots(4, 1, figsize=(max(12, len(df)*0.3), 16), sharex=True)
        
        # Row 1: Identity Score (Conv Only)
        df_conv = df[df['layer_type'].str.contains("Conv", case=False, na=False)]
        if not df_conv.empty:
            sns.barplot(x="layer", y="identity_score", data=df_conv, ax=axes[0], color="mediumpurple")
            axes[0].set_title(f"1. Identity Score (Conv Only) - Higher = Collapse Candidate")
            axes[0].set_ylim(0, 1.05)
            # Annotate top candidates
            for i, row in enumerate(df_conv.itertuples()):
                if row.identity_score > 0.8:
                    axes[0].text(i, row.identity_score, f"{row.identity_score:.2f}", 
                                 ha='center', va='bottom', fontsize=7, fontweight='bold')
        
        # Row 2: Arithmetic Intensity
        sns.barplot(x="layer", y="arithmetic_intensity", data=df, ax=axes[1], color="coral")
        axes[1].set_title(f"2. Arithmetic Intensity (FLOPs/Byte) - Lower = Memory Bound")
        axes[1].set_yscale("log") # Log scale often helps here

        # Row 3: Weight Magnitude (L1)
        sns.barplot(x="layer", y="weight_l1", data=df, ax=axes[2], color="teal")
        axes[2].set_title(f"3. Weight Magnitude (Norm. L1) - Lower = Insignificant Weights")

        # Row 4: Activation Variance
        sns.barplot(x="layer", y="act_var", data=df, ax=axes[3], color="goldenrod")
        axes[3].set_title(f"4. Activation Variance - Lower = Dead/Static Features")
        axes[3].set_yscale("log")

        # Layout cleanup
        for ax in axes:
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            ax.set_xlabel("")
        
        axes[-1].set_xticklabels(axes[-1].get_xticklabels(), rotation=90, fontsize=8)
        axes[-1].set_xlabel("Layer Name")

        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{exp_name}_extended_heuristics.svg")
        plt.savefig(plot_path)
        plt.close()
        print(f"[✓] Extended heuristics saved to {plot_path}")

    return df

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
