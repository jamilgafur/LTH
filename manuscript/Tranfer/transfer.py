# Transfer.py
import os
import torch
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF
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
}

EXPERIMENTS = {
    "VGG16": {
        "Cifar10": {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        },
        "Cifar100": {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        },
        "tinyimagenet": {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        },
        "imagenet": {
            "Original Model": None,
            "Last 2": ("features.conv_12", "features.conv_13"),
            "Stage 5": ("features.conv_11", "features.conv_13"),
            "Stage 4": ("features.conv_8", "features.conv_10"),
            "Stage 3": ("features.conv_5", "features.conv_7"),
            "Stage 4-5": ("features.conv_8", "features.conv_13"),
            "Stage 3-5": ("features.conv_5", "features.conv_13"),
            "Stage 2-5": ("features.conv_3", "features.conv_13"),
        },
    },
    "RegNetX_400MF": {
        "Cifar10": {
            "Original Model": None,

            # Single-stage collapses (single tuples)
            "Last 2": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block6.block.conv3"),
            "Stage 4": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),
            "Stage 3": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),
            "Stage 2": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
            "Stage 1": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 2-4": [
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 1-4": [
                ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),  # Stage 1
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],

            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv2"),
            "Stage 2 first 2 conv": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv2"),
            "Stage 3 first 2 conv": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block1.block.conv1"),
            "Stage 4 first 2 conv": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block1.block.conv1"),

            "Stage 1 last 2 conv": ("stage1.stage1_block0.block.conv2", "stage1.stage1_block0.block.conv3"),
            "Stage 2 last 2 conv": ("stage2.stage2_block0.block.conv2", "stage2.stage2_block0.block.conv3"),
            "Stage 3 last 2 conv": ("stage3.stage3_block2.block.conv3", "stage3.stage3_block3.block.conv3"),
            "Stage 4 last 2 conv": ("stage4.stage4_block4.block.conv3", "stage4.stage4_block5.block.conv3"),
        },
        "Cifar100": {
            "Original Model": None,

            # Single-stage collapses (single tuples)
            "Last 2": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block6.block.conv3"),
            "Stage 4": ("stage4.stage4_block3.block.conv1", "stage4.stage4_block3.block.conv3"),
            "Stage 3": ("stage3.stage3_block1.block.conv1", "stage3.stage3_block1.block.conv3"),
            "Stage 2": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
            "Stage 1": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 2-4": [
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 1-4": [
                ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),  # Stage 1
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],

            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv2"),
            "Stage 2 first 2 conv": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv2"),
            "Stage 3 first 2 conv": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block1.block.conv1"),
            "Stage 4 first 2 conv": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block1.block.conv1"),

            "Stage 1 last 2 conv": ("stage1.stage1_block0.block.conv2", "stage1.stage1_block0.block.conv3"),
            "Stage 2 last 2 conv": ("stage2.stage2_block0.block.conv2", "stage2.stage2_block0.block.conv3"),
            "Stage 3 last 2 conv": ("stage3.stage3_block2.block.conv3", "stage3.stage3_block3.block.conv3"),
            "Stage 4 last 2 conv": ("stage4.stage4_block4.block.conv3", "stage4.stage4_block5.block.conv3"),
        },
        "tinyimagenet": {
            "Original Model": None,

            # Single-stage collapses (single tuples)
            "Last 2": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block6.block.conv3"),
            "Stage 4": ("stage4.stage4_block3.block.conv1", "stage4.stage4_block3.block.conv3"),
            "Stage 3": ("stage3.stage3_block1.block.conv1", "stage3.stage3_block1.block.conv3"),
            "Stage 2": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
            "Stage 1": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 2-4": [
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 1-4": [
                ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),  # Stage 1
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],

            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv2"),
            "Stage 2 first 2 conv": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv2"),
            "Stage 3 first 2 conv": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block1.block.conv1"),
            "Stage 4 first 2 conv": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block1.block.conv1"),

            "Stage 1 last 2 conv": ("stage1.stage1_block0.block.conv2", "stage1.stage1_block0.block.conv3"),
            "Stage 2 last 2 conv": ("stage2.stage2_block0.block.conv2", "stage2.stage2_block0.block.conv3"),
            "Stage 3 last 2 conv": ("stage3.stage3_block2.block.conv3", "stage3.stage3_block3.block.conv3"),
            "Stage 4 last 2 conv": ("stage4.stage4_block4.block.conv3", "stage4.stage4_block5.block.conv3"),
        },
        "imagenet": {
            "Original Model": None,

            # Single-stage collapses (single tuples)
            "Last 2": ("stage4.stage4_block5.block.conv1", "stage4.stage4_block6.block.conv3"),
            "Stage 4": ("stage4.stage4_block3.block.conv1", "stage4.stage4_block3.block.conv3"),
            "Stage 3": ("stage3.stage3_block1.block.conv1", "stage3.stage3_block1.block.conv3"),
            "Stage 2": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),
            "Stage 1": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),

            # Multi-stage collapses (lists of tuples)
            "Stage 3-4": [
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 2-4": [
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],
            "Stage 1-4": [
                ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv3"),  # Stage 1
                ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv3"),  # Stage 2
                ("stage3.stage3_block0.block.conv1", "stage3.stage3_block3.block.conv3"),  # Stage 3
                ("stage4.stage4_block0.block.conv1", "stage4.stage4_block6.block.conv3"),  # Stage 4
            ],

            # Stage-specific first/last conv pairs
            "Stage 1 first 2 conv": ("stage1.stage1_block0.block.conv1", "stage1.stage1_block0.block.conv2"),
            "Stage 2 first 2 conv": ("stage2.stage2_block0.block.conv1", "stage2.stage2_block0.block.conv2"),
            "Stage 3 first 2 conv": ("stage3.stage3_block0.block.conv1", "stage3.stage3_block1.block.conv1"),
            "Stage 4 first 2 conv": ("stage4.stage4_block0.block.conv1", "stage4.stage4_block1.block.conv1"),

            "Stage 1 last 2 conv": ("stage1.stage1_block0.block.conv2", "stage1.stage1_block0.block.conv3"),
            "Stage 2 last 2 conv": ("stage2.stage2_block0.block.conv2", "stage2.stage2_block0.block.conv3"),
            "Stage 3 last 2 conv": ("stage3.stage3_block2.block.conv3", "stage3.stage3_block3.block.conv3"),
            "Stage 4 last 2 conv": ("stage4.stage4_block4.block.conv3", "stage4.stage4_block5.block.conv3"),
        }
    },
    "InceptionNet": {
        "Cifar10": {
            "Original Model": None,
        },
        "Cifar100": {
            "Original Model": None,
        },
        "tinyimagenet": {
            "Original Model": None,
        },
        "imagenet": {
            "Original Model": None,
        },
    },
    "XceptionNet": {
        "Cifar10": {
            "Original Model": None,
        },
        "Cifar100": {
            "Original Model": None,
        },
        "tinyimagenet": {
            "Original Model": None,
        },
        "imagenet": {
            "Original Model": None,
        },
    },"MobileNet": {
        "Cifar10": {
            "Original Model": None,
        },
        "Cifar100": {
            "Original Model": None,
        },
        "tinyimagenet": {
            "Original Model": None,
        },
        "imagenet": {
            "Original Model": None,
        },
    },
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

    # Ensure InceptionNet is not used with JF experiments
    if model_class == InceptionNet and args.JF:
        raise ValueError("JF experiments are not supported for InceptionNet.")
    
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)
    model_kwargs["num_classes"] = num_classes
    model_kwargs["one_batch"] = next(iter(load_dataset(dataset, model_class)[0]))[0]
    
    return train_loader, test_loader, model_class, model_kwargs, input_size, input_channels, num_classes

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
        # Adjust epochs for original model experiment
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

    # Handle special cases for InceptionNet, XceptionNet, and MobileNet
    if model_class in [InceptionNet, XceptionNet, MobileNet]:
        steps = [0]
        epochs = pretrain
        pretrain = 0
    else:
        steps = exponential_decay_list(steps=21)
    print(f"Pruning steps: {steps}")

    # Initialize the dataset
    train_loader, test_loader, input_size, input_channels, num_classes = load_dataset(dataset, model_class)

    # Iterate over experiments and run
    for name, layers in experiments.items():
        print(f"\n--- Running experiment: {name} ---")
        model = run_jf_or_kevin_experiment(
            name, layers, model_class, model_kwargs, input_size, epochs, pretrain, experiment_func, save_path,
            post_compress_epochs, quant, model_path_097, model_path_000, train_loader, test_loader, device, args
        )
    
# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="XceptionNet", choices=["VGG16", "RegNetX_400MF", "InceptionNet", "XceptionNet", "MobileNet"], help="Model architecture to use")
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
