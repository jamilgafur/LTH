import os
import torch
from pyPrune.models.Vgg16 import VGG16
from pyPrune.models.RegNetX import RegNetX_400MF 
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
import os

CHECKPOINT_BASES = {
    "VGG16": {
        "Cifar10": glob.glob("../structured_study/pruning_checkpoints/*Vgg16*datasetcifar10_*")[0] + "/",
        "Cifar100": glob.glob("../structured_study/pruning_checkpoints/*Vgg16*datasetcifar100_*")[0] + "/",
        "TinyImageNet": glob.glob("../structured_study/pruning_checkpoints/*Vgg16*datasettinyimagenet_*")[0] + "/",
        "ImageNet": glob.glob("../structured_study/pruning_checkpoints/*Vgg16*datasetimagenet_*")[0] + "/",
    },
    "RegNetX_400MF": {
        "Cifar10": glob.glob("../structured_study/pruning_checkpoints/*RegNetX*datasetcifar10_*")[0] + "/",
        # "Cifar100": glob.glob("../structured_study/pruning_checkpoints/*RegNetX*datasetcifar100_*")[0] + "/",
        # "TinyImageNet": glob.glob("../structured_study/pruning_checkpoints/*RegNetX*datasettinyimagenet_*")[0] + "/",
        # "ImageNet": glob.glob("../structured_study/pruning_checkpoints/*RegNetX*datasetimagenet_*")[0] + "/",
    }
}# To make sure that the paths are correct, print them out
for model, datasets in CHECKPOINT_BASES.items():
    for dataset, path in datasets.items():
        print(f"{model} - {dataset}: {path}")

CHECKPOINT_FILES = {
    "VGG16": {
        "Cifar10": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "Cifar100": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "TinyImageNet": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "ImageNet": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
    },
    "RegNetX_400MF": {
        "Cifar10": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "Cifar100": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "TinyImageNet": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
        "ImageNet": ("checkpoint_Finetuned_0.000000.pth", "checkpoint_Original_0.000000.pth"),
    }
}

EXPERIMENTS = {
    "VGG16": {
        "Cifar10": {
            "Original Model": None,
            "Last 2": ('features.conv_12', 'features.conv_13'),  # Conv layers at the last stage
            "Stage 5": ('features.conv_11', 'features.conv_13'),  # Last convolution block
            "Stage 4": ('features.conv_8', 'features.conv_10'),  # Mid convolution block
            "Stage 3": ('features.conv_5', 'features.conv_7'),  # Earlier convolution block
            "Stage 4-5": ('features.conv_8', 'features.conv_13'),  # Combine mid and late conv layers
            "Stage 3-5": ('features.conv_5', 'features.conv_13'),  # Combine early and late conv layers
            "Stage 2-5": ('features.conv_3', 'features.conv_13'),  # Combine early and late conv layers
        },
        "Cifar100": {
            "Original Model": None,
            "Last 2": ('features.conv_9', 'features.conv_10'),
            "Stage 5": ('features.conv_11', 'features.conv_13'),  # Last convolution block
            "Stage 4": ('features.conv_8', 'features.conv_10'),  # Mid convolution block
            "Stage 3": ('features.conv_5', 'features.conv_7'),  # Earlier convolution block
            "Stage 4-5": ('features.conv_8', 'features.conv_13'),  # Combine mid and late conv layers
            "Stage 3-5": ('features.conv_5', 'features.conv_13'),  # Combine early and late conv layers
            "Stage 2-5": ('features.conv_3', 'features.conv_13'),  # Combine early and late conv layers
        },
        "TinyImageNet": {
            "Original Model": None,
            "Last 2": ('features.conv9', 'features.conv10'),
            "Stage 5": ('features.conv_11', 'features.conv_13'),  # Last convolution block
            "Stage 4": ('features.conv_8', 'features.conv_10'),  # Mid convolution block
            "Stage 3": ('features.conv_5', 'features.conv_7'),  # Earlier convolution block
            "Stage 4-5": ('features.conv_8', 'features.conv_13'),  # Combine mid and late conv layers
            "Stage 3-5": ('features.conv_5', 'features.conv_13'),  # Combine early and late conv layers
            "Stage 2-5": ('features.conv_3', 'features.conv_13'),  # Combine early and late conv layers
        },
        "ImageNet": {
            "Original Model": None,
            "Last 2": ('features.conv9', 'features.conv10'),
            "Stage 5": ('features.conv8', 'features.conv9'),
            "Stage 5": ('features.conv_11', 'features.conv_13'),  # Last convolution block
            "Stage 4": ('features.conv_8', 'features.conv_10'),  # Mid convolution block
            "Stage 3": ('features.conv_5', 'features.conv_7'),  # Earlier convolution block
            "Stage 4-5": ('features.conv_8', 'features.conv_13'),  # Combine mid and late conv layers
            "Stage 3-5": ('features.conv_5', 'features.conv_13'),  # Combine early and late conv layers
            "Stage 2-5": ('features.conv_3', 'features.conv_13'),  # Combine early and late conv layers
        },
    },
    "RegNetX_400MF": {
        "Cifar10": {
            "Original Model": None,
            "Last 2": ('stage4.stage4_block6.block.conv1', 'stage4.stage4_block6.block.conv3'),
        }
    }
}

# -------------------------------
# IMP Pruning Logic Integration
# -------------------------------

def imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs, finetune_epochs, device, save_dir, strategy, patience, experiment_name=None):
    """
    Function to run pruning based on the IterativePruner class.
    """
    save_dir = os.path.join(save_dir, experiment_name)

    print("\nModel:", model)
    print("Optimizer:", optimizer)
    print("Scheduler:", scheduler)
    print("Criterion:", criterion)
    print("Train Loader:", train_loader)
    print("Test Loader:", test_loader)
    print("Steps:", steps)
    print("Pretrain Epochs:", pretrain_epochs)
    print("Device:", device)
    print("Finetune Epochs:", finetune_epochs)
    print("Save Dir:", save_dir)
    print("Strategy:", strategy)
    # Initialize the IterativePruner
    pruner = IterativePruner(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        steps=steps,
        device=device,
        save_dir=save_dir,
        pretrain_epochs=pretrain_epochs,
        finetune_epochs=10,
        scheduler=scheduler,
        strategy=MagnitudePruningStrategy.MagnitudePruningStrategy(device=device)
    )

    print("Running pruning process...")
    pruner.run()  # Start pruning process
    print("Pruning process complete.")
    
    return pruner

def run_experiments_for_dataset(experiments, dataset, model_path_097, model_path_000, train_loader, test_loader, device, epochs, pretrain, model_class, model_kwargs, post_compress_epochs, run_all, experiment_func):
    """Run all experiments for a given dataset."""
    save_path = f"{model_class.__name__}_{dataset}_{CHECKPOINT_FILES[model_class.__name__][dataset][0]}_epochs{epochs}_pretrain{pretrain}_postcompress{post_compress_epochs}"
    
    # Define optimizer, scheduler, and criterion
    def create_optimizer_scheduler(model, learning_rate=1e-3):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return optimizer, scheduler

    criterion = torch.nn.CrossEntropyLoss()

    # Define the number of pruning steps (this is arbitrary, you can adjust it)
    steps = exponential_decay_list(steps=21)
    steps = [steps[i] for i in range(len(steps)) if i % 3 == 0 or i == len(steps) - 1]
    print(f"Pruning steps: {steps}")

    train_loader, test_loader,input_size, input_channels, num_classes = load_dataset(dataset, model_class.__name__)
    model_kwargs['num_classes'] = num_classes
    input_tensor = next(iter(train_loader))[0]
    model_kwargs['one_batch'] = input_tensor
    
    if run_all:
        run_jf_experiment(experiments, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                        model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path,
                        post_compress_epochs=post_compress_epochs)     
        run_kevin_experiment(experiments, model_path_000, train_loader, test_loader, device, epochs,
                         model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                            save_path=save_path, post_compress_epochs=post_compress_epochs)
    else:
        for name, layers in experiments.items():
            print(f"\n--- Running experiment: {name} ---")
            if args.JF:
                model = run_jf_experiment({name: layers}, model_path_097, train_loader, test_loader, device, epochs, pretrain,
                                        model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size, save_path=save_path,
                                        post_compress_epochs=post_compress_epochs)
                if args.imp:
                    optimizer, scheduler = create_optimizer_scheduler(model)
                    imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs=pretrain, finetune_epochs=pretrain, device=device, save_dir=save_path, strategy="magnitude", patience=5, experiment_name=name)

            elif args.Kevin:
                model = run_kevin_experiment({name: layers}, model_path_000, train_loader, test_loader, device, epochs,
                                    model_class=model_class, model_kwargs=model_kwargs, data_shape=input_size,
                                    save_path=save_path, post_compress_epochs=post_compress_epochs)
                if args.imp:
                    optimizer, scheduler = create_optimizer_scheduler(model)
                    imp_prune(model, optimizer, scheduler, criterion, train_loader, test_loader, steps, pretrain_epochs=pretrain, finetune_epochs=pretrain, device=device, save_dir=save_path, strategy="magnitude", patience=5, experiment_name=name)
            else:
                raise ValueError("You must specify either --JF or --Kevin to run the corresponding experiment.")
# -------------------------------
# Main Function
# -------------------------------

if __name__ == "__main__":
    import argparse

    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RegNetX_400MF", choices=["VGG16", "RegNetX_400MF"], help="Model architecture to use")
    parser.add_argument("--dataset", type=str, help="Dataset to use (Cifar10, Cifar100, ImageNet, TinyImageNet)",default="Cifar10")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--pretrain", type=int, default=30, help="Number of pretraining epochs")
    parser.add_argument("--experiment", type=str, help="Experiment to run",default="Last 2")
    parser.add_argument("--post_compress_epochs", type=int, default=10, help="Number of post-pruning compression epochs")
    parser.add_argument("--run_all", action="store_true", help="Run all experiments")
    parser.add_argument("--imp", action="store_true", help="Apply Iterative Magnitude Pruning")
    parser.add_argument("--JF", action="store_true", help="Run JF experiments")
    parser.add_argument("--Kevin", action="store_true", help="Run Kevin experiments")
    args = parser.parse_args()
    print(args)
    print(f"has GPU: {torch.cuda.is_available()}")
    model_name = args.model
    dataset = args.dataset

    model_class = VGG16 if model_name == "VGG16" else RegNetX_400MF
    model_kwargs = {}  # Customize based on model

    # Select the checkpoint paths
    base_path = CHECKPOINT_BASES[model_name][dataset]
    model_path_097 = os.path.join(base_path, CHECKPOINT_FILES[model_name][dataset][0])
    model_path_000 = os.path.join(base_path, CHECKPOINT_FILES[model_name][dataset][1])

    # Modify the logic for selecting the experiment dictionary
    if args.run_all:
        experiment_dict = EXPERIMENTS[model_name][dataset]  # If running all experiments, use the entire dictionary
    elif args.experiment is not None:
        experiment_dict = {args.experiment: EXPERIMENTS[model_name][dataset][args.experiment]}  # If a specific experiment is passed
    else:
        raise ValueError("You must specify an experiment with --experiment or use --run_all to run all experiments.")

    # Example: Run experiments
    run_experiments_for_dataset(experiment_dict, dataset, model_path_097, model_path_000, None, None, 'cpu', args.epochs, args.pretrain, model_class, model_kwargs, args.post_compress_epochs, args.run_all, experiment_func=imp_prune)
