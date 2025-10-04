import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader

import numpy as np
import copy
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Custom models
from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.ResNet50 import ResNet50
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.EfficientNet import EfficientNetB7
from pyPrune.models.Vgg16 import VGG16

from pyPrune.pruneMethods.IterativePruner import IterativePruner
from pyPrune.strategies import MagnitudePruningStrategy, OptimalBrainDamageStrategy
from pyPrune.utils import *

from experiments.WeightZeroing import WeightZeroing
from experiments.NeuronZeroing import NeuronZeroing
from experiments.NeuronSimilarity import NeuronSimilarity

# -----------------------------
# Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Run pruning and experiments with a specified model, dataset, and experiments.")

parser.add_argument('--model', type=str, default='Vgg16',
                    choices=['LeNet', 'ResNet20', 'Vgg16', 'RegNetX', 'EfficientNet', 'ResNet50'],
                    help="The model architecture to use for pruning.")
parser.add_argument('--dataset', type=str, default='tinyimagenet',
                    choices=['cifar10', 'imagenet', 'tinyimagenet', 'cifar100', 'stl10', 'caltech101', 'fashionmnist', 'mnist'],
                    help="Dataset to use.")
parser.add_argument('--experiments', type=str, nargs='+', default=['None'],
                    choices=['NeuronSimilarity', 'NeuronZeroing', 'WeightZeroing', "None"],
                    help="List of experiments to run.")
parser.add_argument('--steps', type=int, default=21)
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--finetune_epochs', type=int, default=30)
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--save_dir', type=str, default='./pruning_checkpoints/')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--strategy', type=str, default='magnitude', choices=['magnitude', 'brain-damage'])
parser.add_argument('--experimentStep', type=int, default=1)

args = parser.parse_args()

args.save_dir = os.path.join(
    args.save_dir,
    f"{args.model}_dataset{args.dataset}_pretrain{args.pretrain_epochs}_finetune{args.finetune_epochs}"
    f"_steps{args.steps}_batch{args.batch_size}_device{args.device}_strategy_{args.strategy}"
)

print("Experiment configuration:", args)

# -----------------------------
# Learning Rate Functions
# -----------------------------
def poly_lr_with_warmup(epoch):
    max_epochs = args.pretrain_epochs + args.finetune_epochs
    warmup_epochs = max_epochs // 10
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    else:
        decay_epochs = max_epochs - warmup_epochs
        decay_progress = (epoch - warmup_epochs) / decay_epochs
        return (1 - decay_progress) ** 2

def lr_lambda_func(epoch: int) -> float:
    epoch_percentage = epoch / (args.pretrain_epochs + args.finetune_epochs)
    if epoch_percentage < 0.5:
        return 1.0
    elif epoch_percentage < 0.75:
        return 0.1
    else:
        return 0.01

# -----------------------------
# Initialize Pruner
# -----------------------------
def initialize_pruner(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                      steps: list[float], pretrain_epochs: int, finetune_epochs: int,
                      device: str, save_dir: str, model_name: str, total_epochs: int, dataset: str) -> IterativePruner:
    pruner_path = os.path.join(save_dir, 'pruner.pkl')

    if os.path.exists(pruner_path):
        with open(pruner_path, 'rb') as f:
            pruner = pickle.load(f)
        print("Loaded pruner from checkpoint.")
    else:
        # Choose criterion
        if dataset in ['cifar10', 'cifar100', 'imagenet', 'tinyimagenet', 'stl10', 'caltech101']:
            criterion = nn.CrossEntropyLoss()
        elif dataset in ['mnist', 'fashionmnist']:
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

        # Optimizer & Scheduler Selection
        if model_name == 'LeNet':
            optimizer = Adam(model.parameters(), lr=0.001)
            scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

        elif model_name == 'EfficientNet':
            scaled_lr = 0.1 * (args.batch_size / 256)
            optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
            scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_with_warmup)

        elif model_name == 'RegNetX':
            optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

        elif model_name == 'ResNet50':
            scaled_lr = 0.1 * (args.batch_size / 256)
            optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        elif model_name == 'ResNet20':
            optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
            scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        elif model_name == 'Vgg16':
            if dataset in ['cifar10', 'mnist', 'fashionmnist', 'stl10']:
                lr = 0.01
            elif dataset in ['cifar100', 'tinyimagenet', 'caltech101']:
                lr = 0.05
            else:
                lr = 0.01

            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Choose pruning strategy
        if args.strategy == 'magnitude':
            strategy = MagnitudePruningStrategy.MagnitudePruningStrategy(device=device)
        elif args.strategy == 'brain-damage':
            strategy = OptimalBrainDamageStrategy.OptimalBrainDamageStrategy(
                train_loader=train_loader, criterion=criterion, device=device
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        # Print the type and some basic info about each parameter
        print("Model Type:", type(model))
        print("Optimizer Type:", type(optimizer))
        print("Scheduler Type:", type(scheduler))
        print("Criterion Type:", type(criterion))
        print("Train Loader Type:", type(train_loader))
        print("Test Loader Type:", type(test_loader))
        print("Steps Type:", type(steps))
        print("Pretrain Epochs Type:", type(pretrain_epochs))
        print("Device Type:", type(device))
        print("Finetune Epochs Type:", type(finetune_epochs))
        print("Save Dir Type:", type(save_dir))
        print("Strategy Type:", type(strategy))
        print("Early Stopping Type:", type(args.patience))

        # Optionally, print out their values if they are not too large/complex
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
        print("Early Stopping:", args.patience)

        # Initialize Pruner
        pruner = IterativePruner(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            steps=steps,
            pretrain_epochs=pretrain_epochs,
            device=device,
            finetune_epochs=finetune_epochs,
            save_dir=save_dir,
            strategy=strategy,
            early_stopping=args.patience,
        )

    print("Running pruner")
    pruner.run()
    print("Pruning process complete.")
    return pruner

# -----------------------------
# Experiment Execution
# -----------------------------
def run_experiments(pruner: IterativePruner, experiment_names: list[str]) -> None:
    if 'None' in experiment_names:
        print("No experiments to run.")
        return

    if 'NeuronSimilarity' in experiment_names:
        print("Running NeuronSimilarity experiment...")
        NeuronSimilarity(pruner, process_Step=args.experimentStep).run_experiment()

    if 'NeuronZeroing' in experiment_names:
        print("Running NeuronZeroing experiment...")
        NeuronZeroing(pruner, process_Step=args.experimentStep).run_experiment()

    if 'WeightZeroing' in experiment_names:
        print("Running WeightZeroing experiment...")
        sample_fractions = {'linear': .01, 'conv': .01}
        WeightZeroing(pruner, sample_fractions).run_experiment()

# -----------------------------
# Main Logic
# -----------------------------

# -----------------------------
# Main Logic
# -----------------------------
import argparse

def main() -> None:
    set_seed(69917111)

    dataset = args.dataset.lower()

    # Map dataset to the number of classes
    dataset_to_num_classes = {
        'cifar10': 10,
        'imagenet': 10,
        'tinyimagenet': 200,
        'cifar100': 100,
        'stl10': 10,
        'caltech101': 101,
        'fashionmnist': 10,
        'mnist': 10
    }

    # Check if the dataset is valid and get the number of classes
    if dataset not in dataset_to_num_classes:
        raise ValueError(f"Unsupported dataset: {dataset}")

    num_classes = dataset_to_num_classes[dataset]

    # Load model and dataset
    if args.model == 'LeNet':
        model = LeNet()
        train_loader, test_loader = load_mnist(batch_size=args.batch_size, num_workers=args.num_workers)

    elif args.model == 'ResNet20':
        model = ResNet20(num_classes=num_classes)
        if dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for ResNet20: {dataset}")

    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=num_classes)
        if dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for ResNet50: {dataset}")

    elif args.model == 'Vgg16':
        if dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        elif dataset == 'cifar100':
            train_loader, test_loader = load_cifar100(args.batch_size, args.num_workers)
        elif dataset == 'stl10':
            train_loader, test_loader = load_stl10(args.batch_size, args.num_workers)
        elif dataset == 'caltech101':
            train_loader, test_loader = load_caltech101(args.batch_size, args.num_workers)
        elif dataset == 'fashionmnist':
            train_loader, test_loader = load_fashionmnist(args.batch_size, args.num_workers)
        elif dataset == 'mnist':
            train_loader, test_loader = load_mnist(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for VGG16: {dataset}")

        input_size = next(iter(train_loader))[0].shape[2:]  # Get input shape for VGG16
        model = VGG16(one_batch=next(iter(train_loader))[0], num_classes=num_classes)

    elif args.model == 'RegNetX':
        if dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        elif dataset == 'cifar100':
            train_loader, test_loader = load_cifar100(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for RegNetX: {dataset}")
        input_tensor = next(iter(train_loader))[0]
        input_size = input_tensor.shape[1:]  # Includes channels: (C, H, W)
        model = RegNetX_400MF(one_batch=input_tensor,num_classes=num_classes)

    elif args.model == 'EfficientNet':
        model = EfficientNetB7(num_classes=num_classes)
        if dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for EfficientNet: {dataset}")

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    print(model)

    total_epochs = args.pretrain_epochs + args.finetune_epochs
    
    pruner = initialize_pruner(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        steps=exponential_decay_list(steps=args.steps),
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        device=args.device,
        save_dir=args.save_dir,
        model_name=args.model,
        total_epochs=total_epochs,
        dataset=args.dataset
    )

    run_experiments(pruner, args.experiments)


if __name__ == '__main__':
    main()
