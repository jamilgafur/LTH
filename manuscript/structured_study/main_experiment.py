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
parser.add_argument('--dataset', type=str, default='cifar100',
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
parser.add_argument('--batch_size', type=int, default=512)
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
                      device: str, save_dir: str, model_name: str, total_epochs: int, dataset:str) -> IterativePruner:
    pruner_path = os.path.join(save_dir, 'pruner.pkl')

    if os.path.exists(pruner_path):
        with open(pruner_path, 'rb') as f:
            pruner = pickle.load(f)
        print("Loaded pruner from checkpoint.")
    else:
        criterion = nn.CrossEntropyLoss()

        if model_name == 'LeNet':
            optimizer = Adam(model.parameters(), lr=0.0012)
            scheduler = StepLR(optimizer, step_size=1, gamma=1)

        elif model_name == 'EfficientNet':
            scaled_lr = 0.1 * (args.batch_size / 256)
            optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=1e-5, nesterov=True)
            scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_with_warmup)

        elif model_name == 'RegNetX':
            optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs, eta_min=1e-6)

        elif model_name == 'ResNet50':
            scaled_lr = 0.1 * (args.batch_size / 256)
            optimizer = SGD(model.parameters(), lr=scaled_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        else:
            optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
            scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_func)

        if args.strategy == 'magnitude':
            strategy = MagnitudePruningStrategy.MagnitudePruningStrategy(device=device)
        elif args.strategy == 'brain-damage':
            strategy = OptimalBrainDamageStrategy.OptimalBrainDamageStrategy(
                train_loader=train_loader, criterion=criterion, device=device
            )
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")

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
def main() -> None:
    set_seed(69917111)

    dataset = args.dataset.lower()

    # Load model and dataset
    if args.model == 'LeNet':
        model = LeNet()
        train_loader, test_loader = load_mnist(batch_size=args.batch_size, num_workers=args.num_workers)

    elif args.model == 'ResNet20':
        model = ResNet20(num_classes=1000 if dataset == 'imagenet' else 10)
        if dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for ResNet20: {dataset}")

    elif args.model == 'ResNet50':
        model = ResNet50(num_classes=1000 if dataset == 'imagenet' else 10)
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
            num_classes = 10
        elif dataset == 'cifar100':
            train_loader, test_loader = load_cifar100(args.batch_size, args.num_workers)
            num_classes = 100
        elif dataset == 'stl10':
            train_loader, test_loader = load_stl10(args.batch_size, args.num_workers)
            num_classes = 10
        elif dataset == 'caltech101':
            train_loader, test_loader = load_caltech101(args.batch_size, args.num_workers)
            num_classes = 101
        elif dataset == 'fashionmnist':
            train_loader, test_loader = load_fashionmnist(args.batch_size, args.num_workers)
            num_classes = 10
        elif dataset == 'mnist':
            train_loader, test_loader = load_mnist(args.batch_size, args.num_workers)
            num_classes = 10
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
            num_classes = 200
        elif dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
            num_classes = 10
        else:
            raise ValueError(f"Unsupported dataset for VGG16: {dataset}")

        input_size = next(iter(train_loader))[0].shape[2:]
        model = VGG16(num_classes=num_classes, input_size=input_size)

    elif args.model == 'RegNetX':
        model = RegNetX_400MF(num_classes=1000 if dataset == 'imagenet' else 10)
        if dataset == 'cifar10':
            train_loader, test_loader = load_cifar10(args.batch_size, args.num_workers)
        elif dataset == 'tinyimagenet':
            train_loader, test_loader = load_tiny_imagenet(args.batch_size, args.num_workers)
        elif dataset == 'imagenet':
            train_loader, test_loader = load_imagenet(args.batch_size, args.num_workers)
        else:
            raise ValueError(f"Unsupported dataset for RegNetX: {dataset}")

    elif args.model == 'EfficientNet':
        model = EfficientNetB7(num_classes=1000 if dataset == 'imagenet' else 10)
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
