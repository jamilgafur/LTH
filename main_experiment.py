import argparse
import os
import pickle
import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import numpy as np
import copy
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16 import VGG16_CIFAR10 as Vgg16
from pyPrune.models.RegNetX import RegNetX_400MF
from pyPrune.models.EfficientNet import EfficientNetB0
from pyPrune.IterativePruner import IterativePruner
from pyPrune.strategies import MagnitudePruningStrategy, OptimalBrainDamageStrategy
from pyPrune.utils import plot_loss_accuracy_sparsity, set_seed
from experiments.WeightZeroing import WeightZeroing
from experiments.NeuronZeroing import NeuronZeroing
from experiments.NeuronSimilarity import NeuronSimilarity
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import shutil

parser = argparse.ArgumentParser(description="Run pruning and experiments with a specified model and experiments.")

parser.add_argument('--model', type=str, default='LeNet', choices=['LeNet', 'ResNet20', 'Vgg16', 'RegNetX', 'EfficientNet'],
                    help="The model architecture to use for pruning. Default is 'LeNet'.")
parser.add_argument('--experiments', type=str, nargs='+', default=['None'],
                    choices=['NeuronSimilarity', 'NeuronZeroing', 'WeightZeroing', "None"],
                    help="List of experiments to run. Default is all.")
parser.add_argument('--steps', type=int, default=21,
                    help="Number of steps for pruning decay (defaults to exponential decay).")
parser.add_argument('--pretrain_epochs', type=int, default=3,
                    help="Number of pretrain epochs. Default is 10.")
parser.add_argument('--finetune_epochs', type=int, default=1,
                    help="Number of finetune epochs after pruning. Default is 10.")
parser.add_argument('--device', type=str, default='cuda',
                    choices=['cpu', 'cuda'],
                    help="Device to use for training and pruning. Default is 'cuda'.")
parser.add_argument('--save_dir', type=str, default='pruning_checkpoints/',
                    help="Directory to save pruning checkpoints. Default is 'pruning_checkpoints/'.")
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training. Default is 128.")
parser.add_argument('--num_workers', type=int, default=1, help="Number of workers for data loading. Default is 1.")
parser.add_argument('--strategy', type=str, default='brain-damage', choices=['magnitude', 'brain-damage'],
                    help="Pruning strategy to use. Default is 'magnitude'.")

args = parser.parse_args()

# Update save_dir to include model name, pretrain_epochs, and finetune_epochs, also length of steps, and device
args.save_dir = os.path.join(args.save_dir, f"{args.model}_pretrain{args.pretrain_epochs}_finetune{args.finetune_epochs}_steps{args.steps}_batch{args.batch_size}_device{args.device}_strategy_{args.strategy}")

print(f"Experiment configuration: {args}")


def load_cifar10(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_loader = DataLoader(
        datasets.CIFAR10('data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.CIFAR10('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    print("CIFAR-10 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader


def load_mnist(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=1000, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def load_tiny_imagenet(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    data_dir = '/projects/modularai/jgafur/LTH/data/tinyImageNet/tiny-imagenet-200/'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annot_path = os.path.join(val_dir, 'val_annotations.txt')

    # Reorganize validation images into subfolders (only needs to be done once)
    if os.path.exists(val_img_dir):
        with open(val_annot_path, 'r') as f:
            for line in f:
                img_file, label = line.strip().split('\t')[:2]
                label_dir = os.path.join(val_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                src = os.path.join(val_img_dir, img_file)
                dst = os.path.join(label_dir, img_file)
                if os.path.exists(src):
                    shutil.move(src, dst)
        os.rmdir(val_img_dir)

    # Define transforms
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
    ])

    transform_val = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.480, 0.448, 0.398), (0.277, 0.269, 0.282))
    ])

    # Datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def lr_lambda(epoch: int) -> float:
    epoch_percentage = epoch / (args.pretrain_epochs+ args.finetune_epochs)
    if epoch_percentage < 0.5:
        return 1.0
    elif epoch_percentage < 0.75:
        return 0.1
    else:
        return 0.01


def exponential_decay_list(decay_rate: float = 0.8, steps: int = 21) -> list[float]:
    decay_list = [0]
    n = 1
    for _ in range(steps):
        n *= decay_rate
        decay_list.append(1 - n)
    return decay_list


def initialize_pruner(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                      steps: list[float], pretrain_epochs: int, finetune_epochs: int, device: str, save_dir: str, model_name: str, total_epochs: int) -> IterativePruner:
    pruner_path = os.path.join(save_dir, 'pruner.pkl')
    
    if os.path.exists(pruner_path):
        with open(pruner_path, 'rb') as f:
            pruner = pickle.load(f)
        print("Loaded pruner from checkpoint.")
    else:
        if model_name == 'LeNet':
            optimizer = Adam(model.parameters(), lr=0.0012)
            criterion = nn.CrossEntropyLoss()
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

        elif model_name == 'EfficientNet':
            # EfficientNet-specific configuration
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = optim.RMSprop(
                model.parameters(),
                lr=0.05,              # adjusted for smaller batch sizes
                momentum=0.9,
                alpha=0.9,
                eps=1e-3,
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

        elif model_name == 'RegNetX':
            # RegNetX-specific configuration
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.1,               # Initial learning rate for SGD
                momentum=0.9,
                weight_decay=1e-5
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=pretrain_epochs, eta_min=1e-6)

        else:
            # Default config for other models (ResNet, VGG)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
            scheduler = LambdaLR(optimizer, lr_lambda)

        print(f"optimizer: {optimizer}, criterion: {criterion}")
        print(f"Scheduler: {scheduler}")

        # Choose pruning strategy
        if args.strategy == 'magnitude':
            strategy = MagnitudePruningStrategy.MagnitudePruningStrategy(device=device)
        elif args.strategy == 'brain-damage':
            strategy = OptimalBrainDamageStrategy.OptimalBrainDamageStrategy(train_loader=train_loader, criterion=criterion, device=device)
        else:
            raise ValueError(f"Unknown strategy: {args.strategy}")
        
        # Initialize and run the pruner
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
        )
        pruner.run()

    print("Pruning process complete. Saved pruner to checkpoint.")
    return pruner



def run_experiments(pruner: IterativePruner, experiment_names: list[str]) -> None:
    if 'None' in experiment_names:
        print("No experiments to run.")
        return

    if 'NeuronSimilarity' in experiment_names:
        neuron_similarity = NeuronSimilarity(pruner)
        neuron_similarity.run_experiment()

    if 'NeuronZeroing' in experiment_names:
        neuron_zeroing = NeuronZeroing(pruner)
        neuron_zeroing.run_experiment()

    if 'WeightZeroing' in experiment_names:
        sample_fractions = {
            'linear': .01,
            'conv': .01
        }

        weight_zeroing = WeightZeroing(pruner, sample_fractions)
        weight_zeroing.run_experiment()

    
def main() -> None:

    set_seed(69917111)

    if args.model == 'LeNet':
        model = LeNet()
        train_loader, test_loader = load_mnist(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.model == 'ResNet20':
        model = ResNet20()
        train_loader, test_loader = load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.model == 'Vgg16':
        model = Vgg16()
        train_loader, test_loader = load_cifar10(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.model == 'RegNetX':
        model = RegNetX_400MF(num_classes=200)
        train_loader, test_loader = load_tiny_imagenet(batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.model == 'EfficientNet':
        model = EfficientNetB0(num_classes=200)
        train_loader, test_loader = load_tiny_imagenet(batch_size=args.batch_size, num_workers=args.num_workers)
    
    print(model)
    pruner = initialize_pruner(model, train_loader, test_loader, steps=exponential_decay_list(steps=args.steps),
                               pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                               device=args.device, save_dir=args.save_dir, model_name=args.model, total_epochs=args.pretrain_epochs + args.finetune_epochs)
    
    run_experiments(pruner, args.experiments)

if __name__ == '__main__':
    main()
