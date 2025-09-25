import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Tuple, List, Callable, Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import torchvision.transforms as transforms
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader

import os
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import shutil
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def get_pruneable_named_parameters(model: torch.nn.Module, prunable_layers: Tuple) -> Tuple[List[str], List[torch.nn.Parameter]]:
    names = []
    params = []
    last_layer = list(model.modules())[-1]
    for name, param in model.named_parameters():
        module_name = name.rsplit('.', 1)[0]
        module = dict(model.named_modules()).get(module_name, None)
        if module != last_layer and 'weight' in name and module and any(isinstance(module, layer) for layer in prunable_layers):
            names.append(name)
            params.append(param)
    return names, params

def get_pruneable_named_modules(model: torch.nn.Module, prunable_layers: Tuple) -> Tuple[List[str], List[torch.nn.Module]]:
    names = []
    modules = []
    last_layer = list(model.modules())[-1]
    for name, module in model.named_modules():
        if module != last_layer and any(isinstance(module, layer) for layer in prunable_layers):
            names.append(name)
            modules.append(module)
    return names, modules

def get_pruneable_modules(model: torch.nn.Module, prunable_layers: Tuple) -> List[torch.nn.Module]:
    modules = []
    last_layer = list(model.modules())[-1]
    for module in model.modules():
        if module != last_layer and any(isinstance(module, layer) for layer in prunable_layers):
            modules.append(module)
    return modules

def clean_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss_accuracy_sparsity(pruner) -> None:
    metrics = pruner.metrics
    accuracy = metrics['accuracy']
    loss = metrics['loss']
    sparsity = metrics['sparsity']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(sparsity, accuracy, 'g-', label='Accuracy', linewidth=2)
    ax1.scatter(sparsity, accuracy, c='g', marker='o', s=50)
    ax1.set_ylabel('Accuracy', color='g', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=12)
    
    ax2.plot(sparsity, loss, 'b-', label='Loss', linewidth=2)
    ax2.scatter(sparsity, loss, c='b', marker='x', s=50)
    ax2.set_xlabel('Sparsity', fontsize=14)
    ax2.set_ylabel('Loss', color='b', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc='upper left', fontsize=12)
    
    plt.suptitle('Loss and Accuracy vs. Sparsity', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = pruner.save_dir + '/sparsity_vs_loss_and_accuracy.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()

def lr_lambda(epoch: int,experiment) -> float:
    epoch_percentage = epoch / (experiment.pretrain_epochs+ experiment.finetune_epochs)
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

def load_cifar100(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

    train_loader = DataLoader(
        datasets.CIFAR100('data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.CIFAR100('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("CIFAR-100 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

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

def load_fashionmnist(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_loader = DataLoader(
        datasets.FashionMNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.FashionMNIST('data', train=False, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("FashionMNIST data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

def load_stl10(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4467, 0.4398, 0.4066), (0.2241, 0.2215, 0.2239))
    ])

    train_loader = DataLoader(
        datasets.STL10('data', split='train', download=True, transform=transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_loader = DataLoader(
        datasets.STL10('data', split='test', download=True, transform=transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print("STL10 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

def load_caltech101(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.Caltech101('data', download=True, transform=transform)

    # Simple train/test split
    split_ratio = 0.8
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print("Caltech101 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

def load_tiny_imagenet(batch_size: int = 64, num_workers: int = 1) -> tuple[DataLoader, DataLoader]:
    data_dir = '/projects/modularai/jgafur/LTH/manuscript/structured_study/.temp/data/tiny-imagenet-200/'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    val_img_dir = os.path.join(val_dir, 'images')
    val_annot_path = os.path.join(val_dir, 'val_annotations.txt')

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
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def load_imagenet(batch_size: int = 64, num_workers: int = 2):
    """
    Returns train and validation DataLoaders for the Imagenette2-160 dataset using torchvision.datasets.Imagenette.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        (train_loader, val_loader): Tuple of DataLoaders.
    """

    root = '/projects/modularai/jgafur/LTH/manuscript/structured_study/.temp/data/imagenette2'
    dataset_dir = os.path.join(root, 'imagenette2-160')
    
    # Check if the dataset is already downloaded
    already_downloaded = os.path.exists(dataset_dir)

    # Define image transforms (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load datasets using torchvision.datasets.Imagenette
    train_dataset = Imagenette(
        root=root,
        split='train',
        size='160px',
        download=not already_downloaded,
        transform=transform
    )

    val_dataset = Imagenette(
        root=root,
        split='val',
        size='160px',
        download=not already_downloaded,
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
