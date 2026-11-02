import gc
import os
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Imagenette


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

def load_cifar100(batch_size: int = 512, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
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
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        datasets.CIFAR100('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    print("CIFAR-100 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader

def load_cifar10(batch_size: int = 512, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
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
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    test_loader = DataLoader(
        datasets.CIFAR10('data', train=False, transform=test_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    
    print("CIFAR-10 data shape: ", next(iter(train_loader))[0].shape)
    return train_loader, test_loader


_DATA_ROOT = os.environ.get("PYPRUNE_DATA_ROOT", os.path.join(os.path.expanduser("~"), ".cache", "pyprune"))


def _tiny_imagenet_fixed_val_dir(val_dir: str, class_to_idx: dict) -> str:
    """Restructure raw Tiny-ImageNet-200 val/ into per-class folders.

    The raw val/ is a single flat images/ dir plus val_annotations.txt, so
    ImageFolder(val_dir) reads every sample as one class. Build (once, cached)
    a sibling val_fixed/ of per-wnid symlink folders matching the train
    ImageFolder's class order.
    """
    fixed_dir = val_dir.rstrip(os.sep) + "_fixed"
    stamp = os.path.join(fixed_dir, ".done")
    if not os.path.exists(stamp):
        with open(os.path.join(val_dir, "val_annotations.txt")) as f:
            pairs = [line.split("\t")[:2] for line in f]
        for wnid in class_to_idx:
            os.makedirs(os.path.join(fixed_dir, wnid), exist_ok=True)
        for img, wnid in pairs:
            if wnid not in class_to_idx:
                continue
            src = os.path.abspath(os.path.join(val_dir, "images", img))
            dst = os.path.join(fixed_dir, wnid, img)
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)  # filesystem without symlink support
        open(stamp, "w").close()
    return fixed_dir


def load_tiny_imagenet(batch_size: int = 64, num_workers: int = 1) -> tuple[DataLoader, DataLoader]:
    data_dir = os.path.join(_DATA_ROOT, "tiny-imagenet-200")
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

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
    fixed_val_dir = _tiny_imagenet_fixed_val_dir(val_dir, train_dataset.class_to_idx)
    val_dataset = datasets.ImageFolder(fixed_val_dir, transform=transform_val)

    assert val_dataset.class_to_idx == train_dataset.class_to_idx, (
        "tiny-imagenet train/val class order mismatch")

    # sanity: val targets must be real labels, not the single-class artifact
    from collections import Counter
    counts = Counter(val_dataset.targets)
    # the known failure mode is exactly one class (flat raw val/ dir)
    assert len(counts) > 1 and len(set(counts.values())) == 1, (
        f"tiny-imagenet val labels look broken: {len(counts)} classes, "
        f"counts {dict(list(counts.items())[:3])}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))

    return train_loader, val_loader

def load_imagenet(batch_size: int = 64, num_workers: int = 1):
    """
    Returns train and validation DataLoaders for the Imagenette2-160 dataset using torchvision.datasets.Imagenette.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses for data loading.

    Returns:
        (train_loader, val_loader): Tuple of DataLoaders.
    """

    root = os.path.join(_DATA_ROOT, 'imagenette2')
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
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
