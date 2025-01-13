import argparse
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16 import VGG16_CIFAR10 as Vgg16
from pyPrune.pruning import IterativeMagnitudePruning
from pyPrune.utils import plot_loss_accuracy_sparsity
from experiments.WeightZeroing import WeightZeroing
from experiments.NeuronZeroing import NeuronZeroing
from experiments.NeuronSimilarity import NeuronSimilarity

def load_mnist(batch_size: int = 64, num_workers: int = 4) -> tuple[DataLoader, DataLoader]:
    """
    Loads the MNIST dataset and prepares DataLoader objects for training and testing.

    Args:
        batch_size (int): The batch size for training and testing datasets (default: 64).
        num_workers (int): The number of subprocesses to use for data loading (default: 4).

    Returns:
        tuple: A tuple containing the training and test DataLoader objects.
    """
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

def exponential_decay_list(decay_rate: float = 0.8, steps: int = 21) -> list[float]:
    """
    Generates a list of pruning steps based on exponential decay.

    Args:
        decay_rate (float): The decay rate for each step (default: 0.8).
        steps (int): The number of pruning steps to generate (default: 21).

    Returns:
        list: A list of pruning steps representing exponential decay.
    """
    decay_list = [0]
    n = 1
    for _ in range(steps):
        n *= decay_rate
        decay_list.append(1 - n)
    return decay_list

def initialize_pruner(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                      steps: list[float], pretrain_epochs: int, finetune_epochs: int, device: str, save_dir: str) -> IterativeMagnitudePruning:
    """
    Initializes the pruning process, either by loading an existing pruner from checkpoint or by creating a new one.

    Args:
        model (nn.Module): The model to prune (LeNet, ResNet20, Vgg16).
        train_loader (DataLoader): The DataLoader for training data.
        test_loader (DataLoader): The DataLoader for test data.
        steps (list): List of pruning decay steps.
        pretrain_epochs (int): The number of epochs to pretrain the model (default: 0).
        finetune_epochs (int): The number of epochs to finetune the model after pruning (default: 5).
        device (str): The device to use ('cpu' or 'cuda').
        save_dir (str): Directory to save or load the pruning checkpoint.

    Returns:
        IterativeMagnitudePruning: The initialized or loaded pruner object.
    """
    pruner_path = os.path.join(save_dir, 'pruner.pkl')
    
    if os.path.exists(pruner_path):
        with open(pruner_path, 'rb') as f:
            pruner = pickle.load(f)
        print("Loaded pruner from checkpoint.")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        pruner = IterativeMagnitudePruning(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            steps=steps,
            pretrain_epochs=pretrain_epochs,
            device=device,
            finetune_epochs=finetune_epochs,
            save_dir=save_dir,
        )
        pruner.run()
        print("Pruning process complete. Saved pruner to checkpoint.")

    return pruner

def run_experiments(pruner: IterativeMagnitudePruning, experiment_names: list[str]) -> None:
    """
    Runs specified experiments (e.g., NeuronSimilarity, NeuronZeroing, WeightZeroing) after pruning.

    Args:
        pruner (IterativeMagnitudePruning): The pruner object after pruning is completed.
        experiment_names (list): List of experiment names to run (e.g., ['NeuronSimilarity']).

    Returns:
        None
    """
    if 'NeuronSimilarity' in experiment_names:
        print("Starting neuron similarity experiment...")
        neuron_similarity = NeuronSimilarity(pruner)
        neuron_similarity.run_experiment()

    if 'NeuronZeroing' in experiment_names:
        print("Starting neuron zeroing experiment...")
        neuron_zeroing = NeuronZeroing(pruner)
        neuron_zeroing.run_experiment()

    if 'WeightZeroing' in experiment_names:
        print("Starting weight zeroing experiment...")
        weight_zeroing = WeightZeroing(pruner)
        weight_zeroing.run_experiment()

def parse_args() -> tuple:
    """
    Parses command line arguments.

    Returns:
        tuple: A tuple containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run pruning and experiments with a specified model and experiments.")
    
    # Model and experiments arguments
    parser.add_argument('--model', type=str, default='LeNet', choices=['LeNet', 'ResNet20', 'Vgg16'],
                        help="The model architecture to use for pruning. Default is 'LeNet'.")
    parser.add_argument('--experiments', type=str, nargs='+', default=['NeuronSimilarity', 'NeuronZeroing', 'WeightZeroing'],
                        choices=['NeuronSimilarity', 'NeuronZeroing', 'WeightZeroing'],
                        help="List of experiments to run. Default is all.")
    
    # Pruning related arguments
    parser.add_argument('--steps', type=int, default=3,
                        help="Number of steps for pruning decay (defaults to exponential decay).")
    parser.add_argument('--pretrain_epochs', type=int, default=10,
                        help="Number of pretrain epochs. Default is 0.")
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help="Number of finetune epochs after pruning. Default is 5.")
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help="Device to use for training and pruning. Default is 'cuda'.")
    parser.add_argument('--save_dir', type=str, default='pruning_checkpoints/',
                        help="Directory to save pruning checkpoints. Default is 'pruning_checkpoints/'.")
    
    # Other arguments
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training. Default is 64.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading. Default is 4.")
    
    args = parser.parse_args()
    
    # Default decay steps if not provided
    if args.steps is None:
        args.steps = exponential_decay_list()
    # update save_dir to include model name, pretrain_epochs, and finetune_epochs, also length of steps, and device
    args.save_dir = os.path.join(args.save_dir, f"{args.model}_pretrain{args.pretrain_epochs}_finetune{args.finetune_epochs}_steps{len(args.steps)}_{args.device}")
    
    return args

def main() -> None:
    """
    The main function that runs the pruning process and specified experiments.

    Returns:
        None
    """
    args = parse_args()

    # Load MNIST data
    train_loader, test_loader = load_mnist(batch_size=args.batch_size, num_workers=args.num_workers)

    # Initialize model
    if args.model == 'LeNet':
        model = LeNet()
    elif args.model == 'ResNet20':
        model = ResNet20()
    elif args.model == 'Vgg16':
        model = Vgg16()
    else:
        raise ValueError(f"Model '{args.model}' is not supported.")
    
    # Initialize pruning process
    pruner = initialize_pruner(model, train_loader, test_loader, steps=exponential_decay_list(steps=args.steps),
                                pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs,
                                device=args.device, save_dir=args.save_dir)

    # Plot loss, accuracy, and sparsity after pruning
    plot_loss_accuracy_sparsity(pruner)

    # Run the specified experiments
    run_experiments(pruner, args.experiments)

if __name__ == '__main__':
    main()
