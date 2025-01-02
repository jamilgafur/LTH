import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyPrune.pruning import IterativeMagnitudePruning  # Import the pruning class
from pyPrune.analysis import PruningAnalysis  # Import the analysis class

def prune_model(model, train_loader, test_loader, final_sparsity=0.99, steps=9, E=5, pretrain_epochs=0, device=None):
    """
    Perform iterative pruning on the model.
    
    Args:
        model: The model to prune.
        train_loader: The training DataLoader.
        test_loader: The testing DataLoader.
        final_sparsity: The target sparsity (0.0 to 1.0).
        steps: Number of pruning steps.
        E: Number of fine-tuning epochs after each pruning step.
        pretrain_epochs: Number of pretrain epochs before pruning.
        device: Device to run the model on (CPU or GPU).

    Returns:
        pruned_model: The pruned model after pruning and fine-tuning.
    """
    pruner = IterativeMagnitudePruning(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        final_sparsity=final_sparsity,
        steps=steps,
        E=E,
        pretrain_epochs=pretrain_epochs,
        device=device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    pruner.run()  # Run pruning
    return pruner

def analyze_pruning(pruner, output_log='pruning_log.txt', output_dir='results', device=None):
    """
    Perform pruning analysis and plotting.

    Args:
        pruner: The pruner object containing pruning details.
        output_log: Path to save the pruning log.
        output_dir: Directory to save analysis results and plots.
        device: Device to run the analysis on (CPU or GPU).
    
    Returns:
        analysis_results: The results of the pruning analysis.
    """
    analysis = PruningAnalysis(pruner, output_log, output_dir, device=device if device else 'cuda')
    analysis_results = analysis.run_analysis()
    
    return analysis_results

def get_device():
    """
    Get the device to use (GPU or CPU).
    """
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def calculate_accuracy(outputs, targets):
    """
    Calculate the accuracy of the model based on the outputs and targets.

    Args:
        outputs: The model's output tensor.
        targets: The ground truth labels.

    Returns:
        accuracy: The percentage accuracy of the model.
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / len(targets)
    return accuracy
