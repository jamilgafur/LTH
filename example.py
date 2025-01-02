
import torch
from torch import nn
from pyPrune.utils import prune_model, analyze_pruning
from pyPrune.models.LeNet import LeNet


def load_mnist():
    # Load MNIST dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    test_loader = DataLoader(
        datasets.MNIST('data', train=False, transform=transform),
        batch_size=1000, shuffle=False
    )

    return train_loader, test_loader

def main():
        
    # Load MNIST dataset X_train, y_train, X_test, y_test
    train_loader, test_loader = load_mnist()
    
    # Initialize the model
    model = LeNet()

    # Initialize Iterative Magnitude Pruning with gradual pruning
    pruner = prune_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        final_sparsity=0.99,  # Target final sparsity (e.g., 99% sparsity)
        steps=9,  # Prune in 5 steps
        E=5,  # Fine-tuning epochs after each pruning step
        pretrain_epochs=10,  # Pretrain the model for 20 epochs before pruning
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    )
    
    # Perform pruning analysis
    analysis = analyze_pruning(
        pruner=pruner,
        output_log='pruning_log.txt',  # Save pruning log
        output_dir='results',  # Save analysis results and plots
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    )
    
if __name__ == '__main__':
    main()