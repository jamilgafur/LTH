
import torch
from torch import nn
from pyPrune.models.LeNet import LeNet
from pyPrune.pruning import IterativeMagnitudePruning 
import numpy as np
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Initialize Iterative Magnitude Pruning with gradual pruning
    pruner = IterativeMagnitudePruning(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        steps=np.linspace(0, .99, 9)[1:]  , # 9 steps from 1'st step to 99% pruning
        pretrain_epochs=3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        finetune_epochs=2
    )    
    pruner.run()  # Run pruning process
    plot_loss_accuracy_sparsity(pruner)


def plot_loss_accuracy_sparsity(pruner: IterativeMagnitudePruning):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(pruner.metrics['sparsity'], pruner.metrics['accuracy'], label='Accuracy')
    ax[0].set_xlabel('Sparsity')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[1].plot(pruner.metrics['sparsity'], pruner.metrics['loss'], label='Loss')
    ax[1].set_xlabel('Sparsity')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    plt.savefig('metrics.png')
    
if __name__ == '__main__':
    main()