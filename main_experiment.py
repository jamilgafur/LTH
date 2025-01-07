import matplotlib.pyplot as plt
import torch
from torch import nn
from pyPrune.models.LeNet import LeNet
from pyPrune.pruning import IterativeMagnitudePruning 
import numpy as np
from pyPrune.utils import plot_loss_accuracy_sparsity

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

def baseline(pruner: IterativeMagnitudePruning):
    baseline_model = LeNet()
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    logger = pruner.logger

    # train for puner.pretrain_epochs + (pruner.finetune_epochs * len(pruner.steps)) epochs using tqdm and save the model to the checkpoint folder as baseline_model
    baseline_model.to(pruner.device)
    for epoch in range(pruner.pretrain_epochs + (pruner.finetune_epochs * len(pruner.steps))):
        # update the logger
        logger.debug(f'Epoch {epoch + 1}/{pruner.pretrain_epochs + (pruner.finetune_epochs * len(pruner.steps))}')
        baseline_model.train()
        for batch_idx, (data, target) in enumerate(pruner.train_loader):
            data, target = data.to(pruner.device), target.to(pruner.device)
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        logger.debug(f'Train Epoch: {epoch + 1}/{pruner.pretrain_epochs + (pruner.finetune_epochs * len(pruner.steps))} [{batch_idx * len(data)}/{len(pruner.train_loader.dataset)} ({100. * batch_idx / len(pruner.train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    logger.debug('Saving baseline model...')
    torch.save(baseline_model.state_dict(), pruner.save_dir + '/baseline_model.pth')

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
        steps=np.linspace(0, .99, 3), # 9 steps from 1'st step to 99% pruning
        pretrain_epochs=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        finetune_epochs=1, 
    )    
    pruner.run()  # Run pruning process
    plot_loss_accuracy_sparsity(pruner)
    baseline(pruner)  # Train baseline model for comparison


if __name__ == '__main__':
    main()