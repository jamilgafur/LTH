import matplotlib.pyplot as plt
import torch
from torch import nn
from pyPrune.models.LeNet import LeNet
from pyPrune.pruning import IterativeMagnitudePruning 
import numpy as np
from pyPrune.utils import plot_loss_accuracy_sparsity
from experiments.WeightZeroing import WeightZeroing
from experiments.NeuronZeroing import NeuronZeroing
from experiments.NeuronSimilarity import NeuronSimilarity

def load_mnist():
    # Load MNIST dataset
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([
        transforms.Resize((32,32)),
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

def exponential_decay_list(decay_rate = 0.8, steps = 21):
    decay_list = []
    decay_list.append(0)
    n = 1
    for i in range(steps):
        n*= decay_rate
        decay_list.append(1-n)
    return decay_list


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
        steps=exponential_decay_list(), #goes down to 1% weights remaining 20% at a time
        pretrain_epochs=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        finetune_epochs=5, 
    )    
    pruner.run()  # Run pruning process
    plot_loss_accuracy_sparsity(pruner)

    # Initialize NeuronSimilarity class
    print("Starting neuron similarity experiment...")
    neuron_similarity = NeuronSimilarity(pruner)
    neuron_similarity.run_experiment()

    # Initialize NeuronZeroing class
    print("Starting neuron zeroing experiment...")
    neuron_zeroing = NeuronZeroing(pruner)
    neuron_zeroing.run_experiment()

    # Initialize WeightZeroing class
    print("Starting weight zeroing experiment...")
    weight_zeroing = WeightZeroing(pruner)
    weight_zeroing.run_experiment()



if __name__ == '__main__':
    main()