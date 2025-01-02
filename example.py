import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyPrune.utils import prune_model, analyze_pruning
from pyPrune.models.LeNet import LeNet
# Define a simple model for the example (e.g., a small fully connected network)

def main():
     # Load MNIST dataset with transformation
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    X_train, y_train = next(iter(train_loader))

    # Initialize the model
    model = LeNet()


    # Initialize Iterative Magnitude Pruning with gradual pruning
    pruner = prune_model(model, 
                train_loader,
                test_loader,
                final_sparsity=0.99, 
                steps=2, 
                E=0, 
                pretrain_epochs=0, 
                device='cuda'
                )

    analysis_results = analyze_pruning(pruner, 
                                       'pruning_log.txt', 
                                       'results', 
                                       device='cuda'
                                       )

if __name__ == '__main__':
    main()