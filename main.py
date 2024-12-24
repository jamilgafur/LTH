import os
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import VGG16Adaptive, ResNet18Adaptive, LeNetAdaptive  # Assuming these models are defined elsewhere
from IMP import IterativeMagnitudePruning
from models import *

# Function to get data loaders
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing for RGB images
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Split the test set into validation and test sets
    validset, testset = torch.utils.data.random_split(testset, [5000, 5000])  # Split 50-50 for simplicity
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, validloader

# Function to create model based on the input type
def create_model(model_name='VGG16', input_channels=3):
    if model_name == 'VGG16':
        return VGG16Adaptive(input_channels=input_channels)
    elif model_name == 'ResNet':
        return ResNet18Adaptive(input_channels=input_channels)
    elif model_name == 'LeNet':
        return LeNetAdaptive(input_channels=input_channels)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

# Main experiment loop
def run_experiment(model_name='VGG16', pruning_rate=0.05, target_sparsity=0.9, input_channels=3):
    # Prepare dataset and loaders
    trainloader, testloader, validloader = get_data_loaders()

    # Initialize model, optimizer, and criterion
    model = create_model(model_name=model_name, input_channels=input_channels)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Create folder for the experiment run
    experiment_name = f"{model_name}_{target_sparsity:.2f}Sparsity_{pruning_rate:.2f}Rate"
    experiment_folder = os.path.join('experiments', experiment_name)
    os.makedirs(experiment_folder, exist_ok=True)

    # Initialize the pruning framework
    pruning_framework = IterativeMagnitudePruning(
        model, optimizer, criterion, pruning_rate, target_sparsity, device,
        trainloader, testloader, validloader, experiment_folder, pretrain_epochs=5, finetune_epochs=5)

    # Run the pruning and retraining experiment
    pruning_framework.prune_and_retrain()

    # After training, plot accuracy vs sparsity
    pruning_framework._plot_accuracy_vs_sparsity()

    print(f"Experiment completed: {experiment_name}")

if __name__ == "__main__":
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run experiments with different models
    run_experiment(model_name='VGG16', pruning_rate=0.1, target_sparsity=0.99, input_channels=3)  # RGB
    run_experiment(model_name='ResNet', pruning_rate=0.1, target_sparsity=0.99, input_channels=3)  # RGB
    run_experiment(model_name='LeNet', pruning_rate=0.1, target_sparsity=0.99, input_channels=3)  # RGB
