
import torch
from torch import nn
from pyPrune.models.LeNet import LeNet
from pyPrune.pruning import IterativeMagnitudePruning 

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
        final_sparsity=0.99,
        steps=9,
        pretrain_epochs=0,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        finetune_epochs=1
    )    
    pruner.run()  # Run pruning
    import pdb; pdb.set_trace()
    # # Perform pruning analysis
    # analysis = analyze_pruning(
    #     pruner=pruner,
    #     output_log='pruning_log.txt',  # Save pruning log
    #     output_dir='results',  # Save analysis results and plots
    #     device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
    # )
    
if __name__ == '__main__':
    main()