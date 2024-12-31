import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pyPrune.pruning import IterativeMagnitudePruning  # Import the pruning class

# Define a simple model for the example (e.g., a small fully connected network)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for MNIST

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define your training data and labels (X_train and y_train)
X_train, y_train = next(iter(train_loader))
X_train = X_train.view(-1, 28 * 28)  # Flatten the images for the fully connected network

# Initialize the model
model = SimpleNN()

# Initialize Iterative Magnitude Pruning with gradual pruning
pruner = IterativeMagnitudePruning(
    model=model,
    X_train=X_train,
    y_train=y_train,
    final_sparsity=0.99,  # Target final sparsity (e.g., 99% sparsity)
    steps=9,  # Prune in 5 steps
    E=5,  # Fine-tuning epochs after each pruning step
    pretrain_epochs=0,  # Pretrain the model for 20 epochs before pruning
    device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
)

# Run pruning and fine-tuning
pruned_model = pruner.run()

# Access saved checkpoints for each pruning step
import pdb; pdb.set_trace()
print(pruner.checkpoints)

