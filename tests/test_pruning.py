import pytest
import torch
from iterative_pruning.pruning import prune_weights

# Simple model for testing
class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 10)
        self.fc2 = torch.nn.Linear(10, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

@pytest.fixture
def model():
    return SimpleNN()

def test_prune_weights(model):
    # Apply pruning of 50%
    prune_weights(model, prune_percentage=0.5)

    # Check if pruning applied correctly
    zero_count = sum(torch.sum(param == 0) for param in model.parameters())
    total_count = sum(param.numel() for param in model.parameters())
    assert zero_count / total_count == 0.5, f"Expected 50% pruning, but got {zero_count / total_count}"

def test_prune_weights_invalid_percentage(model):
    # Test invalid pruning percentage
    with pytest.raises(ValueError):
        prune_weights(model, prune_percentage=1.5)  # Invalid percentage > 1

    with pytest.raises(ValueError):
        prune_weights(model, prune_percentage=-0.1)  # Invalid percentage < 0
