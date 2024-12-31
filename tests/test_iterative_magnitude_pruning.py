import pytest
import torch
import torch.nn as nn
from iterative_pruning.pruning.iterative_magnitude_pruning import IterativeMagnitudePruning
from iterative_pruning.pruning import prune_weights
from iterative_pruning.utils import train_model

# Simple model for testing
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

@pytest.fixture
def model():
    return SimpleNN()

@pytest.fixture
def input_data():
    return torch.randn(10, 10), torch.randint(0, 2, (10,))

def test_initial_model_sparsity(model):
    # Ensure initial model has no zero weights
    for param in model.parameters():
        assert torch.sum(param == 0) == 0, f"Parameter {param} has zeros initially."

def test_prune_weights(model):
    # Prune 50% of weights
    prune_weights(model, prune_percentage=0.5)

    # Check that 50% of the weights are pruned (set to zero)
    zero_count = sum(torch.sum(param == 0) for param in model.parameters())
    total_count = sum(param.numel() for param in model.parameters())
    assert zero_count / total_count == 0.5, "Pruned weights are not 50%"

def test_iterative_magnitude_pruning(model, input_data):
    X_train, y_train = input_data

    pruner = IterativeMagnitudePruning(
        model, X_train, y_train, final_sparsity=0.8, prune_sparsity=0.1, steps=10
    )
    pruner.run()  # This will prune and fine-tune the model

    # Check the sparsity of the final model after pruning
    zero_count = sum(torch.sum(param == 0) for param in model.parameters())
    total_count = sum(param.numel() for param in model.parameters())
    final_sparsity = zero_count / total_count

    assert 0.78 <= final_sparsity <= 0.82, "Final model sparsity is not as expected."

def test_reset_weights(model):
    # Apply pruning
    pruner = IterativeMagnitudePruning(
        model, None, None, final_sparsity=0.8, prune_sparsity=0.1, steps=10
    )
    pruner.run()

    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Now reset the weights
    pruner.reset()

    # Ensure the weights have been reset to their original values
    for name, param in model.named_parameters():
        assert torch.equal(param, initial_params[name]), f"Weight for {name} was not reset properly."
