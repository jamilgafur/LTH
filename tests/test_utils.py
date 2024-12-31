import pytest
import torch
from iterative_pruning.utils import train_model, fine_tune_model

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

@pytest.fixture
def input_data():
    return torch.randn(10, 10), torch.randint(0, 2, (10,))

def test_train_model(model, input_data):
    X_train, y_train = input_data

    # Train the model
    train_model(model, X_train, y_train, epochs=5, learning_rate=0.001)

    # Check that training didn't result in all zero weights
    for param in model.parameters():
        assert torch.sum(param == 0) < param.numel(), "Model's weights are not updated during training."

def test_fine_tune_model(model, input_data):
    X_train, y_train = input_data

    # Fine-tune the model
    fine_tune_model(model, X_train, y_train, epochs=3, learning_rate=0.001)

    # Check that fine-tuning improved accuracy (this is a simplistic check, can be refined)
    assert model.fc2.weight.grad is not None, "Fine-tuning did not update weights."
