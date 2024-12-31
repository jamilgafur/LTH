import torch
import torch.nn as nn
import torch.optim as optim

# Custom training function (to be used for non-pretrained models)
def train_model(model, X_train, y_train, num_epochs=20, learning_rate=0.001, device=None):
    """
    Train the model from scratch.
    """
    model.to(device)  # Move the model to the selected device
    X_train, y_train = X_train.to(device), y_train.to(device)  # Move data to device

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Fine-tune the model after pruning
def fine_tune_model(model, X_train, y_train, epochs=10, learning_rate=0.001, device=None):
    """
    Fine-tune the model after pruning for a given number of epochs.
    """
    model.to(device)  # Move the model to the selected device
    X_train, y_train = X_train.to(device), y_train.to(device)  # Move data to device

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop for fine-tuning
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
