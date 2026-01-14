import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import glob
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
# -------------------------
# Training and Evaluation
# -------------------------

def train_and_evaluate(model, train_loader, test_loader, optimizer, device, 
                       quant=False, use_autocast=False):
    """
    Performs EXACTLY ONE epoch of training and returns metrics.
    Optimizer is passed in to maintain state between epochs.
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0

    for xb, yb in tqdm.tqdm(train_loader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        if use_autocast:
            with torch.cuda.amp.autocast():
                preds = model(xb)
                loss = loss_fn(preds, yb)
        else:
            preds = model(xb)
            loss = loss_fn(preds, yb)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, predicted = preds.max(1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    avg_loss = total_loss / total
    acc = 100 * correct / total

    return avg_loss, acc
def evaluate(model, loader, device, quant=False):
    model.eval()
    correct = total = 0
    use_autocast = quant and device.type == 'cuda'
    
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_autocast:
                with torch.cuda.amp.autocast():
                    preds = model(xb)
            else:
                preds = model(xb)
            _, predicted = preds.max(1)
            correct += (predicted == yb).sum().item()
            total += yb.size(0)
    return 100 * correct / total
