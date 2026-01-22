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

def train_one_epoch(model, train_loader, optimizer, device, scaler=None, use_autocast=False):
    """
    Trains the model for exactly one epoch.
    scaler: torch.cuda.amp.GradScaler object for mixed precision.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()

    for xb, yb in tqdm.tqdm(train_loader, desc="Training", leave=False):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()

        if use_autocast and scaler is not None:
            with torch.cuda.amp.autocast():
                preds = model(xb)
                loss = loss_fn(preds, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, predicted = preds.max(1)
        correct += (predicted == yb).sum().item()
        total += yb.size(0)

    return total_loss / total, 100. * correct / total

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
