import torch
import torch.nn as nn
import torch.optim as optim
import logging

def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    accuracy = 100 * correct / len(targets)
    return accuracy

