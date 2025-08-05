# measure.py
from SparseLinear import SparseLinear
from SparseConv2d import SparseConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def measure_inference_time(model, x, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
        # move everything to the specified device
    model.to(device)
    x.to(device)
    start_time = time.time()

    with torch.no_grad():
        model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()

    duration = end_time - start_time

    return duration 

