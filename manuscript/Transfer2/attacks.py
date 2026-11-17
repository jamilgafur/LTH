import torch
import os
import glob
import re
import json
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd

import argparse
def fgsm_attack(model, images, labels, device, epsilon=0.03, quant=False):
    """Fast Gradient Sign Method (FGSM).
    
    When quant=True, use fp32 for gradient computation to avoid underflow,
    then clamp and return perturbed images for evaluation on quant model.
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    # For quant models, use autocast to simulate fp16 inference for the forward pass
    # but keep backward in fp32 to avoid gradient underflow
    if quant and device.type == 'cuda':
        with torch.cuda.amp.autocast():
            outputs = model(images)
        # Convert outputs back to fp32 for loss computation
        outputs = outputs.float()
    else:
        outputs = model(images)
    
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # Create the perturbed image by adjusting each pixel by the gradient sign
    perturbed_images = images + epsilon * images.grad.sign()
    return perturbed_images

def pgd_attack(model, images, labels, device, epsilon=0.03, alpha=0.01, iters=10, quant=False):
    """Projected Gradient Descent (PGD).
    
    When quant=True, use fp32 for gradient computation to avoid underflow,
    then clamp and return perturbed images for evaluation on quant model.
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    
    for _ in range(iters):
        images.requires_grad = True
        
        # For quant models, use autocast for forward pass
        if quant and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
            outputs = outputs.float()
        else:
            outputs = model(images)
        
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Adjust by alpha, then project back into the epsilon ball around original
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = (original_images + eta).detach_()
    
    return images
