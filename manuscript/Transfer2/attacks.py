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
import torch.nn as nn

from collapse import collapse_only
import argparse
def fgsm_attack(model, images, labels, device, epsilon=0.03):
    """Fast Gradient Sign Method (FGSM)"""
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # Create the perturbed image by adjusting each pixel by the gradient sign
    perturbed_images = images + epsilon * images.grad.sign()
    return perturbed_images

def pgd_attack(model, images, labels, device, epsilon=0.03, alpha=0.01, iters=10):
    """Projected Gradient Descent (PGD)"""
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    original_images = images.clone().detach()
    
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Adjust by alpha, then project back into the epsilon ball around original
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        images = (original_images + eta).detach_()
        
    return images
