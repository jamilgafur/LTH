from SparseLinear import SparseLinear
from SparseConv2d import SparseConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob 
import os
import re

from pyPrune.models.LeNet import LeNet
from pyPrune.models.ResNet20 import ResNet20
from pyPrune.models.Vgg16ImageNet import VGG16_ImageNet
from pyPrune.models.Vgg16 import VGG16_CIFAR10
from pyPrune.models.RegNetX import RegNetX_400MF

import torch.nn as nn

import torch
import torch.nn as nn

def convert_all_to_sparse(model, threshold=0.0):
    assert 0.0 <= threshold <= 1.0, (
        "Threshold must be between 0 and 1, where 1 means all layers are converted "
        "and 0 means only layers with 100% zeros are converted."
    )

    def is_sparse_enough(tensor, threshold):
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        return (zero_elements / total_elements) >= threshold

    passed_threshold = False

    def convert_layers(module):
        nonlocal passed_threshold
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                weight = child.weight.data
                sparse_condition = is_sparse_enough(weight, threshold)

                # Inverted logic: if threshold is 1.0, we convert all layers
                if sparse_condition or passed_threshold or threshold == 1.0:
                    passed_threshold = passed_threshold or sparse_condition or (threshold == 1.0)
                    
                    # Convert to COO first, then CSR
                    coo = weight.to_sparse().coalesce()
                    csr = coo.to_sparse_csr()
                    
                    bias = child.bias.data if child.bias is not None else None
                    sparse_linear_layer = SparseLinear(csr, bias)
                    setattr(module, name, sparse_linear_layer)

            elif isinstance(child, nn.Conv2d):
                weight = child.weight.data
                out_channels, in_channels, kh, kw = weight.shape
                weight_2d = weight.view(out_channels, -1)

                sparse_condition = is_sparse_enough(weight_2d, threshold)

                # Inverted logic: if threshold is 1.0, we convert all layers
                if sparse_condition or passed_threshold or threshold == 1.0:
                    passed_threshold = passed_threshold or sparse_condition or (threshold == 1.0)
                    
                    # Convert to COO first, then CSR
                    coo = weight_2d.to_sparse().coalesce()
                    csr = coo.to_sparse_csr()
                    
                    bias = child.bias.data if child.bias is not None else None
                    new_layer = SparseConv2d(
                        sparse_weight=csr,
                        bias=bias,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kh,
                        stride=child.stride[0],
                        padding=child.padding[0],
                        dilation=child.dilation[0]
                    )
                    setattr(module, name, new_layer)
            else:
                convert_layers(child)

    convert_layers(model)

def extract_sparsity_from_filename(filename, prefix="checkpoint_Pruned_"):
    pattern = rf"{prefix}(\d+\.\d+).pth"
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None

def get_checkpoints(path, prefix="checkpoint_Pruned_"):
    files = glob.glob(f"{path}/{prefix}*.pth")

    checkpoints = [(extract_sparsity_from_filename(f, prefix), f) for f in files]
    checkpoints = [ckpt for ckpt in checkpoints if ckpt[0] is not None]
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def getModelType(model_name):
    if model_name == "ResNet20":
        return ResNet20
    elif model_name == "Vgg16ImageNet":
        return VGG16_ImageNet
    elif model_name == "Vgg16_":
        return VGG16_CIFAR10
    elif model_name == "LeNet":
        return LeNet
    elif model_name == "RegNetX":
        return RegNetX_400MF
    else:
        raise ValueError(f"Unknown model type: {model_name}")

def load_model_from_checkpoint(model_cls, checkpoint_path: str, device: torch.device, use_compile: bool = True):
    if "Vgg16ImageNet" in checkpoint_path:
        model =model_cls(num_classes=200)  # VGG16 for ImageNet
    elif "Vgg16_" in checkpoint_path:
        model = model_cls()
    elif "ResNet20" in checkpoint_path:
        model = model_cls()
    elif "RegNetX" in checkpoint_path:
        model = model_cls(num_classes=200)
    else:    
        model = model_cls()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    if use_compile:
        model = torch.compile(model)
    return model

def generateData(modelName, batch_size, device):
    if modelName == "ResNet20":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "Vgg16ImageNet":
        x = torch.randn(batch_size, 3, 64, 64, device=device)
    elif modelName == "Vgg16_":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "RegNetX":
        x = torch.randn(batch_size, 3, 64, 64, device=device)
    else:
        raise ValueError(f"Unknown model type: {modelName}")
    return x
