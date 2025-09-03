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

import torch
import torch.nn as nn

def convert_all_to_sparse(model: nn.Module, threshold: float = 1.0):
    """
    Converts Linear and Conv2d layers to sparse CSR format
    if their weight sparsity >= threshold.
    """

    num_linear_checked = num_conv2d_checked = 0
    num_linear_converted = num_conv2d_converted = 0

    if threshold <= 0.0:
        print("Threshold <= 0.0: No layers will be converted.")
        return

    convert_all = threshold > 1.0
    if convert_all:
        print("Threshold > 1.0: All eligible layers will be converted.")

    def layer_sparsity(tensor: Tensor) -> float:
        total = tensor.numel()
        zeros = (tensor == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    def convert_layers(module: nn.Module):
        nonlocal num_linear_checked, num_conv2d_checked
        nonlocal num_linear_converted, num_conv2d_converted

        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                num_linear_checked += 1
                weight = child.weight.detach()
                sparsity = layer_sparsity(weight)
                print(f"[Linear] {name} sparsity={sparsity:.4f}")

                if convert_all or sparsity >= threshold:
                    csr = weight.to_sparse_csr()
                    bias = child.bias.detach() if child.bias is not None else None
                    sparse_layer = SparseLinear(csr, bias)

                    # free dense params
                    del child.weight
                    if child.bias is not None:
                        del child.bias

                    setattr(module, name, sparse_layer)
                    num_linear_converted += 1
                    print(f"  --> converted to SparseLinear")
                else:
                    convert_layers(child)

            elif isinstance(child, nn.Conv2d):
                num_conv2d_checked += 1
                weight = child.weight.detach()
                out_channels, in_channels, kh, kw = weight.shape
                weight_2d = weight.view(out_channels, -1)
                sparsity = layer_sparsity(weight_2d)
                print(f"[Conv2d] {name} sparsity={sparsity:.4f}")

                if convert_all or sparsity >= threshold:
                    csr = weight_2d.to_sparse_csr()
                    bias = child.bias.detach() if child.bias is not None else None
                    sparse_layer = SparseConv2d(
                        sparse_weight=csr,
                        bias=bias,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=(kh, kw),
                        stride=child.stride[0],
                        padding=child.padding[0],
                        dilation=child.dilation[0]
                    )

                    del child.weight
                    if child.bias is not None:
                        del child.bias

                    setattr(module, name, sparse_layer)
                    num_conv2d_converted += 1
                    print(f"  --> converted to SparseConv2d")
                else:
                    convert_layers(child)

            else:
                convert_layers(child)

    convert_layers(model)

    print("\n[SUMMARY] Sparse Conversion Complete:")
    print(f"  Linear Layers Checked: {num_linear_checked}")
    print(f"  Linear Layers Converted: {num_linear_converted}")
    print(f"  Conv2d Layers Checked: {num_conv2d_checked}")
    print(f"  Conv2d Layers Converted: {num_conv2d_converted}")
    print(f"  Sparsity Threshold: {threshold}\n")

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
