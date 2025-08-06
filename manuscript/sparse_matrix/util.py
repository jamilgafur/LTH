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

def convert_all_to_sparse(model, threshold=0.0):
    assert 0.0 <= threshold <= 1.0, "Threshold must be between 0 and 1"

    def is_sparse_enough(tensor, threshold):
        total_elements = tensor.numel()
        zero_elements = (tensor == 0).sum().item()
        return (zero_elements / total_elements) >= threshold

    def convert_linear_to_sparse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                weight = child.weight.data
                if is_sparse_enough(weight, threshold):
                    sparse_weight = weight.to_sparse_csr()
                    bias = child.bias.data if child.bias is not None else None
                    sparse_linear_layer = SparseLinear(sparse_weight, bias)
                    setattr(module, name, sparse_linear_layer)
            else:
                convert_linear_to_sparse(child)

    def convert_conv_to_sparse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                weight = child.weight.data
                out_channels, in_channels, kh, kw = weight.shape
                weight_2d = weight.view(out_channels, -1)
                
                if is_sparse_enough(weight_2d, threshold):
                    sparse_weight = weight_2d.to_sparse_csr()
                    bias = child.bias.data if child.bias is not None else None
                    new_layer = SparseConv2d(
                        sparse_weight=sparse_weight,
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
                convert_conv_to_sparse(child)

    convert_linear_to_sparse(model)
    convert_conv_to_sparse(model)


def validate_original_weights_not_sparse(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            weight = child.weight.data
            print(f"Layer {name}:")
            print(f"  Type: {type(weight)}")
            print(f"  Is sparse? {weight.is_sparse}")
            assert not weight.is_sparse, f"Weight in layer {name} should NOT be sparse!"
        else:
            validate_original_weights_not_sparse(child)

def validate_sparse_csr_format(module):
    for name, child in module.named_children():
        if isinstance(child, SparseLinear) or isinstance(child, SparseConv2d):
            weight = child.sparse_weight
            print(f"Layer {name}:")
            print(f"  Is sparse? {weight.is_sparse}")
            print(f"  Layout: {weight.layout}")
            assert weight.is_sparse, f"Weight in layer {name} is not sparse!"
            assert weight.layout == torch.sparse_csr, f"Weight in layer {name} is not CSR format!"
        else:
            validate_sparse_csr_format(child)

def extract_sparsity_from_filename(filename, prefix="checkpoint_Pruned_"):
    """Extract sparsity value from checkpoint filename."""
    pattern = rf"{prefix}(\d+\.\d+).pth"
    match = re.search(pattern, filename)
    return float(match.group(1)) if match else None

def get_checkpoints(path, prefix="checkpoint_Pruned_"):
    """Return sorted list of (sparsity, full_path) tuples for checkpoints matching prefix."""
    files = glob.glob(os.path.join(path, f"{prefix}*.pth"))
    checkpoints = [(extract_sparsity_from_filename(f, prefix), f) for f in files]  # don't re-join the path
    checkpoints = [ckpt for ckpt in checkpoints if ckpt[0] is not None]
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def getModelType(model_name):
    if model_name == "ResNet20":
        return ResNet20
    elif model_name == "Vgg16_ImageNet":
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
    model = model_cls()
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict if wrapped in 'model'
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if use_compile:
        model = torch.compile(model)  # PyTorch 2.0+ speedup

    return model

def generateData(modelName, batch_size, device):
    if modelName == "ResNet20":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "Vgg16_ImageNet":
        x = torch.randn(batch_size, 3, 224, 224, device=device)
    elif modelName == "Vgg16_":
        x = torch.randn(batch_size, 3, 32, 32, device=device)
    elif modelName == "LeNet":
        x = torch.randn(batch_size, 1, 32, 32, device=device)
    elif modelName == "RegNetX":
        x = torch.randn(batch_size, 3, 224, 224, device=device)
    else:
        raise ValueError(f"Unknown model type: {modelName}")
    return x