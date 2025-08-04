from SparseLinear import SparseLinear
from SparseConv2d import SparseConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_all_to_sparse(model):
    def convert_linear_to_sparse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                sparse_weight = child.weight.data.to_sparse_csr()
                bias = child.bias.data if child.bias is not None else None
                sparse_linear_layer = SparseLinear(sparse_weight, bias)
                setattr(module, name, sparse_linear_layer)
            else:
                convert_linear_to_sparse(child)

    def convert_conv_to_sparse(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                new_layer = SparseConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size[0],
                    stride=child.stride[0],
                    padding=child.padding[0],
                    dilation=child.dilation[0],
                    bias=(child.bias is not None)
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


def sparsify_weights(module, sparsity=0.9):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
            weight = child.weight.data
            num_elements = weight.numel()
            num_zero = int(num_elements * sparsity)
            idx = torch.randperm(num_elements)[:num_zero]
            weight.view(-1)[idx] = 0
            child.weight.data = weight
        else:
            sparsify_weights(child, sparsity)

