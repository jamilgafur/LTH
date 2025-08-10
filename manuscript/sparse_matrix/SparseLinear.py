import torch
import torch.nn as nn
from torch import Tensor

class SparseLinear(nn.Module):
    def __init__(self, sparse_weight: Tensor, bias: Tensor = None):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_coo, "sparse_weight must be COO sparse tensor"
        self.sparse_weight = sparse_weight
        self.bias = bias

    def _sparse_linear(self, input: Tensor, sparse_weight: Tensor, bias: Tensor = None) -> Tensor:
        assert input.dim() == 2, "Input must be 2D (batch_size, in_features)"
        output = torch.sparse.mm(sparse_weight, input.t()).t()
        if bias is not None:
            output += bias
        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._sparse_linear(x, self.sparse_weight, self.bias)

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.sparse_weight = self.sparse_weight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        return super().to(*args, **kwargs)
