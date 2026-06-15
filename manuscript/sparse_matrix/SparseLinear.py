import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import Tensor


class SparseLinear(nn.Module):
    def __init__(self, sparse_weight: torch.Tensor, bias: torch.Tensor = None):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR sparse tensor"
        self.register_buffer("sparse_weight", sparse_weight)
        if bias is not None:
            # If training, consider using nn.Parameter(bias)
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2D (batch_size, in_features)"
        # assume sparse_weight is already on the device
        if self.sparse_weight.device != x.device:
            self.sparse_weight = self.sparse_weight.to(x.device)
        out = torch.sparse.mm(self.sparse_weight, x.t()).t()
        if getattr(self, "bias", None) is not None:
            out = out + self.bias
        return out
    def __init__(self, sparse_weight: Tensor, bias: Tensor = None):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR sparse tensor"
        self.sparse_weight = sparse_weight
        self.bias = bias

    def _sparse_linear(self, input: Tensor, sparse_weight: Tensor, bias: Tensor = None) -> Tensor:
        assert input.dim() == 2, "Input must be 2D (batch_size, in_features)"
        # Ensure weight and input are on the same device
        sparse_weight = sparse_weight.to(input.device)
        output = torch.sparse.mm(sparse_weight, input.t()).t()  # (batch, out_features)
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