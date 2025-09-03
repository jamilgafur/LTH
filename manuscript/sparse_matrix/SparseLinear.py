import torch
import torch.nn as nn
from torch import Tensor

class SparseLinear(nn.Module):
    def __init__(self, sparse_weight: Tensor, bias: Tensor = None):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR"
        self.sparse_weight = sparse_weight  # don’t register_buffer (lighter state_dict)
        self.bias = bias if bias is not None else None

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, in_features)
        out = torch.sparse.mm(self.sparse_weight.to(x.device), x.t()).t()
        if self.bias is not None:
            out.add_(self.bias)  # in-place add
        return out

