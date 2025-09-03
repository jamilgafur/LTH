import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class SparseConv2d(nn.Module):
    def __init__(self, sparse_weight: Tensor, bias: Tensor,
                 in_channels: int, out_channels: int,
                 kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR"

        self.sparse_weight = sparse_weight
        self.bias = bias if bias is not None else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x_unfold = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride)
        B, K, L = x_unfold.shape
        x_unfold = x_unfold.permute(0, 2, 1).reshape(B * L, K)

        weight = self.sparse_weight.to(x_unfold.device)
        out = torch.sparse.mm(weight, x_unfold.T).T
        out = out.view(B, L, self.out_channels).permute(0, 2, 1)

        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        out_h = (H + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (W + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1
        out = out.view(B, self.out_channels, out_h, out_w)

        if self.bias is not None:
            out.add_(self.bias.view(1, -1, 1, 1))

        return out