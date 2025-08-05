import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv2d(nn.Module):
    def __init__(self, sparse_weight, bias, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR format"
        self.sparse_weight = sparse_weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    @torch.jit.ignore  # exclude from compilation to reduce overhead
    def sparse_mm_per_batch(self, x_unfold, B, L):
        outputs = []
        for i in range(B):
            x_i = x_unfold[i*L:(i+1)*L]
            out_i = torch.sparse.mm(self.sparse_weight, x_i.T).T
            outputs.append(out_i)
        return torch.cat(outputs, dim=0)

    def forward(self, x):
        B, C, H, W = x.shape

        x_unfold = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride)
        B, K, L = x_unfold.shape
        x_unfold = x_unfold.permute(0, 2, 1).contiguous().view(B*L, K)

        # Option 1: Dense fallback (uncomment if sparse matmul is bottleneck and memory allows)
        # dense_weight = self.sparse_weight.to_dense()
        # out = x_unfold @ dense_weight.T

        # Option 2: Sparse batch matmul without Python loop in forward (recommended)
        out = self.sparse_mm_per_batch(x_unfold, B, L)

        out = out.view(B, L, self.out_channels).permute(0, 2, 1)

        out_h = (H + 2*self.padding - self.dilation*(self.kernel_size - 1) - 1)//self.stride + 1
        out_w = (W + 2*self.padding - self.dilation*(self.kernel_size - 1) - 1)//self.stride + 1

        out = out.view(B, self.out_channels, out_h, out_w)

        if self.bias is not None:
            out.add_(self.bias.view(1, -1, 1, 1))

        return out