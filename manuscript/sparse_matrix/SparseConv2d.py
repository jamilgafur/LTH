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
    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.sparse_weight = self.sparse_weight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        return super().to(*args, **kwargs)

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract patches using unfold
        x_unfold = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride)  # (B, K, L)
        B, K, L = x_unfold.shape

        # Reshape for matrix multiplication
        x_unfold = x_unfold.permute(0, 2, 1).reshape(B * L, K)  # (B*L, K)

        # Ensure both weight and input are on the same device
        weight = self.sparse_weight.to(x_unfold.device)

        # Perform sparse × dense multiplication: (out_channels x K) @ (K x B*L)^T
        out = torch.sparse.mm(weight, x_unfold.T).T  # (B*L, out_channels)

        # Reshape back to image-like format
        out = out.view(B, L, self.out_channels).permute(0, 2, 1)  # (B, out_channels, L)

        # Calculate output spatial dimensions
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        out_h = (H + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (W + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1

        out = out.view(B, self.out_channels, out_h, out_w)

        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)

        return out
