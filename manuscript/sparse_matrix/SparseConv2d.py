import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv2d(nn.Module):
    def __init__(self, sparse_weight: torch.Tensor, bias: torch.Tensor,
                 in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR format"
        # register sparse weight so .to() moves it automatically
        self.register_buffer("sparse_weight", sparse_weight)
        # bias: if you want it trainable, wrap as Parameter; for inference, use buffer
        if bias is not None:
            if isinstance(bias, torch.nn.Parameter):
                self.bias = bias
            else:
                self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        # assume module and input are already on same device
        B, C, H, W = x.shape

        # Extract patches using unfold
        x_unfold = F.unfold(x, kernel_size=self.kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride)  # (B, K, L)
        B, K, L = x_unfold.shape

        # Reshape for matrix multiplication
        x_unfold = x_unfold.permute(0, 2, 1).reshape(B * L, K)  # (B*L, K)

        # Ensure weights on same device (do not always .to() each forward)
        if self.sparse_weight.device != x_unfold.device:
            # this is a safety fallback: should be unnecessary if you call model.to(device) ahead of time
            self.sparse_weight = self.sparse_weight.to(x_unfold.device)

        # Perform sparse × dense multiplication: (out_channels x K) @ (K x B*L)^T
        out = torch.sparse.mm(self.sparse_weight, x_unfold.T).T  # (B*L, out_channels)

        # Reshape back to image-like format
        out = out.view(B, L, self.out_channels).permute(0, 2, 1)  # (B, out_channels, L)

        # Calculate output spatial dimensions
        kh, kw = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        out_h = (H + 2 * self.padding - self.dilation * (kh - 1) - 1) // self.stride + 1
        out_w = (W + 2 * self.padding - self.dilation * (kw - 1) - 1) // self.stride + 1

        out = out.view(B, self.out_channels, out_h, out_w)

        if getattr(self, "bias", None) is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out

