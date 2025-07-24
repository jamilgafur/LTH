import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseConv2d(nn.Module):
    def __init__(self, sparse_weight, bias, stride=1, padding=0, dilation=1):
        super().__init__()
        assert sparse_weight.layout == torch.sparse_csr, "sparse_weight must be CSR format"
        self.sparse_weight = sparse_weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        batch_size, in_channels, H, W = x.shape
        out_channels, in_channels_kh_kw = self.sparse_weight.shape
        kernel_size_sq = in_channels_kh_kw // in_channels
        kernel_size = int(kernel_size_sq ** 0.5)

        x_unfold = F.unfold(x, kernel_size=kernel_size,
                            dilation=self.dilation,
                            padding=self.padding,
                            stride=self.stride)  # [B, in_channels * kh * kw, L]

        outputs = []
        for i in range(batch_size):
            out = torch.sparse.mm(self.sparse_weight, x_unfold[i])  # [out_channels, L]
            outputs.append(out)
        output = torch.stack(outputs, dim=0)  # [B, out_channels, L]

        out_h = (H + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1
        out_w = (W + 2 * self.padding - self.dilation * (kernel_size - 1) - 1) // self.stride + 1

        output = output.view(batch_size, out_channels, out_h, out_w)

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)
        return output

    def to(self, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device is not None:
            self.sparse_weight = self.sparse_weight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)
        return super().to(*args, **kwargs)
