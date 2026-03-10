import torch
import torch.nn as nn
from collections import OrderedDict

# ---------------------------------------------------
# Helper: LayerNorm for Channels-First (NCHW) Tensors
# ---------------------------------------------------
class LayerNorm2d(nn.Module):
    """
    Standard PyTorch LayerNorm only supports applying normalization over the last dimension(s).
    This helper applies LayerNorm over the channel dimension (dim=1) for spatial NCHW tensors,
    which is required for ConvNeXt's stem and downsampling layers.
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# ---------------------------------------------------
# ConvNeXt Block
# ---------------------------------------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # LayerScale: Learnable parameter to scale block output
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Apply LayerScale
        if self.gamma is not None:
            x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2)   # NHWC → NCHW

        return x + residual

# ---------------------------------------------------
# ConvNeXt Architecture
# ---------------------------------------------------
class ConvNeXt(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]):
        """
        Defaults are set to ConvNeXt-Tiny configuration.
        """
        super().__init__()

        if one_batch is not None:
            _, input_channels, height, width = one_batch.shape
        else:
            input_channels, height, width = 3, 224, 224

        # ---------------------------------------------------
        # Stem (Patch Embedding)
        # ---------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]) # CRITICAL: LayerNorm after stem
        )

        # ---------------------------------------------------
        # Stages & Downsampling
        # ---------------------------------------------------
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        # The stem handles the first downsample, so we only need 3 more between stages
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]), # CRITICAL: LayerNorm before downsampling
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        # Build the 4 ConvNeXt stages
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        # ---------------------------------------------------
        # Classifier
        # ---------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final norm before classifier
        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        
        self._calculate_fc_input_size(one_batch)
        self.classifier = nn.Linear(self.fc_input_size, num_classes)

    def _calculate_fc_input_size(self, one_batch):
        was_training = self.training
        self.eval() # Safely switch to eval mode

        with torch.no_grad():
            if one_batch is None:
                # Assuming 224x224 as standard, though it will adapt to whatever
                dummy = torch.zeros(1, 3, 224, 224) 
            else:
                dummy = one_batch[:1]

            x = self.stem(dummy)
            for i in range(4):
                x = self.stages[i](x)
                if i < 3:
                    x = self.downsample_layers[i](x)
                    
            x = self.avgpool(x)
            self.fc_input_size = x.view(1, -1).size(1)

        if was_training:
            self.train()

    def forward(self, x):
        x = self.stem(x)
        
        for i in range(4):
            x = self.stages[i](x)
            if i < 3:
                x = self.downsample_layers[i](x)
                
        x = self.avgpool(x)
        
        # Flatten and apply the final LayerNorm (requires NHWC shape for nn.LayerNorm)
        x = x.view(x.size(0), -1) 
        x = self.final_norm(x)
        
        x = self.classifier(x)
        return x