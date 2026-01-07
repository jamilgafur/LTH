import torch
import torch.nn as nn
from collections import OrderedDict


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block:
    Depthwise Conv → LayerNorm → Pointwise MLP → residual
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)   # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)   # NHWC → NCHW

        return x + residual


class ConvNeXt(nn.Module):
    """
    ConvNeXt-style CNN implemented in a VGG-like, collapse-friendly form
    """
    def __init__(self, one_batch=None, num_classes=10):
        super().__init__()

        if one_batch is not None:
            _, input_channels, height, width = one_batch.shape
        else:
            input_channels, height, width = 3, 32, 32

        # ---------------------------------------------------
        # Stem (patch embedding)
        # ---------------------------------------------------
        self.features = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(
                input_channels, 64, kernel_size=4, stride=4
            )),
        ]))

        # ---------------------------------------------------
        # Stage 1 (64 channels)
        # ---------------------------------------------------
        self.stage1 = nn.Sequential(OrderedDict([
            ('block1_1', ConvNeXtBlock(64)),
            ('block1_2', ConvNeXtBlock(64)),
        ]))

        self.down1 = nn.Conv2d(64, 128, kernel_size=2, stride=2)

        # ---------------------------------------------------
        # Stage 2 (128 channels)
        # ---------------------------------------------------
        self.stage2 = nn.Sequential(OrderedDict([
            ('block2_1', ConvNeXtBlock(128)),
            ('block2_2', ConvNeXtBlock(128)),
        ]))

        self.down2 = nn.Conv2d(128, 256, kernel_size=2, stride=2)

        # ---------------------------------------------------
        # Stage 3 (256 channels)
        # ---------------------------------------------------
        self.stage3 = nn.Sequential(OrderedDict([
            ('block3_1', ConvNeXtBlock(256)),
            ('block3_2', ConvNeXtBlock(256)),
            ('block3_3', ConvNeXtBlock(256)),
        ]))

        self.down3 = nn.Conv2d(256, 512, kernel_size=2, stride=2)

        # ---------------------------------------------------
        # Stage 4 (512 channels)
        # ---------------------------------------------------
        self.stage4 = nn.Sequential(OrderedDict([
            ('block4_1', ConvNeXtBlock(512)),
            ('block4_2', ConvNeXtBlock(512)),
        ]))

        # ---------------------------------------------------
        # Classifier
        # ---------------------------------------------------
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._calculate_fc_input_size(one_batch)

        self.classifier = nn.Sequential(OrderedDict([
            ('fc', nn.Linear(self.fc_input_size, num_classes))
        ]))

    def _calculate_fc_input_size(self, one_batch):
        """
        Dynamically infer flattened feature size
        """
        with torch.no_grad():
            if one_batch is None:
                dummy = torch.zeros(1, 3, 32, 32)
            else:
                dummy = one_batch[:1]

            x = self.features(dummy)
            x = self.stage1(x)
            x = self.down1(x)
            x = self.stage2(x)
            x = self.down2(x)
            x = self.stage3(x)
            x = self.down3(x)
            x = self.stage4(x)
            x = self.avgpool(x)

            self.fc_input_size = x.view(1, -1).size(1)

    def forward(self, x):
        x = self.features(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.down3(x)
        x = self.stage4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
