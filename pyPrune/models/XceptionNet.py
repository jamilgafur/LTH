import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------
# Separable Convolution Block for Xception
# -------------------------
class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConvBlock, self).__init__()

        # Depthwise separable convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return x

# -------------------------
# XceptionNet (Model Definition)
# -------------------------
class XceptionNet(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000):
        super(XceptionNet, self).__init__()

        if one_batch is not None:
            _, in_channels, H, W = one_batch.shape
            self.input_channels = in_channels
            self.input_size = (in_channels, H, W)
        else:
            self.input_channels = 3
            self.input_size = (3, 224, 224)

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

        # -------------------------
        # Xception Blocks
        # -------------------------
        self.block1 = SeparableConvBlock(64, 128, stride=2)
        self.block2 = SeparableConvBlock(128, 256, stride=2)
        self.block3 = SeparableConvBlock(256, 728, stride=2)

        # Add multiple Xception blocks for deeper network
        self.block4 = SeparableConvBlock(728, 728, stride=1)
        self.block5 = SeparableConvBlock(728, 728, stride=1)

        # -------------------------
        # Classifier
        # -------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_input_features = self._get_flattened_feature_size(one_batch)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    # -------------------------
    # Compute FC feature size dynamically
    # -------------------------
    def _get_flattened_feature_size(self, one_batch):
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if one_batch is None:
                dummy = torch.zeros(1, *self.input_size)
            else:
                _, C, H, W = one_batch.shape
                dummy = torch.zeros(1, C, H, W)

            x = self.stem(dummy)
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.pool(x)
            out_features = x.view(1, -1).size(1)

        if was_training:
            self.train()

        return out_features

    # -------------------------
    # Forward
    # -------------------------
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
