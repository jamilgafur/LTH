import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------
# Depthwise Separable Convolution Block for MobileNetV1
# -------------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise layer with BN and ReLU6
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # Pointwise layer with BN and ReLU6
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# -------------------------
# MobileNetV1 (Model Definition)
# -------------------------
class MobileNet(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000):
        super(MobileNet, self).__init__()

        # Handle dynamic input sizes
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
            ('relu1', nn.ReLU6(inplace=True)),
        ]))

        # -------------------------
        # Full MobileNetV1 Architecture
        # -------------------------
        layers = [
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2)
        ]

        # 5x repeating blocks of 512 channels
        for _ in range(5):
            layers.append(DepthwiseSeparableConv(512, 512, stride=1))

        # Final expansion to 1024 channels
        layers.extend([
            DepthwiseSeparableConv(512, 1024, stride=2),
            DepthwiseSeparableConv(1024, 1024, stride=1)
        ])

        # Pack the layers into an nn.Sequential for cleaner forward pass
        self.features = nn.Sequential(*layers)

        # -------------------------
        # Classifier Setup
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
            x = self.features(x)
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
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
