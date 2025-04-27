import torch
import torch.nn as nn
from collections import OrderedDict

class XBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(XBlock, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels))
        ]))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ('bn', nn.BatchNorm2d(out_channels))
            ]))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class RegNetX_400MF(nn.Module):
    def __init__(self, num_classes=1000):
        super(RegNetX_400MF, self).__init__()

        self.stem = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)),  # 112x112
            ('bn', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # Basic RegNetX-400MF settings
        self.stage1 = self._make_stage(32, 24, num_blocks=1, stride=1)
        self.stage2 = self._make_stage(24, 56, num_blocks=1, stride=2)
        self.stage3 = self._make_stage(56, 152, num_blocks=4, stride=2)
        self.stage4 = self._make_stage(152, 368, num_blocks=7, stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(368, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            layers.append(XBlock(in_channels, out_channels, stride=s))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
