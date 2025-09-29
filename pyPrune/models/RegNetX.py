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

            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),

            ('conv3', nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)),
            ('bn3', nn.BatchNorm2d(out_channels))
        ]))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(OrderedDict([
                ('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ('shortcut_bn', nn.BatchNorm2d(out_channels))
            ]))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))

class RegNetX_400MF(nn.Module):
    def __init__(self, num_classes=200, input_shape=(3, 224, 224)):
        super(RegNetX_400MF, self).__init__()

        self.input_shape = input_shape  # (C, H, W)
        in_channels = input_shape[0]

        # Stem
        self.stem = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('stem_bn', nn.BatchNorm2d(32)),
            ('stem_relu', nn.ReLU(inplace=True))
        ]))

        # Stages
        self.stage1 = self._make_stage('stage1', 32, 24, num_blocks=1, stride=1)
        self.stage2 = self._make_stage('stage2', 24, 56, num_blocks=1, stride=2)
        self.stage3 = self._make_stage('stage3', 56, 152, num_blocks=4, stride=2)
        self.stage4 = self._make_stage('stage4', 152, 368, num_blocks=7, stride=2)

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Placeholder for fc; actual size will be calculated dynamically
        self.fc_input_features = self._get_flattened_feature_size()
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _make_stage(self, stage_name, in_channels, out_channels, num_blocks, stride):
        blocks = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            block_name = f"{stage_name}_block{i}"
            blocks.append((block_name, XBlock(in_channels, out_channels, stride=s)))
            in_channels = out_channels
        return nn.Sequential(OrderedDict(blocks))

    def _get_flattened_feature_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            x = self.stem(dummy_input)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.pool(x)
            return x.view(1, -1).size(1)

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
