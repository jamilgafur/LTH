import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------
# XBlock for RegNetX
# -------------------------
class XBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, group_width=16):
        super(XBlock, self).__init__()
        
        # Calculate the number of groups for the ResNeXt-style 3x3 convolution.
        # RegNetX fixes the bottleneck ratio to 1, so inner channels = out_channels.
        groups = out_channels // group_width

        self.block = nn.Sequential(OrderedDict([
            # 1x1 projection
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(out_channels)),
            ('relu1', nn.ReLU(inplace=True)),

            # 3x3 Group Convolution (CRITICAL FIX)
            ('conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, 
                                padding=1, groups=groups, bias=False)),
            ('bn2', nn.BatchNorm2d(out_channels)),
            ('relu2', nn.ReLU(inplace=True)),

            # 1x1 expansion
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


# -------------------------
# RegNetX-400MF
# -------------------------
class RegNetX_400MF(nn.Module):
    def __init__(self, one_batch=None, num_classes=200):
        super(RegNetX_400MF, self).__init__()

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
            ('stem_conv', nn.Conv2d(self.input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)),
            ('stem_bn', nn.BatchNorm2d(32)),
            ('stem_relu', nn.ReLU(inplace=True))
        ]))

        # -------------------------
        # Stages (Fixed to exact 400MF specs)
        # Depths: [1, 2, 7, 12]
        # Widths: [32, 64, 160, 384]
        # -------------------------
        self.stage1 = self._make_stage('stage1', in_channels=32, out_channels=32, num_blocks=1, stride=2)
        self.stage2 = self._make_stage('stage2', in_channels=32, out_channels=64, num_blocks=2, stride=2)
        self.stage3 = self._make_stage('stage3', in_channels=64, out_channels=160, num_blocks=7, stride=2)
        self.stage4 = self._make_stage('stage4', in_channels=160, out_channels=384, num_blocks=12, stride=2)

        # -------------------------
        # Classifier Setup
        # -------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_input_features = self._get_flattened_feature_size(one_batch)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _make_stage(self, stage_name, in_channels, out_channels, num_blocks, stride):
        blocks = []
        for i in range(num_blocks):
            # Only the first block in the stage handles the downsampling
            s = stride if i == 0 else 1
            block_name = f"{stage_name}_block{i}"
            # group_width=16 is constant across all stages for RegNetX-400MF
            blocks.append((block_name, XBlock(in_channels, out_channels, stride=s, group_width=16)))
            in_channels = out_channels
        return nn.Sequential(OrderedDict(blocks))

    # -------------------------
    # Compute FC feature size dynamically
    # -------------------------
    def _get_flattened_feature_size(self, one_batch):
        # We explicitly set eval() to prevent tracking BatchNorm stats with the dummy batch!
        was_training = self.training
        self.eval()

        with torch.no_grad():
            if one_batch is None:
                dummy_input = torch.zeros(1, *self.input_size)
            else:
                _, C, H, W = one_batch.shape
                dummy_input = torch.zeros(1, C, H, W)

            x = self.stem(dummy_input)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
