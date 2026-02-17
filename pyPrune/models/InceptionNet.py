import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------
# Inception Block (Remains the same)
# -------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, ch1x1, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(ch1x1)),
            ('relu1', nn.ReLU(inplace=True))
        ]))
        self.branch2 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(ch3x3_reduce)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1, bias=False)),
            ('bn2', nn.BatchNorm2d(ch3x3)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.branch3 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1, bias=False)),
            ('bn1', nn.BatchNorm2d(ch5x5_reduce)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2, bias=False)),
            ('bn2', nn.BatchNorm2d(ch5x5)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.branch4 = nn.Sequential(OrderedDict([
            ('pool', nn.MaxPool2d(kernel_size=3, stride=1, padding=1)),
            ('conv', nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(pool_proj)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

# -------------------------
# InceptionNet
# -------------------------
class InceptionNet(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000):
        super(InceptionNet, self).__init__()

        if one_batch is not None:
            _, in_channels, H, W = one_batch.shape
            self.input_channels = in_channels
            self.input_size = (in_channels, H, W)
        else:
            self.input_channels = 3
            self.input_size = (3, 224, 224)

        # Stage 1: Initial Convolution and Pool
        self.stage1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)),
            ('bn1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Stage 2: Local Response Norm (often skipped in modern BN versions) and Conv
        self.stage2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(64, 64, kernel_size=1, bias=False)),
            ('bn2', nn.BatchNorm2d(64)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False)),
            ('bn3', nn.BatchNorm2d(192)),
            ('relu3', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Stage 3
        self.stage3 = nn.Sequential(OrderedDict([
            ('inception_3a', InceptionBlock(192, 64, 96, 128, 16, 32, 32)),
            ('inception_3b', InceptionBlock(256, 128, 128, 192, 32, 96, 64)),
            ('pool3', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Stage 4
        self.stage4 = nn.Sequential(OrderedDict([
            ('inception_4a', InceptionBlock(480, 192, 96, 208, 16, 48, 64)),
            ('inception_4b', InceptionBlock(512, 160, 112, 224, 24, 64, 64)),
            ('inception_4c', InceptionBlock(512, 128, 128, 256, 24, 64, 64)),
            ('inception_4d', InceptionBlock(512, 112, 144, 288, 32, 64, 64)),
            ('inception_4e', InceptionBlock(528, 256, 160, 320, 32, 128, 128)),
            ('pool4', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        # Stage 5
        self.stage5 = nn.Sequential(OrderedDict([
            ('inception_5a', InceptionBlock(832, 256, 160, 320, 32, 128, 128)),
            ('inception_5b', InceptionBlock(832, 384, 192, 384, 48, 128, 128))
        ]))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_input_features = self._get_flattened_feature_size(one_batch)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _get_flattened_feature_size(self, one_batch):
        was_training = self.training
        self.eval()  
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size) if one_batch is None else torch.zeros(1, *one_batch.shape[1:])
            x = self.stage1(dummy)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.stage5(x)
            x = self.pool(x)
            out_features = x.view(1, -1).size(1)
        if was_training: self.train()
        return out_features

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x