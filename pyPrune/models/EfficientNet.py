import torch
import torch.nn as nn
from collections import OrderedDict

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=1, stride=1):
        super(MBConv, self).__init__()
        mid_channels = in_channels * expansion
        self.use_residual = (in_channels == out_channels and stride == 1)

        self.block = nn.Sequential(OrderedDict([
            ('expand_conv', nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)),
            ('expand_bn', nn.BatchNorm2d(mid_channels)),
            ('expand_act', nn.SiLU(inplace=True)),

            ('dw_conv', nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, groups=mid_channels, bias=False)),
            ('dw_bn', nn.BatchNorm2d(mid_channels)),
            ('dw_act', nn.SiLU(inplace=True)),

            ('proj_conv', nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)),
            ('proj_bn', nn.BatchNorm2d(out_channels)),
        ]))

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)),  # 224 -> 112
            ('bn', nn.BatchNorm2d(32)),
            ('act', nn.SiLU(inplace=True))
        ]))

        # EfficientNet-B0 architecture block repeats
        self.stage1 = MBConv(32, 16, expansion=1, stride=1)
        self.stage2 = nn.Sequential(*[MBConv(16, 24, expansion=6, stride=2), MBConv(24, 24, expansion=6)])
        self.stage3 = nn.Sequential(*[MBConv(24, 40, expansion=6, stride=2), MBConv(40, 40, expansion=6)])
        self.stage4 = nn.Sequential(*[MBConv(40, 80, expansion=6, stride=2), MBConv(80, 80, expansion=6)])
        self.stage5 = nn.Sequential(*[MBConv(80, 112, expansion=6), MBConv(112, 112, expansion=6)])
        self.stage6 = nn.Sequential(*[MBConv(112, 192, expansion=6, stride=2), MBConv(192, 192, expansion=6)])
        self.stage7 = MBConv(192, 320, expansion=6)

        self.head = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(320, 1280, kernel_size=1, bias=False)),
            ('bn', nn.BatchNorm2d(1280)),
            ('act', nn.SiLU(inplace=True))
        ]))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
