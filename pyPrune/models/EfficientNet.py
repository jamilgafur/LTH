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

class EfficientNetB7(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetB7, self).__init__()
        self.stem = nn.Sequential(OrderedDict([
            ('stem_conv', nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)),  # 224 -> 112
            ('stem_bn', nn.BatchNorm2d(64)),
            ('stem_act', nn.SiLU(inplace=True))
        ]))

        self.stage1 = nn.Sequential(OrderedDict([
            ('mbconv1', MBConv(64, 64, expansion=1, stride=1))
        ]))

        self.stage2 = nn.Sequential(OrderedDict([
            ('mbconv2_0', MBConv(64, 128, expansion=6, stride=2)),
            ('mbconv2_1', MBConv(128, 128, expansion=6))
        ]))

        self.stage3 = nn.Sequential(OrderedDict([
            ('mbconv3_0', MBConv(128, 256, expansion=6, stride=2)),
            ('mbconv3_1', MBConv(256, 256, expansion=6))
        ]))

        self.stage4 = nn.Sequential(OrderedDict([
            ('mbconv4_0', MBConv(256, 512, expansion=6, stride=2)),
            ('mbconv4_1', MBConv(512, 512, expansion=6))
        ]))

        self.stage5 = nn.Sequential(OrderedDict([
            ('mbconv5_0', MBConv(512, 1024, expansion=6)),
            ('mbconv5_1', MBConv(1024, 1024, expansion=6))
        ]))

        self.stage6 = nn.Sequential(OrderedDict([
            ('mbconv6_0', MBConv(1024, 2048, expansion=6, stride=2)),
            ('mbconv6_1', MBConv(2048, 2048, expansion=6))
        ]))

        self.stage7 = nn.Sequential(OrderedDict([
            ('mbconv7', MBConv(2048, 4096, expansion=6))
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('head_conv', nn.Conv2d(4096, 1536, kernel_size=1, bias=False)),
            ('head_bn', nn.BatchNorm2d(1536)),
            ('head_act', nn.SiLU(inplace=True))
        ]))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1536, num_classes)

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
