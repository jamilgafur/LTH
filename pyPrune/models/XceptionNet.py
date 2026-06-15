import torch
import torch.nn as nn

# -------------------------
# Separable Convolution
# -------------------------
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, 
                                   padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# -------------------------
# Modular Xception Block
# -------------------------
class XceptionBlock(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(XceptionBlock, self).__init__()
        
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=strides, bias=False),
                nn.BatchNorm2d(out_filters)
            )
        else:
            self.skip = nn.Identity()

        rep = []
        filters = in_filters
        
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:] 
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
            
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        return self.skip(x) + self.rep(x)

# -------------------------
# Complete XceptionNet
# -------------------------
class XceptionNet(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000):
        super(XceptionNet, self).__init__()

        # Handle dynamic input sizes based on the user's provided batch
        if one_batch is not None:
            _, in_channels, H, W = one_batch.shape
            self.input_channels = in_channels
            self.input_size = (in_channels, H, W)
        else:
            self.input_channels = 3
            # Standard Xception input size is 299x299, but 224x224 works too!
            self.input_size = (3, 299, 299) 

        # -------------------------
        # Stem
        # -------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(self.input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # -------------------------
        # Entry Flow
        # -------------------------
        self.block1 = XceptionBlock(64, 128, reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = XceptionBlock(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = XceptionBlock(256, 728, reps=2, strides=2, start_with_relu=True, grow_first=True)

        # -------------------------
        # Middle Flow
        # -------------------------
        middle_blocks = []
        for _ in range(8):
            middle_blocks.append(XceptionBlock(728, 728, reps=3, strides=1, start_with_relu=True, grow_first=True))
        self.middle_flow = nn.Sequential(*middle_blocks)

        # -------------------------
        # Exit Flow
        # -------------------------
        self.block4 = XceptionBlock(728, 1024, reps=2, strides=2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)

        # -------------------------
        # Classifier Setup
        # -------------------------
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Compute FC feature size dynamically just like your original code
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
            x = self.middle_flow(x)
            x = self.block4(x)
            
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu3(x)
            
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            
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
        
        x = self.middle_flow(x)
        
        x = self.block4(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

