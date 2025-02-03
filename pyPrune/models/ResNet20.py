import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# Define a Residual Block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        
        # Skip connection (identity mapping)
        self.shortcut = nn.Sequential() #empty sequential acts as skip connection when channels doesn't change
        if stride != 1 or in_channels != out_channels: #this is for when it does change
            self.shortcut = nn.Sequential(
                OrderedDict([
                ('conv1',nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                ('bn1',nn.BatchNorm2d(out_channels))
                ])
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the skip connection (identity mapping)
        out = self.relu2(out)
        return out

# Define ResNet-20
class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        
        # Initial convolution and batch norm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        
        # Create layers with residual blocks
        # chunks are collections of residual blocks with a constant size
        self.chunk1 = self._make_layer(16, 16, 3, stride=1)  # 3 residual blocks
        self.chunk2 = self._make_layer(16, 32, 3, stride=2)  # 3 residual blocks
        self.chunk3 = self._make_layer(32, 64, 3, stride=2)  # 3 residual blocks
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = OrderedDict()
        for i in range(num_blocks):
            layers['block'+str(i)] = BasicBlock(in_channels, out_channels, stride)
            in_channels = out_channels # number of channles changes once, then stays constant for all blocks in a chunk
            stride = 1  # Keep stride=1 for subsequent blocks in the same layer
        return nn.Sequential(layers)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))  # Initial convolution
        x = self.chunk1(x)  # Layer 1
        x = self.chunk2(x)  # Layer 2
        x = self.chunk3(x)  # Layer 3
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten to 1D
        
        # Fully connected layer
        x = self.fc(x)
        return x