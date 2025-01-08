import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a Residual Block for ResNet
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the skip connection (identity mapping)
        out = F.relu(out)
        return out


# Define ResNet-20
class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet20, self).__init__()
        
        # Initial convolution and batch norm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Create layers with residual blocks
        self.layer1 = self._make_layer(16, 16, 3, stride=1)  # 3 residual blocks
        self.layer2 = self._make_layer(16, 32, 3, stride=2)  # 3 residual blocks
        self.layer3 = self._make_layer(32, 64, 3, stride=2)  # 3 residual blocks
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(in_channels, out_channels, stride))
            in_channels = out_channels
            stride = 1  # Keep stride=1 for subsequent blocks in the same layer
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # Initial convolution
        x = self.layer1(x)  # Layer 1
        x = self.layer2(x)  # Layer 2
        x = self.layer3(x)  # Layer 3
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten to 1D
        
        # Fully connected layer
        x = self.fc(x)
        return x
    
