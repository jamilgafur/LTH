import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class VGG16_CIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16_CIFAR10, self).__init__()
        
        # Convolutional layers
        self.features = nn.Sequential(
            OrderedDict([
            # Block 1
            ('conv_1',nn.Conv2d(3, 64, kernel_size=3, padding=1)),  # 32x32x3 -> 32x32x64
            ('relu_1',nn.ReLU(inplace=True)),
            ('bn_1', nn.BatchNorm2d(64)),
            ('conv_2',nn.Conv2d(64, 64, kernel_size=3, padding=1)),  # 32x32x64 -> 32x32x64
            ('relu_2',nn.ReLU(inplace=True)),
            ('bn_2', nn.BatchNorm2d(64)),
            ('pool_1',nn.MaxPool2d(kernel_size=2, stride=2)),  # 32x32x64 -> 16x16x64
            
            # Block 2
            ('conv_3',nn.Conv2d(64, 128, kernel_size=3, padding=1)),  # 16x16x64 -> 16x16x128
            ('relu_3',nn.ReLU(inplace=True)),
            ('bn_3', nn.BatchNorm2d(128)),
            ('conv_4',nn.Conv2d(128, 128, kernel_size=3, padding=1)),  # 16x16x128 -> 16x16x128
            ('relu_4',nn.ReLU(inplace=True)),
            ('bn_4', nn.BatchNorm2d(128)),
            ('pool_2',nn.MaxPool2d(kernel_size=2, stride=2)),  # 16x16x128 -> 8x8x128
            
            # Block 3
            ('conv_5',nn.Conv2d(128, 256, kernel_size=3, padding=1)),  # 8x8x128 -> 8x8x256
            ('relu_5',nn.ReLU(inplace=True)),
            ('bn_5', nn.BatchNorm2d(256)),
            ('conv_6',nn.Conv2d(256, 256, kernel_size=3, padding=1)),  # 8x8x256 -> 8x8x256
            ('relu_6',nn.ReLU(inplace=True)),
            ('bn_6', nn.BatchNorm2d(256)),
            ('conv_7',nn.Conv2d(256, 256, kernel_size=3, padding=1)),  # 8x8x256 -> 8x8x256
            ('relu_7',nn.ReLU(inplace=True)),
            ('bn_7', nn.BatchNorm2d(256)),
            ('pool_3',nn.MaxPool2d(kernel_size=2, stride=2)),  # 8x8x256 -> 4x4x256
            
            # Block 4
            ('conv_8',nn.Conv2d(256, 512, kernel_size=3, padding=1)),  # 4x4x256 -> 4x4x512
            ('relu_8',nn.ReLU(inplace=True)),
            ('bn_8', nn.BatchNorm2d(512)),
            ('conv_9',nn.Conv2d(512, 512, kernel_size=3, padding=1)),  # 4x4x512 -> 4x4x512
            ('relu_9',nn.ReLU(inplace=True)),
            ('bn_9', nn.BatchNorm2d(512)),
            ('conv_10',nn.Conv2d(512, 512, kernel_size=3, padding=1)),  # 4x4x512 -> 4x4x512
            ('relu_10',nn.ReLU(inplace=True)),
            ('bn_10', nn.BatchNorm2d(512)),
            ('pool_4',nn.MaxPool2d(kernel_size=2, stride=2)),  # 4x4x512 -> 2x2x512
            
            # Block 5
            ('conv_11',nn.Conv2d(512, 512, kernel_size=3, padding=1)),  # 2x2x512 -> 2x2x512
            ('relu_11',nn.ReLU(inplace=True)),
            ('bn_11', nn.BatchNorm2d(512)),
            ('conv_12',nn.Conv2d(512, 512, kernel_size=3, padding=1)),  # 2x2x512 -> 2x2x512
            ('relu_12',nn.ReLU(inplace=True)),
            ('bn_12', nn.BatchNorm2d(512)),
            ('conv_13',nn.Conv2d(512, 512, kernel_size=3, padding=1)),  # 2x2x512 -> 2x2x512
            ('relu_13',nn.ReLU(inplace=True)),
            ('bn_13', nn.BatchNorm2d(512)),
            ('pool_5',nn.MaxPool2d(kernel_size=2, stride=2))  # 2x2x512 -> 1x1x512
            ])
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            OrderedDict([
            ('fc_1',nn.Linear(512, 4096)),  # Flatten 1x1x512 -> 512
            ('relu_1',nn.ReLU(inplace=True)),
            ('dropout_1',nn.Dropout(p=0.5)),
            ('fc_2',nn.Linear(4096, 4096)), 
            ('relu_2',nn.ReLU(inplace=True)),
            ('dropout_2',nn.Dropout(p=0.5)),
            ('fc_3',nn.Linear(4096, num_classes)) # Output for CIFAR-10 classes (10 classes)
            ])
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the output of the convolution layers
        x = self.classifier(x)
        return x
