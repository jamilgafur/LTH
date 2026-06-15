import torch
import torch.nn as nn
from collections import OrderedDict

class VGG16(nn.Module):
    def __init__(self, one_batch=None, num_classes=10):
        super(VGG16, self).__init__()

        if one_batch is not None:
            # Extract the input channels and size from the provided batch
            batch_size, input_channels, height, width = one_batch.shape
            self.input_channels = input_channels
            self.input_size = (height, width)
        else:
            self.input_channels = 3  # Default to 3 channels (RGB)
            self.input_size = (32, 32)  # Default to (height, width)

        # Convolutional layers (Reordered to Conv -> BN -> ReLU)
        self.features = nn.Sequential(
            OrderedDict([
                # Block 1
                ('conv_1', nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1)),
                ('bn_1', nn.BatchNorm2d(64)),
                ('relu_1', nn.ReLU(inplace=True)),
                ('conv_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
                ('bn_2', nn.BatchNorm2d(64)),
                ('relu_2', nn.ReLU(inplace=True)),
                ('pool_1', nn.MaxPool2d(kernel_size=2, stride=2)),

                # Block 2
                ('conv_3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
                ('bn_3', nn.BatchNorm2d(128)),
                ('relu_3', nn.ReLU(inplace=True)),
                ('conv_4', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
                ('bn_4', nn.BatchNorm2d(128)),
                ('relu_4', nn.ReLU(inplace=True)),
                ('pool_2', nn.MaxPool2d(kernel_size=2, stride=2)),

                # Block 3
                ('conv_5', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
                ('bn_5', nn.BatchNorm2d(256)),
                ('relu_5', nn.ReLU(inplace=True)),
                ('conv_6', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn_6', nn.BatchNorm2d(256)),
                ('relu_6', nn.ReLU(inplace=True)),
                ('conv_7', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn_7', nn.BatchNorm2d(256)),
                ('relu_7', nn.ReLU(inplace=True)),
                ('pool_3', nn.MaxPool2d(kernel_size=2, stride=2)),

                # Block 4
                ('conv_8', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
                ('bn_8', nn.BatchNorm2d(512)),
                ('relu_8', nn.ReLU(inplace=True)),
                ('conv_9', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn_9', nn.BatchNorm2d(512)),
                ('relu_9', nn.ReLU(inplace=True)),
                ('conv_10', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn_10', nn.BatchNorm2d(512)),
                ('relu_10', nn.ReLU(inplace=True)),
                ('pool_4', nn.MaxPool2d(kernel_size=2, stride=2)),

                # Block 5
                ('conv_11', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn_11', nn.BatchNorm2d(512)),
                ('relu_11', nn.ReLU(inplace=True)),
                ('conv_12', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn_12', nn.BatchNorm2d(512)),
                ('relu_12', nn.ReLU(inplace=True)),
                ('conv_13', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
                ('bn_13', nn.BatchNorm2d(512)),
                ('relu_13', nn.ReLU(inplace=True)),
                ('pool_5', nn.MaxPool2d(kernel_size=2, stride=2))
            ])
        )

        # Calculate the size of the output from the convolutional layers
        self._calculate_fc_input_size(one_batch)

        # Fully connected layers (Added 1D Batch Norm to control massive variance)
        self.classifier = nn.Sequential(
            OrderedDict([
                ('fc_1', nn.Linear(self.fc_input_size, 4096)),
                ('bn_fc1', nn.BatchNorm1d(4096)),
                ('relu_1', nn.ReLU(inplace=True)),
                ('dropout_1', nn.Dropout(p=0.5)),
                
                ('fc_2', nn.Linear(4096, 4096)),
                ('bn_fc2', nn.BatchNorm1d(4096)),
                ('relu_2', nn.ReLU(inplace=True)),
                ('dropout_2', nn.Dropout(p=0.5)),
                
                ('fc_3', nn.Linear(4096, num_classes))
            ])
        )

    def _calculate_fc_input_size(self, one_batch):
        # This method calculates the flattened feature size after passing through the convolutional layers
        with torch.no_grad():
            if one_batch is not None:
                batch_size, input_channels, height, width = one_batch.shape
                dummy_input = torch.zeros(1, input_channels, height, width)
            else:
                dummy_input = torch.zeros(1, self.input_channels, self.input_size[0], self.input_size[1])
            
            feature_map = self.features(dummy_input)
            self.fc_input_size = feature_map.numel() // feature_map.size(0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
