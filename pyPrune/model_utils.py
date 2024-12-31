# iterative_magnitude_pruning/model_utils.py
import torch
import torch.nn as nn

class PrunedLinear(nn.Linear):
    def forward(self, x):
        if hasattr(self, 'mask'):
            self.weight.data *= self.mask.view(self.weight.data.shape)
        return super().forward(x)

class PrunedConv2d(nn.Conv2d):
    def forward(self, x):
        if hasattr(self, 'mask'):
            self.weight.data *= self.mask.view(self.weight.data.shape)
        return super().forward(x)
