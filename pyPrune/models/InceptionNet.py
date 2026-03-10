import torch
import torch.nn as nn
from collections import OrderedDict

# -------------------------
# Basic Conv Block (To clean up repetition)
# -------------------------
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# -------------------------
# Inception Block
# -------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3_reduce, kernel_size=1),
            BasicConv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5_reduce, kernel_size=1),
            # Original used 5x5, though later versions swapped to 3x3s
            BasicConv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2) 
        )
        
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)

# -------------------------
# Auxiliary Classifier
# -------------------------
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)
        
        # We assume standard 224x224 input, which makes the spatial size here 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------------
# GoogLeNet (Inception v1)
# -------------------------
class InceptionNet(nn.Module):
    def __init__(self, one_batch=None, num_classes=1000, aux_logits=True):
        super(InceptionNet, self).__init__()
        self.aux_logits = aux_logits

        if one_batch is not None:
            _, in_channels, H, W = one_batch.shape
            self.input_size = (in_channels, H, W)
        else:
            self.input_size = (3, 224, 224)

        # Stage 1 & 2
        self.stage1 = nn.Sequential(
            BasicConv2d(self.input_size[0], 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage2 = nn.Sequential(
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage 3
        self.stage3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.stage3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 4
        self.stage4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.stage4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.stage4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.stage4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.stage4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 5
        self.stage5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.stage5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        # Auxiliary Classifiers
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes) # Attached to 4a
            self.aux2 = InceptionAux(528, num_classes) # Attached to 4d

        # Classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.4) 
        
        self.fc_input_features = self._get_flattened_feature_size(one_batch)
        self.fc = nn.Linear(self.fc_input_features, num_classes)

    def _get_flattened_feature_size(self, one_batch):
        was_training = self.training
        self.eval()  
        
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_size) if one_batch is None else torch.zeros(1, *one_batch.shape[1:])
            
            x = self.stage1(dummy)
            x = self.stage2(x)
            x = self.stage3a(x)
            x = self.stage3b(x)
            x = self.maxpool3(x)
            
            x = self.stage4a(x)
            x = self.stage4b(x)
            x = self.stage4c(x)
            x = self.stage4d(x)
            x = self.stage4e(x)
            x = self.maxpool4(x)
            
            x = self.stage5a(x)
            x = self.stage5b(x)
            x = self.pool(x)
            out_features = x.view(1, -1).size(1)
            
        if was_training: 
            self.train()
            
        return out_features

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        
        x = self.stage3a(x)
        x = self.stage3b(x)
        x = self.maxpool3(x)
        
        x = self.stage4a(x)
        
        # First Auxiliary Head
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.stage4b(x)
        x = self.stage4c(x)
        x = self.stage4d(x)
        
        # Second Auxiliary Head
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.stage4e(x)
        x = self.maxpool4(x)
        
        x = self.stage5a(x)
        x = self.stage5b(x)
        
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main_out = self.fc(x)

        # During training with aux logits, return all three outputs
        if self.aux_logits and self.training:
            return main_out, aux2, aux1
            
        return main_out

    # -------------------------
    # Built-in Loss Calculator
    # -------------------------
    def compute_loss(self, outputs, targets, criterion):
        """
        Safely computes the loss regardless of whether the model is in 
        training mode (returning a tuple of 3 logits) or eval mode 
        (returning a single tensor).
        
        The original GoogLeNet paper weights the auxiliary losses by 0.3.
        """
        if isinstance(outputs, tuple):
            # We are in training mode and aux_logits=True
            main_out, aux2_out, aux1_out = outputs
            
            loss_main = criterion(main_out, targets)
            loss_aux2 = criterion(aux2_out, targets)
            loss_aux1 = criterion(aux1_out, targets)
            
            return loss_main + 0.3 * loss_aux2 + 0.3 * loss_aux1
        else:
            # We are in eval mode or aux_logits=False
            return criterion(outputs, targets)