import torch
from torchvision import models
from torch import nn


class GarbageEffNetModelV0(nn.Module):
    def __init__(self,
                 pretrained=models.efficientnet_b0(weights='IMAGENET1K_V1'), num_classes=5):
        super(GarbageEffNetModelV0, self).__init__()
        self.model_backbone = 'efficientnet_b0'
        self.pretrained = pretrained
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(1280, 64),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64, 32),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(32, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.pretrained.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.pretrained.classifier[0](x)
        x = self.classifier_layer(x)
        return x
