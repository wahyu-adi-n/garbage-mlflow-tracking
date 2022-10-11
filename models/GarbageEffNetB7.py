import torch
from torchvision import models
from torch import nn

class GarbageEffNetB7(nn.Module):
    def __init__(self,
                 pretrained=models.efficientnet_b0(weights='IMAGENET1K_V1'), num_classes=5):
        super(GarbageEffNetB7, self).__init__()
        self.model_backbone = 'efficientnet_b7'
        self.pretrained = pretrained
        self.classifier_layer = torch.nn.Sequential(
            torch.nn.Linear(1280, 256),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes)
        )
        self.num_classes = num_classes

    def forward(self, x):
        x = self.pretrained.features(x)
        x = self.pretrained.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.pretrained.classifier[0](x)
        x = self.classifier_layer(x)
        return x
