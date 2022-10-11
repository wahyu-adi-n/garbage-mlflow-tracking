import torch
from torchvision import models
from torch import nn

class GarbageResNet50(nn.Module):
    def __init__(self,pretrained=models.resnet50(weights='IMAGENET1K_V1'), num_classes=5):
        super(GarbageResNet50, self).__init__()
        self.model_backbone = 'resnet50' 
        resnet = pretrained
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(in_features=resnet.fc.in_features, out_features=num_classes)
        )
        self.base_model = resnet
        self.feature_layer = self.base_model._modules.get("avgpool")
        self.num_classes = num_classes

    def forward(self, x):
        return self.base_model(x)