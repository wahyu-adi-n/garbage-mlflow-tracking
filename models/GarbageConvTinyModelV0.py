from torchvision import models
from torch import nn


class GarbageConvTinyModelV0(nn.Module):
    def __init__(self,
                 pretrained=models.convnext_tiny(),
                 num_classes=5):
        super(GarbageConvTinyModelV0, self).__init__()
        self.model_backbone = 'convnext_tiny'
        self.num_classes = num_classes
        self.model = pretrained
        num_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.model.forward(x)
