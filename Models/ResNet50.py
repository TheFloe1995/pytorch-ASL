import torch
import torch.nn as nn
from torchvision import models


class ResNet50(nn.Module):
    """
    Pretrained ResNet 50 model for first test purposes.
    """
    def __init__(self, num_classes=24):
        super(ResNet50, self).__init__()

        model = models.resnet50(pretrained=True)

        self.features = nn.Sequential(*list(model.children())[:-1])
        in_features = model.fc.in_features
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path.
        """
        torch.save(self, path)