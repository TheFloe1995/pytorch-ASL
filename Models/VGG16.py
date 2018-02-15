import torch
import torch.nn as nn
from torchvision import models


class VGG16(nn.Module):
    """
    Pretrained VGG16 model for first test purposes.
    """
    def __init__(self, num_classes=24, p=0.5):
        super(VGG16, self).__init__()
        self.features = models.vgg16(pretrained=True).features
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(4096, num_classes)
        )
        
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
