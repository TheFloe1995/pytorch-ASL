import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DenseNet161(nn.Module):
    """
    DenseNet 161 module, pretrained on ImageNet. The fully connected classifier is removed and replaced by a new one
    that fits to the number of classes.
    The net exposes a user defined number of dense layers from the last dense block for training.
    All other layers are frozen, i.e. the gradients are disabled to improve training efficiency.
    """
    
    def __init__(self, freeze_point=24, num_classes=24):
        super(DenseNet161, self).__init__()
        
        model = models.densenet161(pretrained=True)

        final_batchnorm_num_features = model.features.norm5.num_features
        final_fc_in_features = model.classifier.in_features

        # Remove dense_block 4 and following.
        features = nn.Sequential(*list(model.features.children())[:-2])
        
        # Split denseblock 4 into a part to be trained and part to be frozen.
        # freeze_point element [0,...,24],
        # ... where freeze_point = 0 means all layers of block 4 will be trained,
        # ... where freeze_point = 24 means only the classifier will be trained.
        if not freeze_point in range(0,25):
            raise Exception(
                "Error: The index of the layer from which training should be enabled (arg freeze_point) is not in the valid range between 0 and 25!")
        feature4 = nn.Sequential(*list(model.features.denseblock4.children())[:freeze_point])
        trainyou_wewill = nn.Sequential(*list(model.features.denseblock4.children())[freeze_point:])

        # Frozen layers are added to the features module.
        self.features = nn.Sequential(features,feature4)

        # All layers that are ment to be trained are bundled in a separate module.
        self.trainyou_wewill = nn.Sequential(trainyou_wewill, nn.BatchNorm2d(final_batchnorm_num_features))
        self.classifier = nn.Sequential(nn.Linear(final_fc_in_features, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.trainyou_wewill(x)

        x = F.relu(x, inplace=True)
        x = F.avg_pool2d(x, kernel_size=7).view(x.size(0), -1)
        
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

    
    
