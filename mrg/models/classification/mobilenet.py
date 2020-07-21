import torch
import torch.nn as nn
from torchvision import models

class MobileNetV2CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False,
                 pretrained_cnn=None, **kwargs):
        """VGG-19.
        
        The head is the original one (except from the last layer).
        """
        super().__init__()

        self.labels = list(labels)

        self.base_cnn = models.mobilenet_v2(pretrained=imagenet)

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())

        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )

        self.features_size = 1280
        self.prediction = nn.Linear(self.features_size, len(labels))



    def forward(self, x, features=False):
        # x shape: batch_size, 3, height, width

        x = self.base_cnn.features(x)
        # shape: batch_size, n_features = 1280, h, w

        if features:
            return x

        x = self.global_pool(x)
        # shape: batch_size, n_features = 1280

        x = self.prediction(x)
        # shape: batch_size, n_diseases

        return x,
