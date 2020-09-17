import torch
import torch.nn as nn
from torchvision import models

from medai.models.common import (
    get_adaptive_pooling_layer,
    get_linear_layers,
)

class MobileNetV2CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False,
                 pretrained_cnn=None, gpool='max', fc_layers=(),
                 **unused):
        """MobileNet-v2."""
        super().__init__()

        self.labels = list(labels)

        self.base_cnn = models.mobilenet_v2(pretrained=imagenet)

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())

        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.features_size = 1280
        self.prediction = get_linear_layers(
            self.features_size,
            len(self.labels),
            fc_layers,
        )


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
