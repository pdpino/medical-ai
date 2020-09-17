import torch
import torch.nn as nn
from torchvision import models

from medai.models.common import (
    get_adaptive_pooling_layer,
    get_linear_layers,
)

class Densenet121CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False,
                 pretrained_cnn=None, gpool='max', fc_layers=(),
                 **unused):
        super().__init__()
        self.base_cnn = models.densenet121(pretrained=imagenet)
        
        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())

        self.labels = list(labels)

        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.features_size = 1024
        self.prediction = get_linear_layers(
            self.features_size,
            len(self.labels),
            fc_layers,
        )
        
    def forward(self, x, features=False):
        x = self.base_cnn.features(x)
        
        if features:
            return x

        x = self.global_pool(x)

        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases

        return x, embedding
