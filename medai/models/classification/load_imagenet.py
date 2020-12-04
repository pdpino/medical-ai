import torch
import torch.nn as nn
from torchvision import models

from medai.models.common import (
    get_adaptive_pooling_layer,
    get_linear_layers,
)

def _extract_densenet_121_features(densenet_121):
    return densenet_121.features, 1024

def _extract_resnet_50_features(resnet_50):
    features = nn.Sequential(
        resnet_50.conv1,
        resnet_50.bn1,
        resnet_50.relu,
        resnet_50.maxpool,
        resnet_50.layer1,
        resnet_50.layer2,
        resnet_50.layer3,
        resnet_50.layer4,
    )
    return features, 2048

def _extract_mobilenet_features(mobilenet):
    return mobilenet.features, 1280

_LOADERS = {
    'densenet-121': (models.densenet121, _extract_densenet_121_features),
    'resnet-50': (models.resnet50, _extract_resnet_50_features),
    'mobilenet': (models.mobilenet_v2, _extract_mobilenet_features),
}

class ImageNetModel(nn.Module):
    """Loads a torchvision model."""
    def __init__(self, labels=[], model_name='densenet-121',
                 imagenet=True, freeze=False,
                 pretrained_cnn=None, gpool='max', fc_layers=(),
                 **unused):
        super().__init__()
        # Config by model
        model_constructor, extractor = _LOADERS[model_name]

        # Load base CNN
        base_cnn = model_constructor(pretrained=imagenet)
        if pretrained_cnn is not None:
            base_cnn.load_state_dict(pretrained_cnn.state_dict())

        # Extract feature layers only
        self.features, self.features_size = extractor(base_cnn)

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.labels = list(labels)
        self.base_name = model_name

        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.prediction = get_linear_layers(
            self.features_size,
            len(self.labels),
            fc_layers,
        )

    def forward(self, x, features=False):
        x = self.features(x)
        # shape: batch_size, n_features, height, width

        if features:
            return x

        x = self.global_pool(x)
        # shape: batch_size, n_features

        embedding = x

        x = self.prediction(x) # shape: batch_size, n_diseases

        return x, embedding
