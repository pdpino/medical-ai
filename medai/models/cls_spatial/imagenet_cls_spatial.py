import torch.nn as nn
from torch.nn import functional as F

from medai.models.common import (
    get_activation_layer,
    get_adaptive_pooling_layer,
    load_imagenet_model,
)

class ImageNetClsSpatialModel(nn.Module):
    def __init__(self, cl_labels, model_name='densenet-121',
                 imagenet=True, freeze=False, gpool='max', dropout=0,
                 dropout_features=0, activation='relu', **unused_kwargs):
        super().__init__()
        self.features, self.features_size = load_imagenet_model(
            model_name,
            imagenet=imagenet,
            dropout=dropout,
        )

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.cl_labels = list(cl_labels)
        self.model_name = model_name

        self.dropout_features = dropout_features

        self.activation = get_activation_layer(activation)

        self.spatial_classifier = nn.Conv2d(
            self.features_size, len(cl_labels), 1, 1, 0,
        )

        self.cl_reduction = get_adaptive_pooling_layer(gpool, drop=0)


    def forward(self, x):
        features = self.features(x)
        # shape: bs, n_features, f-height, f-width

        features = self.activation(features)

        if self.dropout_features:
            features = F.dropout2d(features, self.dropout_features, training=self.training)

        spatial_cls = self.spatial_classifier(features)
        # shape: bs, n_labels, f-height, f-width

        classification = self.cl_reduction(spatial_cls)
        # shape: bs, n_labels

        return classification, spatial_cls
