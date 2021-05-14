import torch.nn as nn
import torch.nn.functional as F

from medai.models.common import (
    get_adaptive_pooling_layer,
    get_linear_layers,
    load_imagenet_model,
)

class ImageNetModel(nn.Module):
    """Loads a torchvision model."""
    def __init__(self, labels=[], model_name='densenet-121',
                 imagenet=True, freeze=False, gpool='max', fc_layers=(), dropout=0,
                 dropout_features=0,
                 **unused_kwargs):
        super().__init__()
        # Extract feature layers only
        self.features, self.features_size = load_imagenet_model(
            model_name,
            imagenet=imagenet,
            dropout=dropout,
        )

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.labels = list(labels)
        self.model_name = model_name

        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.dropout_features = dropout_features

        # Create additional FC layers, if any
        pred_input_size = self.features_size
        if fc_layers:
            self.fc = get_linear_layers(
                self.features_size,
                fc_layers,
            )
            pred_input_size = fc_layers[-1]
        else:
            self.fc = None

        self.prediction = nn.Linear(pred_input_size, len(self.labels))

    def forward(self, x, features=False):
        x = self.features(x)
        # shape: batch_size, n_features, height, width

        if features:
            return x

        x = self.global_pool(x)
        # shape: batch_size, n_features

        embedding = x

        if self.dropout_features > 0:
            x = F.dropout(x, self.dropout_features, training=self.training)

        if self.fc:
            x = self.fc(x) # shape: batch_size, n_fc

        x = self.prediction(x) # shape: batch_size, n_diseases

        return x, embedding

    @property
    def cl_labels(self):
        return self.labels

    @property
    def classifier(self):
        return self.prediction
