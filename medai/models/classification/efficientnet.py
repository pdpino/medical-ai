from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNet(nn.Module):
    def __init__(self, labels=[], model_name='efficientnet_b0',
                 imagenet=True, freeze=False, gpool='avg', dropout=0,
                 dropout_features=0,
                 **unused_kwargs):
        super().__init__()
        if model_name != 'efficientnet_b0':
            raise Exception(f'Only implemented model is "efficientnet_b0", got {model_name}')

        model = timm.create_model(
            model_name,
            pretrained=imagenet,
            global_pool=gpool,
            in_chans=3,
            drop_path_rate=dropout,
        )

        self.features = nn.Sequential(OrderedDict([
            (layer_name, getattr(model, layer_name))
            for layer_name in ['conv_stem', 'bn1', 'act1', 'blocks', 'conv_head', 'bn2', 'act2']
        ]))

        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

        self.labels = list(labels)
        self.model_name = model_name

        self.global_pool = model.global_pool

        self.dropout_features = dropout_features

        self.classifier = nn.Linear(model.classifier.in_features, len(self.labels))


    def forward(self, images):
        features = self.features(images)
        # shape: bs, n_features=1280, f-height, f-width

        features = self.global_pool(features)
        # shape: bs, n_features

        if self.dropout_features > 0:
            features = F.dropout(features, p=self.dropout_features, training=self.training)

        classification = self.classifier(features)
        # shape: bs, n_diseases

        return (classification,)

    @property
    def cl_labels(self):
        return self.labels
