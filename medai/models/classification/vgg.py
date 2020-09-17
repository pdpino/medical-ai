import torch
import torch.nn as nn
from torchvision import models

class VGG19CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False,
                 pretrained_cnn=None, **unused):
        """VGG-19.
        
        The head is the original one (except from the last layer).
        """
        super().__init__()

        self.labels = list(labels)

        self.base_cnn = models.vgg19(pretrained=False, num_classes=len(labels))

        self.flatten = nn.Flatten()

        if imagenet:
            imagenet_model = models.vgg19(pretrained=True)
            self.load_pretrained(imagenet_model)

        if pretrained_cnn is not None:
            self.load_pretrained(pretrained_cnn)

        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        self.features_size = 512


    def forward(self, x, features=False):
        # x shape: batch_size, 3, height, width

        x = self.base_cnn.features(x)
        # shape: batch_size, n_features, h, w

        if features:
            return x

        x = self.base_cnn.avgpool(x)
        # shape: batch_size, n_features, 7, 7

        x = self.flatten(x)
        # shape: batch_size, 25088

        x = self.base_cnn.classifier(x)
        # shape: batch_size, n_diseases

        return x,

    def load_pretrained(self, model):
        weights = model.state_dict()

        if len(self.labels) != 1000:
            # Monkeypatch last FC layer weights to use pretrained weights
            state_dict = self.base_cnn.state_dict()
            for key in ['classifier.6.weight', 'classifier.6.bias']:
                weights[key] = state_dict[key]

        self.base_cnn.load_state_dict(weights)