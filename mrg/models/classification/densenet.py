import torch
import torch.nn as nn
from torchvision import models

class Densenet121CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False,
                 pretrained_cnn=None, **kwargs):
        super().__init__()
        self.base_cnn = models.densenet121(pretrained=imagenet)
        
        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())

        self.labels = list(labels)

        self.global_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )

        self.features_size = 1024
        self.prediction = nn.Linear(self.features_size, len(self.labels))
        
    def forward(self, x, features=False):
        x = self.base_cnn.features(x)
        
        if features:
            return x

        x = self.global_pool(x)

        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases

        return x, embedding
