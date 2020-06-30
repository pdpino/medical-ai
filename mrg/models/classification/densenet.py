import torch
import torch.nn as nn
from torchvision import models

class Densenet121CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False, multilabel=True,
                 pretrained_cnn=None):
        super().__init__()
        self.base_cnn = models.densenet121(pretrained=imagenet)
        
        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())

        self.labels = list(labels)
        # self.multilabel = multilabel 

        n_densenet_features = 1024
        n_densenet_output_size = 16 # With input of 512

        self.global_pool = nn.Sequential(
            nn.MaxPool2d(n_densenet_output_size),
        )

        self.flatten = nn.Flatten()

        linear = nn.Linear(n_densenet_features, len(self.labels))

        if multilabel:
            self.prediction = nn.Sequential(
                linear,
                nn.Sigmoid()
            )
        else:
            self.prediction = linear

        self.features_size = n_densenet_features * n_densenet_output_size * n_densenet_output_size

        
    def forward(self, x):
        x = self.base_cnn.features(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)

        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases

        return x, embedding


    def features(self, x):
        return self.base_cnn.features(x)