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

        n_densenet_features = 1024
        # TODO: calculate this size
        output_size = 16 # With input of 512

        self.global_pool = nn.Sequential(
            # nn.MaxPool2d(output_size),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
        )


        self.prediction = nn.Linear(n_densenet_features, len(self.labels))

        self.features_size = (n_densenet_features, output_size, output_size)

        
    def forward(self, x, features=False):
        x = self.base_cnn.features(x)
        
        if features:
            return x

        x = self.global_pool(x)
        x = self.flatten(x)

        embedding = x
        
        x = self.prediction(x) # n_samples, n_diseases

        return x, embedding
