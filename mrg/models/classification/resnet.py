import torch
import torch.nn as nn
from torchvision import models

class Resnet50CNN(nn.Module):
    def __init__(self, labels, imagenet=True, freeze=False, multilabel=True,
                 pretrained_cnn=None, dropout=None):
        """Resnet-50."""
        super().__init__()

        self.base_cnn = models.resnet50(pretrained=imagenet)

        if pretrained_cnn is not None:
            self.base_cnn.load_state_dict(pretrained_cnn.state_dict())
        
        self.labels = list(labels)
        self.multilabel = multilabel

        if freeze:
            for param in self.base_cnn.parameters():
                param.requires_grad = False

        # TODO: calculate this size!
        n_resnet_output_size = 16 # With input of 512

        self.global_pool = nn.Sequential(
            nn.MaxPool2d(n_resnet_output_size)
        )

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(dropout) if dropout else None

        n_diseases = len(labels)
        n_resnet_features = 2048

        linear = nn.Linear(n_resnet_features, n_diseases)

        if multilabel:
            # Custom losses do not include sigmoid on the calculation
            self.prediction = nn.Sequential(
                linear,
                nn.Sigmoid(),
            )
        else:
            # Cross entropy loss includes softmax
            self.prediction = linear

        self.features_size = n_resnet_features * n_resnet_output_size * n_resnet_output_size
        
    def forward(self, x):
        # x shape: batch_size, 3, height, width
        # 3 as in RGB, heigth and width are usually 512 for CXR14

        x = self.features(x)
        # shape: batch_size, n_features, height = 16, width = 16
        
        x = self.global_pool(x)
        # shape: batch_size, n_features, height = 1, width = 1

        x = self.flatten(x)
        # shape: batch_size, n_features
        
        if self.dropout:
            x = self.dropout(x)
            # shape: batch_size, n_features

        x = self.prediction(x)
        # shape: batch_size, n_diseases

        return x,

    def features(self, x):
        x = self.base_cnn.conv1(x)
        x = self.base_cnn.bn1(x)
        x = self.base_cnn.relu(x)
        x = self.base_cnn.maxpool(x)

        x = self.base_cnn.layer1(x)
        x = self.base_cnn.layer2(x)
        x = self.base_cnn.layer3(x)
        x = self.base_cnn.layer4(x)

        # batch_size, 2048, height=16, width=16
        return x


    def forward_with_cam(self, x):
        ### DEPRECATED, use captum

        x = self.features(x)
        # shape: batch_size, n_features, height = 16, width = 16
        

        # Calculate CAM
        pred_weights, pred_bias_unused = list(self.prediction.parameters())
        # pred_weights size: n_diseases, n_features = 2048

        # x: activations from prev layer
        # bbox: for each sample, multiply n_features dimensions
        activations = torch.matmul(pred_weights, x.transpose(1, 2)).transpose(1, 2)
        # shape: batch_size, n_diseases, height, width
        
        x = self.global_pool(x)
        # shape: batch_size, n_features, 1, 1
        
        x = x.view(x.size(0), -1)
        # shape: batch_size, n_features
        
        embedding = x
        
        x = self.prediction(x)
        # shape: batch_size, n_diseases
        
        return x, embedding, activations