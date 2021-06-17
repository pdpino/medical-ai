import torch
from torch import nn
from torchvision import models

from medai.models.common import load_imagenet_model

class VisualFeatureExtractor(nn.Module):
    def __init__(self, model_name='densenet201', pretrained=False):
        super(VisualFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.features, self.out_features, self.avg_func = self.__get_model()
        # self.activation = nn.ReLU()

    def __get_model(self):
        model = None
        out_features = None
        func = None
        if self.model_name == 'resnet152':
            resnet = models.resnet152(pretrained=self.pretrained)
            modules = list(resnet.children())[:-2]
            model = nn.Sequential(*modules)
            out_features = resnet.fc.in_features
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        elif self.model_name == 'densenet201':
            densenet = models.densenet201(pretrained=self.pretrained)
            modules = list(densenet.features)
            model = nn.Sequential(*modules)
            func = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
            out_features = densenet.classifier.in_features
        elif self.model_name == 'densenet-121':
            model, out_features = load_imagenet_model(
                self.model_name,
                imagenet=self.pretrained,
            )
            func = torch.nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise Exception(f'CNN not recognized: {self.model_name}')

        return model, out_features, func

    def forward(self, images):
        """
        :param images:
        :return:
        """
        visual_features = self.features(images)
        avg_features = self.avg_func(visual_features).squeeze()
        return visual_features, avg_features


class MLC(nn.Module):
    def __init__(self,
                 classes=range(210),
                 sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()

        self.cl_labels = list(classes)

        self.classifier = nn.Linear(in_features=fc_in_features, out_features=len(classes))
        self.embed = nn.Embedding(len(classes), sementic_features_dim)
        self.k = k
        # self.softmax = nn.Softmax(dim=-1)

        # TODO: do not initialize with this???
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self, avg_features):
        tags = self.classifier(avg_features)
        semantic_features = self.embed(torch.topk(tags, self.k)[1])
        return tags, semantic_features
