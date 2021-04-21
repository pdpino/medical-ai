from torch import nn
from torchvision import models

AVAILABLE_POOLING_REDUCTIONS = ['max', 'avg']

def get_adaptive_pooling_layer(reduction, drop=0):
    """Returns a torch layer with AdaptivePooling2d, plus dropout if needed."""
    if reduction in ('avg', 'mean'):
        reduction_step = nn.AdaptiveAvgPool2d((1, 1))
    elif reduction == 'max':
        reduction_step = nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise Exception(f'No such reduction {reduction}')

    layers = [reduction_step, nn.Flatten()]

    if drop > 0:
        layers.append(nn.Dropout(drop))

    return nn.Sequential(*layers)


def _build_linear_layers(input_size, layers_def):
    layers = []

    current_size = input_size
    for layer_size in layers_def:
        layers.extend([
            nn.Linear(current_size, layer_size),
            nn.ReLU(),
        ])
        current_size = layer_size

    return layers

def get_linear_layers(input_size, layers):
    if not layers or len(layers) == 0:
        return None

    return nn.Sequential(
        *_build_linear_layers(input_size, layers),
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

def load_imagenet_model(model_name, imagenet=True, dropout=0):
    # Config by model
    model_constructor, extractor = _LOADERS[model_name]

    # Load base CNN
    kwargs = {'pretrained': imagenet}
    if dropout != 0 and model_name == 'densenet-121':
        kwargs['drop_rate'] = dropout
    base_cnn = model_constructor(**kwargs)

    # Extract feature layers only
    features, features_size = extractor(base_cnn)

    return features, features_size
