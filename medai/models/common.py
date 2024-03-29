from torch import nn
from torch.nn.functional import adaptive_max_pool2d, adaptive_avg_pool2d
from torchvision import models

AVAILABLE_POOLING_REDUCTIONS = ['max', 'avg', 'adapt']

class GlobalMaxMeanPool2d(nn.Module):
    def forward(self, x):
        # x shape: bs, n_channels, height, width

        if self.training:
            return adaptive_avg_pool2d(x, (1, 1))
        return adaptive_max_pool2d(x, (1, 1))


def get_adaptive_pooling_layer(reduction, drop=0):
    """Returns a torch layer with AdaptivePooling2d, plus dropout if needed."""
    if reduction in ('avg', 'mean'):
        reduction_step = nn.AdaptiveAvgPool2d((1, 1))
    elif reduction == 'max':
        reduction_step = nn.AdaptiveMaxPool2d((1, 1))
    elif reduction == 'adapt':
        reduction_step = GlobalMaxMeanPool2d()
    else:
        raise Exception(f'No such reduction {reduction}')

    layers = [reduction_step, nn.Flatten()]

    if drop > 0:
        layers.append(nn.Dropout(drop))

    return nn.Sequential(*layers)

_ACTIVATIONS_LAYERS = {
    'relu': nn.ReLU,
    'lrelu': nn.LeakyReLU,
    'prelu': nn.PReLU,
    'selu': nn.SELU,
}

AVAILABLE_ACTIVATION_LAYERS = list(_ACTIVATIONS_LAYERS)

def get_activation_layer(activation, *args, **kwargs):
    if activation not in _ACTIVATIONS_LAYERS:
        raise Exception('Activation not found: ', activation)
    return _ACTIVATIONS_LAYERS[activation](*args, **kwargs)


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

def _extract_vgg19_features(vgg19):
    return vgg19.features, 512

_LOADERS = {
    'densenet-121': (models.densenet121, _extract_densenet_121_features),
    'resnet-50': (models.resnet50, _extract_resnet_50_features),
    'mobilenet': (models.mobilenet_v2, _extract_mobilenet_features),
    'vgg-19': (models.vgg19, _extract_vgg19_features),
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
