from torch import nn

AVAILABLE_POOLING_REDUCTIONS = ['max', 'avg']

def get_adaptive_pooling_layer(reduction, flatten=True):
    """Returns a torch layer with AdaptivePooling2d, plus flatten if needed."""
    if reduction in ('avg', 'mean'):
        reduction_step = nn.AdaptiveAvgPool2d((1, 1))
    elif reduction == 'max':
        reduction_step = nn.AdaptiveMaxPool2d((1, 1))
    else:
        raise Exception(f'No such reduction {reduction}')

    if flatten:
        layer = nn.Sequential(
            reduction_step,
            nn.Flatten(),
        )
    else:
        layer = reduction_step

    return layer

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
