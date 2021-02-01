from torch import nn

from medai.utils.conv import calc_module_output_size


def _cbr_layer(in_ch, out_ch, kernel_size=(7, 7), stride=1, max_pool=True):
    modules = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
    ]
    if max_pool:
        modules.append(nn.MaxPool2d(3, stride=2))

    return modules


_CONFIGS = {
    # 'name': kernel_size, [layer1, layer2,...],
    'tall': ((7, 7), [32, 64, 128, 256, 512]),
    'wide': ((7, 7), [64, 128, 256, 512]),
    'small': ((7, 7), [32, 64, 128, 256]),
    'tiny': ((5, 5), [64, 128, 256, 512]),
}

def _conv_config(name, in_ch=3):
    if name not in _CONFIGS:
        raise Exception(f'Transfusion CNN not found: {name}')
    kernel_size, layers_def = _CONFIGS[name]

    n_layers = len(layers_def)

    layers = []
    for idx, out_ch in enumerate(layers_def):
        is_last = (idx == n_layers-1)

        modules = _cbr_layer(in_ch, out_ch, kernel_size, max_pool=not is_last)
        layers.extend(modules)

        in_ch = out_ch

    return layers


def class_wrapper(name):
    def constructor(*args, **kwargs):
        return TransfusionCBRCNN(*args, name=name, **kwargs)
    return constructor


class TransfusionCBRCNN(nn.Module):
    """Based on CBR models seen on Transfusion paper.

    Paper: Transfusion: Understanding Transfer Learning for medical imaging
    """
    def __init__(self, labels, pretrained_cnn=None,
                 n_channels=3, name='tall', **unused_kwargs):
        super().__init__()

        self.labels = list(labels)
        self.name = name

        self.features = nn.Sequential(
            *_conv_config(name, in_ch=n_channels),
        )

        # NOTE: height and width passed are dummy, only number of channels is relevant
        out_channels, _ = calc_module_output_size(self.features, (512, 512))
        self.features_size = out_channels

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.prediction = nn.Linear(out_channels, len(self.labels))

        if pretrained_cnn is not None:
            self.load_state_dict(pretrained_cnn.state_dict())



    def forward(self, x, features=False):
        # x shape: batch_size, channels, height, width

        x = self.features(x)
        # x shape: batch_size, out_channels, h2, w2

        if features:
            return x

        x = self.global_pool(x)
        # x shape: batch_size, out_channels

        x = self.prediction(x)
        # x shape: batch_size, n_labels

        return (x,)
