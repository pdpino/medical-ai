from functools import partial
from torch import nn
from torchvision.models import densenet as dn

from medai.models.common import (
    get_adaptive_pooling_layer,
)

class CustomDenseNetCNN(nn.Module):
    def __init__(self, labels, gpool='avg', model_name='densenet',
                 growth_rate=12, block_config=(6, 6, 6, 12),
                 num_init_features=64, bn_size=4, drop_rate=0,
                 **unused_kwargs):
        super().__init__()

        self.model_name = model_name

        densenet = dn.DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=len(labels),
        )

        self.labels = list(labels)

        self.features = densenet.features
        self.relu = nn.ReLU()
        self.global_pool = get_adaptive_pooling_layer(gpool)

        self.prediction = densenet.classifier

    def forward(self, x):
        x = self.features(x)

        x = self.relu(x) # Pytorch implementation also uses this Relu
        x = self.global_pool(x)

        x = self.prediction(x)

        return (x,)


# Num of params: 371,156
TinyDenseNetCNN = partial(
    CustomDenseNetCNN,
    growth_rate=12,
    block_config=(6, 6, 6, 12),
    num_init_features=64, # TODO: create version with only 32?
    bn_size=4,
    drop_rate=0,
    model_name='tiny-densenet',
)

# Num of params: 1,096,181
SmallDenseNetCNN = partial(
    CustomDenseNetCNN,
    growth_rate=15,
    block_config=(12, 12, 12, 12),
    num_init_features=32,
    bn_size=4,
    drop_rate=0,
    model_name='tiny-densenet-v2',
)
