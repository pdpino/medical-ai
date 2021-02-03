from torch import nn
from torchvision.models import densenet as dn

from medai.models.common import (
    get_adaptive_pooling_layer,
)

class TinyDenseNetCNN(nn.Module):
    def __init__(self, labels, gpool='avg', **unused_kwargs):
        super().__init__()

        densenet = dn.DenseNet(
            growth_rate=12,
            block_config=(6, 6, 6, 12),
            num_init_features=64,
            bn_size=4,
            drop_rate=0,
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
