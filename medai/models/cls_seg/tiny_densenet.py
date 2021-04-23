from functools import partial
from torch import nn
from torchvision.models import densenet as dn

from medai.models.common import (
    get_adaptive_pooling_layer,
)
from medai.models.classification.tiny_densenet import TINY_DENSENET_CONFIGS

_ASSERT_IN_OUT_IMAGE_SIZE = False

class CustomDenseNetClsSeg(nn.Module):
    def __init__(self, cl_labels, seg_labels,
                 gpool='avg', model_name='densenet',
                 growth_rate=12, block_config=(6, 6, 6, 12),
                 num_init_features=64, bn_size=4, dropout=0, dropout_features=0,
                 _features_size=None,
                 **unused_kwargs):
        super().__init__()

        self.model_name = model_name

        densenet = dn.DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=dropout or 0,
            num_classes=len(cl_labels),
        )

        self.cl_labels = list(cl_labels)
        self.seg_labels = list(seg_labels)

        self.features = densenet.features

        self.relu = nn.ReLU()
        self.global_pool = get_adaptive_pooling_layer(gpool, drop=dropout_features)

        self.features_size = _features_size

        self.segmentator = nn.Sequential(
            # in: features_size, f-height, f-width
            nn.ConvTranspose2d(self.features_size, 4, 4, 2, padding=1),
            # out: 4, 2x fheight, 2x fwidth
            nn.ConvTranspose2d(4, len(seg_labels), 32, 16, padding=8),
            # out: n_seg_labels, in_size, in_size
        )

        self.classifier = densenet.classifier

    def forward(self, x):
        in_size = x.size()[-2:]

        features = self.features(x)
        # shape: batch_size, n_features, f-height, f-width

        x = self.relu(features) # Pytorch implementation also uses this Relu
        x = self.global_pool(x)
        # shape: batch_size, n_features

        classification = self.classifier(x)
        # shape: batch_size, n_diseases

        segmentation = self.segmentator(features)
        # shape: batch_size, n_diseases, height, width

        if _ASSERT_IN_OUT_IMAGE_SIZE:
            out_size = segmentation.size()[-2:]
            assert in_size == out_size, f'Image sizes do not match: in={in_size} vs out={out_size}'

        return classification, segmentation


def _build_class(name):
    densenet_params = TINY_DENSENET_CONFIGS[name]
    return partial(
        CustomDenseNetClsSeg,
        model_name=name,
        **densenet_params,
    )


# Num params: 409,070
TinyDenseNetCNN = _build_class('tiny-densenet')

# Num params: 939,292
SmallDenseNetCNN = _build_class('small-densenet')
