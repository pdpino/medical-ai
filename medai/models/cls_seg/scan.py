import torch.nn as nn

from medai.models.common import get_adaptive_pooling_layer
from medai.models.segmentation.scan import _ParallelResBlocks, _ResBlock

class ScanClsSeg(nn.Module):
    """Resnet-like CNN, using SCAN residual-blocks, with SEG and CLS output."""
    def __init__(self, cl_labels, seg_labels, gpool='max',
                 dropout_features=0,
                 **unused_kwargs):
        super().__init__()

        # height and width sizes are considering input size (400 x 400)
        self.features = nn.Sequential(
            _ParallelResBlocks(8, 1, 7), # output: 8 x 400 x 400
            _ResBlock(8, 3),
            nn.AvgPool2d((2, 2), (2, 2)), # output: 8 x 200 x 200
            _ParallelResBlocks(2, 8, 3), # output: 16 x 200 x 200
            nn.AvgPool2d((2, 2), (2, 2)), # output: 16 x 100 x 100
            _ParallelResBlocks(2, 16, 3), # output: 32 x 100 x 100
            nn.AvgPool2d((2, 2), (2, 2)), # output: 32 x 50 x 50
            _ParallelResBlocks(2, 32, 3), # output: 64 x 50 x 50
            nn.AvgPool2d((2, 2), (2, 2)), # output: 64 x 25 x 25
            _ResBlock(64, 1),
            _ResBlock(64, 3),
            _ResBlock(64, 1),
            _ResBlock(64, 3),
            _ResBlock(64, 1), # output: 64 x 25 x 25
        )

        self.dropout_features = dropout_features

        self.cl_labels = list(cl_labels)
        self.seg_labels = list(seg_labels)
        self.features_size = 64

        self.segmentator = nn.Sequential(
            nn.Conv2d(self.features_size, 4, 1, 1),# output: 4 x 25 x 25
            nn.ConvTranspose2d(4, len(seg_labels), 32, 16, padding=8),
            # output: n_seg_labels x 400 x 400
        )

        self.classifier = nn.Sequential(
            get_adaptive_pooling_layer(gpool, drop=self.dropout_features),
            nn.Linear(self.features_size, len(cl_labels)),
        )

    def forward(self, x):
        # x shape: batch_size, n_channels=1, height, width

        x = self.features(x) # shape: bs, n_features=64, f-height, f-width

        classification = self.classifier(x) # shape: bs, n_cl_labels

        segmentation = self.segmentator(x) # shape: bs, n_seg_labels, height, width

        return (classification, segmentation)
