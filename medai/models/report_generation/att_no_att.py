from torch import nn

from medai.models.common import get_adaptive_pooling_layer

class NoAttention(nn.Module):
    """Has the same API as an Attention network, but does not implement attention."""
    def __init__(self, reduction='avg'):
        super().__init__()

        self.features_reduction = get_adaptive_pooling_layer(reduction)

    def forward(self, features, unused_h_state):
        # features shape: batch_size, n_features, height, width
        # unused_h_state shape: batch_size, lstm_size

        reduced = self.features_reduction(features)
        # shape: batch_size, n_features

        scores = None

        return reduced, scores
