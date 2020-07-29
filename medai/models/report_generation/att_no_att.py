import torch
from torch import nn

class NoAttention(nn.Module):
    """Has the same API as an Attention network, but does not implement attention."""
    def __init__(self, reduction='avg'):
        super().__init__()

        if reduction == 'avg' or reduction == 'mean':
            reduction_step = nn.AdaptiveAvgPool2d((1, 1))
        elif reduction == 'max':
            reduction_step = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise Exception(f'No such reduction {reduction}')

        self.features_reduction = nn.Sequential(
            reduction_step,
            nn.Flatten(),
        )

    def forward(self, features, unused_h_state):
        # features shape: batch_size, n_features, height, width
        # unused_h_state shape: batch_size, lstm_size
        
        reduced = self.features_reduction(features)
        # shape: batch_size, n_features

        scores = None

        return reduced, scores