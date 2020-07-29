import torch
from torch import nn

class CNN2Seq(nn.Module):
    """Encoder-decoder architecture container.

    Encoder can be any CNN that accepts features=True.
    Decoder can be: `LSTMDecoder`, `LSTMAttDecoder`, `HierarchicalLSTMAttDecoder`.
    """
    def __init__(self, cnn, decoder):
        super().__init__()

        self.cnn = cnn
        self.decoder = decoder

    def forward(self, images, reports=None, **kwargs):
        """Forward method.

        images -- shape: batch_size, n_channels, height, width
        reports -- shape: batch_size, max_sentence_len
        """
        features = self.cnn(images, features=True)
        # shape: batch_size, n_features, height, width

        result = self.decoder(features, reports=reports, **kwargs)

        return result