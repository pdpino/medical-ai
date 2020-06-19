import torch
from torch import nn

class CNN2Seq(nn.Module):
    """Encoder-decoder architecture.

    Decoder can be: `LSTMDecoder`.
    """
    def __init__(self, cnn, decoder):
        super().__init__()

        self.cnn = cnn
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(cnn.features_size, decoder.hidden_size)
        self.decoder = decoder

    def forward(self, images, max_sentence_len, reports=None):
        """Forward method.

        images -- shape: batch_size, 3, height, width
            (3 for RGB)
        max_sentence_len -- int
        reports -- shape: batch_size, max_sentence_len
        """
        features = self.cnn.features(images)
        # shape: batch_size, n_features, height, width

        features = self.flatten(features)
        # shape: batch_size, n_features * height * width
        # n_features * height * width = cnn.features_size

        initial_decoder_state = self.linear(features)
        # shape: batch_size, hidden_size

        decoded, = self.decoder(initial_decoder_state, max_sentence_len, reports=reports)
        # shape: batch_size, max_sentence_len, vocab_size

        return decoded,