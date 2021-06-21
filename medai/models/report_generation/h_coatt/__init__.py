"""2nd implementation of the CoAtt Model.

Based on the https://github.com/ZexinYan/Medical-Report-Generation
repo, but improved and adapted to medai code.
"""
from torch import nn

from medai.models.report_generation.h_coatt.encoder import HCoAttEncoder
from medai.models.report_generation.h_coatt.decoder import HCoAttDecoder


class HCoAtt(nn.Module):
    def __init__(self, encoder_kwargs, decoder_kwargs):
        super().__init__()

        self.encoder = HCoAttEncoder(**encoder_kwargs)

        decoder_kwargs['features_size'] = self.encoder.features_size
        self.decoder = HCoAttDecoder(**decoder_kwargs)

    def forward(self, images, **kwargs):
        # images shape: batch_size, 3, height, width

        local_features, tags, tags_embeddings = self.encoder(images)

        reports, stops, scores_v, scores_t, topics = self.decoder(
            local_features,
            tags_embeddings,
            **kwargs,
        )

        return reports, stops, scores_v, scores_t, topics, tags
