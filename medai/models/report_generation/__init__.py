import logging
from functools import partial
from medai.models.report_generation.decoder_lstm import LSTMDecoder
from medai.models.report_generation.decoder_lstm_v2 import LSTMDecoderV2
from medai.models.report_generation.decoder_lstm_att import LSTMAttDecoder
from medai.models.report_generation.decoder_lstm_att_v2 import LSTMAttDecoderV2
from medai.models.report_generation.decoder_lstm_att_v3 import LSTMAttDecoderV3
from medai.models.report_generation.decoder_h_lstm_att import HierarchicalLSTMAttDecoder
from medai.models.report_generation.decoder_h_lstm_att_v2 import HierarchicalLSTMAttDecoderV2

LOGGER = logging.getLogger(__name__)

_MODELS_DEF = {
    'lstm': LSTMDecoder,
    'lstm-v2': LSTMDecoderV2,
    'lstm-att': LSTMAttDecoder,
    'lstm-att-v2': LSTMAttDecoderV2,
    'lstm-att-v3': LSTMAttDecoderV3,
    'h-lstm': partial(HierarchicalLSTMAttDecoder, attention=False),
    'h-lstm-att': partial(HierarchicalLSTMAttDecoder, attention=True),
    'h-lstm-v2': partial(HierarchicalLSTMAttDecoderV2, attention=False),
    'h-lstm-att-v2': partial(HierarchicalLSTMAttDecoderV2, attention=True),
}

AVAILABLE_DECODERS = list(_MODELS_DEF)

DEPRECATED_DECODERS = set([
    'lstm',
    'lstm-att',
    'lstm-att-v3', # not used for now
    'h-lstm',
    'h-lstm-att',
])

def is_decoder_hierarchical(decoder_name):
    return 'h-lstm' in decoder_name


def create_decoder(decoder_name, **kwargs):
    if decoder_name not in _MODELS_DEF:
        raise Exception(f'Decoder not found: {decoder_name}')

    # NOTE: for backward compatibility
    # features_size used to be a tuple/list with (n_features, height, width)
    # now both are separated into n_features and image_size
    # (though notice most/all decoders do not use image_size)
    if 'features_size' in kwargs:
        features_size = kwargs['features_size']
        if isinstance(features_size, (tuple, list)):
            kwargs['features_size'] = features_size[0]
            kwargs['image_size'] = features_size[1:]

    info_str = ' '.join(f'{k}={v}' for k, v in kwargs.items())
    LOGGER.info('Creating decoder: %s, %s', decoder_name, info_str)

    ModelClass = _MODELS_DEF[decoder_name]

    if not getattr(ModelClass, 'implemented_dropout', False):
        ignored_options = [
            k
            for k, v in kwargs.items()
            if 'dropout' in k and v != 0
        ]
        if ignored_options:
            LOGGER.warning(
                'Dropout not implemented in %s, ignoring %s', decoder_name, ignored_options,
            )

    model = ModelClass(**kwargs)

    return model
