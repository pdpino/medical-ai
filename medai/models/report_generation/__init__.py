import logging
import functools
from medai.models.report_generation.decoder_lstm import LSTMDecoder
from medai.models.report_generation.decoder_lstm_v2 import LSTMDecoderV2
from medai.models.report_generation.decoder_lstm_att import LSTMAttDecoder
from medai.models.report_generation.decoder_lstm_att_v2 import LSTMAttDecoderV2
from medai.models.report_generation.decoder_lstm_att_v3 import LSTMAttDecoderV3
from medai.models.report_generation.decoder_h_lstm_att import HierarchicalLSTMAttDecoder
from medai.models.report_generation.decoder_h_lstm_att_v2 import HierarchicalLSTMAttDecoderV2
from medai.utils import partialclass

LOGGER = logging.getLogger(__name__)

_MODELS_DEF = {
    'lstm': LSTMDecoder,
    'lstm-v2': LSTMDecoderV2,
    'lstm-att': LSTMAttDecoder,
    'lstm-att-v2': LSTMAttDecoderV2,
    'lstm-att-v3': LSTMAttDecoderV3,
    'h-lstm': partialclass(HierarchicalLSTMAttDecoder, attention=False),
    'h-lstm-att': partialclass(HierarchicalLSTMAttDecoder, attention=True),
    'h-lstm-v2': partialclass(HierarchicalLSTMAttDecoderV2, attention=False),
    'h-lstm-att-v2': partialclass(HierarchicalLSTMAttDecoderV2, attention=True),
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


def _get_info_str(**kwargs):
    _printable_kwargs = {}
    for k, v in kwargs.items():
        if k == 'vocab':
            v = len(v)
        elif k == 'embedding_kwargs':
            continue

        _printable_kwargs[k] = v
    return ' '.join(f'{k}={v}' for k, v in _printable_kwargs.items())


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

    # NOTE: backward compatibility for older models that do not specify this
    if decoder_name == 'h-lstm-att-v2':
        if 'double_bias' not in kwargs:
            kwargs['double_bias'] = True

    # NOTE: backward compatibility
    # Before: only vocab_size was passed
    # After: whole vocab
    if 'vocab_size' in kwargs:
        # Pass a dummy vocabulary as vocab
        # This does not break anything, because older implementations will only use len(vocab)
        kwargs['vocab'] = range(kwargs['vocab_size'])
        del kwargs['vocab_size']

    LOGGER.info('Creating decoder: %s, %s', decoder_name, _get_info_str(**kwargs))

    ModelClass = _MODELS_DEF[decoder_name]

    # Check if dropout is implemented
    if not getattr(ModelClass, 'implemented_dropout', False):
        ignored_options = [
            k
            for k, v in kwargs.items()
            if 'dropout' in k and v != 0
        ]
        if ignored_options:
            LOGGER.error(
                'Dropout not implemented in %s, ignoring %s', decoder_name, ignored_options,
            )

    model = ModelClass(**kwargs)

    return model
