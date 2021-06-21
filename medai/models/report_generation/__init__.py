import logging
from medai.models.classification import create_cnn
from medai.models.cls_seg import create_cls_seg_model

from medai.models.report_generation.h_coatt import HCoAtt
from medai.models.report_generation.coatt import CoAttModel
from medai.models.report_generation.cnn_to_seq import CNN2Seq
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
    return decoder_name.startswith('h-')


def _get_info_str(**kwargs):
    _printable_kwargs = {}
    for k, v in kwargs.items():
        if k in ('vocab', 'labels'):
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


def create_cnn_rg(task='cls', **kwargs):
    if task == 'cls':
        return create_cnn(**kwargs)
    if task == 'cls-seg':
        return create_cls_seg_model(**kwargs)

    raise Exception('CNN task not supported: ', task)


def create_rg_model(name, **model_kwargs):
    def _get_kwarg_and_assert(key):
        value = model_kwargs.get(key, None)
        assert value is not None, f'{key} are not present in model_kwargs'
        return value

    if name == 'coatt':
        model = CoAttModel(**model_kwargs)
    elif name == 'h-coatt':
        encoder_kwargs = _get_kwarg_and_assert('encoder_kwargs')
        decoder_kwargs = _get_kwarg_and_assert('decoder_kwargs')

        LOGGER.info('Creating h-coatt-encoder: %s', _get_info_str(**encoder_kwargs))
        LOGGER.info('Creating h-coatt-decoder: %s', _get_info_str(**decoder_kwargs))

        model = HCoAtt(encoder_kwargs, decoder_kwargs)
    else:
        # Create CNN
        cnn_kwargs = _get_kwarg_and_assert('cnn_kwargs')

        if 'task' not in cnn_kwargs:
            # HACK: hotfix for backward compatibility
            if 'cls-seg' in cnn_kwargs['model_name']:
                task = 'cls-seg'
            else:
                task = 'cls'
            cnn_kwargs['task'] = task
        cnn = create_cnn_rg(**cnn_kwargs)

        # Create Decoder
        decoder = create_decoder(**model_kwargs['decoder_kwargs'])

        # Create CNN2Seq model and optimizer
        model = CNN2Seq(cnn, decoder)
    return model
