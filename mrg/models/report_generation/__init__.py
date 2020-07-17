from mrg.models.report_generation.decoder_lstm import LSTMDecoder
from mrg.models.report_generation.decoder_lstm_att import LSTMAttDecoder
from mrg.models.report_generation.decoder_h_lstm_att import h_lstm_wrapper

_MODELS_DEF = {
    'lstm': LSTMDecoder,
    'lstm-att': LSTMAttDecoder,
    'h-lstm': h_lstm_wrapper(attention=False),
    'h-lstm-att': h_lstm_wrapper(attention=True),
}

AVAILABLE_DECODERS = list(_MODELS_DEF)


def is_decoder_hierarchical(decoder_name):
    return 'h-lstm' in decoder_name


def create_decoder(decoder_name, **kwargs):
    if decoder_name not in _MODELS_DEF:
        raise Exception(f'Decoder not found: {decoder_name}')

    ModelClass = _MODELS_DEF[decoder_name]
    model = ModelClass(**kwargs)

    return model
