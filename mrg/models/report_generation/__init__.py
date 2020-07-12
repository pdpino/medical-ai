from mrg.models.report_generation.decoder_lstm import LSTMDecoder
from mrg.models.report_generation.decoder_lstm_att import LSTMAttDecoder
from mrg.models.report_generation.decoder_h_lstm_att import HierarchicalLSTMAttDecoder

_MODELS_DEF = {
    'lstm': LSTMDecoder,
    'lstm-att': LSTMAttDecoder,
    'h-lstm-att': HierarchicalLSTMAttDecoder,
}

AVAILABLE_DECODERS = list(_MODELS_DEF)


def is_decoder_hierarchical(decoder_name):
    return decoder_name == 'h-lstm-att'


def create_decoder(decoder_name, **kwargs):
    if decoder_name not in _MODELS_DEF:
        raise Exception(f'Decoder not found: {decoder_name}')

    ModelClass = _MODELS_DEF[decoder_name]
    model = ModelClass(**kwargs)

    return model
