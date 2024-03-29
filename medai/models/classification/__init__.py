from functools import partial
import logging
from torch import nn

from medai.models.classification import (
    resnet,
    densenet,
    transfusion,
    vgg,
    mobilenet,
    tiny_res_scan,
    tiny_densenet,
    efficientnet,
)
from medai.models.classification.load_imagenet import ImageNetModel

LOGGER = logging.getLogger(__name__)

_MODELS_DEF = {
    'resnet-50': resnet.Resnet50CNN,
    'densenet-121': densenet.Densenet121CNN,
    'tfs-small': partial(transfusion.TransfusionCBRCNN, model_name='small'),
    'tfs-tall': partial(transfusion.TransfusionCBRCNN, model_name='tall'),
    'tfs-wide': partial(transfusion.TransfusionCBRCNN, model_name='wide'),
    'tfs-tiny': partial(transfusion.TransfusionCBRCNN, model_name='tiny'),
    'vgg-19': vgg.VGG19CNN,
    'mobilenet': mobilenet.MobileNetV2CNN,
    'resnet-50-v2': partial(ImageNetModel, model_name='resnet-50'),
    'densenet-121-v2': partial(ImageNetModel, model_name='densenet-121'),
    'mobilenet-v2': partial(ImageNetModel, model_name='mobilenet'),
    'tiny-res-scan': tiny_res_scan.TinyResScanCNN,
    'tiny-densenet': tiny_densenet.TinyDenseNetCNN,
    'small-densenet': tiny_densenet.SmallDenseNetCNN,
    'effnet': efficientnet.EfficientNet,
}

_DEPRECATED_CNNS = set([
    'resnet-50', 'densenet-121', 'mobilenet', # Use v2 instead
])

AVAILABLE_CLASSIFICATION_MODELS = list(_MODELS_DEF)


_MODELS_WITH_DROPOUT_IMPLEMENTED = (
    'tiny-densenet',
    'small-densenet',
    'densenet-121-v2',
    'effnet',
)

def _get_printable_kwargs(kwargs):
    info_str = ' '.join(f'{k}={v}' for k, v in kwargs.items() if k != 'labels')
    info_str += f' n_labels={len(kwargs.get("labels", []))}'
    return info_str


def create_cnn(model_name=None, allow_deprecated=False, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    if not allow_deprecated and model_name in _DEPRECATED_CNNS:
        raise Exception(f'CNN is deprecated: {model_name}')

    dropout = kwargs.get('dropout', 0)
    if dropout != 0 and model_name not in _MODELS_WITH_DROPOUT_IMPLEMENTED:
        LOGGER.error('Dropout not implemented for %s, ignoring', model_name)

    LOGGER.info('Creating CNN: %s, %s', model_name, _get_printable_kwargs(kwargs))

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model


def find_cnn_name_in_run_name(run_name):
    for model_name in AVAILABLE_CLASSIFICATION_MODELS:
        if f'_{model_name}_' in run_name:
            return model_name
    return 'precnn'
