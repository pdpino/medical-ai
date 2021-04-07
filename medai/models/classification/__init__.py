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
)
from medai.models.classification.load_imagenet import ImageNetModel
from medai.models.cls_seg.imagenet import ImageNetClsSegModel

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
}

_DEPRECATED_CNNS = set([
    'resnet-50', 'densenet-121', 'mobilenet', # Use v2 instead
])

AVAILABLE_CLASSIFICATION_MODELS = list(_MODELS_DEF)


_MODELS_WITH_DROPOUT_IMPLEMENTED = (
    'tiny-densenet',
    'small-densenet',
    'densenet-121-v2',
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


def get_last_layer(model):
    """Returns the last layer of a model, to be used for Grad-CAM."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    model_name = model.model_name

    layer = None

    if isinstance(model, (ImageNetModel, ImageNetClsSegModel)):
        if model_name == 'mobilenet':
            layer = model.features[-1][0] # -1
        if model_name == 'densenet-121':
            layer = model.features.denseblock4.denselayer16.conv2 # norm5
        if model_name == 'resnet-50':
            layer = model.features[-1][-1].conv3 # relu
    elif isinstance(model, tiny_densenet.CustomDenseNetCNN):
        if model_name == 'tiny-densenet':
            layer = model.features.denseblock4.denselayer12.conv2
        if model_name == 'small-densenet':
            layer = model.features.denseblock4.denselayer12.conv2

    # DEPRECATED MODELS
    else:
        if model_name == 'mobilenet':
            # layer = model.base_cnn.features[-1][0] # Last conv
            layer = model.base_cnn.features[-1][-1] # Actual last
        if model_name == 'densenet-121':
            # layer = model.base_cnn.features.denseblock4.denselayer16.conv2 # Last conv
            layer = model.base_cnn.features.norm5 # Actual last
        if model_name == 'resnet-50':
            # layer = model.base_cnn.layer4[-1].conv3 # Last conv
            layer = model.base_cnn.layer4[-1].relu # Actual last

    if layer is None:
        raise Exception(f'Last layer not defined for: {model_name}')
    return layer
