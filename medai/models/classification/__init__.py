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

DEPRECATED_CNNS = set([
    'resnet-50', 'densenet-121', 'mobilenet', # Use v2 instead
])

AVAILABLE_CLASSIFICATION_MODELS = list(_MODELS_DEF)


_MODELS_WITH_DROPOUT_IMPLEMENTED = (
    'tiny-densenet',
    'small-densenet',
    'densenet-121-v2',
)

def create_cnn(model_name, labels, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    dropout = kwargs.get('dropout', 0)
    if dropout != 0 and model_name not in _MODELS_WITH_DROPOUT_IMPLEMENTED:
        LOGGER.error('Dropout not implemented for %s, ignoring', model_name)

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(labels, **kwargs)

    return model


def get_last_layer(model):
    """Returns the last layer of a model, to be used for Grad-CAM."""
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    model_name = model.model_name

    if isinstance(model, ImageNetModel):
        if model_name == 'mobilenet':
            return model.features[-1][0] # -1
        if model_name == 'densenet-121':
            return model.features.denseblock4.denselayer16.conv2 # norm5
        if model_name == 'resnet-50':
            return model.features[-1][-1].conv3 # relu
    elif isinstance(model, tiny_densenet.CustomDenseNetCNN):
        if model_name == 'tiny-densenet':
            return model.features.denseblock4.denselayer12.conv2
        if model_name == 'small-densenet':
            pass # TODO: define this one

    # DEPRECATED MODELS
    else:
        if model_name == 'mobilenet':
            # return model.base_cnn.features[-1][0] # Last conv
            return model.base_cnn.features[-1][-1] # Actual last
        if model_name == 'densenet-121':
            # return model.base_cnn.features.denseblock4.denselayer16.conv2 # Last conv
            return model.base_cnn.features.norm5 # Actual last
        if model_name == 'resnet-50':
            # return model.base_cnn.layer4[-1].conv3 # Last conv
            return model.base_cnn.layer4[-1].relu # Actual last

    raise Exception(f'Last layer not defined for: {model_name}')
