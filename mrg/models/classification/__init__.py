import torch

from mrg.models.classification import (
    resnet,
    densenet,
    transfusion,
    vgg,
    mobilenet,
)

_MODELS_DEF = {
    'resnet-50': resnet.Resnet50CNN,
    'densenet-121': densenet.Densenet121CNN,
    'tfs-small': transfusion.class_wrapper('small'),
    'tfs-tall': transfusion.class_wrapper('tall'),
    'tfs-wide': transfusion.class_wrapper('wide'),
    'tfs-tiny': transfusion.class_wrapper('tiny'),
    'vgg-19': vgg.VGG19CNN,
    'mobilenet': mobilenet.MobileNetV2CNN,
}

AVAILABLE_CLASSIFICATION_MODELS = list(_MODELS_DEF)


def init_empty_model(model_name, labels, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(labels, **kwargs)

    return model
