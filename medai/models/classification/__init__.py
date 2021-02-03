from functools import partial
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


def create_cnn(model_name, labels, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(labels, **kwargs)

    return model
