from torch import nn


from medai.models.classification import tiny_densenet
from medai.models.cls_seg import tiny_densenet as tiny_densenet_cls_seg
from medai.models.classification.load_imagenet import ImageNetModel
from medai.models.cls_seg.imagenet import ImageNetClsSegModel

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
    elif isinstance(model, (
        tiny_densenet.CustomDenseNetCNN, tiny_densenet_cls_seg.CustomDenseNetClsSeg,
        )):
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
