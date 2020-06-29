import torch
# from torch.nn import DataParallel
# import os

# from cxr8 import utils
# from cxr8.training import optimizers
from mrg.models.classification import resnet

_MODELS_DEF = {
    'resnet': resnet.Resnet50CNN,
}

AVAILABLE_MODELS = list(_MODELS_DEF)


def init_empty_model(model_name, labels, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')
    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(labels, **kwargs)

    return model
