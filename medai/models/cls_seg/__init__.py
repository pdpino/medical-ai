from functools import partial
import logging

from medai.models.cls_seg.scan import ScanClsSeg
from medai.models.cls_seg.imagenet import ImageNetClsSegModel

LOGGER = logging.getLogger(__name__)

_MODELS_DEF = {
    'scan-cls-seg': ScanClsSeg,
    'densenet-121-cls-seg': partial(ImageNetClsSegModel, model_name='densenet-121'),
    'resnet-50-cls-seg': partial(ImageNetClsSegModel, model_name='resnet-50'),
    'mobilenet-cls-seg': partial(ImageNetClsSegModel, model_name='mobilenet'),
}

AVAILABLE_CLS_SEG_MODELS = list(_MODELS_DEF)

def create_cls_seg_model(model_name, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    dropout = kwargs.get('dropout', 0)
    if dropout != 0 and 'densenet' not in model_name:
        LOGGER.error('Dropout not implemented for %s, ignoring', model_name)

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model
