import logging

from medai.models.cls_spatial.imagenet_cls_spatial import ImageNetClsSpatialModel
from medai.utils import partialclass

LOGGER = logging.getLogger(__name__)

_CLS_SPATIAL_MODELS_DEF = {
    'densenet-121-cls-spatial': partialclass(ImageNetClsSpatialModel, model_name='densenet-121'),
    'resnet-50-cls-spatial': partialclass(ImageNetClsSpatialModel, model_name='resnet-50'),
    'mobilenet-cls-spatial': partialclass(ImageNetClsSpatialModel, model_name='mobilenet'),
}

AVAILABLE_CLS_SPATIAL_MODELS = list(_CLS_SPATIAL_MODELS_DEF)

def _get_printable_kwargs(kwargs):
    info_str = ' '.join(f'{k}={v}' for k, v in kwargs.items() if 'labels' not in k)
    info_str += f' n_cl={len(kwargs.get("cl_labels", []))}'
    return info_str

def create_cls_spatial_model(model_name, **kwargs):
    if model_name not in _CLS_SPATIAL_MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    LOGGER.info('Creating CLS-spatial: %s, %s', model_name, _get_printable_kwargs(kwargs))

    ModelClass = _CLS_SPATIAL_MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model
