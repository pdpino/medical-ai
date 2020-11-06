from medai.models.segmentation.scan import ScanFCN

_MODELS_DEF = {
    'scan': ScanFCN,
}

AVAILABLE_SEGMENTATION_MODELS = list(_MODELS_DEF)

def create_fcn(model_name, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model
