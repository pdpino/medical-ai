from medai.models.detection.scan import ResScanFull

_MODELS_DEF = {
    'resnet-scan': ResScanFull,
}

AVAILABLE_DETECTION_SEG_MODELS = list(_MODELS_DEF)

def create_detection_seg_model(model_name, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model
