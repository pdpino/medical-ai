from medai.models.cls_seg.scan import ScanClsSeg

_MODELS_DEF = {
    'scan-cls-seg': ScanClsSeg,
}

AVAILABLE_CLS_SEG_MODELS = list(_MODELS_DEF)

def create_cls_seg_model(model_name, **kwargs):
    if model_name not in _MODELS_DEF:
        raise Exception(f'Model not found: {model_name}')

    ModelClass = _MODELS_DEF[model_name]
    model = ModelClass(**kwargs)

    return model
