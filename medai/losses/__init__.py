import torch
from torch import nn

from medai.losses.focal import FocalLoss
from medai.losses.wbce import WeigthedBCELoss, WeigthedBCEByDiseaseLoss

_LOSS_CLASSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce': WeigthedBCELoss,
    'wbce_by_disease': WeigthedBCEByDiseaseLoss,
    'focal': FocalLoss,
    'cross-entropy': nn.CrossEntropyLoss, # Use only for multilabel=False models!
}

AVAILABLE_LOSSES = list(_LOSS_CLASSES)

def get_loss_function(loss_name, **loss_params):
    if loss_name not in _LOSS_CLASSES:
        raise Exception(f'Loss not found: {loss_name}')
        
    LossClass = _LOSS_CLASSES[loss_name]
        
    loss = LossClass(**loss_params)
    return loss