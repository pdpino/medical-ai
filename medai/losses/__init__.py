import torch
from torch import nn

from medai.datasets.common import CXR14_POS_WEIGHTS
from medai.losses.focal import FocalLoss
from medai.losses.wbce import WeigthedBCELoss, WeigthedBCEByDiseaseLoss

_LOSS_CLASSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce': WeigthedBCELoss,
    'wbce-d': WeigthedBCEByDiseaseLoss,
    'focal': FocalLoss,
    'cross-entropy': nn.CrossEntropyLoss, # Use only for multilabel=False models!
}

AVAILABLE_LOSSES = list(_LOSS_CLASSES)

POS_WEIGHTS_BY_DATASET = {
    'cxr14': CXR14_POS_WEIGHTS,
}

def get_loss_function(loss_name, **loss_kwargs):
    if loss_name not in _LOSS_CLASSES:
        raise Exception(f'Loss not found: {loss_name}')

    LossClass = _LOSS_CLASSES[loss_name]

    pos_weight = loss_kwargs.get('pos_weight')
    if pos_weight is not None and isinstance(pos_weight, (list, tuple)):
        # pylint: disable=not-callable
        loss_kwargs['pos_weight'] = torch.tensor(pos_weight)

    loss = LossClass(**loss_kwargs)
    return loss
