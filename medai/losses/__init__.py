import torch
from torch import nn

from medai.datasets.common import CXR14_POS_WEIGHTS
from medai.losses.focal import FocalLoss
from medai.losses.wbce import WeigthedBCELoss, WeigthedBCEByDiseaseLoss
from medai.losses.out_of_target import OutOfTargetSumLoss

_CLS_LOSS_CLASSES = {
    'bce': nn.BCEWithLogitsLoss,
    'wbce': WeigthedBCELoss,
    'wbce-d': WeigthedBCEByDiseaseLoss,
    'focal': FocalLoss,
    'cross-entropy': nn.CrossEntropyLoss, # Use only for multilabel=False models!
}

# TODO: rename this to AVAILABLE_CL_LOSSES, or alike
AVAILABLE_LOSSES = list(_CLS_LOSS_CLASSES)

POS_WEIGHTS_BY_DATASET = {
    'cxr14': CXR14_POS_WEIGHTS,
}

def get_loss_function(loss_name, **loss_kwargs):
    """Returns a classification loss function for the CL task."""
    # TODO: rename to get_cl_loss or alike
    if loss_name not in _CLS_LOSS_CLASSES:
        raise Exception(f'Loss not found: {loss_name}')

    LossClass = _CLS_LOSS_CLASSES[loss_name]

    pos_weight = loss_kwargs.get('pos_weight')
    if pos_weight is not None and isinstance(pos_weight, (list, tuple)):
        # pylint: disable=not-callable
        loss_kwargs['pos_weight'] = torch.tensor(pos_weight)

    loss = LossClass(**loss_kwargs)
    return loss



_HINT_LOSS_CLASSES = {
    'wbce': WeigthedBCELoss,
    'oot': OutOfTargetSumLoss,
    'bce': nn.BCELoss,
}
AVAILABLE_HINT_LOSSES = list(_HINT_LOSS_CLASSES)


def get_detection_hint_loss(loss_name):
    """Returns a HINT loss function for the DET task."""
    if loss_name not in _HINT_LOSS_CLASSES:
        raise Exception(f'HINT loss not available: {loss_name}')

    LossClass = _HINT_LOSS_CLASSES[loss_name]

    loss_kwargs = {}

    if loss_name == 'wbce':
        loss_kwargs['sigmoid'] = False

    return LossClass(**loss_kwargs)
