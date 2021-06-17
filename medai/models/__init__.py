import logging

from medai.models.checkpoint import load_compiled_model
from medai.utils.files import RunId

LOGGER = logging.getLogger(__name__)

def load_pretrained_weights_cnn_(target_model, pretrained,
                                 features=True,
                                 cls_weights=False, seg_weights=False,
                                 spatial_weights=False,
                                 device='cuda'):
    """Loads state_dict from a pretrained_model into a target_model.

    Works for cls and cls-seg models
    """
    if pretrained is None:
        return

    _info = {
        'features': features,
        'cls': cls_weights,
        'seg': seg_weights,
        'spatial': spatial_weights,
    }

    if isinstance(pretrained, RunId):
        pretrained_model = load_compiled_model(pretrained, device=device).model
        _info.update({
            'run': pretrained.short_name,
            'task': pretrained.task,
            'dataset': pretrained.get_dataset_name(),
        })
    else:
        # Assume is a model
        pretrained_model = pretrained

    LOGGER.info(
        'Using pretrained model: %s',
        ' '.join(f"{k}={v}" for k, v in _info.items()),
    )

    def _check_weights_exist(key):
        prev_has = hasattr(pretrained_model, key)
        target_has = hasattr(target_model, key)
        if not prev_has:
            LOGGER.error('Pretrained model does not have %s!', key)
        if not target_has:
            LOGGER.error('Target model does not have %s!', key)
        return prev_has and target_has

    # Copy features
    if features and _check_weights_exist('features'):
        target_model.features.load_state_dict(pretrained_model.features.state_dict())

    if cls_weights and _check_weights_exist('classifier'):
        new_labels = target_model.cl_labels
        n_new_labels = len(new_labels)
        old_labels = pretrained_model.cl_labels
        n_old_labels = len(old_labels)
        if n_old_labels != n_new_labels:
            raise Exception(f'N-labels do not match: old={n_old_labels} vs now={n_new_labels}')
        if old_labels != new_labels:
            LOGGER.warning(
                'Labels used do not match with pretrained: pretrained=%s vs this=%s',
                old_labels, new_labels,
            )

        target_model.classifier.load_state_dict(
            pretrained_model.classifier.state_dict(),
        )

    if seg_weights and _check_weights_exist('segmentator'):
        target_model.segmentator.load_state_dict(pretrained_model.segmentator.state_dict())

    if spatial_weights and _check_weights_exist('spatial_classifier'):
        target_model.spatial_classifier.load_state_dict(
            pretrained_model.spatial_classifier.state_dict(),
        )


def freeze_cnn(cnn):
    frozen_layers = []

    for name, param in cnn.named_parameters():
        param.requires_grad = False
        frozen_layers.append(name)

    LOGGER.info('Froze %d cnn layers', len(frozen_layers))
    LOGGER.debug('Frozen layers: %s', frozen_layers)
