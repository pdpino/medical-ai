import os
import json
import logging

from medai.models.checkpoint import load_compiled_model
from medai.utils.files import get_checkpoint_folder

LOGGER = logging.getLogger(__name__)

def save_training_stats(run_id,
                        batch_size,
                        epochs,
                        secs_per_epoch,
                        hw_options,
                        initial_epoch=0,
                        dryrun=False,
                        ):
    if dryrun:
        return
    training_stats = {
        'secs_per_epoch': secs_per_epoch,
        'batch_size': batch_size,
        'initial_epoch': initial_epoch,
        'n_epochs': epochs,
        'hw_options': hw_options,
    }
    folder = get_checkpoint_folder(run_id, save_mode=True)
    final_epoch = initial_epoch + epochs
    fpath = os.path.join(folder, f'training-stats-{initial_epoch}-{final_epoch}.json')

    with open(fpath, 'w') as f:
        json.dump(training_stats, f, indent=2)

    LOGGER.debug('Saved training stats to %s', fpath)


def load_pretrained_weights_cnn_(target_model, pretrained_run_id,
                                 cls_weights=False, seg_weights=False, device='cuda'):
    """Loads state_dict from a pretrained_model into a target_model.

    Works for cls and cls-seg models
    """
    if pretrained_run_id is None:
        return

    _info = {
        'run': pretrained_run_id.short_name,
        'task': pretrained_run_id.task,
        'dataset': pretrained_run_id.get_dataset_name(),
        'cls': cls_weights,
        'seg': seg_weights,
    }
    LOGGER.info(
        'Using pretrained model: %s',
        ' '.join(f"{k}={v}" for k, v in _info.items()),
    )

    pretrained_model = load_compiled_model(pretrained_run_id, device=device).model

    # Copy features
    target_model.features.load_state_dict(pretrained_model.features.state_dict())

    if cls_weights:
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

        prev_layers = pretrained_model.classifier
        target_model.classifier.load_state_dict(prev_layers.state_dict())

    if seg_weights:
        if not hasattr(pretrained_model, 'segmentator'):
            LOGGER.error('Pretrained model does not have a segmentator!')
        elif not hasattr(target_model, 'segmentator'):
            LOGGER.error('Target model does not have a segmentator!')
        else:
            target_model.segmentator.load_state_dict(pretrained_model.segmentator.state_dict())
