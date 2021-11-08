from collections import namedtuple
import logging
import os
import re

LOGGER = logging.getLogger(__name__)

_CHECKPOINT_REGEX = re.compile(r'^[A-Za-z]+_(\d+)(?:_([\w\-]+)=([\d\.]+)\.pt)?')

CheckpointInfo = namedtuple('CheckpointInfo', ('fname', 'epoch', 'metric', 'value'))

def _split_checkpoint_name(name):
    matched = _CHECKPOINT_REGEX.match(name)
    if matched is None:
        LOGGER.error('Checkpoint name does not match regex: %s', name)
        return CheckpointInfo(epoch=-1, metric_value=-100)

    epoch, metric_name, metric_value = matched.groups()
    return CheckpointInfo(name, epoch, metric_name, metric_value)


def _get_last(infos):
    return max(infos, key=lambda x: x.epoch)

def _get_best(infos, metric):
    infos = [info for info in infos if info.metric == metric]
    if len(infos) == 0:
        return None
    return max(infos, key=lambda x: x.value)


def get_checkpoint_metrics_from_folder(folder):
    checkpoint_names = [
        fname
        for fname in os.listdir(folder)
        if fname.endswith('.pt')
    ]

    if len(checkpoint_names) == 0:
        return []

    checkpoint_infos = [
        _split_checkpoint_name(name)
        for name in checkpoint_names
    ]

    return sorted(list(set(
        info.metric for info in checkpoint_infos if info.metric is not None
    )))


def get_checkpoint_filepath(folder, mode='best'):
    """Return the checkpoint fpath for in a folder.

    For the same model, multiple checkpoints can be saved.
    """
    checkpoint_names = [
        fname
        for fname in os.listdir(folder)
        if fname.endswith('.pt')
    ]

    if len(checkpoint_names) == 0:
        raise Exception('Model filepath empty:', folder)

    checkpoint_infos = [
        _split_checkpoint_name(name)
        for name in checkpoint_names
    ]

    if mode == 'last':
        checkpoint = _get_last(checkpoint_infos)
    elif mode == 'best':
        available_metrics = list(set(
            info.metric for info in checkpoint_infos if info.metric is not None
        ))
        if len(available_metrics) == 0:
            LOGGER.error(
                'Loading best-checkpoint: no metrics found, fallback to last',
            )
            checkpoint = _get_last(checkpoint_infos)
        elif len(available_metrics) == 1:
            checkpoint = _get_best(checkpoint_infos, available_metrics[0])
        else:
            available_metrics = sorted(available_metrics) # Make it deterministic
            first_metric = available_metrics[0]
            LOGGER.warning(
                'Loading best-checkpoint: multiple metrics found %s, using %s',
                available_metrics, first_metric,
            )
            checkpoint = _get_best(checkpoint_infos, first_metric)
    else:
        # Assume "mode" is the name of the metric
        checkpoint = _get_best(checkpoint_infos, mode)
        if checkpoint is None:
            raise Exception(
                f'Loading checkpoint-{mode}: metric not found, available={checkpoint_infos}',
            )

    if checkpoint is None:
        raise Exception(
            f'Could not find a suitable checkpoint, mode={mode}, available={checkpoint_infos}',
        )

    LOGGER.info('Loading from checkpoint: %s', checkpoint)
    LOGGER.debug('Available checkpoints: %s', checkpoint_infos)
    return os.path.join(folder, checkpoint.fname)
