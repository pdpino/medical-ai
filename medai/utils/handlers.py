"""Common handlers for any engine."""
import json
import logging
import numbers
import os
import time

import torch
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from medai.utils import duration_to_str
from medai.utils.files import get_checkpoint_folder

LOGGER = logging.getLogger(__name__)


_shorter_names = {
    'roc_auc': 'roc',
    'pr_auc': 'pr',
    'cl_loss': 'cl',
    'hamming': 'ham',
    'hint_loss': 'hint',
    'seg_loss': 'seg',
    'spatial_loss': 'spatial',
    'word_loss': 'word',
    'stop_loss': 'stop',
    'att_loss': 'att',
    'sentence_loss': 'sent',
    'mse-total': 'mse-t',
    'mse-pos': 'mse-p',
    'mse-neg': 'mse-n',
    'ciderD': 'C-D',
    'n-shapes-gen': 'shapes',
    'n-holes-gen': 'holes',
    'chex_f1_woNF': 'f1_woNF',
    'lighter-chex_f1': 'l-chex_f1',
    'chex_labeler_misses': 'miss',
    'chex_labeler_misses_unique': 'miss_uniq',
    'chex_labeler_misses_perc': 'miss_perc',
}

def _shorten(metric_name):
    return _shorter_names.get(metric_name, metric_name)


def _prettify(value):
    """Prettify a metric value."""
    if value is None:
        return -1
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, numbers.Number):
        return f'{round(value, 3):<5}'
    return value


def attach_log_metrics(trainer,
                       validator,
                       compiled_model,
                       val_dataloader,
                       tb_writer,
                       timer,
                       logger=LOGGER,
                       initial_epoch=0,
                       print_metrics=['loss'],
                       ):
    """Attaches a function to log metrics after each epoch."""
    def log_metrics(trainer):
        """Performs a step on the end of each epoch."""
        # Run on validation
        if val_dataloader is not None:
            validator.run(val_dataloader, 1)

        # State
        epoch = trainer.state.epoch + initial_epoch
        max_epochs = trainer.state.max_epochs + initial_epoch
        train_metrics = trainer.state.metrics
        val_metrics = validator.state.metrics

        # Save state
        compiled_model.save_current_epoch(epoch)

        # Walltime
        wall_time = time.time()

        # Log to TB
        tb_writer.write_histogram(compiled_model.model, epoch, wall_time)
        tb_writer.write_metrics(train_metrics, 'train', epoch, wall_time)
        tb_writer.write_metrics(val_metrics, 'val', epoch, wall_time)

        # Log to stdout
        metrics_str = ', '.join(
            f'{_shorten(m)} {_prettify(train_metrics.get(m))} {_prettify(val_metrics.get(m))}'
            for m in print_metrics
        )

        duration = duration_to_str(timer._elapsed()) # pylint: disable=protected-access

        logger.info(
            'Epoch %d/%d, %s, %s',
            epoch, max_epochs, metrics_str, duration,
        )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics)


def attach_early_stopping(trainer,
                          validator,
                          attach=True,
                          metric='loss',
                          patience=10,
                          **kwargs,
                          ):
    """Attaches an early stopping handler to a trainer.

    Notes:
        - The handler should be attached after every other handler,
        so those will get executed completely
        - The handler is attached to the trainer, not the validator (as in most examples),
        so the stop signal is sent at the very end of the epoch (i.e. after every handler is run),
        and not after the `validator.run(...)` is run.
        - If a metric value is not present or is -1, the early-stopping handler will be called
        anyway, implying that the run may be terminated even if no values are observed. (This
        is relevant for metrics that may not be calculated on every epoch, such as chex_f1).
    """
    if not attach:
        LOGGER.info('No early stopping')
        return
    info = {
        'metric': metric,
        'patience': patience,
        **kwargs,
    }
    info_str = ' '.join(f'{k}={v}' for k, v in info.items())
    LOGGER.info('Early stopping: %s', info_str)

    # Set early-stopping to info level
    es_logger = logging.getLogger('ignite.handlers.early_stopping.EarlyStopping')
    es_logger.setLevel(logging.INFO)

    def score_fn(unused_engine):
        value = validator.state.metrics.get(metric, -1)
        if metric == 'loss':
            value = -value
        return value

    early_stopping = EarlyStopping(patience=patience,
                                   score_function=score_fn,
                                   trainer=trainer,
                                   **kwargs,
                                   )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, early_stopping)


def attach_save_training_stats(trainer, run_id, timer, epochs, hw_options,
                               initial_epoch=0, dryrun=False):
    if dryrun:
        return

    folder = get_checkpoint_folder(run_id, save_mode=True)
    final_epoch = initial_epoch + epochs
    fpath = os.path.join(folder, f'training-stats-{initial_epoch}-{final_epoch}.json')

    def _save_training_stats(unused_engine):
        dataloader = trainer.state.dataloader
        secs_per_epoch = timer.value()
        training_stats = {
            'secs_per_epoch': secs_per_epoch,
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'initial_epoch': initial_epoch,
            'final_epoch': final_epoch,
            'current_epoch': trainer.state.epoch,
            'hw_options': hw_options,
        }

        with open(fpath, 'w') as f:
            json.dump(training_stats, f, indent=2)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _save_training_stats)
