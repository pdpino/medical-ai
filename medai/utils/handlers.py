"""Common handlers for any engine."""
import time
import logging
import numbers
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from medai.utils import duration_to_str

LOGGER = logging.getLogger(__name__)


_shorter_names = {
    'roc_auc': 'roc',
    'cl_loss': 'cl',
    'hint_loss': 'hint',
    'seg_loss': 'seg',
    'mse-total': 'mse-t',
    'mse-pos': 'mse-p',
    'mse-neg': 'mse-n',
    'n-shapes-gen': 'shapes',
    'n-holes-gen': 'holes',
}

def _shorten(metric_name):
    return _shorter_names.get(metric_name, metric_name)


def _prettify(value):
    """Prettify a metric value."""
    if value is None:
        return -1
    if isinstance(value, numbers.Number):
        return str(round(value, 3))
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


def attach_lr_scheduler_handler(lr_scheduler,
                                trainer,
                                validator,
                                target_metric='loss',
                                ):
    """Attaches a callback that updates the lr_scheduler.

    Note: if the metric value is -1, the scheduler will not be called.
        For metrics that may not be calculated on every epoch, such as chex_f1,
        this implies the scheduler patience will be accounted only in the epochs
        where the value is indeed calculated; other epochs will be skipped.
        For metrics that are calculated on every epoch, this has no effect.
    """
    _IGNORE_WARNING_METRICS = ('chex_f1',)

    def _update_scheduler(unused_engine):
        val_metrics = validator.state.metrics
        if target_metric not in val_metrics:
            if target_metric not in _IGNORE_WARNING_METRICS:
                LOGGER.warning(
                    'Cannot step LR-scheduler, %s not found in val_metrics', target_metric,
                )
            return
        value = val_metrics[target_metric]

        # NOTE: ignores -1 values
        if value != -1:
            lr_scheduler.step(value)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, _update_scheduler)
