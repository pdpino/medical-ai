"""Common handlers for any engine."""
import time
import logging
import numbers
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping

from medai.utils import duration_to_str

def _prettify(value):
    """Prettify a metric value."""
    if value is None:
        return -1
    if isinstance(value, numbers.Number):
        return str(round(value, 4))
    return value

def attach_log_metrics(trainer,
                       validator,
                       compiled_model,
                       val_dataloader,
                       tb_writer,
                       timer,
                       logger=logging,
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
            f'{metric} {_prettify(train_metrics.get(metric))} {_prettify(val_metrics.get(metric))}'
            for metric in print_metrics
        )

        duration = duration_to_str(timer._elapsed())

        logger.info(f'Finished epoch {epoch}/{max_epochs}, {metrics_str}, {duration}')

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics)

    return


def attach_early_stopping(trainer,
                          validator,
                          metric='loss',
                          patience=10,
                          **kwargs,
                          ):
    """Attaches an early stopping handler to a trainer.

    NOTEs:
        - The handler should be attached after every other handler, so those will get executed completely
        - The handler is attached to the trainer, not the validator (as in most examples), so the stop signal is sent at the very end of the epoch (i.e. after every handler is run), and not after the `validator.run(...)` is run.
    """
    # Set early-stopping to info level
    es_logger = logging.getLogger('ignite.handlers.early_stopping.EarlyStopping')
    es_logger.setLevel(logging.INFO)

    def score_fn(unused_engine):
        value = validator.state.metrics[metric]
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
    """Attaches a callback that updates the lr_scheduler."""
    def update_scheduler(unused):
        val_metrics = validator.state.metrics
        lr_scheduler.step(val_metrics[target_metric])

    trainer.add_event_handler(Events.EPOCH_COMPLETED, update_scheduler)
