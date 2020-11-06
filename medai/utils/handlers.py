import time
import logging
from ignite.engine import Engine, Events

from medai.utils import duration_to_str

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

        metrics_str = ''
        for metric in print_metrics:
            train_value = train_metrics.get(metric, -1)
            val_value = val_metrics.get(metric, -1)
            metrics_str += f' {metric} {train_value:.3f} {val_value:.3f},'

        logger.info(f'Finished epoch {epoch}/{max_epochs}, {metrics_str} {duration_to_str(timer._elapsed())}')

    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_metrics)

    return