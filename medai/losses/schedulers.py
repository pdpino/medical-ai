import abc
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from ignite.engine import Events

LOGGER = logging.getLogger(__name__)

class LRSchedulerHandler(abc.ABC):
    """Provides an uniform interface for all torch.optim.lr_schedulers."""
    def attach(self, trainer, validator):
        """Attaches a callback that updates the lr_scheduler."""
        # pylint: disable=attribute-defined-outside-init
        self._trainer = trainer
        self._validator = validator

        trainer.add_event_handler(Events.EPOCH_COMPLETED, self.step)

    def step(self, unused_engine):
        # pylint: disable=no-member
        self.scheduler.step()


class PlateauHandler(LRSchedulerHandler):
    """Handles a ReduceLROnPlateau

    Note: if the metric value is -1, the scheduler will not be called.
        For metrics that may not be calculated on every epoch, such as chex_f1,
        this implies the scheduler patience will be accounted only in the epochs
        where the value is indeed calculated; other epochs will be skipped.
        For metrics that are calculated on every epoch, this has no effect.
    """
    def __init__(self, optimizer, metric='loss', **kwargs):
        self.scheduler = ReduceLROnPlateau(optimizer, **kwargs)

        self.metric = metric

    def step(self, unused_engine):
        should_ignore_warning = self.metric.startswith('chex_')

        val_metrics = self._validator.state.metrics
        if self.metric not in val_metrics:
            if not should_ignore_warning:
                LOGGER.warning(
                    'Cannot step LR-scheduler, %s not found in val_metrics', self.metric,
                )
            return
        value = val_metrics[self.metric]

        # NOTE: ignores -1 values
        if value != -1:
            self.scheduler.step(value)


class StepLRHandler(LRSchedulerHandler):
    def __init__(self, optimizer, **kwargs):
        self.scheduler = StepLR(optimizer, **kwargs)

    def step(self, unused_engine):
        # HACK: StepLR with verbose=True prints something every epoch,
        # instead of only when the
        pre_lr = self.scheduler.optimizer.param_groups[-1]['lr']
        self.scheduler.step()
        post_lr = self.scheduler.optimizer.param_groups[-1]['lr']

        if pre_lr != post_lr:
            LOGGER.info('Reducing LR from %f to %f', pre_lr, post_lr)


class NoSchedulerHandler(LRSchedulerHandler):
    def __init__(self, unused_optimizer, **unused_kwargs):
        self.scheduler = None

    def attach(self, unused_trainer, unused_validator):
        pass

    def step(self, unused_engine):
        pass


_SCHEDULERS = {
    'plateau': PlateauHandler,
    'step': StepLRHandler,
    '': NoSchedulerHandler, # To be able to use `--lr-scheduler ""` from the command line
    None: NoSchedulerHandler, # For backward compatibility
}

AVAILABLE_SCHEDULERS = list(_SCHEDULERS)


def create_lr_sch_handler(optimizer, quiet=False, name=None, **kwargs):
    if name not in _SCHEDULERS:
        raise Exception(f'Scheduler not found: {name}')

    SchedulerClass = _SCHEDULERS[name]
    lr_scheduler = SchedulerClass(optimizer, **kwargs)

    if not quiet:
        if isinstance(lr_scheduler, NoSchedulerHandler):
            LOGGER.warning('Not using a LR-scheduler')
        else:
            info = {
                'name': name,
                **kwargs,
            }
            info_str = ' '.join(f'{k}={v}' for k, v in info.items())
            LOGGER.info('Using LR-scheduler: %s', info_str)

    return lr_scheduler
