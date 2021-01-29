from functools import partial
import logging
import numpy as np
from ignite.engine import Events
from ignite.metrics import MetricsLambda

from medai.metrics.report_generation.labeler_correctness.metric import MedicalLabelerCorrectness
from medai.metrics.report_generation.labeler_correctness.light_labeler import ChexpertLightLabeler
from medai.metrics.report_generation.labeler_correctness.labeler_timer import LabelerTimerMetric
from medai.metrics.report_generation.labeler_correctness.cache import LABELER_CACHE_DIR
from medai.metrics.report_generation.transforms import get_flat_reports
from medai.utils.lock import SyncLock
from medai.utils.metrics_usage import EveryNEpochs

LOGGER = logging.getLogger(__name__)


_LOCK_FOLDER = LABELER_CACHE_DIR
_LOCK_NAME = 'medical-correctness-cache'


def _attach_labeler(engine, labeler, basename, run_every_n_steps=None, timer=True):
    """Attaches MedicalLabelerCorrectness metrics to an engine.

    It will attach metrics in the form <basename>_<metric_name>_<disease>

    Args:
        engine -- ignite engine to attach metrics to
        labeler -- labeler instance to pass to the MedicalLabelerCorrectness metric
        basename -- to use when attaching metrics
    """
    if run_every_n_steps:
        kwargs = {
            'usage': EveryNEpochs(run_every_n_steps),
        }
    else:
        kwargs = {}


    if timer:
        timer_metric = LabelerTimerMetric(labeler=labeler)
        timer_metric.attach(engine, f'{basename}_timer', **kwargs)

    metric_obj = MedicalLabelerCorrectness(labeler, output_transform=get_flat_reports)

    def _disease_metric_getter(result, metric_name, metric_index):
        """Given the MedicalLabelerCorrectness output returns a disease metric value.

        The metric obj returns a dict(key: metric_name, value: tensor/array of size n_diseases)
        e.g.: {
          'acc': tensor of 14 diseases,
          'prec': tensor of 14 diseases,
          etc
        }
        """
        return result[metric_name][metric_index].item()

    def _macro_avg_getter(result, metric_name):
        return np.mean(result[metric_name])

    for metric_name in metric_obj.METRICS:
        # Attach diseases' macro average
        macro_avg = MetricsLambda(
            partial(_macro_avg_getter, metric_name=metric_name),
            metric_obj,
        )
        macro_avg.attach(engine, f'{basename}_{metric_name}', **kwargs)

        # Attach for each disease
        for index, disease in enumerate(labeler.diseases):
            disease_metric = MetricsLambda(
                partial(_disease_metric_getter, metric_name=metric_name, metric_index=index),
                metric_obj,
            )
            disease_metric.attach(engine, f'{basename}_{metric_name}_{disease}', **kwargs)

    return metric_obj


def attach_medical_correctness(trainer, validator, vocab, after=None, steps=None):
    """Attaches medical correctness metrics to engines.

    It uses a SyncLock to assure not to engines use the inner Cache at the same time.

    Notes:
        - allows attaching the metrics after n epochs, by delaying the attaching to
            a later epoch. This is a HACKy way!
            If after == 3, in the epoch=4 the metric will be started
            for validation, and in epoch=5 the metric will be started for training
            (the metric is attached in 4, so is too late to be run in epoch=4).
        - allows running the metrics every_n epochs in the training set
            (though is not working right now)

    Args:
        trainer -- ignite.Engine
        validator -- ignite.Engine or None
        vocab -- dataset vocabulary (dict)
    """
    if steps:
        # FIXME
        LOGGER.warning('Setting med-steps is not working, ignoring')
        steps = None

    def _actually_attach():
        LOGGER.info('Attaching medical correctness metrics')

        lock = SyncLock(_LOCK_FOLDER, _LOCK_NAME, verbose=True)

        if not lock.acquire():
            LOGGER.warning(
                'Cannot attach medical correctness metric, cache is locked',
            )
            return

        for engine in (trainer, validator):
            if engine is None:
                continue

            if engine is trainer:
                run_every_n_steps = steps
            else:
                run_every_n_steps = None

            labeler = ChexpertLightLabeler(vocab)
            _attach_labeler(engine, labeler, 'chex', run_every_n_steps)

            # TODO: apply for MIRQI as well
            # _attach_labeler(engine, MirqiLightLabeler(vocab), 'mirqi')


        def _release_locks(engine, err=None):
            lock.release()

            if err is not None:
                LOGGER.error('Error in trainer=%s', engine is trainer)
                raise err

        trainer.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)
        trainer.add_event_handler(Events.COMPLETED, _release_locks, err=None)

        if validator is not None:
            validator.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)

    if not after or after <= trainer.state.epoch:
        _actually_attach()
    else:
        trainer.add_event_handler(
            Events.EPOCH_STARTED(once=after+1), _actually_attach, # pylint: disable=not-callable
        )
