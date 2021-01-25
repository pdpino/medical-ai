from functools import partial
import logging
import numpy as np
from ignite.engine import Events
from ignite.metrics import MetricsLambda

from medai.metrics.report_generation.labeler_correctness.metric import MedicalLabelerCorrectness
from medai.metrics.report_generation.labeler_correctness.light_labeler import ChexpertLightLabeler
from medai.metrics.report_generation.labeler_correctness.labeler_timer import LabelerTimerMetric
from medai.metrics.report_generation.labeler_correctness.cache import LABELER_CACHE_DIR
from medai.metrics.report_generation.transforms import _get_flat_reports
from medai.utils.lock import SyncLock

LOGGER = logging.getLogger(__name__)


_LOCK_FOLDER = LABELER_CACHE_DIR
_LOCK_NAME = 'medical-correctness-cache'


def _attach_medical_labeler_correctness(engine, labeler, basename, timer=True):
    """Attaches MedicalLabelerCorrectness metrics to an engine.

    It will attach metrics in the form <basename>_<metric_name>_<disease>

    Args:
        engine -- ignite engine to attach metrics to
        labeler -- labeler instance to pass to the MedicalLabelerCorrectness metric
        basename -- to use when attaching metrics
    """
    if timer:
        timer_metric = LabelerTimerMetric(labeler=labeler)
        timer_metric.attach(engine, f'{basename}_timer')

    metric_obj = MedicalLabelerCorrectness(labeler, output_transform=_get_flat_reports)

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
        macro_avg.attach(engine, f'{basename}_{metric_name}')

        # Attach for each disease
        for index, disease in enumerate(labeler.diseases):
            disease_metric = MetricsLambda(
                partial(_disease_metric_getter, metric_name=metric_name, metric_index=index),
                metric_obj,
            )
            disease_metric.attach(engine, f'{basename}_{metric_name}_{disease}')

    return metric_obj


def attach_medical_correctness(trainer, validator, vocab):
    """Attaches medical correctness metrics to engines.

    It uses a SyncLock to assure not to engines use the inner Cache at the same time.

    Args:
        trainer -- ignite.Engine
        validator -- ignite.Engine or None
        vocab -- dataset vocabulary (dict)
    """
    lock = SyncLock(_LOCK_FOLDER, _LOCK_NAME, verbose=True)

    if not lock.acquire():
        LOGGER.warning(
            'Cannot attach medical correctness metric, cache is locked',
        )
        return

    for engine in (trainer, validator):
        if engine is None:
            continue

        labeler = ChexpertLightLabeler(vocab)
        _attach_medical_labeler_correctness(engine, labeler, 'chex')


    def _release_locks(unused_engine, err=None):
        lock.release()

        if err is not None:
            raise err

    trainer.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)
    trainer.add_event_handler(Events.COMPLETED, _release_locks, err=None)

    if validator is not None:
        validator.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)


    ## TODO: awake metrics only after N epochs,
    ## to avoid calculating for non-sense random reports
    # @trainer.on(Events.EPOCH_STARTED(once=5))
    # def _awake_after_epochs():
    #     LOGGER.info('Awaking metrics...')
    #     chexpert.has_started = True

    # TODO: apply for MIRQI as well
    # _attach_medical_labeler_correctness(engine, MirqiLightLabeler(vocab), 'mirqi')
