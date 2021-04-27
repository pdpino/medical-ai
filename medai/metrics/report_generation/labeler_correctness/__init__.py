import operator
from functools import partial
import logging
import numpy as np
from ignite.engine import Events
from ignite.metrics import MetricsLambda

from medai.metrics.report_generation.labeler_correctness.metric import MedicalLabelerCorrectness
from medai.metrics.report_generation.labeler_correctness.light_labeler import ChexpertLightLabeler
from medai.metrics.report_generation.labeler_correctness.labeler_timer import LabelerTimerMetric
from medai.metrics.report_generation.labeler_correctness.cache import LABELER_CACHE_DIR
from medai.metrics.report_generation.labeler_correctness.hit_counter_metric import HitCounterMetric
from medai.metrics.report_generation.transforms import get_flat_reports
from medai.utils.lock import SyncLock
from medai.utils.metrics import EveryNAfterMEpochs

LOGGER = logging.getLogger(__name__)


_LOCK_FOLDER = LABELER_CACHE_DIR
_LOCK_NAME = 'medical-correctness-cache'


def _attach_hit_counter(engine, labeler, basename, device='cuda'):
    hit_counter_metric = HitCounterMetric(labeler, device=device)

    for i, subvalue in enumerate(['misses', 'misses_unique', 'misses_perc']):
        metric = MetricsLambda(operator.itemgetter(i), hit_counter_metric)
        metric.attach(engine, f'{basename}_labeler_{subvalue}')


def _attach_labeler(engine, labeler, basename, usage=None,
                    timer=True, device='cuda'):
    """Attaches MedicalLabelerCorrectness metrics to an engine.

    It will attach metrics in the form <basename>_<metric_name>_<disease>

    Args:
        engine -- ignite engine to attach metrics to
        labeler -- labeler instance to pass to the MedicalLabelerCorrectness metric
        basename -- to use when attaching metrics
        device -- passed to Metrics
    """
    if timer:
        timer_metric = LabelerTimerMetric(labeler=labeler, device=device)
        timer_metric.attach(engine, f'{basename}_timer')

    _attach_hit_counter(engine, labeler, basename, device=device)

    metric_obj = MedicalLabelerCorrectness(
        labeler, output_transform=get_flat_reports, device=device,
    )

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

    ignore_no_finding_mask = np.zeros(len(labeler.diseases))
    if labeler.no_finding_idx is not None:
        ignore_no_finding_mask[labeler.no_finding_idx] = 1

    def _macro_avg_getter_fonly(result, metric_name):
        values = result[metric_name] # shape: n_findings
        return np.ma.array(values, mask=ignore_no_finding_mask).mean()


    if usage:
        kwargs = {
            'usage': usage,
        }
    else:
        kwargs = {}

    for metric_name in metric_obj.METRICS:
        # Attach diseases' macro average
        macro_avg = MetricsLambda(
            partial(_macro_avg_getter, metric_name=metric_name),
            metric_obj,
        )
        macro_avg.attach(engine, f'{basename}_{metric_name}', **kwargs)

        if labeler.no_finding_idx is not None:
            # Attach macro-avg removing "no-finding"
            macro_avg_fonly = MetricsLambda(
                partial(_macro_avg_getter_fonly, metric_name=metric_name),
                metric_obj,
            )
            macro_avg_fonly.attach(engine, f'{basename}_{metric_name}_woNF', **kwargs)

        # Attach for each disease
        for index, disease in enumerate(labeler.diseases):
            disease_metric = MetricsLambda(
                partial(_disease_metric_getter, metric_name=metric_name, metric_index=index),
                metric_obj,
            )
            disease_metric.attach(engine, f'{basename}_{metric_name}_{disease}', **kwargs)

    return metric_obj


def attach_medical_correctness(trainer, validator, vocab, after=None, steps=None, val_steps=None,
                               device='cuda'):
    """Attaches medical correctness metrics to engines.

    It uses a SyncLock to assure not two engines use the inner Cache at the same time.

    Notes:
        - allows calculating the metrics after m epochs and every n epochs

    Args:
        trainer -- ignite.Engine
        validator -- ignite.Engine or None
        vocab -- dataset vocabulary (dict)
        device -- passed to Metrics
    """
    info = {
        'after': after,
        'train_steps': steps,
        'val_steps': val_steps,
    }
    info_str = ' '.join(f"{k}={v}" for k, v in info.items())
    LOGGER.info('Using medical correctness metrics %s', info_str)

    lock = SyncLock(_LOCK_FOLDER, _LOCK_NAME, verbose=True)

    if not lock.acquire():
        LOGGER.warning(
            'Cannot attach medical correctness metric, cache is locked',
        )
        return

    if after or steps or val_steps:
        train_usage = EveryNAfterMEpochs(steps, after, trainer)
        val_usage = EveryNAfterMEpochs(val_steps or steps, after, trainer)
    else:
        train_usage = None
        val_usage = None

    for engine in (trainer, validator):
        if engine is None:
            continue

        if engine is trainer:
            usage = train_usage
        else:
            usage = val_usage

        labeler = ChexpertLightLabeler(vocab)
        _attach_labeler(engine, labeler, 'chex', device=device, usage=usage)

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
