import operator
from functools import partial
import logging
import numpy as np
from ignite.engine import Events
from ignite.metrics import MetricsLambda

from medai.metrics.report_generation.labeler.metric import MedicalLabelerCorrectness
from medai.metrics.report_generation.labeler.light_labeler import (
    ChexpertLightLabeler,
    DummyLabeler,
)
from medai.metrics.report_generation.labeler.labeler_timer import LabelerTimerMetric
from medai.metrics.report_generation.labeler.cache import LABELER_CACHE_DIR
from medai.metrics.report_generation.labeler.hit_counter_metric import HitCounterMetric
from medai.metrics.report_generation.abn_match.schemas.chexpert import (
    ChexpertLighterLabeler,
)
from medai.metrics.report_generation.transforms import get_flat_reports
from medai.utils.lock import SyncLock
from medai.utils.metrics import EveryNAfterMEpochs

LOGGER = logging.getLogger(__name__)


_LOCK_FOLDER = LABELER_CACHE_DIR
_LOCK_NAME = 'medical-correctness-cache'

_LABELER_CLASSES = {
    'dummy': DummyLabeler,
    'lighter-chexpert': ChexpertLighterLabeler,
    'light-chexpert': ChexpertLightLabeler,
}

AVAILABLE_MED_LABELERS = list(_LABELER_CLASSES)


def _attach_hit_counter(engine, labeler, basename, device='cuda'):
    hit_counter_metric = HitCounterMetric(labeler, device=device)

    for i, subvalue in enumerate(['misses', 'misses_unique', 'misses_perc']):
        metric = MetricsLambda(operator.itemgetter(i), hit_counter_metric)
        metric.attach(engine, f'{basename}_labeler_{subvalue}')


def _attach_labeler(engine, labeler, basename, usage=None, device='cuda'):
    """Attaches MedicalLabelerCorrectness metrics to an engine.

    It will attach metrics in the form <basename>_<metric_name>_<disease>

    Args:
        engine -- ignite engine to attach metrics to
        labeler -- labeler instance to pass to the MedicalLabelerCorrectness metric
        basename -- to use when attaching metrics
        device -- passed to Metrics
    """
    if labeler.use_timer:
        timer_metric = LabelerTimerMetric(labeler=labeler, device=device)
        timer_metric.attach(engine, f'{basename}_timer')

    if labeler.use_cache:
        _attach_hit_counter(engine, labeler, basename, device=device)

    metric_obj = MedicalLabelerCorrectness(
        labeler, output_transform=get_flat_reports, device=device,
    )

    def _disease_metric_getter(result, metric_name, metric_index):
        """Given the MedicalLabelerCorrectness output returns a disease metric value.

        The metric obj returns a dict(key: metric_name, value: tensor/array of size n_diseases)
        e.g.: {
          'acc': tensor/ndarray of 14 diseases,
          'prec': tensor/ndarray of 14 diseases,
          etc
        }
        """
        return result[metric_name][metric_index].item()

    def _macro_avg_getter(result, metric_name):
        return result[metric_name].mean().item()

    if labeler.no_finding_idx is not None:
        if not labeler.use_numpy:
            # To use this, the tensors should be moved to CPU to use np.ma.array
            # REVIEW: can something like array[mask == 0].mean().item() be used?
            raise Exception('Internal error: cannot attach woNF metrics if use_numpy is False')

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


def attach_medical_correctness(trainer, validator, vocab,
                               after=None, steps=None, val_after=None, val_steps=None,
                               metric='lighter-chexpert',
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
    val_after = val_after if val_after is not None else after
    val_steps = val_steps if val_steps is not None else steps
    info = {
        'metric': metric,
        'train_after': after,
        'train_steps': steps,
        'val_after': val_after,
        'val_steps': val_steps,
    }
    info_str = ' '.join(f"{k}={v}" for k, v in info.items())
    LOGGER.info('Using medical correctness: %s', info_str)

    if metric not in _LABELER_CLASSES:
        raise Exception(f'Metric not found {metric}')
    LabelerClass = _LABELER_CLASSES[metric]

    if metric == 'dummy':
        LOGGER.warning('Attaching DUMMY med metrics!!')

    needs_lock = metric.startswith('light-')

    if needs_lock:
        lock = SyncLock(_LOCK_FOLDER, _LOCK_NAME, verbose=True)

        if not lock.acquire():
            LOGGER.warning(
                'Cannot attach medical correctness metric, cache is locked',
            )
            return

    train_usage = EveryNAfterMEpochs(steps, after, trainer) if after or steps else None
    val_usage = EveryNAfterMEpochs(val_steps, val_after, trainer) \
        if val_after or val_steps else None

    for engine in (trainer, validator):
        if engine is None:
            continue

        if engine is trainer:
            usage = train_usage
        else:
            usage = val_usage

        if isinstance(usage, EveryNAfterMEpochs) and usage.m_after == -1:
            continue

        labeler = LabelerClass(vocab, device=device)
        _attach_labeler(engine, labeler, labeler.metric_name, device=device, usage=usage)

    if needs_lock:
        def _release_locks(engine, err=None):
            lock.release()

            if err is not None:
                LOGGER.error('Error in trainer=%s', engine is trainer)
                raise err

        trainer.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)
        trainer.add_event_handler(Events.COMPLETED, _release_locks, err=None)

        if validator is not None:
            validator.add_event_handler(Events.EXCEPTION_RAISED, _release_locks)
