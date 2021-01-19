import os
from functools import partial
import operator
import logging
import numpy as np
import torch
from torch.nn.functional import interpolate
from ignite.engine import Events
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.metrics.report_generation.word_accuracy import WordAccuracy
from medai.metrics.report_generation.bleu import Bleu
from medai.metrics.report_generation.rouge import RougeL
from medai.metrics.report_generation.cider import CiderD
from medai.metrics.report_generation.distinct_sentences import DistinctSentences
from medai.metrics.report_generation.distinct_words import DistinctWords
from medai.metrics.report_generation.labeler_correctness import MedicalLabelerCorrectness
from medai.metrics.report_generation.labeler_correctness.light_labeler import ChexpertLightLabeler
from medai.metrics.report_generation.labeler_correctness.labeler_timer import LabelerTimerMetric
from medai.utils import WORKSPACE_DIR
from medai.utils.csv import CSVWriter
from medai.utils.files import get_results_folder
from medai.utils.nlp import ReportReader, trim_rubbish


def _get_flat_reports(outputs):
    """Transforms the output to arrays of words indexes.

    Args:
        outputs: dict with at least:
            ['flat_reports']: shape: batch_size, n_words
            ['flat_reports_gen']: shape: batch_size, n_words
    """
    flat_reports = outputs['flat_reports']
    flat_reports_gen = outputs['flat_reports_gen']

    return flat_reports_gen, flat_reports


def _attach_bleu(engine, up_to_n=4,
                 output_transform=_get_flat_reports):
    bleu_up_to_n = Bleu(n=up_to_n, output_transform=output_transform)
    for i in range(up_to_n):
        bleu_n = MetricsLambda(operator.itemgetter(i), bleu_up_to_n)
        bleu_n.attach(engine, f'bleu{i+1}')

    bleu_avg = MetricsLambda(np.mean, bleu_up_to_n)
    bleu_avg.attach(engine, 'bleu')


def attach_attention_vs_masks(engine):
    """Attaches metrics that evaluate attention scores vs gt-masks."""
    def _get_masks_and_attention(outputs):
        """Extracts generated and GT masks.

        Args:
            outputs: dict with tensors:
                ['gen_masks']: shape batch_size, n_sentences, features-height, features-width
                ['gt_masks']: shape batch_size, n_sentences, original-height, original-width
                ['gt_stops']: shape batch_size, n_sentences (optional)

            Notice features-* sizes will probably be smaller than original-* sizes,
            as the former are extracted from the last layer of a CNN,
            and the latter are the original GT masks.

        Returns:
            tuple with two (optional three) tensors
        """
        gen_masks = outputs['gen_masks']
        gt_masks = outputs['gt_masks']

        # Reduce gt_masks to gen_masks size
        features_dimensions = gen_masks.size()[-2:]
        gt_masks = interpolate(gt_masks.float(), features_dimensions, mode='nearest').long()

        # Include stops if present
        if 'gt_stops' in outputs:
            gt_stops = outputs['gt_stops']

            # Transform stops into valid
            # stop == 1 indicates stopping --> sentence not valid --> valid == 0
            # stop == 0 indicates dont-stop --> sentence valid --> valid == 1
            gt_valid = 1 - gt_stops
        else:
            gt_valid = None

        return gen_masks, gt_masks, gt_valid


    iou = IoU(reduce_sum=True, output_transform=_get_masks_and_attention)
    iou.attach(engine, 'att_iou')

    iobb = IoBB(reduce_sum=True, output_transform=_get_masks_and_attention)
    iobb.attach(engine, 'att_iobb')


def attach_metrics_report_generation(engine, hierarchical=False, free=False,
                                     supervise_attention=False):
    losses = ['loss']
    if hierarchical:
        losses.extend(['word_loss', 'stop_loss'])
    if supervise_attention:
        losses.append('att_loss')

    # Attach losses
    for loss_name in losses:
        loss = RunningAverage(output_transform=operator.itemgetter(loss_name))
        loss.attach(engine, loss_name)

    # Attach word accuracy
    if not free:
        word_acc = WordAccuracy(output_transform=_get_flat_reports)
        word_acc.attach(engine, 'word_acc')

    # Attach multiple bleu
    _attach_bleu(engine, 4)

    rouge = RougeL(output_transform=_get_flat_reports)
    rouge.attach(engine, 'rougeL')

    cider = CiderD(output_transform=_get_flat_reports)
    cider.attach(engine, 'ciderD')

    # Attach variability
    distinct_words = DistinctWords(output_transform=_get_flat_reports)
    distinct_words.attach(engine, 'distinct_words')

    distinct_sentences = DistinctSentences(output_transform=_get_flat_reports)
    distinct_sentences.attach(engine, 'distinct_sentences')


def _attach_medical_labeler_correctness(engine, labeler, basename, timer=True):
    """Attaches MedicalLabelerCorrectness metrics to an engine.

    It will attach metrics in the form <basename>_<metric_name>_<disease>

    Args:
        engine -- ignite engine to attach metrics to
        labeler -- labeler instance to pass to the MedicalLabelerCorrectness metric
        basename -- to use when attaching metrics
    """
    if timer:
        timer_metric = LabelerTimerMetric(labeler)
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
    for engine in (trainer, validator):
        if engine is None:
            continue
        _attach_medical_labeler_correctness(engine, ChexpertLightLabeler(vocab), 'chex')

    ## TODO: awake metrics only after N epochs,
    ## to avoid calculating for non-sense random reports
    # @trainer.on(Events.EPOCH_STARTED(once=5))
    # def _awake_after_epochs():
    #     print('Awaking metrics...')
    #     chexpert.has_started = True

    # TODO: apply for MIRQI as well
    # _attach_medical_labeler_correctness(engine, MirqiLightLabeler(vocab), 'mirqi')
