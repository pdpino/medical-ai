# import torch
import os
from functools import partial
import operator
import numpy as np
import torch
from ignite.engine import Events
from ignite.metrics import RunningAverage, MetricsLambda

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

    bleu_avg = MetricsLambda(lambda x: np.mean(x), bleu_up_to_n)
    bleu_avg.attach(engine, 'bleu')


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


def attach_report_writer(engine, vocab, run_name, debug=True, free=False):
    """Attach a report-writer to an engine.

    For each example in the dataset writes to a CSV the generated report and ground truth.
    """
    report_reader = ReportReader(vocab)

    folder = get_results_folder(run_name,
                                task='rg',
                                debug=debug,
                                save_mode=True)
    suffix = 'free' if free else 'notfree'
    path = os.path.join(folder, f'outputs-{suffix}.csv')

    writer = CSVWriter(path, columns=[
        'filename',
        'epoch',
        'dataset_type',
        'ground_truth',
        'generated',
    ])

    @engine.on(Events.STARTED)
    def _open_writer():
        writer.open()

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_text(engine):
        output = engine.state.output
        filenames = engine.state.batch.report_fnames
        gt_reports = output['flat_reports']
        gen_reports = output['flat_reports_gen']

        epoch = engine.state.epoch
        dataset_type = engine.state.dataloader.dataset.dataset_type

        # Save result
        for report, generated, filename in zip(
            gt_reports,
            gen_reports,
            filenames,
        ):
            # Remove padding and END token
            report = trim_rubbish(report)
            generated = trim_rubbish(generated)

            # Pass to text
            report = report_reader.idx_to_text(report)
            generated = report_reader.idx_to_text(generated)

            # Add quotes to avoid issues with commas
            report = f'"{report}"'
            generated = f'"{generated}"'

            writer.write(
                filename,
                epoch,
                dataset_type,
                report,
                generated,
            )

    @engine.on(Events.COMPLETED)
    def _close_writer():
        writer.close()