# import torch
import os
import operator
import numpy as np
from ignite.engine import Events
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics import get_results_folder
from medai.metrics.report_generation.word_accuracy import WordAccuracy
from medai.metrics.report_generation.bleu import Bleu
from medai.metrics.report_generation.rouge import RougeL
from medai.metrics.report_generation.cider import CiderD
from medai.metrics.report_generation.distinct_sentences import DistinctSentences
from medai.metrics.report_generation.distinct_words import DistinctWords
from medai.utils import WORKSPACE_DIR
from medai.utils.csv import CSVWriter
from medai.utils.nlp import ReportReader


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


def attach_metrics_report_generation(engine, hierarchical=False, free=False):
    losses = ['loss']
    if hierarchical:
        losses.extend(['word_loss', 'stop_loss'])

    # Attach losses
    for loss_name in losses:
        loss = RunningAverage(output_transform=lambda x: x[loss_name])
        loss.attach(engine, loss_name)

    # Attach word accuracy
    if not free:
        word_acc = WordAccuracy(output_transform=_get_flat_reports)
        word_acc.attach(engine, 'word_acc')

    # Attach multiple bleu
    _attach_bleu(engine, 4) # , output_transform=output_transform)

    rouge = RougeL(output_transform=_get_flat_reports)
    rouge.attach(engine, 'rougeL')

    cider = CiderD(output_transform=_get_flat_reports)
    cider.attach(engine, 'ciderD')

    # Attach variability
    distinct_words = DistinctWords(output_transform=_get_flat_reports)
    distinct_words.attach(engine, 'distinct_words')

    distinct_sentences = DistinctSentences(output_transform=_get_flat_reports)
    distinct_sentences.attach(engine, 'distinct_sentences')


def attach_report_writer(engine, vocab, run_name, debug=True):
    """Attach a report-writer to an engine.
    
    For each example in the dataset writes to a CSV the generated report and ground truth.
    """
    report_reader = ReportReader(vocab)

    folder = get_results_folder(run_name,
                                classification=False,
                                debug=debug,
                                save_mode=True)
    path = os.path.join(folder, 'outputs.csv')
    
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
        filenames = engine.state.batch.filenames
        batch_report = output['flat_reports']
        batch_generated = output['flat_reports_gen']
        
        epoch = engine.state.epoch
        dataset_type = engine.state.dataloader.dataset.dataset_type

        # Save result
        for report, generated, filename in zip(batch_report, batch_generated, filenames):
            original_report = f'"{report_reader.idx_to_text(report)}"'
            generated_report = f'"{report_reader.idx_to_text(generated)}"'
            writer.write(
                filename,
                epoch,
                dataset_type,
                original_report,
                generated_report,
            )
            
    @engine.on(Events.COMPLETED)
    def _close_writer():
        writer.close()