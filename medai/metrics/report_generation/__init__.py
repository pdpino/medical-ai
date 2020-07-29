# import torch
import operator
import numpy as np
from ignite.metrics import RunningAverage, MetricsLambda

from medai.metrics.report_generation.word_accuracy import WordAccuracy
from medai.metrics.report_generation.bleu import Bleu
from medai.metrics.report_generation.rouge import RougeL
from medai.metrics.report_generation.cider import CiderD


def _transform_score_to_indexes(outputs):
    """Transforms the output to arrays of words indexes.
    
    Args:
        outputs: tuple
            [1]: generated_scores -- shape: batch_size, *, vocab_size
            [2]: reports -- shape: batch_size, *
    """
    generated_scores = outputs[1]
    seq = outputs[2]

    _, words_predicted = generated_scores.max(dim=-1)
    # words_predicted shape: batch_size, *

    return words_predicted, seq


def _transform_get_flatten(outputs):
    # FIXME: receives flattened reports, which should be fixed!
    flattened_reports = outputs[3]
    seq = outputs[4]

    return flattened_reports, seq


def _attach_bleu(engine, up_to_n=4, output_transform=_transform_score_to_indexes):
    bleu_up_to_n = Bleu(n=up_to_n, output_transform=output_transform)
    for i in range(up_to_n):
        bleu_n = MetricsLambda(operator.itemgetter(i), bleu_up_to_n)
        bleu_n.attach(engine, f'bleu{i+1}')

    bleu_avg = MetricsLambda(lambda x: np.mean(x), bleu_up_to_n)
    bleu_avg.attach(engine, 'bleu')


def attach_metrics_report_generation(engine, hierarchical=False):
    loss = RunningAverage(output_transform=lambda x: x[0])
    loss.attach(engine, 'loss')

    word_acc = WordAccuracy(output_transform=_transform_score_to_indexes)
    word_acc.attach(engine, 'word_acc')

    if hierarchical:
        output_transform = _transform_get_flatten
    else:
        output_transform = _transform_score_to_indexes

    # Attach multiple bleu
    _attach_bleu(engine, 4, output_transform=output_transform)

    rouge = RougeL(output_transform=output_transform)
    rouge.attach(engine, 'rougeL')

    cider = CiderD(output_transform=output_transform)
    cider.attach(engine, 'ciderD')
