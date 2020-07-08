# import torch
import operator
import numpy as np
from ignite.metrics import RunningAverage, MetricsLambda

from mrg.metrics.report_generation.word_accuracy import WordAccuracy
from mrg.metrics.report_generation.bleu import Bleu
from mrg.metrics.report_generation.rouge import RougeL
from mrg.metrics.report_generation.cider import CiderD


def _transform_score_to_indexes(outputs):
    """Transforms the output to arrays of words indexes.
    
    Args:
        outputs = loss_unused, generated_scores, reports
        generated_scores -- shape: batch_size, max_sentence_len, vocab_size
        reports -- shape: batch_size, max_sentence_len
    """
    _, generated_scores, seq = outputs
    _, words_predicted = generated_scores.max(dim=2)
    # words_predicted shape: batch_size, max_sentence_len

    return words_predicted, seq


def _attach_bleu(engine, up_to_n=4):
    bleu_up_to_n = Bleu(n=up_to_n, output_transform=_transform_score_to_indexes)
    for i in range(up_to_n):
        bleu_n = MetricsLambda(operator.itemgetter(i), bleu_up_to_n)
        bleu_n.attach(engine, f'bleu{i+1}')

    bleu_avg = MetricsLambda(lambda x: np.mean(x), bleu_up_to_n)
    bleu_avg.attach(engine, 'bleu')


def attach_metrics_report_generation(engine):
    loss = RunningAverage(output_transform=lambda x: x[0])
    loss.attach(engine, 'loss')
    
    word_acc = WordAccuracy(output_transform=_transform_score_to_indexes)
    word_acc.attach(engine, 'word_acc')

    # Attach multiple bleu
    _attach_bleu(engine, 4)

    rouge = RougeL(output_transform=_transform_score_to_indexes)
    rouge.attach(engine, 'rougeL')

    cider = CiderD(output_transform=_transform_score_to_indexes)
    cider.attach(engine, 'ciderD')
