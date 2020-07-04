from ignite.metrics import RunningAverage

from mrg.metrics.report_generation.word_accuracy import WordAccuracy


def _transform_words_indexes(outputs):
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


def attach_metrics_report_generation(engine):
    loss = RunningAverage(output_transform=lambda x: x[0])
    loss.attach(engine, 'loss')
    
    word_acc = WordAccuracy(output_transform=_transform_words_indexes)
    word_acc.attach(engine, 'word_acc')
