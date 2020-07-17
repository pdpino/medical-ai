import torch

# Common tokens
PAD_TOKEN = 'PAD'
PAD_IDX = 0
END_TOKEN = 'END'
END_IDX = 1
START_TOKEN = 'START'
START_IDX = 2
UNKNOWN_TOKEN = 'UNK'
UNKNOWN_IDX = 3
END_OF_SENTENCE_TOKEN = '.'
END_OF_SENTENCE_IDX = 4


def compute_vocab(reports_iterator):
    """Computes a vocabulary, given a set of reports."""
    word_to_idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
        END_OF_SENTENCE_TOKEN: END_OF_SENTENCE_IDX,
    }

    for report in reports_iterator:
        for token in report:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)

    return word_to_idx


class ReportReader:
    """Translates idx to words for generated reports."""

    def __init__(self, vocab):
        self._idx_to_word = {v: k for k, v in vocab.items()}

    def idx_to_text(self, report):
        if isinstance(report, torch.Tensor):
            report = report.view(-1).tolist()

        return ' '.join([self._idx_to_word[int(g)] for g in report])