import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

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


def count_sentences(report):
    """Counts the amount of sentences in a report."""
    if isinstance(report, torch.Tensor):
        report = report.detach().tolist()

    n_sentences = report.count(END_OF_SENTENCE_IDX)

    if report[-1] != END_OF_SENTENCE_IDX:
        n_sentences += 1

    return n_sentences


def split_sentences_and_pad(report, end_of_sentence_idx=END_OF_SENTENCE_IDX):
    """Splits a report into sentences and pads them.

    Args:
        report -- list of shape (n_words)
        end_of_sentence_idx -- int indicating idx of the end-of-sentence token
    Returns:
        report (tensor) of shape (n_sentences, n_words)
    """
    if not isinstance(report, list):
        raise Exception(f'Report should be list, got: {type(report)}')

    # Last sentence must end with a dot
    if report[-1] != END_OF_SENTENCE_IDX:
        report = report + [END_OF_SENTENCE_IDX]

    report = torch.tensor(report)

    # Index positions of end-of-sentence tokens
    end_positions = (report == end_of_sentence_idx).nonzero(as_tuple=False).view(-1)

    # Transform it to count of items
    end_counts = end_positions + 1

    # Calculate sentence sizes, by subtracting index positions to the one before
    shifted_counts = torch.cat((torch.zeros(1).long(), end_counts), dim=0)[:-1]
    split_sizes = (end_counts - shifted_counts).tolist()

    # Split into sentences
    sentences = torch.split(report, split_sizes)

    return pad_sequence(sentences, batch_first=True)


class ReportReader:
    """Translates idx to words for generated reports."""

    def __init__(self, vocab):
        self._idx_to_word = {v: k for k, v in vocab.items()}
        self._vocab = dict(vocab)

    def idx_to_text(self, report):
        if isinstance(report, torch.Tensor):
            report = report.view(-1).tolist()

        if not isinstance(report, (list, np.ndarray)):
            return 'ERROR'

        return ' '.join([self._idx_to_word[int(g)] for g in report])

    def text_to_idx(self, report):
        assert isinstance(report, (list, str)), f'Report must be list or str, got: {type(report)}'

        if isinstance(report, str):
            report = report.split()

        return [
            self._vocab.get(word, UNKNOWN_IDX)
            for word in report
        ]


class SentenceToOrgans:
    def __init__(self, dataset):
        needed_attrs = ['organs_by_sentence_df', 'get_vocab', 'organs']
        for attr_name in needed_attrs:
            if not hasattr(dataset, attr_name):
                raise Exception(f'dataset must have a {attr_name} attribute')

        report_reader = ReportReader(dataset.get_vocab())

        self._sentence_to_organ_mapping = dict()

        organs_df = dataset.organs_by_sentence_df
        organs_names = dataset.organs

        # Save these to notify sentences that are not found
        self.report_reader = report_reader
        self.n_organs = len(organs_names)

        for _, sample in organs_df.iterrows():
            sentence_str = sample['sentences']
            organs = sample[organs_names].tolist()

            sentence_idxs = report_reader.text_to_idx(sentence_str)
            sentence_hash = self._sentence_to_hash(sentence_idxs)

            self._sentence_to_organ_mapping[sentence_hash] = organs

    def _sentence_to_hash(self, sentence):
        sentence = trim_rubbish(sentence)
        return ','.join(str(word_idx) for word_idx in sentence)

    def get_organs(self, sentence):
        """Returns a one-hot list of organs for a sentence.

        Args:
            sentence -- tensor of word indices, shape n_words
        """
        sentence = sentence.tolist()

        sentence_hash = self._sentence_to_hash(sentence)

        if sentence_hash not in self._sentence_to_organ_mapping:
            self._sentence_to_organ_mapping[sentence_hash] = [1] * self.n_organs
            sentence_str = self.report_reader.idx_to_text(sentence)
            print(f'Organs not found for sentence: {sentence_str}')

        return self._sentence_to_organ_mapping[sentence_hash]



def trim_rubbish(report):
    """Trims padding and END token of a report.

    Receives a report list/array/tensor of word indexes.
    Assumes pad_idx is 0, otherwise np.trim_zeros() function would be uglier
    """
    if isinstance(report, torch.Tensor):
        report = report.cpu().detach().numpy()

    if report is None or len(report) == 0:
        return np.array([])

    # Trim padding from the end of sentences
    report = np.trim_zeros(report, 'b')

    if len(report) > 0 and report[-1] == END_IDX:
        report = report[:-1]

    return report


def indexes_to_strings(candidate, ground_truth):
    """Receives two word-indexes tensors, and returns candidate and gt strings.

    Args:
        candidate -- torch.Tensor of shape n_words
        ground_truth -- torch.Tensor of shape n_words
    Returns:
        candidate_str, ground_truth_strs
        - candidate_str: string of concatenated indexes
        - ground_truth_strs: list of strings of concatenated indexes
    """
    candidate = trim_rubbish(candidate)
    ground_truth = trim_rubbish(ground_truth)

    # Join as string
    candidate = ' '.join(str(val) for val in candidate)
    ground_truth = ' '.join(str(val) for val in ground_truth)

    return candidate, ground_truth