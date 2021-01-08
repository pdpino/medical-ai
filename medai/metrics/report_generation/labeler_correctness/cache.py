import os
import pandas as pd
import numpy as np

from medai.utils import CACHE_DIR
from medai.utils.csv import CSVWriter
from medai.utils.nlp import PAD_IDX, END_OF_SENTENCE_IDX

LABELER_CACHE_DIR = folder = os.path.join(CACHE_DIR, 'labeled_sentences')

class _SentencesLabelCache:
    """Class to handle sentence to labels relation.

    Note: the information is duplicated in memory (a dict) and disk (a csv file).
        When writing new information, both are updated.
        This is error-prone, but more efficient than transforming dict-to-csv every time new entries are added.
        The disk-version is saved as csv to comply with the format of sentences_with_chexpert_labels.csv file, and the memory-version is a dict to optimize access times.
    """
    def __init__(self, name, labels):
        os.makedirs(LABELER_CACHE_DIR, exist_ok=True)

        self.fpath = os.path.join(LABELER_CACHE_DIR, f'sentences_{name}.csv')

        if not os.path.isfile(self.fpath):
            # Init empty state
            self.state = {}
        else:
            # Load previous state from csv
            df = pd.read_csv(self.fpath, index_col=0)

            assert list(df.columns) == labels, f'Cache-labels do not match: {labels} vs {list(df.columns)}'

            self.state = df.transpose().to_dict(orient='list')
            # key: sentence
            # value: list of labels

        # Create CSV writer to write on the fly
        columns = ['sentences'] + labels
        self.writer = CSVWriter(self.fpath, columns, assert_folder=False)

    def __getitem__(self, sentence):
        return np.array(self.state[sentence])

    def __contains__(self, sentence):
        return sentence in self.state

    def insert(self, sentences, labels):
        """Insert into the cache a list of sentences and its labels.

        Args:
            sentences -- list of strs, of len batch_size
            labels -- list, tensor or np.array of shape batch_size, n_labels
        """
        with self.writer:
            for sentence, sentence_labels in zip(sentences, labels):
                if sentence in self.state:
                    continue
                self.state[sentence] = sentence_labels
                self.writer.write(sentence, *sentence_labels, quote=True)

_instances = {}

def SentencesLabelCache(name, *args):
    """Enforce loading one instance per name of the labeler."""
    if name not in _instances:
        _instances[name] = _SentencesLabelCache(name, *args)
    return _instances[name]