import os
import logging
import pandas as pd
import numpy as np

from medai.utils import CACHE_DIR
from medai.utils.csv import CSVWriter

LABELER_CACHE_DIR = os.path.join(CACHE_DIR, 'labeler')

LOGGER = logging.getLogger(__name__)

class _ReportLabelsCache:
    """Class to handle report to labels relation.

    Note: the information is duplicated in memory (a dict) and disk (a csv file).
        When writing new information, both are updated.
        This is error-prone, but should be more efficient than writing to disk every
        time new entries are added. Also, this implies the need of a lock, i.e.
        only one process can use this cache at the same time.
        The disk-version is saved as csv to comply with the format of
        sentences_with_chexpert_labels.csv file, and the memory-version
        is a dict to optimize access times.
    """
    def __init__(self, name, labels):
        os.makedirs(LABELER_CACHE_DIR, exist_ok=True)

        self.fpath = os.path.join(LABELER_CACHE_DIR, f'{name}.csv')

        LOGGER.info('Using labeler-cache saved in %s', self.fpath)

        if not os.path.isfile(self.fpath):
            # Init empty state
            self.state = {}
        else:
            # Load previous state from csv
            df = pd.read_csv(self.fpath, index_col=0)

            assert list(df.columns) == labels, (
                f'Cache-labels do not match: {labels} vs {list(df.columns)}'
            )

            self.state = df.transpose().to_dict(orient='list')
            # key: sentence
            # value: list of labels

        # Create CSV writer to write on the fly
        columns = ['text'] + labels
        self.writer = CSVWriter(self.fpath, columns, assert_folder=False)

    def __getitem__(self, text):
        return np.array(self.state[text])

    def __contains__(self, text):
        return text in self.state

    def insert(self, texts, labels):
        """Insert into the cache a list of texts and its labels.

        Args:
            texts -- list of strs, of len batch_size
            labels -- list, tensor or np.array of shape batch_size, n_labels
        """
        with self.writer:
            for text, text_labels in zip(texts, labels):
                if text in self.state:
                    continue
                self.state[text] = text_labels
                self.writer.write(text, *text_labels, quote=True)

_instances = {}

def ReportLabelsCache(name, level, *args):
    """Enforce loading one instance per name and level of the labeler."""
    full_name = f'{level}_{name}'
    if full_name not in _instances:
        _instances[full_name] = _ReportLabelsCache(full_name, *args)
    return _instances[full_name]
