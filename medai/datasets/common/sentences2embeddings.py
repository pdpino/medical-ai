import os
import logging
import pandas as pd
import torch

from medai.utils.nlp import remove_garbage_tokens, ReportReader

LOGGER = logging.getLogger(__name__)

class _SentenceToEmbeddings:
    def __init__(self, dataset):
        report_reader = ReportReader(dataset.get_vocab())

        self._sentence_to_embedding = dict()

        _name = 'radglove-average'
        _df_fpath = os.path.join(
            dataset.reports_dir,
            'sentences-embeddings',
            f'{_name}.csv',
        )
        if not os.path.isfile(_df_fpath):
            raise FileNotFoundError(f'Sentence-embeddings not available for dataset! {_df_fpath}')
        emb_df = pd.read_csv(_df_fpath)
        embedding_cols = sorted([c for c in emb_df.columns if c.startswith('emb')])

        self.embedding_size = 100

        LOGGER.info('Preparing sentence2embedding dict')

        for _, sample in emb_df.iterrows():
            sentence_str = sample['sentence']
            vector = torch.tensor(sample[embedding_cols]) # pylint: disable=not-callable

            sentence_idxs = report_reader.text_to_idx(sentence_str)
            sentence_hash = self._sentence_to_hash(sentence_idxs)

            self._sentence_to_embedding[sentence_hash] = vector

    def _sentence_to_hash(self, sentence):
        sentence = remove_garbage_tokens(sentence)
        return ','.join(str(word_idx) for word_idx in sentence)

    def __getitem__(self, sentence):
        sentence_hash = self._sentence_to_hash(sentence)

        return self._sentence_to_embedding[sentence_hash]


_instance = None

def SentenceToEmbeddings(*args, **kwargs):
    # pylint: disable=global-statement
    global _instance

    if _instance is None:
        _instance = _SentenceToEmbeddings(*args, **kwargs)

    return _instance
