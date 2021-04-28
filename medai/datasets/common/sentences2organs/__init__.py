import os
import logging
import pandas as pd
import torch

from medai.datasets.common.constants import JSRT_ORGANS
from medai.utils.nlp import ReportReader, trim_rubbish

LOGGER = logging.getLogger(__name__)

class _SentenceToOrgans:
    def __init__(self, dataset):
        report_reader = ReportReader(dataset.get_vocab())

        self._sentence_to_organ_mapping = dict()

        _df_fpath = os.path.join(dataset.dataset_dir, 'reports', 'sentences_with_organs.csv')
        organs_df = pd.read_csv(_df_fpath)

        # Save these to notify sentences that are not found
        # self.report_reader = report_reader
        self.n_organs = len(JSRT_ORGANS)

        LOGGER.info('Preparing sentence2organ dict')

        for _, sample in organs_df.iterrows():
            sentence_str = sample['sentence']
            organs = sample[JSRT_ORGANS].tolist()

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
            # If does not appear in mapping, assumed to point to all organs
            return [1] * self.n_organs
            # self._sentence_to_organ_mapping[sentence_hash] = [1] * self.n_organs
            # sentence_str = self.report_reader.idx_to_text(sentence)
            # LOGGER.warning(
            #     'Organs not found for sentence: %s, (%s)',
            #     sentence_str, sentence_hash,
            # )

        return self._sentence_to_organ_mapping[sentence_hash]


    def get_mask_for_sentence(self, sentence, image_masks):
        """Returns the presence-mask for a given sentence.

        Args:
            sentence -- tensor of word indices, shape n_words
            image_masks -- tensor of shape n_organs, height, width
        Returns:
            mask -- tensor of shape height, width (binary)
        """
        organs = self.get_organs(sentence)
        # shape: n_organs (one-hot encoded)

        # pylint: disable=not-callable
        organ_indeces = torch.tensor([
            organ_idx
            for organ_idx, organ_presence in enumerate(organs)
            if organ_presence
        ])
        # shape: n_selected_organs

        sentence_mask = image_masks.index_select(dim=0, index=organ_indeces)
        # shape: n_selected_organs, height, width

        sentence_mask = sentence_mask.sum(dim=0) # NOTE: assumes organs do not overlap
        # shape: height, width

        return sentence_mask


_instance = None

def SentenceToOrgans(*args, **kwargs):
    # pylint: disable=global-statement
    global _instance

    if _instance is None:
        _instance = _SentenceToOrgans(*args, **kwargs)

    return _instance
