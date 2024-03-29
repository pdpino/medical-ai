import os
import logging
from functools import partial
import torch
from torch import nn
import torchtext

from medai.utils import WORKSPACE_DIR, CACHE_DIR
from medai.utils.nlp import PAD_IDX

_TEXT_CACHE_DIR = os.path.join(CACHE_DIR, 'torchtext')

LOGGER = logging.getLogger(__name__)

class RadGlove:
    def __init__(self, unk_init=None):
        self._load()
        self.dim = 100

        self.unk_init = torch.Tensor.zero_ if unk_init is None else unk_init

    def _load(self):
        rad_glove_filepath = os.path.join(WORKSPACE_DIR, 'utils', 'radglove.800M.100d.txt')
        self._vectors_by_token = dict()

        with open(rad_glove_filepath, 'r') as f:
            for line in f:
                elements = line.split()
                assert len(elements) == 101

                token = elements[0]
                vector = torch.Tensor(list(map(float, elements[1:])))
                assert vector.size() == (100,)
                self._vectors_by_token[token] = vector

    def __len__(self):
        return len(self._vectors_by_token)

    def __contains__(self, token):
        return token in self._vectors_by_token

    def __getitem__(self, token):
        if token in self._vectors_by_token:
            return self._vectors_by_token[token]
        return self.unk_init(torch.Tensor(self.dim))

    def get_vecs_by_tokens(self, tokens):
        return torch.stack([
            self[token]
            for token in tokens
        ])


_PRETRAINED_EMBEDDINGS = {
    'glove': partial(torchtext.vocab.GloVe, name='6B', cache=_TEXT_CACHE_DIR, dim=100),
    'radglove': RadGlove,
}

AVAILABLE_PRETRAINED_EMBEDDINGS = list(_PRETRAINED_EMBEDDINGS)

def create_word_embedding(vocab, embedding_size,
                          pretrained=None, freeze=False,
                          scale_grad_by_freq=False, batch_normalization=False,
                          ):
    """Creates layers needed for word-embedding.

    Args:
        vocab -- dictionary with (word: idx) pairs
        embedding_size -- int
        pretrained -- specify a pretrained embedding
        freeze -- freeze the embedding (only useful if pretrained is specified)
        scale_grad_by_freq -- nn.Embedding option
        batch_normalization -- whether or not to include a batch_normalization layer
    Returns:
        tuple (embedding_layer, bn_layer)
        If batch_normalization == False, bn_layer is a dummy lambda function.
        Notice layers are returned separated, since the embeddings may be calculated without BN
        during analysis (hence, it is useful to have them separated).
    """
    info = {
        'scale_grad_by_freq': scale_grad_by_freq,
        'bn': batch_normalization,
    }

    ## Create embedding layer
    if pretrained is None:
        emb_layer = nn.Embedding(
            len(vocab),
            embedding_size,
            padding_idx=PAD_IDX,
            scale_grad_by_freq=scale_grad_by_freq,
        )
        info.update({
            'vocab_size': len(vocab),
            'embedding_size': embedding_size,
        })
    else:
        emb_handler = _PRETRAINED_EMBEDDINGS[pretrained]()
        idx_to_word = {idx: word for word, idx in vocab.items()}
        words = [idx_to_word[idx] for idx in range(len(vocab))]

        # Transform necessary words
        words = [
            word if word != 'NUMBER' else 'number'
            for word in words
        ]

        vectors = emb_handler.get_vecs_by_tokens(words)
        # shape: n_words, embbeding_dim

        shape = (len(vocab), embedding_size)
        assert vectors.size() == shape, f'Emb size mismatch: {vectors.size()} vs {shape}'

        emb_layer = nn.Embedding.from_pretrained(
            vectors,
            padding_idx=PAD_IDX,
            freeze=freeze,
            scale_grad_by_freq=scale_grad_by_freq,
        )
        info.update({
            'freeze': freeze,
            'pretrained': pretrained,
        })

    ## Create BN layer
    if batch_normalization:
        bn_layer = nn.BatchNorm1d(embedding_size)
    else:
        bn_layer = lambda x: x

    LOGGER.info('Created word-embedding layer: %s', info)

    return emb_layer, bn_layer
