import json
import os

from medai.utils.nlp import (
    PAD_TOKEN,
    PAD_IDX,
    START_TOKEN,
    START_IDX,
    END_TOKEN,
    END_IDX,
    UNKNOWN_TOKEN,
    UNKNOWN_IDX,
    END_OF_SENTENCE_TOKEN,
    END_OF_SENTENCE_IDX,
)

def _get_this_folder():
    return os.path.dirname(os.path.realpath(__file__))

def _get_vocab_fname(name):
    return os.path.join(
        _get_this_folder(),
        f'{name}.vocab.json'
    )

def _get_syn_fname(name):
    return os.path.join(
        _get_this_folder(),
        f'{name}.syn.json'
    )


def load_vocab(name):
    filepath = _get_vocab_fname(name)

    if not os.path.isfile(filepath):
        raise Exception('Vocabulary not found: ', filepath)
        # return None

    with open(filepath) as f:
        return json.load(f)

def save_vocab(name, vocab):
    filepath = _get_vocab_fname(name)

    with open(filepath, 'w') as f:
        json.dump(vocab, f, indent=2)


def compute_vocab(reports):
    """Computes a vocabulary given an iteration of reports."""
    word_to_idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
        END_OF_SENTENCE_TOKEN: END_OF_SENTENCE_IDX,
    }

    for report in reports:
        for token in report:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)

    return word_to_idx


def save_synonyms(name, synonyms):
    filepath = _get_syn_fname(name)

    with open(filepath, 'w') as f:
        json.dump(synonyms, f, indent=2)


def load_synonyms(name):
    filepath = _get_syn_fname(name)

    with open(filepath, 'r') as f:
        return json.load(f)
