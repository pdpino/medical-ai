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

def _get_vocab_fname(name, greater_than=None):
    fname = name
    if greater_than is not None and greater_than > 0:
        fname = f'{name}.greater{greater_than}'
    fname = f'{fname}.vocab.json'
    return os.path.join(_get_this_folder(), fname)

def _get_syn_fname(name):
    return os.path.join(
        _get_this_folder(),
        f'{name}.syn.json'
    )


def _assert_correlative_ids(vocab):
    ids = vocab.values()
    n_words = len(vocab)

    max_id = max(ids)
    min_id = min(ids)
    n_ids = len(ids)
    n_unique_ids = len(set(ids))

    err = 'Ids are not be correlative'
    assert max_id == n_words - 1, f'{err}, max failed: {max_id} != {n_words - 1}'
    assert min_id == 0, f'{err}, min failed: {min_id} != 0'
    assert n_ids == n_unique_ids, f'{err}, duplicated failed: {n_ids} vs {n_unique_ids}'


def load_vocab(name, greater_than=None):
    filepath = _get_vocab_fname(name, greater_than)

    if not os.path.isfile(filepath):
        raise Exception('Vocabulary not found: ', filepath)
        # return None

    with open(filepath) as f:
        vocab = json.load(f)

    _assert_correlative_ids(vocab)

    return vocab


def _save_vocab(name, vocab, greater_than=None):
    filepath = _get_vocab_fname(name, greater_than)

    _assert_correlative_ids(vocab)

    with open(filepath, 'w') as f:
        json.dump(vocab, f, indent=2)

    print(f'Vocab with {len(vocab):,} tokens saved to: {filepath}')


def _compute_vocab(reports, token_appearances, greater_than=0):
    """Computes a vocabulary given an iterator of reports."""
    word_to_idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
        END_OF_SENTENCE_TOKEN: END_OF_SENTENCE_IDX,
    }

    for report in reports:
        for token in report:
            if greater_than is None or token_appearances[token] > greater_than:
                if token not in word_to_idx:
                    word_to_idx[token] = len(word_to_idx)

    return word_to_idx


def save_vocabs(name, reports_dict, token_appearances, greater_values):
    for greater_than in greater_values:
        vocab = _compute_vocab(
            (r['clean_text'].split() for r in reports_dict.values()),
            token_appearances,
            greater_than,
        )

        _save_vocab(name, vocab, greater_than)



def save_synonyms(name, synonyms):
    filepath = _get_syn_fname(name)

    with open(filepath, 'w') as f:
        json.dump(synonyms, f, indent=2)


def load_synonyms(name):
    filepath = _get_syn_fname(name)

    with open(filepath, 'r') as f:
        return json.load(f)
