import json
import os

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


def save_synonyms(name, synonyms):
    filepath = _get_syn_fname(name)

    with open(filepath, 'w') as f:
        json.dump(synonyms, f, indent=2)


def load_synonyms(name):
    filepath = _get_syn_fname(name)

    with open(filepath, 'r') as f:
        return json.load(f)

