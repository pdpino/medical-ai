import json
import os

def load_vocab(name):
    this_folder = os.path.dirname(os.path.realpath(__file__))

    filepath = os.path.join(this_folder, f'{name}.vocab.json')

    if not os.path.isfile(filepath):
        raise Exception('Vocabulary not found: ', filepath)
        # return None

    with open(filepath) as f:
        return json.load(f)