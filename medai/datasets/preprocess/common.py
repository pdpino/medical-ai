import os
import json
import pandas as pd

from medai.utils.nlp import get_sentences_appearances

def _get_clean_reports_fpath(reports_dir, version):
    return os.path.join(reports_dir, f'reports.clean.{version}.json')


def assert_reports_not_exist(reports_dir, version, override=False):
    fpath = _get_clean_reports_fpath(reports_dir, version)
    if os.path.isfile(fpath):
        msg = f'Reports version {version} already exist, override={override}'
        if override:
            print(msg)
        else:
            raise Exception(msg)

def save_clean_reports(reports_dir, reports_dict, version):
    fpath = _get_clean_reports_fpath(reports_dir, version)

    with open(fpath, 'w') as f:
        json.dump(reports_dict, f)

    print('Saved reports to: ', fpath)


def load_clean_reports(reports_dir, version):
    fpath = _get_clean_reports_fpath(reports_dir, version)

    with open(fpath, 'r') as f:
        return json.load(f)


def split_sentences_and_save_csv(reports_dir, reports_dict):
    # Count sentences and appearances
    sentence_counter = get_sentences_appearances(r['clean_text'] for r in reports_dict.values())

    # Create DF
    columns = ['sentence', 'appearances']
    df_sentences = pd.DataFrame(sentence_counter.items(), columns=columns)

    # Save to file
    fpath = os.path.join(reports_dir, 'sentences.csv')
    df_sentences.to_csv(fpath, index=False)

    print(f'Saved sentences with appearances to {fpath}')


def load_sentences_metadata(reports_dir):
    fpath = os.path.join(reports_dir, 'sentences.csv')
    if not os.path.isfile(fpath):
        raise FileNotFoundError('sentences.csv not created yet')

    df_sentences = pd.read_csv(fpath)
    return df_sentences
