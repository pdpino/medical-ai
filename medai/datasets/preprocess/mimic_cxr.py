"""Pipeline to preprocess MIMIC CXR reports.

* Clean and tokenizes reports
* Creates a reports.clean.version.json file
* Creates a common vocabulary for the dataset
"""
import os
import json
from collections import defaultdict, Counter
from tqdm.auto import tqdm

import pandas as pd

from medai.datasets.preprocess.tokenize import text_to_tokens
from medai.datasets.mimic_cxr import DATASET_DIR
from medai.datasets.vocab import compute_vocab, save_vocab


IGNORE_TOKENS = set(['p.m.', 'pm', 'am'])

REPORTS_DIR = os.path.join(DATASET_DIR, 'reports')

def load_raw_reports_df():
    # Load processed text
    fpath = os.path.join(REPORTS_DIR, 'mimic_cxr_sections.csv')
    reports_df = pd.read_csv(fpath, header=None, names=['study', 'text'])
    reports_df.dropna(axis=0, inplace=True, how='any')

    # As a fallback, load sectioned DF,
    fpath = os.path.join(REPORTS_DIR, 'mimic_cxr_sectioned.csv')
    sectioned_df = pd.read_csv(fpath)
    # Columns: study, impression, findings, last_paragraph, comparison

    # Merge with original
    reports_df = reports_df.merge(sectioned_df, how='left', on='study')

    return reports_df


def save_clean_reports(reports_dict):
    version = 'v1'
    fname = os.path.join(REPORTS_DIR, f'reports.clean.{version}.json')
    with open(fname, 'w') as f:
        json.dump(reports_dict, f)

    print('Saved reports to: ', fname)


def clean_reports(reports_df):
    errors = defaultdict(list)
    token_appearances = Counter()

    cleaned_reports = dict()

    pbar = tqdm(total=len(reports_df))

    for _, row in reports_df.iterrows():
        pbar.update(1)

        study_id = int(row['study'].strip('s'))

        text = ''
        tokens = []
        # Try with different values
        for key in ['text', 'findings', 'impression', 'last_paragraph', 'comparison']:
            if key not in row:
                continue

            text = row[key]
            tokens = text_to_tokens(text, IGNORE_TOKENS)

            if tokens:
                break

        if not tokens:
            errors['tokens-empty'].append(study_id)
            continue

        for token in tokens:
            token_appearances[token] += 1

        cleaned_reports[study_id] = {
            'study_id': study_id,
            'clean_text': ' '.join(tokens),
            'text': text,
        }

    print('Errors: ', {k: len(v) for k, v in errors.items()})
    print('Different tokens: ', len(token_appearances))

    n_reports_1 = len(reports_df)
    n_reports_2 = len(cleaned_reports) + len(errors['tokens-empty'])
    assert n_reports_1 == n_reports_2, f'N reports incorrect: {n_reports_1} vs {n_reports_2}'

    return cleaned_reports, token_appearances, errors


def add_report_len_to_master_df(reports_dict):
    fpath = os.path.join(DATASET_DIR, 'master_metadata.csv')
    master_df = pd.read_csv(fpath)

    report_lens_by_id = {
        study_id: len(report['clean_text'].split())
        for study_id, report in reports_dict.items()
    }

    report_lens = [
        report_lens_by_id[study_id]
        for study_id in master_df['study_id']
    ]

    master_df['report_length'] = report_lens

    master_df.to_csv(fpath, index=False)


def preprocess_mimic_cxr():
    reports_df = load_raw_reports_df()

    reports_dict, token_appearances, errors = clean_reports(reports_df)

    save_clean_reports(reports_dict)

    vocab = compute_vocab(r['clean_text'].split() for r in reports_dict.values())
    save_vocab('mimic_cxr', vocab)

    add_report_len_to_master_df(reports_dict)

    return reports_dict, token_appearances, errors


if __name__ == '__main__':
    preprocess_mimic_cxr()
