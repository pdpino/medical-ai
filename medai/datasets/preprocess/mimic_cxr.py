"""Pipeline to preprocess MIMIC CXR reports.

* Clean and tokenizes reports
* Creates a reports.clean.version.json file
* Creates a common vocabulary for the dataset
"""
import os
from collections import defaultdict, Counter
from tqdm.auto import tqdm

import pandas as pd

from medai.datasets.common.sentences2organs.compute import save_sentences_with_organs
from medai.datasets.preprocess.tokenize import text_to_tokens
from medai.datasets.preprocess.common import (
    assert_reports_not_exist,
    save_clean_reports,
    load_clean_reports,
)
from medai.datasets.mimic_cxr import DATASET_DIR
from medai.datasets.vocab import save_vocabs
from medai.utils.nlp import get_sentences_appearances


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
            if token:
                token_appearances[token] += 1

        cleaned_reports[study_id] = {
            'study_id': study_id,
            'clean_text': ' '.join(tokens),
            'text': text,
        }

    print('Errors: ', {k: len(v) for k, v in errors.items()})
    print(f'Different tokens: {len(token_appearances):,}')

    n_reports_1 = len(reports_df)
    n_reports_2 = len(cleaned_reports) + len(errors['tokens-empty'])
    assert n_reports_1 == n_reports_2, f'N reports incorrect: {n_reports_1} vs {n_reports_2}'

    return cleaned_reports, token_appearances, errors


def add_report_len_to_master_df(reports_dict, errors):
    fpath = os.path.join(DATASET_DIR, 'master_metadata.csv')
    master_df = pd.read_csv(fpath)

    report_lens_by_id = {
        study_id: len(report['clean_text'].split())
        for study_id, report in reports_dict.items()
    }

    report_lens = []
    for study_id in master_df['study_id']:
        n_words = report_lens_by_id.get(study_id, 0)

        if n_words == 0:
            errors['words-0'].append(study_id)

        report_lens.append(n_words)


    master_df['report_length'] = report_lens

    master_df.to_csv(fpath, index=False)


def preprocess_mimic_cxr(version, greater_values=[0, 5, 10], override=False):
    """Preprocess reports, saves reports and vocabularies as JSON."""
    assert_reports_not_exist(REPORTS_DIR, version, override)

    reports_df = load_raw_reports_df()

    reports_dict, token_appearances, errors = clean_reports(reports_df)

    save_clean_reports(REPORTS_DIR, reports_dict, version)

    save_vocabs(REPORTS_DIR, version, reports_dict, token_appearances, greater_values)

    add_report_len_to_master_df(reports_dict, errors)

    return reports_dict, token_appearances, errors


def create_sentences_with_organs(version, studies=None, **kwargs):
    """Computes sentence-to-organs mapping and save it as CSV."""
    reports = load_clean_reports(REPORTS_DIR, version)
    sentence_counter = get_sentences_appearances(
        r['clean_text']
        for study_id, r in reports.items()
        if studies is None or study_id in studies
    )

    sentences = list(sentence_counter.keys())

    df_organs, errors = save_sentences_with_organs(DATASET_DIR, sentences, **kwargs)

    return df_organs, errors
