"""Pipeline to preprocess IU x-ray reports.

* Clean and tokenizes reports
* Creates a reports.clean.version.json file
* Creates a common vocabulary for the dataset
"""
import os
import json
from collections import defaultdict, Counter

from medai.datasets.common.sentences2organs.compute import save_sentences_with_organs
from medai.datasets.preprocess.tokenize import text_to_tokens
from medai.datasets.preprocess.common import (
    assert_reports_not_exist,
    save_clean_reports,
    split_sentences_and_save_csv,
    load_sentences_metadata,
)
from medai.datasets.iu_xray import DATASET_DIR
from medai.datasets.vocab import save_vocabs
from medai.metrics.report_generation.chexpert import _concat_df_matrix, apply_labeler_to_column


_REPLACEMENTS_BY_REPORT = {
    # Hardcode some fixes
    '3368.xml': {
        'findings': [('Pression:', '')],
    },
    '1448.xml': {
        'findings': [('02/010/XXXX', 'xxxx')]
    },
    '793.xml': {
        'findings': [('31 17 XXXX', 'xxxx')],
    },
}


IGNORE_TOKENS = set(['p.m.', 'pm', 'am'])
REPORTS_DIR = os.path.join(DATASET_DIR, 'reports')

def load_raw_reports(first_clean=True):
    reports_fname = os.path.join(REPORTS_DIR, 'reports.json')
    with open(reports_fname, 'r') as f:
        reports_dict = json.load(f)

    if first_clean:
        for filename, replacements in _REPLACEMENTS_BY_REPORT.items():
            for text_type, to_replace in replacements.items():
                for target, replace_with in to_replace:
                    text = reports_dict[filename][text_type]
                    reports_dict[filename][text_type] = text.replace(target, replace_with)


    return reports_dict

def clean_reports(reports_dict, impression_fallback=True):
    token_appearances = Counter()
    errors = defaultdict(list)

    clean_reports_dict = dict()

    for report in reports_dict.values():
        filename = report['filename']
        findings = report['findings']
        impression = report['impression']

        n_images = len(report['images'])
        if n_images == 0:
            errors['no-images'].append(filename)
            continue

        text = findings
        if findings is None and impression_fallback:
            text = impression

        if text is None:
            errors['text-none'].append(filename)
            continue

        if findings is None:
            errors['findings-none'].append(filename)
        elif impression is None:
            errors['impression-none'].append(filename)

        # Clean and tokenize text
        tokens = []
        for token in text_to_tokens(text, IGNORE_TOKENS):
            tokens.append(token)
            token_appearances[token] += 1

        cleaned_report = dict(report)
        cleaned_report['clean_text'] = ' '.join(tokens)

        clean_reports_dict[filename] = cleaned_report

    print('Errors: ', {k: len(v) for k, v in errors.items()})
    print(f'Different tokens: {len(token_appearances):,}')

    n_reports_1 = len(reports_dict)
    n_reports_2 = len(clean_reports_dict) + len(errors['text-none']) + len(errors['no-images'])
    assert n_reports_1 == n_reports_2, f'N reports incorrect: {n_reports_1} vs {n_reports_2}'

    return clean_reports_dict, token_appearances, errors


def load_info():
    info_fname = os.path.join(DATASET_DIR, 'info.json')
    with open(info_fname, 'r') as f:
        info = json.load(f)
    return info


def _add_image_info(reports_dict):
    info = load_info()

    wrong_images = set(info['marks']['wrong'])
    broken_images = set(info['marks']['broken'])

    for report_name, report_dict in reports_dict.items():
        new_images_info = []
        for image_info in report_dict['images']:
            image_name = image_info['id']

            if not image_name.endswith('.png'):
                image_name = f'{image_name}.png'

            image_info['side'] = info['classification'][image_name]
            image_info['wrong'] = image_name in wrong_images
            image_info['broken'] = image_name in broken_images

            new_images_info.append(image_info)

        report_dict['images'] = new_images_info
        reports_dict[report_name] = report_dict

    return reports_dict


def preprocess_iu_x_ray(version, greater_values=[0, 5, 10], override=False, **kwargs):
    assert_reports_not_exist(REPORTS_DIR, version, override)

    reports_dict = load_raw_reports()

    reports_dict, token_appearances, errors = clean_reports(reports_dict, **kwargs)

    reports_dict = _add_image_info(reports_dict)

    save_clean_reports(REPORTS_DIR, reports_dict, version)

    save_vocabs(REPORTS_DIR, version, reports_dict, token_appearances, greater_values)

    split_sentences_and_save_csv(REPORTS_DIR, reports_dict)

    return reports_dict, token_appearances, errors


def create_sentences_with_chexpert_labels():
    df_sentences = load_sentences_metadata(REPORTS_DIR)

    labels = apply_labeler_to_column(list(df_sentences['sentence']),
                                     fill_empty=-2, fill_uncertain=-1,
                                     caller_id='iu-preprocess-chexpert')

    if labels.shape != (len(df_sentences), 14):
        raise Exception(f'Chexpert labels shape failed: {labels.shape} vs {len(df_sentences)}')

    df_sentences = _concat_df_matrix(df_sentences, labels)

    out_fpath = os.path.join(REPORTS_DIR, 'sentences_with_chexpert_labels.csv')
    df_sentences.to_csv(out_fpath, index=False)
    print(f'Saved to {out_fpath}')

    return df_sentences


def create_sentences_with_organs(show=True, ignore_all_ones=True):
    df_sentences = load_sentences_metadata(REPORTS_DIR)
    sentences = list(df_sentences['sentence'])

    return save_sentences_with_organs(
        DATASET_DIR, sentences,show=show, ignore_all_ones=ignore_all_ones,
    )
