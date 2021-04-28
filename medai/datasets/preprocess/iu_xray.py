"""Pipeline to preprocess IU x-ray reports.

* Clean and tokenizes reports
* Creates a reports.clean.version.json file
* Creates a common vocabulary for the dataset
"""
import os
import json
from collections import defaultdict, Counter

from medai.datasets.preprocess.tokenize import text_to_tokens
from medai.datasets.preprocess.common import assert_reports_not_exist, save_clean_reports
from medai.datasets.iu_xray import DATASET_DIR
from medai.datasets.vocab import save_vocabs


IGNORE_TOKENS = set(['p.m.', 'pm', 'am'])
REPORTS_DIR = os.path.join(DATASET_DIR, 'reports')

def load_raw_reports():
    reports_fname = os.path.join(REPORTS_DIR, 'reports.json')
    with open(reports_fname, 'r') as f:
        reports_dict = json.load(f)

    return reports_dict

def clean_reports(reports_dict):
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
        if findings is None and impression is None:
            errors['text-none'].append(filename)
            continue

        if findings is None:
            errors['findings-none'].append(filename)
            text = impression
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


def add_image_info(reports_dict):
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


def preprocess_iu_x_ray(version, greater_values=[0, 5, 10], override=False):
    assert_reports_not_exist(REPORTS_DIR, version, override)

    reports_dict = load_raw_reports()

    reports_dict, token_appearances, errors = clean_reports(reports_dict)

    reports_dict = add_image_info(reports_dict)

    save_clean_reports(REPORTS_DIR, reports_dict, version)

    save_vocabs('iu_xray', reports_dict, token_appearances, greater_values)

    return reports_dict, token_appearances, errors
