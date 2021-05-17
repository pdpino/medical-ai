"""Optimize classification thresholds, using roc and pr curves.

Based on this post:
https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
"""
import json
import logging
import os
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from medai.utils import divide_arrays
from medai.utils.files import get_results_folder

LOGGER = logging.getLogger(__name__)

def _calculate_optimal_roc(gt, pred):
    assert len(gt) == len(pred)
    fpr, tpr, thresholds = roc_curve(gt, pred)

    J_stat = tpr - fpr
    best_idx = J_stat.argmax()

    return thresholds[best_idx], J_stat[best_idx]


def _calculate_optimal_pr(gt, pred):
    assert len(gt) == len(pred)
    precision, recall, thresholds = precision_recall_curve(gt, pred)

    f1 = divide_arrays(2*precision*recall, precision + recall)
    best_idx = f1.argmax()

    return thresholds[best_idx], f1[best_idx], precision[best_idx], recall[best_idx]


def _get_diseases_from_results_df(df):
    return [
        col[:-3]
        for col in df.columns
        if col.endswith('-gt')
    ]


def calculate_optimal_threshold(run_id, split='val'):
    """Calculates optimal thresholds for a classification run."""
    results_folder = get_results_folder(run_id)

    fpath = os.path.join(results_folder, 'outputs.csv')
    if not os.path.isfile(fpath):
        raise FileNotFoundError('Need to calculate outputs first')

    df = pd.read_csv(fpath)
    # columns: filename, epoch, dataset_type, <diseases>-gt, <diseases>-pred

    # Get diseases names
    diseases = _get_diseases_from_results_df(df)

    # Filter by split
    df = df.loc[df['dataset_type'] == split]

    _n_unique_filenames = len(df['filename'].unique())
    if _n_unique_filenames != len(df):
        raise Exception(f'Could not filter by split: {_n_unique_filenames} vs {len(df)}')

    # Calculate optimals
    optimal_thresh_roc = {}
    optimal_thresh_pr = {}
    best_values = {}

    for disease in diseases:
        gt = df[f'{disease}-gt'].to_numpy()
        pred = df[f'{disease}-pred'].to_numpy()

        thresh_roc, best_J = _calculate_optimal_roc(gt, pred)
        thresh_pr, best_f1, best_prec, best_recall = _calculate_optimal_pr(gt, pred)

        optimal_thresh_roc[disease] = thresh_roc
        optimal_thresh_pr[disease] = thresh_pr
        best_values[disease] = {
            'f1': best_f1,
            'prec': best_prec,
            'recall': best_recall,
            'J': best_J,
        }

    # Save to file
    for name, values in zip(['roc', 'pr'], [optimal_thresh_roc, optimal_thresh_pr]):
        fpath = os.path.join(results_folder, f'thresholds-{name}.json')
        with open(fpath, 'w') as f:
            json.dump(values, f)
            LOGGER.info('Saved thresholds to %s', fpath)

    return optimal_thresh_roc, optimal_thresh_pr, best_values


def load_optimal_threshold(run_id, name):
    if name not in ('pr', 'roc'):
        raise Exception(f'Threshold name not recognized: {name}')

    results_folder = get_results_folder(run_id)
    fpath = os.path.join(results_folder, f'thresholds-{name}.json')

    if not os.path.isfile(fpath):
        raise Exception(f'Best thresholds not calculated: {fpath}')

    with open(fpath, 'r') as f:
        return json.load(f)
