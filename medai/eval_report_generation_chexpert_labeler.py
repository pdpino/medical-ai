import csv
import os
import json
import subprocess
import argparse
import pandas as pd
import numpy as np
import hashlib
from sklearn.metrics import precision_recall_fscore_support as prf1s, roc_auc_score, accuracy_score
from pprint import pprint

from medai.datasets.common import CHEXPERT_LABELS
from medai.utils import WORKSPACE_DIR
from medai.metrics import get_results_folder


CHEXPERT_FOLDER = '~/chexpert/chexpert-labeler'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

TMP_FOLDER = os.path.join(WORKSPACE_DIR, 'tmp', 'chexpert-labeler')
CACHE_FOLDER = os.path.join(WORKSPACE_DIR, 'cache', 'chexpert-labeler')

GT_LABELED_FILEPATH = os.path.join(CACHE_FOLDER, 'gt-reports-labeled.csv')


def _labels_with_suffix(suffix):
    """Returns the chexpert labels with a suffix appended to each."""
    if not suffix:
        return list(CHEXPERT_LABELS)
    return [f'{label}-{suffix}' for label in CHEXPERT_LABELS]


def _hash_report(report, n=20):
    """Hash a report to avoid recalculations."""
    if not isinstance(report, str):
        raise Exception('Report must be string')
    payload = report.encode('utf-8')
    return hashlib.sha1(payload).hexdigest()[:n]


def _load_cached_gt_labels(df):
    """Loads cached file with labeled ground truth."""
    if not os.path.isfile(GT_LABELED_FILEPATH):
        print('Cached file for GT not found in', GT_LABELED_FILEPATH)
        return None

    # Load cached CSV
    cached = pd.read_csv(GT_LABELED_FILEPATH)

    # Calculate a hash for each report
    df = df.assign(hash=[
        _hash_report(report)
        for report in df['ground_truth']
    ])

    # Determine if it has all necessary hashes
    target_hashes = set(df['hash'])
    cached_hashes = set(cached['hash'])
    if not target_hashes.issubset(cached_hashes):
        print('Cached file for GT does not have all reports')
        return None

    # Merge on hashes
    merged = df.merge(cached, how='left', on='hash')

    # Return only np.array with labels
    labels = _labels_with_suffix('gt')
    return merged[labels].to_numpy()


def _save_cached_gt_labels(df):
    """Saves calculated labels for ground-truth reports to a CSV file."""
    # TODO: merge with previous cache, in case 

    # Select only useful columns
    columns = ['ground_truth'] + _labels_with_suffix('gt')
    cache_df = df[columns].copy()

    # Calculate a hash for each report
    if 'hash' not in df:
        cache_df = cache_df.assign(hash=[
            _hash_report(report)
            for report in cache_df['ground_truth']
        ])

    # Remove duplicate rows
    cache_df.drop_duplicates(subset='hash', inplace=True)

    # Save to file
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    cache_df.to_csv(GT_LABELED_FILEPATH, index=False)
    print(f'\tSaved labels cache to {GT_LABELED_FILEPATH}')
    return


def _get_custom_env():
    """Adds a necessary environment variable to run the labeler."""
    custom_env = os.environ.copy()
    prev = custom_env.get('PYTHONPATH', '')
    custom_env['PYTHONPATH'] = f'~/chexpert/NegBio:{prev}'

    return custom_env


def _concat_df_matrix(df, results, suffix=None):
    """Concats a DF with a matrix."""
    labels = _labels_with_suffix(suffix)
    return pd.concat([df, pd.DataFrame(results, columns=labels)],
                     axis=1, join='inner')


def _apply_labeler_to_column(dataframe, column_name):
    """Apply chexpert-labeler to a column of a dataframe."""
    # Grab reports
    reports_only = dataframe[column_name]
    
    # Tmp folder can be removed afterwards
    os.makedirs(TMP_FOLDER, exist_ok=True)
    
    # Create input file
    input_path = os.path.join(TMP_FOLDER, 'reports-input.csv')
    reports_only.to_csv(input_path, header=False, index=False, quoting=csv.QUOTE_ALL)
    
    # Call chexpert-labeler
    output_path = os.path.join(TMP_FOLDER, 'reports-output.csv')
    cmd_cd = f'cd {CHEXPERT_FOLDER}'
    cmd_call = f'{CHEXPERT_PYTHON} label.py --reports_path {input_path} --output_path {output_path}'
    cmd = f'{cmd_cd} && {cmd_call}'

    try:
        print(f'Labelling {column_name}...')
        completed_process = subprocess.run(cmd, shell=True, check=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env=_get_custom_env(),
                                           )
    except subprocess.CalledProcessError as e:
        print('Exception: ', e.stderr)
        return None
    
    # Read chexpert-labeler output
    out_df = pd.read_csv(output_path)
    
    # Nan are unknown: mark as negative
    out_df = out_df.fillna(0)
    
    # -1 are Uncertain, mark as negative (REVIEW)
    out_df = out_df.replace(-1, 0)
    
    return out_df[CHEXPERT_LABELS].to_numpy()


def _apply_labeler_to_df(df):
    """Calculates chexpert labels for a set of GT and generated reports."""
    # Calculate labels for ground truth
    ground_truth = _load_cached_gt_labels(df)
    cache_missed = ground_truth is None
    if cache_missed:
        # Cache miss, actually calculate and save
        ground_truth = _apply_labeler_to_column(df, 'ground_truth')
    else:
        print('Using cached labels for ground truth')


    # Calculate labels for generated
    generated = _apply_labeler_to_column(df, 'generated')

    # Concat in main dataframe
    df = _concat_df_matrix(df, ground_truth, 'gt')
    df = _concat_df_matrix(df, generated, 'gen')
    
    if cache_missed:
        _save_cached_gt_labels(df)

    return df


def _calculate_metrics(df):
    """Calculate metrics for a set of reports."""
    labels_gt = _labels_with_suffix('gt')
    labels_gen = _labels_with_suffix('gen')
    
    ground_truth = df[labels_gt].to_numpy()
    generated = df[labels_gen].to_numpy()
    
    acc = np.array([
        accuracy_score(ground_truth[:, i], generated[:, i])
        for i in range(len(CHEXPERT_LABELS))
    ])

    precision, recall, f1, s = prf1s(ground_truth, generated, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(ground_truth, generated, average=None)
    except ValueError as e:
        print(e)
        roc_auc = np.array([-1]*len(CHEXPERT_LABELS))
        
    return acc, precision, recall, f1, roc_auc


def _calculate_metrics_dict(df):
    """Calculates metrics for all dataset_types (train, val, test)."""
    all_metrics = {}

    for dataset_type in set(df['dataset_type']):
        sub_df = df[df['dataset_type'] == dataset_type]
        
        acc, precision, recall, f1, roc_auc = _calculate_metrics(sub_df)
        
        metrics = {}
        
        def _add_to_results(array, prefix):
            # Add mean value to dict
            metrics[prefix] = array.mean()
            
            # Add values for each label
            array = array.tolist() # Avoid numpy-not-serializable issues
            for label, value in zip(CHEXPERT_LABELS, array):
                metrics[f'{prefix}-{label}'] = value
        
        _add_to_results(acc, 'acc')
        _add_to_results(precision, 'prec')
        _add_to_results(recall, 'recall')
        _add_to_results(f1, 'f1')
        _add_to_results(roc_auc, 'roc_auc')

        all_metrics[dataset_type] = metrics

    return all_metrics


def evaluate_run(run_name,
                 debug=True,
                 override=False,
                 max_samples=None,
                 free=False,
                 ):
    """Evaluates a run with the Chexpert-labeler."""
    # Folder containing run results
    results_folder = get_results_folder(run_name, classification=False, debug=debug)

    # Output file at the end of this process
    suffix = 'free' if free else 'notfree'
    labeled_output_path = os.path.join(results_folder, f'outputs-labeled-{suffix}.csv')

    if not override and os.path.isfile(labeled_output_path):
        print('Skipping run, already calculated: ', run_name)
        return

    model_output_path = os.path.join(results_folder, f'outputs-{suffix}.csv')
    
    if not os.path.isfile(model_output_path):
        print('Need to compute outputs for run first: ', model_output_path)
        return

    # Read outputs
    df = pd.read_csv(model_output_path)

    _n_distinct_epochs = set(df['epoch'])
    if len(_n_distinct_epochs) != 1:
        raise NotImplementedError('Only works for one epoch, found: ', _n_distinct_epochs)

    # Debugging purposes
    if max_samples is not None:
        df = df.head(max_samples)
        df.reset_index(inplace=True, drop=True)

    # Compute labels for both GT and generated
    df = _apply_labeler_to_df(df)

    # Save to file, to avoid heavy recalculations
    df.to_csv(labeled_output_path, index=False)

    # Calculate metrics over train, val and test
    metrics = _calculate_metrics_dict(df)

    # Save metrics to file
    chexpert_metrics_path = os.path.join(results_folder, f'chexpert-metrics-{suffix}.json')
    with open(chexpert_metrics_path, 'w') as f:
        json.dump(metrics, f)
    print('Saved to file: ', chexpert_metrics_path)

    pprint(metrics)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run name to evaluate')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override previous calculation')
    parser.add_argument('--free', action='store_true',
                        help='If present, use outputs freely generated')
    parser.add_argument('--no-debug', action='store_true',
                        help='If present, the run is non-debug')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug: use a max amount of samples')

    args = parser.parse_args()

    args.debug = not args.no_debug

    return args


if __name__ == '__main__':
    args = parse_args()

    evaluate_run(args.run_name,
                 debug=args.debug,
                 override=args.override,
                 max_samples=args.max_samples,
                 free=args.free,
                 )