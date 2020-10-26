import csv
import os
import json
import subprocess
import argparse
import re
import pandas as pd
import numpy as np
from pprint import pprint

from medai.utils import TMP_DIR
from medai.metrics import get_results_folder


MIRQI_FOLDER = '~/software/MIRQI'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

METRICS_KEYS = ['MIRQI-r', 'MIRQI-p', 'MIRQI-f'] # Outputed by MIRQI script

TMP_FOLDER = os.path.join(TMP_DIR, 'mirqi')


def _create_input_file(df, col_name, filepath):
    """Creates an input file for the MIRQI script."""
    reports_only = df[col_name]

    reports_only.to_csv(filepath, header=False, index=False, quoting=csv.QUOTE_ALL)


def _apply_mirqi_to_df(df, gt_col_name='ground_truth', gen_col_name='generated'):
    """Applies MIRQI scorer for a set of GT and generated reports."""
    # Tmp folder can be removed afterwards
    os.makedirs(TMP_FOLDER, exist_ok=True)

    # Temp filenames
    GT_INPUT_PATH = os.path.join(TMP_FOLDER, 'gt-input.csv')
    GEN_INPUT_PATH = os.path.join(TMP_FOLDER, 'gen-input.csv')
    OUTPUT_PATH = os.path.join(TMP_FOLDER, 'output.csv')

    # Create input files
    _create_input_file(df, gt_col_name, GT_INPUT_PATH)
    _create_input_file(df, gen_col_name, GEN_INPUT_PATH)

    # Call MIRQI
    cmd_cd = f'cd {MIRQI_FOLDER}'
    cmd_call = f'{CHEXPERT_PYTHON} evaluate.py --reports_path_gt {GT_INPUT_PATH} --reports_path_cand {GEN_INPUT_PATH} --output_path {OUTPUT_PATH}'
    cmd = f'{cmd_cd} && {cmd_call}'

    try:
        print(f'Evaluating reports with MIRQI...')
        completed_process = subprocess.run(cmd, shell=True, check=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print('MIRQI script failed: ', e.stderr)
        raise

    # Read MIRQI output # contains reports, "graph" and scores
    out_df = pd.read_csv(OUTPUT_PATH)

    clean_spaces = lambda s: re.sub(r'\s+', ' ', s)
    out_df['attributes'] = out_df['attributes'].apply(clean_spaces)

    # Merge with original DF
    target_cols = ['attributes'] + METRICS_KEYS
    df = df.merge(out_df[target_cols], left_index=True, right_index=True)

    return df


def _calculate_metrics_dict(df):
    """Averages metrics for a dataset."""
    all_metrics = {}

    for dataset_type in set(df['dataset_type']):
        sub_df = df[df['dataset_type'] == dataset_type]

        averages_in_subdf = {
            metric_key: np.mean(sub_df[metric_key])
            for metric_key in METRICS_KEYS
        }

        all_metrics[dataset_type] = averages_in_subdf

    return all_metrics


def evaluate_run(run_name,
                 debug=True,
                 override=False,
                 max_samples=None,
                 free=False,
                 quiet=False,
                 ):
    """Evaluates a run with the MIRQI metric."""
    # Folder containing run results
    results_folder = get_results_folder(run_name, classification=False, debug=debug)

    # Output file at the end of this process
    suffix = 'free' if free else 'notfree'
    labeled_output_path = os.path.join(results_folder, f'outputs-mirqi-{suffix}.csv')

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

    # Compute MIRQI metrics, plus graph for generated text
    df = _apply_mirqi_to_df(df)

    # Save to file, to avoid heavy recalculations
    df.to_csv(labeled_output_path, index=False)

    # Compute metrics
    metrics = _calculate_metrics_dict(df)

    # Save metrics to file
    mirqi_metrics_path = os.path.join(results_folder, f'mirqi-metrics-{suffix}.json')
    with open(mirqi_metrics_path, 'w') as f:
        json.dump(metrics, f)
    print('Saved to file: ', mirqi_metrics_path)

    if not quiet:
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
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics to stdout')

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
                 quiet=args.quiet,
                 )