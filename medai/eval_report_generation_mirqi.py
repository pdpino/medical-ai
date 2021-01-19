import csv
import os
import json
import subprocess
import argparse
import logging

from pprint import pprint
import pandas as pd
import numpy as np

from medai.utils import TMP_DIR, timeit_main
from medai.utils.files import get_results_folder
from medai.metrics import load_rg_outputs

LOGGER = logging.getLogger('medai.rg.eval.mirqi')

MIRQI_FOLDER = '~/software/MIRQI'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

# METRICS_KEYS = ['MIRQI-r', 'MIRQI-p', 'MIRQI-f'] # Outputed by MIRQI script

TMP_FOLDER = os.path.join(TMP_DIR, 'mirqi')

def _attributes_to_list(all_attrs):
    """Given a list of all attributes as string, parse to lists.

    Args:
      all_attrs -- list of n_samples, with strings having tuples in format:
      (mention|category|valoration|attributes) (tuple2) ...
    Returns:
      List of n_samples, each sample having a list of lists:
      [mention, category, valoration, attributes], [tuple2], ...
    """
    return [
        [
            entity.strip(')').strip('(').split('|')
            for entity in sample_attrs.split(') (')
            if len(entity) > 0
        ]
        for sample_attrs in all_attrs
    ]


def MIRQI_v2(gt_list, cand_list, epsilon=1e-6):
    """Compute a v2 of the MIRQI metric.
       It returns scores: MIRQI-r, MIRQI-p, MIRQI_sp, MIRQI_np, MIRQI_f, MIRQI_attr_p, MIRQI_attr_r
    """

    MIRQI_r = [] # Recall
    MIRQI_p = [] # Precision (Positive predictive value)
    MIRQI_sp = [] # Specificity = "Negative recall"
    MIRQI_np = [] # Negative predictive value
    MIRQI_f = [] # F1

    MIRQI_attr_p = [] # Precision in attributes
    MIRQI_attr_r = [] # Recall in attributes

    for gt_report_entry, cand_report_entry in zip(gt_list, cand_list):
        pos_count_in_gt = 0
        pos_count_in_cand = 0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        tp_attr = 0
        fp_attr = 0
        fn_attr = 0

        for gt_entity in gt_report_entry:
            if len(gt_entity) < 3:
                LOGGER.debug('Len less than 3: %s %s', gt_report_entry, cand_report_entry)
            if gt_entity[2] == 'NEGATIVE':
                continue
            pos_count_in_gt = pos_count_in_gt + 1
        neg_count_in_gt = len(gt_report_entry) - pos_count_in_gt

        for _, cand_entity in enumerate(cand_report_entry):
            if cand_entity[2] == 'NEGATIVE':
                for _, gt_entity in enumerate(gt_report_entry):
                    if  gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            tn = tn + 1     # true negative hits
                            break
                        else:
                            fn = fn + 1     # false negative hits
                            break
            else:
                pos_count_in_cand = pos_count_in_cand + 1
                for _, gt_entity in enumerate(gt_report_entry):
                    if gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            fp = fp + 1     # false positive hits
                            break
                        else:
                            tp = tp + 1

                            # count attribute hits
                            if gt_entity[3] == '':
                                break
                            gt_attrs = set(gt_entity[3].split('/'))
                            cand_attrs = set(cand_entity[3].split('/'))

                            tp_attr += len(gt_attrs.intersection(cand_attrs))
                            fp_attr += len(cand_attrs - gt_attrs)
                            fn_attr += len(gt_attrs - cand_attrs)

                            break
        neg_count_in_cand = len(cand_report_entry) - pos_count_in_cand

        # Compute recall and precision
        if pos_count_in_gt == 0 and pos_count_in_cand == 0:
            score_r = 1.0
            score_p = 1.0

            score_attr_r = 1.0
            score_attr_p = 1.0
        elif pos_count_in_gt == 0 or pos_count_in_cand == 0:
            score_r = 0.0
            score_p = 0.0

            score_attr_r = 0.0
            score_attr_p = 0.0
        else:
            score_r = tp / (tp + fn + epsilon)
            score_p = tp / (tp + fp + epsilon)

            score_attr_r = tp_attr / (tp_attr + fn_attr + epsilon)
            score_attr_p = tp_attr / (tp_attr + fp_attr + epsilon)

        # Compute spec and negative predictive value
        if neg_count_in_gt == 0 and neg_count_in_cand == 0:
            score_sp = 1.0
            score_np = 1.0
        elif neg_count_in_gt == 0 or neg_count_in_cand == 0:
            score_sp = 0.0
            score_np = 0.0
        else:
            score_sp = tn / (tn + fp + epsilon)
            score_np = tn / (tn + fn + epsilon)

        MIRQI_r.append(score_r)
        MIRQI_p.append(score_p)
        MIRQI_sp.append(score_sp)
        MIRQI_np.append(score_np)

        rec_prec = (score_r + score_p)
        MIRQI_f.append(2*(score_r * score_p) / rec_prec if rec_prec != 0.0 else 0.0)

        MIRQI_attr_p.append(score_attr_p)
        MIRQI_attr_r.append(score_attr_r)

    scores = {
        'MIRQI-v2-r': MIRQI_r,
        'MIRQI-v2-p': MIRQI_p,
        'MIRQI-v2-sp': MIRQI_sp,
        'MIRQI-v2-np': MIRQI_np,
        'MIRQI-v2-f': MIRQI_f,
        'MIRQI-v2-attr-p': MIRQI_attr_p,
        'MIRQI-v2-attr-r': MIRQI_attr_r,
    }

    return scores


def _create_input_file(df, col_name, filepath):
    """Creates an input file for the MIRQI script."""
    reports_only = df[col_name]

    reports_only.to_csv(filepath, header=False, index=False, quoting=csv.QUOTE_ALL)


def _apply_mirqi_to_df(df, gt_col_name='ground_truth', gen_col_name='generated', extra=None):
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
    cmd_call = f'{CHEXPERT_PYTHON} evaluate.py --reports_path_gt {GT_INPUT_PATH} \
        --reports_path_cand {GEN_INPUT_PATH} --output_path {OUTPUT_PATH} '
    if extra:
        cmd_call += ' '.join(f'--{e}' for e in extra)
    cmd = f'{cmd_cd} && {cmd_call}'

    try:
        LOGGER.info('Evaluating reports with MIRQI...')
        LOGGER.info('Calling %s', cmd_call)
        subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        LOGGER.error('MIRQI script failed: %s', e.stderr)
        raise

    # Read MIRQI output # contains reports, "graph" and scores
    out_df = pd.read_csv(OUTPUT_PATH)

    # Merge with original DF
    target_cols = [
        col for col in out_df.columns
        if col.startswith('MIRQI') or col.startswith('attributes')
    ]
    df = df.merge(out_df[target_cols], left_index=True, right_index=True)
    df.fillna('', inplace=True)

    return df


def _calculate_metrics_dict(df):
    """Averages metrics for a dataset."""
    all_metrics = {}

    metric_names = [col for col in df.columns if col.startswith('MIRQI')]

    for dataset_type in set(df['dataset_type']):
        sub_df = df[df['dataset_type'] == dataset_type]

        averages_in_subdf = {
            metric_key: np.mean(sub_df[metric_key])
            for metric_key in metric_names
        }

        all_metrics[dataset_type] = averages_in_subdf

    return all_metrics


@timeit_main
def evaluate_run(run_name,
                 debug=True,
                 override=False,
                 max_samples=None,
                 free=False,
                 quiet=False,
                 extra=None,
                 ):
    """Evaluates a run with the MIRQI metric."""
    # Folder containing run results
    results_folder = get_results_folder(run_name, task='rg', debug=debug)

    # Output file at the end of this process
    suffix = 'free' if free else 'notfree'
    labeled_output_path = os.path.join(results_folder, f'outputs-mirqi-{suffix}.csv')

    if not override and os.path.isfile(labeled_output_path):
        LOGGER.info('Skipping calculation, already calculated: %s', run_name)

        df = pd.read_csv(labeled_output_path)
    else:
        # Read outputs
        df = load_rg_outputs(run_name, debug=debug, free=free)

        if df is None:
            LOGGER.error('Need to compute outputs for run first: %s', run_name)
            return

        _n_distinct_epochs = set(df['epoch'])
        if len(_n_distinct_epochs) != 1:
            raise NotImplementedError('Only works for one epoch, found: ', _n_distinct_epochs)

        # Debugging purposes
        if max_samples is not None:
            df = df.head(max_samples)
            df.reset_index(inplace=True, drop=True)

        # Compute MIRQI metrics, plus graph for generated text
        df = _apply_mirqi_to_df(df, extra=extra)

        # Save to file, to avoid heavy recalculations
        df.to_csv(labeled_output_path, index=False)

    # Compute MIRQI v2
    attributes_gt = _attributes_to_list(df['attributes-gt'])
    attributes_gen = _attributes_to_list(df['attributes-gen'])
    scores_v2 = MIRQI_v2(attributes_gt, attributes_gen)

    # Add to dataframe
    df = df.assign(**scores_v2)

    # Compute metrics
    metrics = _calculate_metrics_dict(df)

    # Save metrics to file
    mirqi_metrics_path = os.path.join(results_folder, f'mirqi-metrics-{suffix}.json')
    with open(mirqi_metrics_path, 'w') as f:
        json.dump(metrics, f)
    LOGGER.info('Saved to file: %s', mirqi_metrics_path)

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
                        help='Do not print final metrics to stdout')
    parser.add_argument('--extra', type=str, nargs='+',
                        help='Arguments passed to MIRQI evaluate script')

    args = parser.parse_args()

    args.debug = not args.no_debug

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    evaluate_run(ARGS.run_name,
                 debug=ARGS.debug,
                 override=ARGS.override,
                 max_samples=ARGS.max_samples,
                 free=ARGS.free,
                 quiet=ARGS.quiet,
                 extra=ARGS.extra,
                 )
