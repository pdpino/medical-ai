import csv
import os
import json
import subprocess
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf1s, roc_auc_score, accuracy_score
from pprint import pprint

from medai.datasets.common import CHEXPERT_LABELS
from medai.datasets.iu_xray import DATASET_DIR
from medai.utils import TMP_DIR
from medai.utils.files import get_results_folder
from medai.metrics import load_rg_outputs

_NEGBIO_PATH_KEY = 'NEGBIO_PATH'
assert _NEGBIO_PATH_KEY in os.environ, f'You must export {_NEGBIO_PATH_KEY}'

NEGBIO_PATH = os.environ[_NEGBIO_PATH_KEY] # '~/chexpert/NegBio'
CHEXPERT_FOLDER = '~/chexpert/chexpert-labeler'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

TMP_FOLDER = os.path.join(TMP_DIR, 'chexpert-labeler')
GT_LABELS_FILEPATH = os.path.join(DATASET_DIR, 'reports', 'reports_with_chexpert_labels.csv')


def labels_with_suffix(suffix):
    """Returns the chexpert labels with a suffix appended to each."""
    if not suffix:
        return list(CHEXPERT_LABELS)
    return [f'{label}-{suffix}' for label in CHEXPERT_LABELS]


def _load_gt_labels(df):
    if not os.path.isfile(GT_LABELS_FILEPATH):
        raise Exception('Ground truth labels not found: ', GT_LABELS_FILEPATH)

    # Load CSV
    gt_with_labels = pd.read_csv(GT_LABELS_FILEPATH, index_col=0)
    gt_with_labels.replace((-1, -2), 0, inplace=True)

    # Assure it has all necessary reports
    target_reports = set(df['filename'])
    saved_reports = set(gt_with_labels['filename'])
    if not target_reports.issubset(saved_reports):
        # import pdb
        # pdb.set_trace()
        missing = saved_reports.difference(target_reports)
        raise Exception(f'GT missing {len(missing)} reports')

    # Merge on filenames
    merged = df.merge(gt_with_labels, how='left', on='filename')

    # Return only np.array with labels
    labels = CHEXPERT_LABELS
    return merged[labels].to_numpy()


def _get_custom_env():
    """Adds a necessary environment variable to run the labeler."""
    custom_env = os.environ.copy()
    prev = custom_env.get('PYTHONPATH', '')
    custom_env['PYTHONPATH'] = f'{NEGBIO_PATH}:{prev}'

    return custom_env


def _concat_df_matrix(df, results, suffix=None):
    """Concats a DF with a matrix."""
    labels = labels_with_suffix(suffix)
    return pd.concat([df, pd.DataFrame(results, columns=labels)],
                     axis=1, join='inner')


def apply_labeler_to_column(dataframe, column_name,
                            fill_empty=None, fill_uncertain=None,
                            quiet=False):
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
        if not quiet: print(f'Labelling {column_name}...')
        completed_process = subprocess.run(cmd, shell=True, check=True,
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env=_get_custom_env(),
                                           )
    except subprocess.CalledProcessError as e:
        print('Labeler failed, stdout and stderr:')
        print(e.stdout)
        print(e.stderr)
        raise

    # Read chexpert-labeler output
    out_df = pd.read_csv(output_path)

    if fill_empty is not None:
        # Mark nan as -2
        out_df = out_df.fillna(fill_empty)

    if fill_uncertain is not None and fill_uncertain != -1:
        # -1 are Uncertain, mark as positive
        out_df = out_df.replace(-1, fill_uncertain)

    return out_df[CHEXPERT_LABELS].to_numpy()


def apply_labeler_to_df(df):
    """Calculates chexpert labels for a set of GT and generated reports.

    Args:
        df -- DataFrame with columns 'filename', 'generated'
    """
    # Load labels for ground truth
    ground_truth = _load_gt_labels(df)

    # Calculate labels for generated
    generated = apply_labeler_to_column(df, 'generated',
                                         fill_empty=0,
                                         fill_uncertain=1)

    # Concat in main dataframe
    df = _concat_df_matrix(df, ground_truth, 'gt')
    df = _concat_df_matrix(df, generated, 'gen')

    return df
