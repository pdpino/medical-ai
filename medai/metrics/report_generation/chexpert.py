import csv
import os
import subprocess
import logging
import pandas as pd
import numpy as np

from medai.metrics.report_generation.labeler import (
    HolisticLabeler,
    CacheLookupLabeler,
    NBatchesLabeler,
    AvoidDuplicatedLabeler,
)
from medai.datasets.common import CHEXPERT_DISEASES
from medai.datasets.iu_xray import DATASET_DIR as IU_DIR
from medai.datasets.mimic_cxr import DATASET_DIR as MIMIC_DIR
from medai.utils import TMP_DIR
from medai.utils.lock import with_lock

_NEGBIO_PATH_KEY = 'NEGBIO_PATH'
assert _NEGBIO_PATH_KEY in os.environ, f'You must export {_NEGBIO_PATH_KEY}'

NEGBIO_PATH = os.environ[_NEGBIO_PATH_KEY] # '~/chexpert/NegBio'
CHEXPERT_FOLDER = '~/chexpert/chexpert-labeler'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

TMP_FOLDER = os.path.join(TMP_DIR, 'chexpert-labeler')


LOGGER = logging.getLogger(__name__)

def labels_with_suffix(suffix):
    """Returns the chexpert labels with a suffix appended to each."""
    if not suffix:
        return list(CHEXPERT_DISEASES)
    return [f'{label}-{suffix}' for label in CHEXPERT_DISEASES]


def _load_gt_df(dataset_name, fill_uncertain=1, fill_empty=0):
    gt_labels_filepath = os.path.join(
        MIMIC_DIR if 'mimic' in dataset_name else IU_DIR,
        'reports', 'reports_with_chexpert_labels.csv',
    )
    if not os.path.isfile(gt_labels_filepath):
        raise Exception('Ground truth labels not found: ', gt_labels_filepath)

    # Load CSV
    gt_with_labels = pd.read_csv(gt_labels_filepath)
    gt_with_labels = gt_with_labels.replace({
        -2: fill_empty,
        -1: fill_uncertain,
    })
    return gt_with_labels


def _fetch_gt_labels(target_df, gt_with_labels):
    """Given a target_df and a gt_df, get the chexpert-labels for the target reports."""
    # Assure it has all necessary reports
    target_reports = set(target_df['filename'])
    saved_reports = set(gt_with_labels['filename'])
    if not target_reports.issubset(saved_reports):
        missing = saved_reports.difference(target_reports)
        raise Exception(f'GT missing {len(missing)} reports')

    # Merge on filenames
    merged = target_df.merge(gt_with_labels, how='left', on='filename')

    assert len(merged) == len(target_df), \
        f'Size mismatch: {len(merged)} vs {len(target_df)}'

    # Return only np.array with labels
    return merged[CHEXPERT_DISEASES].to_numpy().astype(np.int8)


def _get_custom_env():
    """Adds a necessary environment variable to run the labeler."""
    custom_env = os.environ.copy()
    prev = custom_env.get('PYTHONPATH', '')
    custom_env['PYTHONPATH'] = f'{NEGBIO_PATH}:{prev}'

    return custom_env


# TODO: make this public?? is used outside of here
def _concat_df_matrix(df, results, suffix=None):
    """Concats a DF with a matrix."""
    labels = labels_with_suffix(suffix)
    return pd.concat([df, pd.DataFrame(results, columns=labels)],
                     axis=1, join='inner')


# TODO: use a more appropiate name
@with_lock(TMP_FOLDER, 'caller_id', raise_error=True)
def apply_labeler_to_column(reports,
                            fill_empty=-2, fill_uncertain=None,
                            quiet=False, caller_id='main'):
    """Apply chexpert-labeler to a column of a dataframe."""
    # Pass reports to a CSV
    reports_only = pd.DataFrame(reports)

    # Tmp folder can be removed afterwards
    os.makedirs(TMP_FOLDER, exist_ok=True)

    # Create input file
    input_path = os.path.join(TMP_FOLDER, f'reports-input_{caller_id}.csv')
    reports_only.to_csv(input_path, header=False, index=False, quoting=csv.QUOTE_ALL)

    # Call chexpert-labeler
    output_path = os.path.join(TMP_FOLDER, f'reports-output_{caller_id}.csv')
    cmd_cd = f'cd {CHEXPERT_FOLDER}'
    cmd_call = f'{CHEXPERT_PYTHON} label.py --reports_path {input_path} --output_path {output_path}'
    cmd = f'{cmd_cd} && {cmd_call}'

    try:
        if not quiet:
            LOGGER.info('Labelling %s reports...', f'{len(reports_only):,}')
        subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_get_custom_env(),
        )
    except subprocess.CalledProcessError as e:
        LOGGER.error('Labeler failed, stdout and stderr:')
        LOGGER.error(e.stdout)
        LOGGER.error(e.stderr)
        raise

    # Read chexpert-labeler output
    out_df = pd.read_csv(output_path)

    if fill_empty is not None:
        # Mark nan as -2
        out_df = out_df.fillna(fill_empty)

    if fill_uncertain is not None and fill_uncertain != -1:
        # -1 are Uncertain, mark as positive
        out_df = out_df.replace(-1, fill_uncertain)

    return out_df[CHEXPERT_DISEASES].to_numpy().astype(np.int8)


def apply_labeler_to_df(df, caller_id='main', batches=None, dataset_name='iu-x-ray'):
    """Calculates chexpert labels for a set of GT and generated reports.

    Args:
        df -- DataFrame with columns 'filename', 'generated'
    """
    # Load labels for ground truth
    gt_df = _load_gt_df(dataset_name, fill_empty=0, fill_uncertain=1)
    ground_truth = _fetch_gt_labels(df, gt_df)

    # Concat GT in main dataframe
    df = _concat_df_matrix(df, ground_truth, 'gt')

    # Create labeler
    labeler = ChexpertLabeler(fill_empty=0, fill_uncertain=1, caller_id=caller_id)
    labeler = CacheLookupLabeler(labeler, gt_df)
    labeler = NBatchesLabeler(labeler, batches)
    labeler = AvoidDuplicatedLabeler(labeler)

    # Label generated reports
    gen_reports = df['generated'].tolist()
    generated = labeler(gen_reports)
    df = _concat_df_matrix(df, generated, 'gen')
    return df


class ChexpertLabeler(HolisticLabeler):
    diseases = CHEXPERT_DISEASES

    def __init__(self, **kwargs):
        super().__init__(None)

        self.kwargs = kwargs

    def forward(self, reports):
        labels = apply_labeler_to_column(reports, **self.kwargs)

        return labels
