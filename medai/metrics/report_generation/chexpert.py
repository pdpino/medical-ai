import csv
import os
import subprocess
import logging
import pandas as pd

from medai.datasets.common import CHEXPERT_LABELS
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
        return list(CHEXPERT_LABELS)
    return [f'{label}-{suffix}' for label in CHEXPERT_LABELS]


def _load_gt_labels(df, dataset_name):
    gt_labels_filepath = os.path.join(
        MIMIC_DIR if 'mimic' in dataset_name else IU_DIR,
        'reports', 'reports_with_chexpert_labels.csv',
    )
    if not os.path.isfile(gt_labels_filepath):
        raise Exception('Ground truth labels not found: ', gt_labels_filepath)

    # Load CSV
    gt_with_labels = pd.read_csv(gt_labels_filepath, index_col=0)
    gt_with_labels.replace(-2, 0, inplace=True)
    gt_with_labels.replace(-1, 1, inplace=True)

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


@with_lock(TMP_FOLDER, 'caller_id', raise_error=True)
def apply_labeler_to_column(dataframe, column_name,
                            fill_empty=None, fill_uncertain=None,
                            quiet=False, caller_id='main'):
    """Apply chexpert-labeler to a column of a dataframe."""
    # Grab reports
    reports_only = dataframe[column_name]

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
            LOGGER.info('Labelling %s: %s reports...', column_name, f'{len(reports_only):,}')
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

    return out_df[CHEXPERT_LABELS].to_numpy()


def apply_labeler_to_df(df, caller_id='main', avoid_duplicated=True, dataset_name='iu-x-ray'):
    """Calculates chexpert labels for a set of GT and generated reports.

    Args:
        df -- DataFrame with columns 'filename', 'generated'
    """
    # Load labels for ground truth
    ground_truth = _load_gt_labels(df, dataset_name)

    # Concat GT in main dataframe
    df = _concat_df_matrix(df, ground_truth, 'gt')

    kwargs = {
        'fill_empty': 0,
        'fill_uncertain': 1,
        'caller_id': caller_id,
    }

    if avoid_duplicated:
        # Calculate labels for unique generated
        n_samples = len(df)

        unique_reports = df['generated'].unique()
        # np.array of shape: n_unique_reports

        LOGGER.info(
            'Reduced duplicated reports: from %s to %s unique',
            f'{n_samples:,}',
            f'{len(unique_reports):,}',
        )

        df_unique = pd.DataFrame(unique_reports, columns=['gen-unique'])

        unique_generated = apply_labeler_to_column(df_unique, 'gen-unique', **kwargs)
        # shape: n_unique_reports, n_labels

        df_unique = _concat_df_matrix(df_unique, unique_generated, 'gen')

        df = df.merge(df_unique, how='inner', left_on='generated', right_on='gen-unique')

        del df['gen-unique']

        assert len(df) == n_samples, f'Merge failed amounts: final={len(df)} original={n_samples}'
    else:
        # Calculate labels for all generated
        generated = apply_labeler_to_column(df, 'generated', **kwargs)
        df = _concat_df_matrix(df, generated, 'gen')

    return df
