import csv
import os
import subprocess
import logging
import pandas as pd
import numpy as np

from medai.metrics.report_generation.labeler.utils import (
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
        raise FileNotFoundError('Ground truth labels not found: ', gt_labels_filepath)

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
        raise AssertionError(f'GT missing {len(missing)} reports')

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
    n_in = len(df)
    labels = labels_with_suffix(suffix)
    result = pd.concat(
        [df, pd.DataFrame(results, columns=labels)], axis=1, join='inner',
        # use index=df.index to avoid problems???
    )

    n_out = len(result)
    assert n_in == n_out, f'Concat failed: in={n_in} vs out={n_out}'

    return result


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

# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_fscore_support as prf1s,
    precision_recall_curve as pr_curve,
    roc_auc_score,
)

def calculate_metrics(ground_truth, generated, average=None):
    """Calculate metrics for a set of reports.

    Args:
        ground_truth -- np.array (n_samples, n_labels)
        generated -- np.array (n_samples, n_labels)
        labels -- list (defaults to CHEXPERT_DISEASES)
    """
    n_labels = ground_truth.shape[1]

    acc = np.array([
        accuracy_score(ground_truth[:, i], generated[:, i])
        for i in range(n_labels)
    ])

    precision, recall, f1, _ = prf1s(ground_truth, generated, zero_division=0, average=average)

    # Calculate ROC-AUC
    try:
        roc_auc = roc_auc_score(ground_truth, generated, average=None)
    except ValueError as e:
        # FIXME: calculate independently for each disease,
        # so if one disease fails, the other values can be computed anyway
        LOGGER.warning(e)
        roc_auc = np.array([-1] * n_labels)

    # Calculate PR-AUC
    pr_auc = []
    for i in range(n_labels):
        gt = ground_truth[:, i]
        gen = generated[:, i]

        prec_values, rec_values, unused_thresholds = pr_curve(gt, gen)
        pr = auc(rec_values, prec_values)

        if np.isnan(pr):
            LOGGER.warning('PR-auc is nan for disease index=%s', i)
            pr = -1

        pr_auc.append(pr)

    pr_auc = np.array(pr_auc)

    return acc, precision, recall, f1, roc_auc, pr_auc


class ChexpertScorer():
    # Comply with BLEUScorer API
    metric_names = ['acc', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc']
    n_metrics = 6

    def __init__(self, caller_id='chexpert-scorer'):
        self.abnormality = None

        self.labeler = ChexpertLabeler(fill_empty=0, fill_uncertain=1, caller_id=caller_id)

        self.reports_gt = []
        self.reports_gen = []

    def update(self, generated, gt):
        self.reports_gt.append(gt)
        self.reports_gen.append(generated)

    def __iadd__(self, tup):
        assert isinstance(tup, tuple) and len(tup) == 2
        gen, gt_list = tup
        assert len(gt_list) == 1
        self.update(gen, gt_list[0])
        return self

    def compute_score(self):
        if self.abnormality is None:
            raise AssertionError('cannot compute score without setting abnormality')

        all_labels = self.labeler(self.reports_gt + self.reports_gen)

        # Keep only labels of interest
        target_index = CHEXPERT_DISEASES.index(self.abnormality)
        all_labels = all_labels[:, [target_index]] # shape: (n_samples, 1)

        ground_truth = all_labels[:len(self.reports_gt), :]
        generated = all_labels[len(self.reports_gt):, :]
        # shapes: n_samples, 1

        results = calculate_metrics(ground_truth, generated, average='binary')
        # acc, precision, recall, f1, roc_auc, pr_auc = results
        results = [r.item() for r in results]

        score_per_sample = np.expand_dims((generated == ground_truth).astype(int).squeeze(), axis=0)
        # shape: (1, n_samples)

        return np.array(results), score_per_sample

    def use_cache(self, sentences_df):
        if sentences_df is not None:
            self.labeler = CacheLookupLabeler(
                self.labeler, sentences_df.replace(-1, 1).replace(-2, 0), 'sentence',
            )

        self.labeler = NBatchesLabeler(self.labeler, None)
        self.labeler = AvoidDuplicatedLabeler(self.labeler)

    def set_abnormality(self, abnormality):
        assert abnormality in CHEXPERT_DISEASES
        self.abnormality = abnormality
