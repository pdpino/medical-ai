import os
import json
import argparse
import logging
import pprint
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prf1s, roc_auc_score, accuracy_score

from medai.datasets.common import CHEXPERT_LABELS
from medai.metrics import load_rg_outputs
from medai.metrics.report_generation import chexpert
from medai.utils.files import get_results_folder
from medai.utils import timeit_main, config_logging, get_timestamp, RunId


LOGGER = logging.getLogger('medai.rg.eval.chexpert')


def _calculate_metrics(df):
    """Calculate metrics for a set of reports."""
    labels_gt = chexpert.labels_with_suffix('gt')
    labels_gen = chexpert.labels_with_suffix('gen')

    ground_truth = df[labels_gt].to_numpy()
    generated = df[labels_gen].to_numpy()

    acc = np.array([
        accuracy_score(ground_truth[:, i], generated[:, i])
        for i in range(len(CHEXPERT_LABELS))
    ])

    precision, recall, f1, _ = prf1s(ground_truth, generated, zero_division=0)

    try:
        roc_auc = roc_auc_score(ground_truth, generated, average=None)
    except ValueError as e:
        LOGGER.warning(e)
        roc_auc = np.array([-1]*len(CHEXPERT_LABELS))

    return acc, precision, recall, f1, roc_auc


def _calculate_metrics_dict(df):
    """Calculates metrics for all dataset_types (train, val, test)."""
    all_metrics = {}

    ignore_no_finding_mask = np.zeros(len(CHEXPERT_LABELS))
    no_finding_idx = CHEXPERT_LABELS.index('No Finding')
    ignore_no_finding_mask[no_finding_idx] = 1

    def _add_to_results(metrics, array, prefix):
        # Add mean value to dict
        metrics[prefix] = array.mean()

        # Add mean value without NF
        metrics[f'{prefix}-woNF'] = np.ma.array(array, mask=ignore_no_finding_mask).mean()

        # Add values for each label
        array = array.tolist() # Avoid numpy-not-serializable issues
        for label, value in zip(CHEXPERT_LABELS, array):
            metrics[f'{prefix}-{label}'] = value

    for dataset_type in set(df['dataset_type']):
        sub_df = df[df['dataset_type'] == dataset_type]

        acc, precision, recall, f1, roc_auc = _calculate_metrics(sub_df)

        metrics = {}
        _add_to_results(metrics, acc, 'acc')
        _add_to_results(metrics, precision, 'prec')
        _add_to_results(metrics, recall, 'recall')
        _add_to_results(metrics, f1, 'f1')
        _add_to_results(metrics, roc_auc, 'roc_auc')

        all_metrics[dataset_type] = metrics

    return all_metrics


def _find_dataset_name(run_name):
    if 'mini-mimic' in run_name:
        return 'mini-mimic'
    if '_mimic' in run_name:
        return 'mimic-cxr'
    return 'iu-x-ray'


@timeit_main(LOGGER)
def evaluate_run(run_id,
                 override=False,
                 max_samples=None,
                 free=False,
                 quiet=False,
                 ):
    """Evaluates a run with the Chexpert-labeler."""
    # Folder containing run results
    results_folder = get_results_folder(run_id)

    # Output file at the end of this process
    suffix = 'free' if free else 'notfree'
    labeled_output_path = os.path.join(results_folder, f'outputs-labeled-{suffix}.csv')

    if not override and os.path.isfile(labeled_output_path):
        LOGGER.info('Skipping run, already calculated: %s', run_id)
        return

    # Read outputs
    df = load_rg_outputs(run_id, free=free)

    if df is None:
        LOGGER.error('Need to compute outputs for run first: %s', run_id)
        return

    n_samples = len(df)
    LOGGER.info('%d samples found in outputs, free=%s', n_samples, free)

    _n_distinct_epochs = set(df['epoch'])
    if len(_n_distinct_epochs) != 1:
        LOGGER.error('Only works for one epoch, found: %d', _n_distinct_epochs)
        return

    # Debugging purposes
    if max_samples is not None:
        df = df.head(max_samples)
        df.reset_index(inplace=True, drop=True)

        n_samples = len(df)
        LOGGER.info('Only using max_samples = %d', n_samples)

    # Get dataset_name
    dataset_name = _find_dataset_name(run_id.full_name)

    # Compute labels for both GT and generated
    caller_id = f'{run_id.short_name}_eval{get_timestamp()}'
    df = chexpert.apply_labeler_to_df(df, caller_id=caller_id, dataset_name=dataset_name)

    if len(df) != n_samples:
        LOGGER.error(
            'Internal error: n_samples does not match, initial=%d vs final=%d',
            n_samples, len(df),
        )
        return

    # Save to file, to avoid heavy recalculations
    df.to_csv(labeled_output_path, index=False)

    # Calculate metrics over train, val and test
    metrics = _calculate_metrics_dict(df)

    # Save metrics to file
    chexpert_metrics_path = os.path.join(results_folder, f'chexpert-metrics-{suffix}.json')
    with open(chexpert_metrics_path, 'w') as f:
        json.dump(metrics, f)
    LOGGER.info('Saved to file: %s', chexpert_metrics_path)

    if not quiet:
        LOGGER.info(pprint.pformat(metrics))


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
    ARGS = parse_args()

    config_logging()

    evaluate_run(RunId(ARGS.run_name, ARGS.debug, 'rg'),
                 override=ARGS.override,
                 max_samples=ARGS.max_samples,
                 free=ARGS.free,
                 quiet=ARGS.quiet,
                 )
