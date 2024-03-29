import os
import json
import argparse
import logging
import numpy as np
import pandas as pd

from medai.datasets.common import CHEXPERT_LABELS
from medai.metrics.report_generation import (
    chexpert,
    print_rg_metrics,
    build_suffix,
)
from medai.metrics.report_generation.writer import (
    load_rg_outputs,
    get_best_outputs_info,
)
from medai.utils.files import get_results_folder
from medai.utils import (
    timeit_main,
    config_logging,
    RunId,
    parsers,
)

LOGGER = logging.getLogger('medai.rg.eval.chexpert')


def _non_null_average(array):
    non_null_values = array[array != -1]
    if len(non_null_values) == 0:
        return 0
    return non_null_values.mean().item()


def _calculate_metrics_for_splits(df):
    """Calculates metrics for all dataset_types (train, val, test)."""
    all_metrics = {}

    ignore_no_finding_mask = np.zeros(len(CHEXPERT_LABELS))
    no_finding_idx = CHEXPERT_LABELS.index('No Finding')
    ignore_no_finding_mask[no_finding_idx] = 1

    def _add_to_results(metrics, array, prefix):
        # Add mean value to dict
        metrics[prefix] = _non_null_average(array)

        # Add mean value without NF
        array_woNF = np.ma.array(array, mask=ignore_no_finding_mask)
        metrics[f'{prefix}-woNF'] = _non_null_average(array_woNF)

        # Add values for each label
        array = array.tolist() # Avoid numpy-not-serializable issues
        for label, value in zip(CHEXPERT_LABELS, array):
            metrics[f'{prefix}-{label}'] = value

    LABELS_GT = chexpert.labels_with_suffix('gt')
    LABELS_GEN = chexpert.labels_with_suffix('gen')

    for dataset_type in set(df['dataset_type']):
        sub_df = df[df['dataset_type'] == dataset_type]

        ground_truth = sub_df[LABELS_GT].to_numpy()
        generated = sub_df[LABELS_GEN].to_numpy()
        acc, precision, recall, f1, roc_auc, pr_auc = chexpert.calculate_metrics(
            ground_truth, generated,
        )

        metrics = {}
        _add_to_results(metrics, acc, 'acc')
        _add_to_results(metrics, precision, 'prec')
        _add_to_results(metrics, recall, 'recall')
        _add_to_results(metrics, f1, 'f1')
        _add_to_results(metrics, roc_auc, 'roc_auc')
        _add_to_results(metrics, pr_auc, 'pr_auc')

        all_metrics[dataset_type] = metrics

    return all_metrics


@timeit_main(LOGGER, sep='-', sep_times=50)
def evaluate_run(run_id,
                 override_outputs=False,
                 max_samples=None,
                 free=False,
                 best=None,
                 beam_size=0,
                 quiet=False,
                 batches=None,
                 ):
    """Evaluates a run with the Chexpert-labeler."""
    # Folder containing run results
    results_folder = get_results_folder(run_id)

    # Output file at the end of this process
    suffix = build_suffix(free, best, beam_size)
    labeled_output_path = os.path.join(results_folder, f'outputs-labeled-{suffix}.csv')

    if not override_outputs and os.path.isfile(labeled_output_path):
        LOGGER.info('Outputs already calculated: %s', suffix)
        df = pd.read_csv(labeled_output_path)
    else:
        # Read outputs
        df = load_rg_outputs(run_id, free=free, best=best, beam_size=beam_size)

        if df is None:
            LOGGER.error('Need to compute outputs for run first: %s', run_id)
            return

        n_samples = len(df)
        LOGGER.info('%s samples found in outputs, suffix=%s', f'{n_samples:,}', suffix)

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
        dataset_name = run_id.get_dataset_name()

        # Compute labels for both GT and generated
        caller_id = f'runid-{run_id.short_name}_suffix-{suffix}'
        df = chexpert.apply_labeler_to_df(
            df, caller_id=caller_id, dataset_name=dataset_name, batches=batches,
        )

        if len(df) != n_samples:
            LOGGER.error(
                'Internal error: n_samples does not match, initial=%d vs final=%d',
                n_samples, len(df),
            )
            return

        # Save to file, to avoid heavy recalculations
        df.to_csv(labeled_output_path, index=False)

    # Calculate metrics over train, val and test
    metrics = _calculate_metrics_for_splits(df)

    # Save metrics to file
    chexpert_metrics_path = os.path.join(results_folder, f'chexpert-metrics-{suffix}.json')
    with open(chexpert_metrics_path, 'w') as f:
        json.dump(metrics, f)
    LOGGER.info('Saved to file: %s', chexpert_metrics_path)

    if not quiet:
        print_rg_metrics(metrics, ignore=CHEXPERT_LABELS)

    LOGGER.info('Finished run: %s', run_id)


@timeit_main(LOGGER)
def evaluate_run_with_free_values(run_id, free_values, only_best, only_beam, **kwargs):
    LOGGER.info('Evaluating chexpert %s', run_id)

    chosen, leftout = get_best_outputs_info(run_id, free_values, only_best, only_beam)

    LOGGER.info('\tChosen suffixes: %s, leftout: %s', chosen, leftout)

    for free_value, best_metric, beam_size in chosen:
        evaluate_run(
            run_id,
            free=free_value,
            best=best_metric,
            beam_size=beam_size,
            **kwargs,
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run name to evaluate')
    parser.add_argument('--override-outputs', action='store_true',
                        help='Whether to override previous calculation')
    parser.add_argument('--no-debug', action='store_true',
                        help='If present, the run is non-debug')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug: use a max amount of samples')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics to stdout')
    parser.add_argument('--batches', type=int, default=None,
                        help='Process the reports in batches')
    parser.add_argument('--only-best', type=str, nargs='*',
                        help='Only eval best by certain metrics')
    parser.add_argument('--only-beam', type=int, nargs='*', default=None,
                        help='Eval only in some beam sizes')

    parsers.add_args_free_values(parser)

    args = parser.parse_args()

    parsers.build_args_free_values_(args, parser)

    args.debug = not args.no_debug

    return args


if __name__ == '__main__':
    config_logging()

    ARGS = parse_args()

    evaluate_run_with_free_values(
        RunId(ARGS.run_name, ARGS.debug, 'rg'),
        free_values=ARGS.free_values,
        only_best=ARGS.only_best,
        only_beam=ARGS.only_beam,
        override_outputs=ARGS.override_outputs,
        max_samples=ARGS.max_samples,
        quiet=ARGS.quiet,
        batches=ARGS.batches,
    )
