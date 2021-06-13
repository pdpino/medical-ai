import os
import json
import argparse
import logging
from pprint import pprint

import pandas as pd
import numpy as np

from medai.utils import timeit_main, RunId, config_logging, parsers
from medai.utils.files import get_results_folder
from medai.metrics import load_rg_outputs
from medai.metrics.report_generation import mirqi

LOGGER = logging.getLogger('medai.rg.eval.mirqi')

# METRICS_KEYS = ['MIRQI-r', 'MIRQI-p', 'MIRQI-f'] # Outputed by MIRQI script

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
        ] if isinstance(sample_attrs, str) else []
        for sample_attrs in all_attrs
    ]


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


@timeit_main(LOGGER, sep='-', sep_times=50)
def evaluate_run(run_id,
                 override=False,
                 max_samples=None,
                 free=False,
                 quiet=False,
                 ):
    """Evaluates a run with the MIRQI metric."""
    # Folder containing run results
    results_folder = get_results_folder(run_id)

    # Output file at the end of this process
    suffix = 'free' if free else 'notfree'
    labeled_output_path = os.path.join(results_folder, f'outputs-labeled-mirqi-{suffix}.csv')
    # The ones "output-mirqi-suffix.csv" are deprecated!!

    if not override and os.path.isfile(labeled_output_path):
        LOGGER.info('Skipping calculation, already calculated: %s', run_id)

        df = pd.read_csv(labeled_output_path)
    else:
        # Read outputs
        df = load_rg_outputs(run_id, free=free)

        if df is None:
            LOGGER.error('Need to compute outputs for run first: %s', run_id)
            return

        _n_distinct_epochs = set(df['epoch'])
        if len(_n_distinct_epochs) != 1:
            raise NotImplementedError(f'Only works for one epoch, found: {_n_distinct_epochs}')

        # Debugging purposes
        if max_samples is not None:
            df = df.head(max_samples)
            df.reset_index(inplace=True, drop=True)

        # Compute MIRQI metrics, plus graph for generated text
        df = mirqi.apply_mirqi_to_df(
            df,
            timestamp=run_id.short_name,
            dataset_name=run_id.get_dataset_name(),
        )

        # Save to file, to avoid heavy recalculations
        df.to_csv(labeled_output_path, index=False)

    # Compute MIRQI v2
    attributes_gt = _attributes_to_list(df['attributes-gt'])
    attributes_gen = _attributes_to_list(df['attributes-gen'])
    scores_v1 = mirqi.MIRQI(attributes_gt, attributes_gen)
    scores_v2 = mirqi.MIRQI_v2(attributes_gt, attributes_gen)

    # Add to dataframe
    df = df.assign(**scores_v1)
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

@timeit_main(LOGGER)
def evaluate_run_with_free_values(run_id, free_values, **kwargs):
    LOGGER.info('Evaluating MIRQI %s', run_id)

    for free_value in free_values:
        evaluate_run(
            run_id,
            free=free_value,
            **kwargs
        )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run name to evaluate')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override previous calculation')
    parser.add_argument('--no-debug', action='store_true',
                        help='If present, the run is non-debug')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Debug: use a max amount of samples')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print final metrics to stdout')

    parsers.add_args_free_values(parser)

    args = parser.parse_args()

    parsers.build_args_free_values_(args, parser)

    args.debug = not args.no_debug

    return args


if __name__ == '__main__':
    config_logging()

    ARGS = parse_args()

    evaluate_run_with_free_values(
        RunId(ARGS.run_name, ARGS.debug, 'rg').resolve(),
        override=ARGS.override,
        max_samples=ARGS.max_samples,
        free_values=ARGS.free_values,
        quiet=ARGS.quiet,
    )
