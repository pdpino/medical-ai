import os
import json
import argparse
import logging
from pprint import pprint

import pandas as pd
import numpy as np

from medai.utils import timeit_main, RunId, config_logging, parsers
from medai.utils.files import get_results_folder
from medai.metrics.report_generation.writer import (
    load_rg_outputs,
    get_best_outputs_info,
)
from medai.metrics.report_generation import build_suffix, mirqi

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


def dict_with_suffix(scores, suffix):
    """Given a dict of MIRQI scores, add a suffix to the keys."""
    new_name = f'MIRQI-{suffix}'
    return { k.replace('MIRQI', new_name): v for k, v in scores.items() }

@timeit_main(LOGGER, sep='-', sep_times=50)
def evaluate_run(run_id,
                 override=False,
                 max_samples=None,
                 free=False,
                 best=None,
                 beam_size=0,
                 quiet=False,
                 ):
    """Evaluates a run with the MIRQI metric."""
    # Folder containing run results
    results_folder = get_results_folder(run_id)

    # Output file at the end of this process
    suffix = build_suffix(free, best, beam_size)
    labeled_output_path = os.path.join(results_folder, f'outputs-labeled-mirqi-{suffix}.csv')
    # The ones "output-mirqi-suffix.csv" are deprecated!!

    if not override and os.path.isfile(labeled_output_path):
        LOGGER.info('Skipping MIRQI labelling, already calculated: %s', run_id)

        df = pd.read_csv(labeled_output_path)
    else:
        # Read outputs
        df = load_rg_outputs(run_id, free=free, best=best, beam_size=beam_size)

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
            caller_id=f'runid-{run_id.short_name}_suffix-{suffix}',
            dataset_name=run_id.get_dataset_name(),
        )

        # Save to file, to avoid heavy recalculations
        df.to_csv(labeled_output_path, index=False)

    # Compute MIRQI v1 and v2
    attributes_gt = _attributes_to_list(df['attributes-gt'])
    attributes_gen = _attributes_to_list(df['attributes-gen'])
    scores_v1 = mirqi.MIRQI(attributes_gt, attributes_gen)
    scores_v2 = mirqi.MIRQI_v2(attributes_gt, attributes_gen)

    scores_v3 = mirqi.MIRQI(attributes_gt, attributes_gen, pos_weight=1, attribute_weight=0)
    scores_v3 = dict_with_suffix(scores_v3, 'v3-clean')

    scores_v4 = mirqi.MIRQI(attributes_gt, attributes_gen, pos_weight=1)
    scores_v4 = dict_with_suffix(scores_v4, 'v4-pos')

    ###
    # v5-game: Remove negative findings when there is one positive
    attributes_gamed_v5 = []
    for report_entry in attributes_gen:
        # neg_findings = [finding for finding in report_entry if finding[2] == 'NEGATIVE']
        pos_findings = [finding for finding in report_entry if finding[2] != 'NEGATIVE']
        if len(pos_findings) > 0:
            attributes_gamed_v5.append(pos_findings) # keep only pos-findings
        else:
            attributes_gamed_v5.append(report_entry)

    scores_v5 = mirqi.MIRQI(attributes_gt, attributes_gamed_v5)
    scores_v5 = dict_with_suffix(scores_v5, 'v5-game')

    ###
    # v6-game: Fill with negative findings
    attributes_gamed_v6 = []
    ALL_ABNS = ["Other Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
              "Lung Lesion", "Airspace Opacity", "Edema", "Consolidation",
              "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
              "Pleural Other", "Fracture", "Support Devices",
              "Emphysema", "Cicatrix", "Hernia", "Calcinosis", "Airspace Disease",
              "Hypoinflation"]
    for report_entry in attributes_gen:
        abns_mentioned = set(finding[1] for finding in report_entry)
        gamed_entry = [list(finding) for finding in report_entry] # copy
        for abn in ALL_ABNS:
            if abn not in abns_mentioned:
                gamed_entry.append([abn.lower(), abn, 'NEGATIVE', ''])

        attributes_gamed_v6.append(gamed_entry)

    scores_v6 = mirqi.MIRQI(attributes_gt, attributes_gamed_v6)
    scores_v6 = dict_with_suffix(scores_v6, 'v6-game')

    scores_v7 = mirqi.MIRQI(attributes_gt, attributes_gen, attribute_weight=1)
    scores_v7 = dict_with_suffix(scores_v7, 'v7-attr-only')

    # Add to dataframe
    df = df.assign(**scores_v1)
    df = df.assign(**scores_v2)
    df = df.assign(**scores_v3)
    df = df.assign(**scores_v4)
    df = df.assign(**scores_v5)
    df = df.assign(**scores_v6)
    df = df.assign(**scores_v7)

    # Compute metrics
    metrics = _calculate_metrics_dict(df)

    # Save metrics to file
    mirqi_metrics_path = os.path.join(results_folder, f'mirqi-metrics-{suffix}.json')
    with open(mirqi_metrics_path, 'w') as f:
        json.dump(metrics, f)
    LOGGER.info('Saved to file: %s', mirqi_metrics_path)

    if not quiet:
        pprint(metrics['test'])

@timeit_main(LOGGER)
def evaluate_run_with_free_values(run_id, free_values, only_best, only_beam, **kwargs):
    LOGGER.info('Evaluating MIRQI %s', run_id)

    chosen, leftout = get_best_outputs_info(run_id, free_values, only_best, only_beam)

    LOGGER.info('\tChosen suffixes: %s, leftout: %s', chosen, leftout)

    for free_value, best, beam_size in chosen:
        evaluate_run(
            run_id,
            free=free_value,
            best=best,
            beam_size=beam_size,
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
        RunId(ARGS.run_name, ARGS.debug, 'rg').resolve(),
        free_values=ARGS.free_values,
        only_best=ARGS.only_best,
        only_beam=ARGS.only_beam,
        override=ARGS.override,
        max_samples=ARGS.max_samples,
        quiet=ARGS.quiet,
    )
