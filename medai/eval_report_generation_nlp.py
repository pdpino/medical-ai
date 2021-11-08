## DEPRECATED
# import argparse
# import os
# import logging
# import pprint
# import pandas as pd
# import numpy as np
# from pycocoevalcap.bleu import bleu_scorer
# from pycocoevalcap.cider import cider_scorer
# from pycocoevalcap.rouge import rouge as rouge_lib

# from medai.datasets.iu_xray import DATASET_DIR
# from medai.metrics import save_results, load_rg_outputs
# from medai.utils import timeit_main, RunId

# LOGGER = logging.getLogger('medai.rg.eval.nlp')

# class RougeLScorer:
#     # TODO: reuse this in rouge.py?? is copied
#     def __init__(self):
#         self._n_samples = 0
#         self._current_score = 0

#         self._scorer = rouge_lib.Rouge()

#     def update(self, generated, gt):
#         self._current_score += self._scorer.calc_score([generated], [gt])
#         self._n_samples += 1

#     def compute(self):
#         return self._current_score / self._n_samples if self._n_samples > 0 else 0.0


# def _split_reports_normal_abnormal():
#     fpath = os.path.join(DATASET_DIR, 'reports', 'reports_with_chexpert_labels.csv')
#     reports_gt = pd.read_csv(fpath)
#     reports_gt = reports_gt.replace({-1: 1, -2: 0})
#     reports_gt = reports_gt[['filename', 'No Finding']]

#     def _get_filenames_with_finding(value):
#         df = reports_gt[reports_gt['No Finding'] == value]
#         return set(df['filename'])

#     normal_reports = _get_filenames_with_finding(1)
#     abnormal_reports = _get_filenames_with_finding(0)

#     return normal_reports, abnormal_reports


# @timeit_main
# def run_evaluation(run_id,
#                    free=False,
#                    quiet=False,
#                    ):
#     # Split by normal and abnormal
#     normal_reports, abnormal_reports = _split_reports_normal_abnormal()
#     subsets = [
#         ('normal', normal_reports),
#         ('abnormal', abnormal_reports),
#     ]

#     # Read outputted reports
#     df = load_rg_outputs(run_id, free=free)

#     if df is None:
#         LOGGER.error('Need to compute outputs for run first: %s', run_id)
#         return

#     # Calculate metrics
#     metrics = {}


#     for dataset_type in set(df['dataset_type']):
#         df_by_dataset_type = df[df['dataset_type'] == dataset_type]

#         for normality_label, reports_list in subsets:
#             sub_df = df_by_dataset_type
#             sub_df = sub_df[sub_df['filename'].isin(reports_list)]

#             subset_name = f'{dataset_type}-{normality_label}'
#             if len(sub_df) == 0:
#                 LOGGER.warning('Empty subset %s', subset_name)
#                 continue

#             LOGGER.info('Evaluating for subset %s...', subset_name)

#             # Init scorers
#             bleu = bleu_scorer.BleuScorer(n=4)
#             cider = cider_scorer.CiderScorer(n=4)
#             rouge = RougeLScorer()

#             for _, row in sub_df.iterrows():
#                 gt = row['ground_truth']
#                 generated = row['generated']

#                 if not isinstance(generated, str) or not isinstance(gt, str):
#                     LOGGER.warning('\tSkipping non-str sample: %s', row)
#                     continue

#                 bleu += (generated, [gt])
#                 cider += (generated, [gt])
#                 rouge.update(generated, gt)

#             bleu_scores, _ = bleu.compute_score()
#             cider_score, _ = cider.compute_score()
#             rouge_score = rouge.compute()

#             metrics[subset_name] = {
#                 'bleu1': bleu_scores[0],
#                 'bleu2': bleu_scores[1],
#                 'bleu3': bleu_scores[2],
#                 'bleu4': bleu_scores[3],
#                 'bleu': np.mean(bleu_scores),
#                 'rougeL': rouge_score,
#                 'ciderD': cider_score,
#             }

#     # Save metrics to file
#     suffix = 'free' if free else 'notfree'
#     # DEPRECATED: use build_suffix!
#     # save_results(metrics, run_id, suffix=suffix)

#     if not quiet:
#         LOGGER.info(pprint.pformat(metrics))


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--run-name', type=str, default=None,
#                         help='Resume from a previous run')
#     parser.add_argument('--no-debug', action='store_true',
#                         help='If is a non-debugging run')
#     parser.add_argument('--free', action='store_true',
#                         help='Use free outputs or not')
#     parser.add_argument('--quiet', action='store_true',
#                         help='Do not print metrics to stdout')

#     args = parser.parse_args()

#     return args


# if __name__ == '__main__':
#     ARGS = parse_args()

#     run_evaluation(RunId(ARGS.run_name, not ARGS.no_debug, 'rg').resolve(),
#                    free=ARGS.free,
#                    quiet=ARGS.quiet,
#                    )
