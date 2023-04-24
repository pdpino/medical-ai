import os
import random
import argparse
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

from medai.utils import timeit_main, config_logging
from medai.utils.files import WORKSPACE_DIR
from medai.notebooks.analyze_nlp_chexpert.nlp_in_chexpert_groups import (
    init_dataset_info,
    _SCORERS,
)

LOGGER = logging.getLogger("medai.analyze.nlp-chex-roc")


def run_roc_auc_experiment(dataset, metrics_with_index, abnormalities, tasks, k_sample, show=True):
    includes_cider = any('cider' in m for m, _ in metrics_with_index)
    dataset_info = init_dataset_info(dataset, skip_docfreq=not includes_cider)

    results = []

    for abnormality in abnormalities:
        sentences_df = dataset_info.sentences_df.replace(-2, 0).replace(-1, 1)
        grouped = sentences_df.groupby(abnormality)["sentence"].apply(
            lambda x: sorted(list(x), key=len),
        )

        for task in tasks:
            # Sample
            if task == 'recall':
                gt_universe = grouped[1]
            elif task == 'spec':
                gt_universe = grouped[0]
            else:
                gt_universe = list(dataset_info.sentences_df['sentence'])
            gen_universe = list(dataset_info.sentences_df['sentence'])

            k_effective_sample = min(len(gt_universe), len(gen_universe), k_sample)

            gt_samples = random.sample(gt_universe, k_effective_sample)
            gen_samples = random.sample(gen_universe, k_effective_sample)

            # Build gold values
            df2 = sentences_df.set_index('sentence')[abnormality]
            gold = (df2.loc[gt_samples].values == df2.loc[gen_samples].values).astype(int)

            for metric_name, metric_i in tqdm(
                metrics_with_index, disable=not show, desc=f"{abnormality} at {task}",
            ):
                ScorerClass, _ = _SCORERS[metric_name]
                if hasattr(ScorerClass, 'preload'):
                    ScorerClass.preload()

                scorer = ScorerClass()
                if metric_name == "cider-IDF":
                    # pylint: disable=assigning-non-slot
                    scorer.document_frequency = dataset_info.doc_freq
                    # Also needs to update ref_len
                    scorer.ref_len = dataset_info.log_ref_len

                # Useful for ChexpertScorer
                if hasattr(scorer, 'use_cache'):
                    scorer.use_cache(dataset_info.chexpert_cache_sentences)
                if hasattr(scorer, 'set_abnormality'):
                    scorer.set_abnormality(abnormality)

                for gt, gen in zip(gt_samples, gen_samples):
                    scorer += (gen, [gt])
                _, scores = scorer.compute_score()
                scores = np.array(scores) # shape: [n_metrics], n_samples
                if scores.ndim > 1:
                    scores = scores[metric_i]

                roc = roc_auc_score(gold, scores)

                results.append((
                    abnormality, metric_name, metric_i, k_effective_sample, task, roc,
                ))

    columns = ['abnormality', 'metric_name', 'metric_i', 'k_samples', 'task', 'roc']
    return pd.DataFrame(results, columns=columns)


@timeit_main(LOGGER)
def run_experiments(
    dataset="mimic-expert1",
    abnormalities=[],
    metrics=["bleu", "rouge", "cider-IDF"],
    k_sample=50,
    n_times=100,
    suffix="base",
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    tasks = ['full', 'recall', 'spec']

    _results_fpath = os.path.join(
        WORKSPACE_DIR,
        'report_generation',
        'nlp-controlled-corpus',
        f'roc-random-{dataset}-{suffix}.csv',
    )

    if os.path.isfile(_results_fpath):
        results = pd.read_csv(_results_fpath)
    else:
        results = pd.DataFrame()

    for _ in tqdm(range(n_times)):
        new_results = run_roc_auc_experiment(
            'mimic-expert1', metrics, abnormalities, tasks, k_sample, show=False,
        )
        results = pd.concat(
            [results, new_results]
        )

    results.to_csv(_results_fpath, index=False)

if __name__ == "__main__":
    config_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, required=True, help='Select set of metrics',
                        choices=["base", "bertscore", "bleurt"])
    parser.add_argument('-k', '--k-sample', type=int, default=100, help='Sample k sentences')
    parser.add_argument('-n', '--n-times', type=int, default=10, help='Run experiment n times')
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    args = parser.parse_args()

    CHEXPERT_LABELS_6 = [
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Pleural Effusion',
        'Atelectasis',
        'Lung Opacity',
    ]

    if args.metric == "bleurt":
        METRICS = [('bleurt', 0)]
    elif args.metric == "bertscore":
        METRICS = [('bertscore', 2)]
    else:
        METRICS = [
            ('bleu', 0),
            ('bleu', 3),
            ('rouge', 0),
            ('cider-IDF', 0),
            ('chexpert', 0),
        ]

    run_experiments(
        "mimic-expert1",
        abnormalities=CHEXPERT_LABELS_6,
        metrics=METRICS,
        k_sample=args.k_sample,
        n_times=args.n_times,
        suffix=args.metric,
        seed=args.seed,
    )
