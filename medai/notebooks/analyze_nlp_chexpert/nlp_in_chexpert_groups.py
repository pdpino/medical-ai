import os
import random
import pickle
import argparse
import logging
from collections import namedtuple, defaultdict
from itertools import product
from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pycocoevalcap.bleu import bleu_scorer
from pycocoevalcap.cider import cider_scorer

from medai.datasets.iu_xray import DATASET_DIR as IU_DIR
from medai.datasets.common.constants import CHEXPERT_DISEASES
from medai.datasets.mimic_cxr import DATASET_DIR as MIMIC_DIR
from medai.metrics.report_generation.nlp.rouge import RougeLScorer
from medai.metrics.report_generation.nlp.huggingface import (
    BLEURT,
    BertScore,
)
from medai.metrics.report_generation.nlp.cider_idf import (
    CiderScorerIDFModified,
    compute_doc_freq,
)
from medai.metrics.report_generation.chexpert import ChexpertScorer
from medai.utils import timeit_main, config_logging
from medai.utils.files import WORKSPACE_DIR
from medai.notebooks.analyze_nlp_chexpert.nlp_in_chexpert_groups_utils import (
    Experiment,
    MatrixResult,
    save_experiment_pickle,
    exist_experiment_pickle,
    load_experiment_pickle,
    get_pretty_metric,
    get_cmap_by_metric,
    get_pretty_valuation,
)

LOGGER = logging.getLogger("medai.analyze.nlp-chex-groups")

###### Scorers
_SCORERS = {
    "bleu": (bleu_scorer.BleuScorer, 4),
    "rouge": (RougeLScorer, 1),
    "cider": (cider_scorer.CiderScorer, 1),
    "cider-IDF": (CiderScorerIDFModified, 1),
    "bleurt": (BLEURT, BLEURT.n_metrics),
    "bertscore": (BertScore, BertScore.n_metrics),
    "chexpert": (ChexpertScorer, ChexpertScorer.n_metrics),
}


###### Sampling classes


class BaseSampler:
    def __init__(self, sentences_gen, sentences_gt):
        self.sentences_gen = sentences_gen
        self.sentences_gt = sentences_gt

    def __repr__(self):
        return self.__str__()


class SampleAllPairwise(BaseSampler):
    def __iter__(self):
        content = product(self.sentences_gen, self.sentences_gt)
        for sentence_gen, sentence_gt in content:
            yield sentence_gen, [sentence_gt]

    def __len__(self):
        return len(self.sentences_gen) * len(self.sentences_gt)

    def __str__(self):
        return "all"


class SampleKRandomGen(BaseSampler):
    def __init__(self, sentences_gen, sentences_gt, k_times=100, max_n=500):
        super().__init__(sentences_gen, sentences_gt)

        self.k = k_times
        self.max_n = max_n

    def __iter__(self):
        if self.max_n is not None and self.max_n < len(self.sentences_gt):
            gts = random.sample(self.sentences_gt, self.max_n)
        else:
            gts = self.sentences_gt

        for sentence_gt in gts:
            if self.k < len(self.sentences_gen):
                gens = random.sample(self.sentences_gen, self.k)
            else:
                gens = self.sentences_gen  # Generate all available pairs instead
            for sentence_gen in gens:
                yield sentence_gen, [sentence_gt]

    def __len__(self):
        n_gts = min(len(self.sentences_gt), self.max_n)
        n_gens = min(len(self.sentences_gen), self.k)
        return n_gts * n_gens

    def __str__(self):
        return f"random-gen_k{self.k}_n{self.max_n}"


class SampleKRandomGT(BaseSampler):
    def __init__(self, sentences_gen, sentences_gt, k_times=100, k_gts=5, max_n=500):
        super().__init__(sentences_gen, sentences_gt)

        self.k_times = k_times
        self.k_gts = k_gts
        self.max_n = max_n

    def __iter__(self):
        if self.max_n < len(self.sentences_gen):
            gens = random.sample(self.sentences_gen, self.max_n)
        else:
            gens = self.sentences_gen

        grab_n_samples = self.k_times * self.k_gts
        for sentence_gen in gens:
            if grab_n_samples < len(self.sentences_gt):
                gts = random.sample(self.sentences_gt, grab_n_samples)
            else:
                gts = self.sentences_gt
            for i in range(0, len(gts), self.k_gts):
                yield sentence_gen, gts[i : i + self.k_gts]

    def __len__(self):
        n_gens = min(self.max_n, len(self.sentences_gen))
        n_gts = min(self.k_times, len(self.sentences_gt) // self.k_gts)
        return n_gens * n_gts

    def __str__(self):
        return f"random-gt_k{self.k_times}_n{self.max_n}_lgts{self.k_gts}"


_SAMPLERS = {
    "all": SampleAllPairwise,
    "random-gen": SampleKRandomGen,
    "random-gt": SampleKRandomGT,
}

###### Matrices functions

def calc_score_matrices(
    grouped,
    dataset_info,
    abnormality=None,
    groups=(-2, 0, -1, 1),
    metric="bleu",
    show="groups",
    sampler="all",
    seed=None,
    **sampler_kwargs,
):
    if seed is not None:
        random.seed(seed)

    for g in groups:
        if g not in grouped:
            LOGGER.warning("%s not in grouped!", g)

    ScorerClass, n_metrics = _SCORERS[metric]
    SamplerClass = _SAMPLERS[sampler]

    n_groups = len(groups)
    out_cube = np.zeros((n_metrics, n_groups, n_groups))

    out_dists = dict()

    if hasattr(ScorerClass, 'preload'):
        ScorerClass.preload()

    for (row_i, group_gen), (col_j, group_gt) in tqdm(product(
        enumerate(groups), enumerate(groups)
    ), disable=show != "groups", total=len(groups) ** 2):
        scorer = ScorerClass()

        if metric == "cider-IDF":
            # pylint: disable=assigning-non-slot
            scorer.document_frequency = dataset_info.doc_freq
            # Also needs to update ref_len
            scorer.ref_len = dataset_info.log_ref_len

        # Useful for ChexpertScorer
        if hasattr(scorer, 'use_cache'):
            scorer.use_cache(dataset_info.chexpert_cache_sentences)
        if hasattr(scorer, 'set_abnormality'):
            scorer.set_abnormality(abnormality)

        sentences_gen = grouped.get(group_gen, [])
        sentences_gt = grouped.get(group_gt, [])

        content = SamplerClass(sentences_gen, sentences_gt, **sampler_kwargs)

        for sentence_gen, sentences_gt in tqdm(content, disable=show != "samples", desc=f"{group_gen:>2} vs {group_gt:>2}"):
            scorer += (sentence_gen, sentences_gt)

        summary, all_scores = scorer.compute_score()
        # all_scores shape: n_metrics, n_samples=n_group1 x n_group2
        # summary shape: n_metrics

        if len(all_scores) == 0:
            LOGGER.error('Empty scores: %s', metric)
            continue

        # REMEMBER THEY ARE FLIPPED!! GT first, then GEN
        # out_cube[:, row_i, col_j] = np.array(summary)
        out_cube[:, col_j, row_i] = np.array(summary)

        # key = (group_gen, group_gt)
        key = (group_gt, group_gen)
        out_dists[key] = np.array(all_scores)

    return MatrixResult(
        cube=out_cube,
        dists=out_dists,
        metric=metric,
        groups=groups,
        # dummy sampler to get a string representing the sampler
        sampler=f"{str(SamplerClass([], [], **sampler_kwargs))}--{seed}",
        # HACK: seed is concated at the end (instead of a separate attribute)
        # to avoid re-creating pickle objects already dumped
    )


#### Experiment functions
def init_experiment(abnormality, dataset_info):
    grouped = dataset_info.sentences_df.groupby(abnormality)["sentence"].apply(
        lambda x: sorted(list(x), key=len),
    )
    # print([(valuation, len(sentences)) for valuation, sentences in grouped.iteritems()])

    return Experiment(
        abnormality=abnormality,
        grouped=grouped,
        dataset=dataset_info.name,
    )


# Load experiments
def load_experiments(dataset_name):
    exp_by_abn = {}
    errors = defaultdict(list)
    for abnormality in CHEXPERT_DISEASES[1:]:
        fname = f'{dataset_name}-{abnormality.replace(" ", "-").lower()}'
        if not exist_experiment_pickle(fname):
            errors["not-found"].append(fname)
            continue
        exp = load_experiment_pickle(fname)
        exp_by_abn[abnormality] = exp

    if len(errors["not-found"]):
        print("Not found: ", errors["not-found"])

    return exp_by_abn


### Plot matrix functions
def plot_heatmap(
    exp,
    result_i=-1,
    metric_i=0,
    ax=None,
    title=True,
    title_fontsize=12,
    ylabel_fontsize=12,
    xlabel_fontsize=12,
    ticks_fontsize=12,
    xlabel=True,
    ylabel=True,
    **heatmap_kwargs,
):
    if ax is None:
        ax = plt.gca()

    # Select result to plot
    result = exp.results[result_i]

    # This is useful for BLEU-1, -2, -3, -4
    if metric_i > result.cube.shape[0]:
        err = f"metric_i={metric_i} too large for cube of shape {result.cube.shape}, using 0"
        LOGGER.error(err)
        metric_i = 0

    # Prettier
    ticks = [get_pretty_valuation(k) for k in result.groups]
    pretty_metric = get_pretty_metric(result.metric, metric_i=metric_i)

    sns.heatmap(
        result.cube[metric_i],
        annot=True,
        square=True,
        cmap=get_cmap_by_metric(result.metric),
        xticklabels=ticks,
        yticklabels=ticks,
        fmt=".3f",
        ax=ax,
        **heatmap_kwargs,
    )

    if title:
        ax.set_title(
            f"{pretty_metric} in {exp.abnormality} ({result.sampler})",
            fontsize=title_fontsize,
        )
    if xlabel:
        ax.set_xlabel("Generated", fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel("Ground-Truth", fontsize=ylabel_fontsize)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticks_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=ticks_fontsize)


#### Plot hist functions
def plot_hists(
    exp,
    keys,
    result_i=-1,
    metric_i=0,
    title=True,
    xlabel=True,
    ylabel=True,
    bins=15,
    alpha=0.5,
    add_n_to_label=False,
    legend_fontsize=12,
    title_fontsize=12,
    ylabel_fontsize=12,
    xlabel_fontsize=12,
    xlog=False,
    ax=None,
    verbose=False,
    **hist_kwargs,
):
    if ax is None:
        ax = plt.gca()

    result = exp.results[result_i]

    for key in keys:
        dist = result.dists[key]
        if dist.ndim > 1:
            values = dist[metric_i]  # useful for bleu-1, -2, -3, -4
        else:
            values = dist

        assert len(key) == 2
        gt_key, gen_key = key
        label = f"GT={get_pretty_valuation(gt_key, short=True)}, Gen={get_pretty_valuation(gen_key, short=True)}"
        if add_n_to_label:
            label += f" / (N={len(values):,})"
        ax.hist(
            values, label=label, alpha=alpha, bins=bins, density=True, **hist_kwargs
        )

        if verbose:
            print(f"{label} -- mean={values.mean():.4f} -- n={len(values):,}")

    pretty_metric = get_pretty_metric(result.metric, metric_i)

    ax.legend(fontsize=legend_fontsize)
    if title:
        dataset = "IU" if exp.dataset == "iu" else "MIMIC"
        ax.set_title(
            f"{pretty_metric} scores in {exp.abnormality} sentences ({dataset})",
            fontsize=title_fontsize,
        )
    if xlabel:
        ax.set_xlabel(f"{pretty_metric}", fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel("Frequency", fontsize=ylabel_fontsize)
    if xlog:
        ax.set_xscale("log")


#### Plot distributions as boxplots
# Useful when there are many keys!
def plot_boxplots(
    exp,
    keys,
    result_i=-1,
    metric_i=0,
    title=True,
    xlabel=True,
    ylabel=True,
    # add_n_to_label=False,
    title_fontsize=12,
    ylabel_fontsize=12,
    xlabel_fontsize=12,
    labels_fontsize=12,
    labelrot=0,
    ylog=False,
    correct_color="green",
    incorrect_color="red",
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    result = exp.results[result_i]

    data = []
    labels = []
    colors = []
    for key in keys:
        dist = result.dists.get(key, np.zeros((1,)))
        if dist.ndim > 1:
            values = dist[metric_i]  # useful for bleu-1, -2, -3, -4
        else:
            values = dist
        data.append(values)

        assert len(key) == 2
        gt_key, gen_key = key
        label = f"{get_pretty_valuation(gt_key, short=True)}-{get_pretty_valuation(gen_key, short=True)}"
        # if add_n_to_label:
        #     label += f' / (N={len(values):,})'
        labels.append(label)

        colors.append(correct_color if gt_key == gen_key else incorrect_color)

    ax.boxplot(data)
    ax.set_xticklabels(labels, fontsize=labels_fontsize, rotation=labelrot)

    xticks = ax.get_xticklabels()
    assert len(xticks) == len(colors), f"{len(xticks)} vs {len(colors)}"
    for color, xtick in zip(colors, xticks):
        xtick.set_color(color)

    pretty_metric = get_pretty_metric(result.metric, metric_i)

    # ax.legend(fontsize=legend_fontsize)
    if title:
        dataset = "IU" if exp.dataset == "iu" else "MIMIC"
        ax.set_title(
            f"{pretty_metric} scores in {exp.abnormality} sentences ({dataset})",
            fontsize=title_fontsize,
        )
    if ylabel:
        ax.set_ylabel(pretty_metric, fontsize=ylabel_fontsize)
    if xlabel:
        ax.set_xlabel("Corpus", fontsize=xlabel_fontsize)
    if ylog:
        ax.set_yscale("log")


####### Dataset loading
DatasetInfo = namedtuple(
    "DatasetInfo",
    [
        "name",
        "reports_df",
        "sentences_df",
        "doc_freq",
        "log_ref_len",
        "chexpert_cache_sentences",
    ],
)

_CACHED_DOC_FREQ = {}

def init_dataset_info(name, skip_docfreq=False):
    dataset_dir = IU_DIR if name == "iu" else MIMIC_DIR

    fpath = os.path.join(dataset_dir, "reports", "sentences_with_chexpert_labels.csv")
    sentences_df_chex = pd.read_csv(fpath)

    if 'expert' in name:
        fpath = f'{WORKSPACE_DIR}/report_generation/nlp-chex-gold-sentences/{name}.csv'
        sentences_df = pd.read_csv(fpath)
    else:
        sentences_df = sentences_df_chex

    fpath = os.path.join(dataset_dir, "reports", "reports_with_chexpert_labels.csv")
    reports_df = pd.read_csv(fpath)

    if skip_docfreq:
        doc_freq = {}
    else:
        dataset_simple = "iu" if name == "iu" else "mimic"
        if dataset_simple in _CACHED_DOC_FREQ:
            # in-memory cache
            doc_freq = _CACHED_DOC_FREQ[dataset_simple]
        else:
            # read instead of re-computing each time
            docfreq_fpath = f'{WORKSPACE_DIR}/report_generation/nlp-chex-gold-sentences/docfreq-{dataset_simple}.data'
            if os.path.isfile(docfreq_fpath):
                with open(docfreq_fpath, 'rb') as f:
                    doc_freq = pickle.load(f)
            else:
                doc_freq = compute_doc_freq(list(reports_df["Reports"]))
                with open(docfreq_fpath, 'wb') as f:
                    pickle.dump(doc_freq, f)
            _CACHED_DOC_FREQ[dataset_simple] = doc_freq

    log_ref_len = np.log(len(reports_df))

    return DatasetInfo(
        name=name,
        reports_df=reports_df,
        sentences_df=sentences_df,
        doc_freq=doc_freq,
        log_ref_len=log_ref_len,
        chexpert_cache_sentences=sentences_df_chex,
    )


@timeit_main(LOGGER)
def run_experiments(
    dataset="iu",
    abns=[],
    metrics=["bleu", "rouge", "cider-IDF"],
    k_times=50,
    max_n=50,
    chex2=True,
    chex4=True,
    suffix=None,
    seed=None,
):
    dataset_info = init_dataset_info(dataset)

    kwargs = {
        "sampler": "random-gen",
        "k_times": k_times,
        # 'k_gts': 1,
        "max_n": max_n,
    }

    for abn in abns:
        fname = f'{dataset_info.name}-{abn.replace(" ", "-").lower()}'
        if suffix:
            fname += f'-{suffix}'
        LOGGER.info("Computing %s", fname)

        # Save only one per dataset per abnormality, with a list of results
        exist = exist_experiment_pickle(fname)
        if exist:
            exp = load_experiment_pickle(fname)
        else:
            exp = init_experiment(abn, dataset_info)

        for metric in metrics:
            if chex4:
                LOGGER.info("\tcomputing 4x4 for %s", metric)
                exp.append(
                    calc_score_matrices(
                        exp.grouped,
                        dataset_info,
                        abnormality=abn,
                        metric=metric,
                        seed=seed,
                        **kwargs,
                    )
                )
            if chex2:
                LOGGER.info("\tcomputing 2x2 for %s", metric)
                exp.append(
                    calc_score_matrices(
                        exp.grouped_2,
                        dataset_info,
                        abnormality=abn,
                        groups=(0, 1),
                        metric=metric,
                        seed=seed,
                        **kwargs,
                    )
                )

        save_experiment_pickle(exp, fname, overwrite=exist)


if __name__ == "__main__":
    config_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, required=True,
                        default=None, choices=_SCORERS.keys(), help='Select metric')
    parser.add_argument('--dataset', type=str, default="",
                        choices=["iu", "mimic", "mimic-expert1", "mimic-expert2"], help='Select dataset')
    parser.add_argument('-n', '--n-times', type=int, default=10, help='Choose k_times and max_n')
    # parser.add_argument('--abnormality', type=str, default=None, choices=CHEXPERT_DISEASES,
    #                     help='Select abnormality')
    args = parser.parse_args()

    CHEXPERT_LABELS_5 = [
        'Cardiomegaly',
        'Consolidation',
        'Edema',
        'Pleural Effusion',
        'Atelectasis',
        'Lung Opacity',
    ]

    run_experiments(
        args.dataset,
        #abns=CHEXPERT_DISEASES[2:-1],
        # abns=[args.abnormality],
        abns=CHEXPERT_LABELS_5,
        # metrics=["bleu", "rouge", "cider-IDF"], #
        metrics=[args.metric],
        k_times=args.n_times,
        max_n=args.n_times,
        # suffix=args.metric,
        seed=1234,
    )
