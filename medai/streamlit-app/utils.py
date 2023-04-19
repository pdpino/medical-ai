import os
import pickle
import logging
from collections import namedtuple

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

LOGGER = logging.getLogger('medai.streamlit-app')

WORKSPACE_DIR = os.environ['MED_AI_WORKSPACE_DIR']


MatrixResult = namedtuple('MatrixResult', ['cube', 'dists', 'metric', 'groups', 'sampler'])

class Experiment:
    def __init__(self, abnormality, grouped, dataset):
        self.abnormality = abnormality
        self.grouped = grouped
        self.dataset = dataset

        self.grouped_2 = dict()
        self.grouped_2[0] = grouped.get(0, []) + grouped.get(-2, [])
        self.grouped_2[1] = grouped.get(1, []) + grouped.get(-1, [])
        self.results = []

    def add_result(self, result):
        assert isinstance(result, MatrixResult)
        self.results.append(result)

    def append(self, result):
        self.add_result(result)

    def __getitem__(self, idx):
        return self.results[idx]

    def __str__(self):
        lens = tuple([len(self.grouped.get(group, [])) for group in (-2, 0, -1, 1)])
        return f'{self.abnormality} data={self.dataset} n_sent={lens} n_results={len(self.results)}'

    def __repr__(self):
        return self.__str__()


### Plot matrix functions

KEY_TO_LABEL = {-2: 'None', 0: 'Neg', 1: 'Pos', -1: 'Unc'}
PRETTIER_METRIC = {
    'bleu': 'BLEU',
    'cider-IDF': 'CIDEr-D',
    'cider': 'CIDEr-D-NONIDF',
    'rouge': 'ROUGE-L',
    "bleurt": "bleurt",
    "bertscore": "bertscore",
}


def get_pretty_metric(metric, metric_i=0, include_range=False):
    pretty_metric = PRETTIER_METRIC.get(metric, metric)
    if pretty_metric == 'BLEU':
        pretty_metric += f'-{metric_i+1}'
    if include_range:
        max_value = 10 if 'cider' in metric else 1
        pretty_metric += f' (0-{max_value})'
    return pretty_metric


def get_cmap_by_metric(metric):
    return 'Blues' if 'cider' in metric else 'YlOrRd'


def plot_heatmap(exp, result_i=-1, metric_i=0, ax=None,
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
        err = f'metric_i={metric_i} too large for cube of shape {result.cube.shape}, using 0'
        LOGGER.error(err)
        metric_i = 0

    # Prettier
    ticks = [KEY_TO_LABEL[k] for k in result.groups]
    pretty_metric = get_pretty_metric(result.metric, metric_i=metric_i)

    sns.heatmap(result.cube[metric_i], annot=True, square=True,
                cmap=get_cmap_by_metric(result.metric),
                xticklabels=ticks, yticklabels=ticks, fmt='.3f', ax=ax,
                **heatmap_kwargs
                )

    if title:
        ax.set_title(
            f'{pretty_metric} in {exp.abnormality} ({result.sampler})',
            fontsize=title_fontsize,
        )
    if xlabel:
        ax.set_xlabel('Generated', fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel('Ground-Truth', fontsize=ylabel_fontsize)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=ticks_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=ticks_fontsize)


#### Plot hist functions
def plot_hists(exp, keys, result_i=-1, metric_i=0,
               title=True, xlabel=True, ylabel=True, bins=15, alpha=0.5,
               add_n_to_label=False,
               legend_fontsize=12,
               title_fontsize=12,
               ylabel_fontsize=12,
               xlabel_fontsize=12,
               xlog=False,
               ax=None,
               verbose=False,
               **hist_kwargs):
    if ax is None:
        ax = plt.gca()

    result = exp.results[result_i]

    for key in keys:
        dist = result.dists[key]
        if dist.ndim > 1:
            values = dist[metric_i] # useful for bleu-1, -2, -3, -4
        else:
            values = dist

        assert len(key) == 2
        gt_key, gen_key = key
        label = f'GT={KEY_TO_LABEL[gt_key]}, Gen={KEY_TO_LABEL[gen_key]}'
        if add_n_to_label:
            label += f' / (N={len(values):,})'
        ax.hist(values, label=label, alpha=alpha, bins=bins, density=True, **hist_kwargs)

        if verbose:
            print(f'{label} -- mean={values.mean():.4f} -- n={len(values):,}')

    pretty_metric = get_pretty_metric(result.metric, metric_i)

    ax.legend(fontsize=legend_fontsize)
    if title:
        dataset = 'IU' if exp.dataset == 'iu' else 'MIMIC'
        ax.set_title(
            f'{pretty_metric} scores in {exp.abnormality} sentences ({dataset})',
            fontsize=title_fontsize,
        )
    if xlabel:
        ax.set_xlabel(f'{pretty_metric} score', fontsize=xlabel_fontsize)
    if ylabel:
        ax.set_ylabel('Frequency', fontsize=ylabel_fontsize)
    if xlog:
        ax.set_xscale('log')

#### Plot distributions as boxplots
# Useful when there are many keys!
def plot_boxplots(exp, keys, result_i=-1, metric_i=0,
                  title=True, xlabel=True, ylabel=True,
                  # add_n_to_label=False,
                  title_fontsize=12,
                  ylabel_fontsize=12,
                  xlabel_fontsize=12,
                  labels_fontsize=12,
                  labelrot=0,
                  ylog=False,
                  correct_color='green',
                  incorrect_color='red',
                  ax=None):
    if ax is None:
        ax = plt.gca()

    result = exp.results[result_i]

    data = []
    labels = []
    colors = []
    for key in keys:
        dist = result.dists.get(key, np.zeros((1,)))
        if dist.ndim > 1:
            values = dist[metric_i] # useful for bleu-1, -2, -3, -4
        else:
            values = dist
        data.append(values)

        assert len(key) == 2
        gt_key, gen_key = key
        label = f'{KEY_TO_LABEL[gt_key]}-{KEY_TO_LABEL[gen_key]}'
        # if add_n_to_label:
        #     label += f' / (N={len(values):,})'
        labels.append(label)

        colors.append(correct_color if gt_key == gen_key else incorrect_color)

    ax.boxplot(data)
    ax.set_xticklabels(labels, fontsize=labels_fontsize, rotation=labelrot)

    xticks = ax.get_xticklabels()
    assert len(xticks) == len(colors), f'{len(xticks)} vs {len(colors)}'
    for color, xtick in zip(colors, xticks):
        xtick.set_color(color)

    pretty_metric = get_pretty_metric(result.metric, metric_i)

    # ax.legend(fontsize=legend_fontsize)
    if title:
        dataset = 'IU' if exp.dataset == 'iu' else 'MIMIC'
        ax.set_title(
            f'{pretty_metric} scores in {exp.abnormality} sentences ({dataset})',
            fontsize=title_fontsize,
        )
    if ylabel:
        ax.set_ylabel(pretty_metric, fontsize=ylabel_fontsize)
    if xlabel:
        ax.set_xlabel('Corpus', fontsize=xlabel_fontsize)
    if ylog:
        ax.set_yscale('log')



#### Load/save pickle functions

_EXP_FOLDER = os.path.join(WORKSPACE_DIR, 'report_generation', 'nlp-controlled-corpus')
def save_experiment_pickle(exp, name, overwrite=False):
    fpath = os.path.join(_EXP_FOLDER, f'{name}.data')
    if not overwrite and os.path.isfile(fpath):
        raise Exception(f'{fpath} file exists!')

    with open(fpath, 'wb') as f:
        pickle.dump(exp, f)
    print('Saved to %s', fpath)

def exist_experiment_pickle(name):
    fpath = os.path.join(_EXP_FOLDER, f'{name}.data')
    return os.path.isfile(fpath)

def load_experiment_pickle(name):
    fpath = os.path.join(_EXP_FOLDER, f'{name}.data')
    if not os.path.isfile(fpath):
        return None

    with open(fpath, 'rb') as f:
        exp = pickle.load(f)
    return exp



