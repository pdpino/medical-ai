####################
# NOTICE:
# This file should not use medai code
# --> streamlit should avoid medai
####################
import logging
from collections import namedtuple

MatrixResult = namedtuple(
    "MatrixResult", ["cube", "dists", "metric", "groups", "sampler"]
)

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
        return f"{self.abnormality} data={self.dataset} n_sent={lens} n_results={len(self.results)}"

    def __repr__(self):
        return self.__str__()



_VALUATION_TO_LABEL = {
    -2: ("Unmention", "Unm"),
    0: ("Healthy", "Heal"),
    1: ("Abnormal", "Abn"),
    -1: ("Uncertain", "Unc"),
}
def get_pretty_valuation(valuation, short=False):
    short_index = 1 if short else 0
    return _VALUATION_TO_LABEL[valuation][short_index] if valuation in _VALUATION_TO_LABEL else valuation

def get_pretty_valuation_pair(val_pair):
    gt, gen = val_pair
    return f"{get_pretty_valuation(gt)}-{get_pretty_valuation(gen)}"

AVAILABLE_METRICS = {
    "bleu": "BLEU",
    "cider-IDF": "CIDEr-D",
    "cider": "CIDEr-D-NONIDF",
    "rouge": "ROUGE-L",
    "bleurt": "BLEURT",
    "bertscore": "BERTscore",
    "chexpert": "CheXpert",
}

# Copied from medai classes
_CHEXPERT_SCORER_METRICS = ['acc', 'precision', 'recall', 'F1', 'roc_auc', 'pr_auc']
_BERT_SCORE_METRICS = ['prec', 'recall', 'F1']

def get_pretty_metric(metric, metric_i=0, include_range=False):
    pretty_metric = AVAILABLE_METRICS.get(metric, metric)
    if metric == "chexpert":
        pretty_metric += f"-{_CHEXPERT_SCORER_METRICS[metric_i]}"
    if metric == "bertscore":
        pretty_metric += f"-{_BERT_SCORE_METRICS[metric_i]}"
    if metric == "bleu":
        pretty_metric += f"-{metric_i+1}"
    if include_range:
        max_value = 10 if "cider" in metric else 1
        pretty_metric += f" (0-{max_value})"
    return pretty_metric


def get_cmap_by_metric(metric):
    return "Blues" if "cider" in metric else "YlOrRd"
