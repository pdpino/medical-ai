"""Utils for these notebooks."""
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from pycocoevalcap.bleu.bleu_scorer import BleuScorer
from pycocoevalcap.cider.cider_scorer import CiderScorer

from medai.datasets.common.constants import CHEXPERT_DISEASES
from medai.metrics.report_generation.chexpert import labels_with_suffix
from medai.metrics.report_generation.nlp.rouge import RougeLScorer


#### NLP STUFF

def add_nlp_metrics_to_df(df, show=True):
    """Given a RG-outputs file compute NLP metrics for each sample."""
    # Create new scorers
    scorers = {
        'bleu': BleuScorer(),
        'rougeL': RougeLScorer(),
        'ciderD': CiderScorer(),
    }

    # Compute metrics
    for _, row in tqdm(df.iterrows(), disable=not show, total=len(df)):
        gt = row['ground_truth']
        gen = row['generated']
        for scorer in scorers.values():
            scorer += (gen, [gt])

    # Save metric values
    metric_values = {}
    for name, scorer in scorers.items():
        unused_summaries, details = scorer.compute_score()
        details = np.array(details)
        if details.ndim == 2:
            for i_metric in range(len(details)):
                metric_values[f'{name}{i_metric+1}'] = details[i_metric,:]
        else:
            assert details.ndim == 1, f'Shape: {details.shape}'
            metric_values[name] = details

    # Double check lens
    for k, v in metric_values.items():
        assert len(v) == len(df), f'Wrong: {len(df)} vs {len(v)} (in {k})'

    # Add to the DF
    return df.assign(**metric_values)


#### CHEXPERT STUFF
def _smart_division(a, b):
    if b == 0:
        return 0
    return a / b

def _row_values_to_labels(row, labels, suffix):
    r = row[labels]
    pos_findings = list(s.replace(suffix, '') for s in r[r == 1].index)
    assert set(pos_findings).issubset(CHEXPERT_DISEASES), pos_findings
    return pos_findings


_GT_LABELS = labels_with_suffix('gt')
_GEN_LABELS = labels_with_suffix('gen')

def add_chex_metrics_to_df(df, show=True):
    chex_values = defaultdict(list)

    for _, row in tqdm(df.iterrows(), disable=not show, total=len(df)):
        # Compute per-sample accuracy
        gtv = row[_GT_LABELS].to_numpy()
        genv = row[_GEN_LABELS].to_numpy()
        acc = (gtv == genv).sum() / 14
        chex_values['chex-acc'].append(acc)

        # Compute per-sample f1
        gtl = _row_values_to_labels(row, _GT_LABELS, '-gt')
        genl = _row_values_to_labels(row, _GEN_LABELS, '-gen')

        intersection = len(set(gtl).intersection(set(genl)))

        precision = _smart_division(intersection, len(genl))
        recall = _smart_division(intersection, len(gtl))
        f1 = _smart_division(2 * precision * recall, precision + recall)

        chex_values['chex-prec'].append(precision)
        chex_values['chex-recall'].append(recall)
        chex_values['chex-f1'].append(f1)

    for k, v in chex_values.items():
        assert len(v) == len(df), f'Wrong chex-values {len(v)} vs {len(df)} (name {k})'

    return df.assign(**chex_values)
