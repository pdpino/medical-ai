from functools import partial
import numpy as np
from sklearn.metrics import roc_auc_score
from ignite.metrics import EpochMetric

class RocAucWarning(Warning):
    pass


def roc_auc_compute_fn(y_preds, y_true, assert_binary=True, **kwargs):
    y_true = y_true.cpu().numpy()
    if assert_binary and len(np.unique(y_true)) != 2:
        # warnings.warn("ROC AUC = 0", RocAucWarning)
        return 0

    y_pred = y_preds.cpu().numpy()

    return roc_auc_score(y_true, y_pred, **kwargs)


def RocAucMetric(roc_kwargs={}, **kwargs):
    return EpochMetric(partial(roc_auc_compute_fn, **roc_kwargs), **kwargs)
