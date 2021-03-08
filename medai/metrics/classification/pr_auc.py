import numpy as np
from sklearn.metrics import auc, precision_recall_curve as pr_curve
from ignite.metrics import EpochMetric


def pr_auc_compute_fn(y_preds, y_true):
    y_true = y_true.cpu().numpy()
    if len(np.unique(y_true)) != 2:
        return 0

    y_pred = y_preds.cpu().numpy()

    precision, recall, unused_thresholds = pr_curve(y_true, y_pred)

    return auc(recall, precision)


def PRAucMetric(**kwargs):
    return EpochMetric(pr_auc_compute_fn, **kwargs)
