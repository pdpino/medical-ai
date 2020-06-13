import torch
from ignite.metrics import Accuracy, Precision, Recall, RunningAverage, \
                           ConfusionMatrix # VariableAccumulation
from ignite.utils import to_onehot

from mrg.metrics.classification.roc_auc import RocAucMetric


ALL_LOGGABLE_METRICS = ["roc_auc", "prec", "recall", "acc"]

def _get_transform_one_label(label_index, use_round=True):
    """Creates a transform function to extract one label from a multi-label output."""
    def transform_fn(output):
        _, y_pred, y_true = output
        
        y_pred = y_pred[:, label_index]
        y_true = y_true[:, label_index]
        
        if use_round:
            y_pred = torch.round(y_pred)

        return y_pred, y_true
    return transform_fn


def _get_transform_cm(label_index, num_classes=2):
    """Creates a transform function to prepare the input for the ConfusionMatrix metric."""
    def transform_fn(output):
        _, y_pred, y_true = output
        
        y_pred = to_onehot(torch.round(y_pred[:, label_index]).long(), num_classes)
        y_true = y_true[:, label_index]

        return y_pred, y_true
    
    return transform_fn


def _get_count_positives(label_index):
    def count_positives_fn(result):
        """Count positive examples in a batch (for a given disease index)."""
        _, _, labels = result
        return torch.sum(labels[:, label_index]).item()

    return count_positives_fn


def _attach_by_disease(engine, chosen_diseases, metric_name, MetricClass,
                            use_round=True,
                            get_transform_fn=None,
                            metric_args=()):
    """Attaches one metric per label to an engine."""
    for index, disease in enumerate(chosen_diseases):
        if get_transform_fn:
            transform_fn = get_transform_fn(index)
        else:
            transform_fn = _get_transform_one_label(index, use_round=use_round)

        metric = MetricClass(*metric_args, output_transform=transform_fn)
        metric.attach(engine, f'{metric_name}_{disease}')


def attach_metrics_classification(engine, chosen_diseases, loss_name):
    """Attach classification metrics to an engine."""
    loss = RunningAverage(output_transform=lambda x: x[0], alpha=1)
    loss.attach(engine, loss_name)
    
    _attach_by_disease(engine, chosen_diseases, 'prec', Precision, True)
    _attach_by_disease(engine, chosen_diseases, 'recall', Recall, True)
    _attach_by_disease(engine, chosen_diseases, 'acc', Accuracy, True)
    _attach_by_disease(engine, chosen_diseases, 'roc_auc', RocAucMetric, False)
    _attach_by_disease(engine, chosen_diseases, 'cm', ConfusionMatrix,
                       get_transform_fn=_get_transform_cm, metric_args=(2,))
    _attach_by_disease(engine, chosen_diseases, 'positives', RunningAverage,
                       get_transform_fn=_get_count_positives)