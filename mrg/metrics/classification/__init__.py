import torch
from torch.nn.functional import binary_cross_entropy
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    RunningAverage,
    ConfusionMatrix,
    # VariableAccumulation,
    Loss,
)
from ignite.utils import to_onehot

from mrg.metrics.classification.accuracy import MultilabelAccuracy
from mrg.metrics.classification.hamming import Hamming
from mrg.metrics.classification.roc_auc import RocAucMetric
from mrg.metrics.classification.specificity import Specificity


def _get_transform_one_label(label_index, use_round=True):
    """Creates a transform function to extract one label from a multilabel output.
    
    Works only for multilabel=True.
    """
    def transform_fn(output):
        """Transform multilabel arrays into binary class arrays.
        
        Args:
            y_pred shape: batch_size, n_labels
            y_true shape: batch_size, n_labels
        Returns:
            binary arrays:
            y_pred shape: batch_size
            y_true shape: batch_size
        """
        _, y_pred, y_true = output
        
        y_pred = y_pred[:, label_index]
        y_true = y_true[:, label_index]
        
        if use_round:
            y_pred = torch.round(y_pred)

        return y_pred, y_true
    return transform_fn


def _get_transform_one_class(label_index):
    """Creates a transform function to extract one label

    Works only for multilabel=False.
    """
    def transform_fn(output):
        """Transform multiclass arrays into binary class arrays.

        Args:
            y_pred shape: batch_size, n_labels
            y_true shape: batch_size

        Returns:
            binary arrays:
            y_pred shape: batch_size
            y_true shape: batch_size
        """
        _, y_pred, y_true = output
        
        _, y_pred = y_pred.max(dim=1)
        y_pred = (y_pred == label_index).long()
        y_true = (y_true == label_index).long()

        return y_pred, y_true
    return transform_fn


def _get_transform_cm(label_index, num_classes=2):
    """Creates a transform function to prepare the input for the ConfusionMatrix metric.
    
    Works for multilabel outputs only.
    """
    def transform_fn(output):
        _, y_pred, y_true = output
        
        y_pred = to_onehot(torch.round(y_pred[:, label_index]).long(), num_classes)
        y_true = y_true[:, label_index].long()

        return y_pred, y_true
    
    return transform_fn


def _attach_binary_metrics(engine, labels, metric_name, MetricClass,
                           use_round=True,
                           get_transform_fn=None,
                           metric_args=()):
    """Attaches one metric per label to an engine."""
    for index, disease in enumerate(labels):
        if get_transform_fn:
            transform_fn = get_transform_fn(index)
        else:
            transform_fn = _get_transform_one_label(index, use_round=use_round)

        metric = MetricClass(*metric_args, output_transform=transform_fn)
        metric.attach(engine, f'{metric_name}_{disease}')


def _transform_remove_loss(output):
    """Simple transform to remove the loss from the output."""
    _, y_pred, y_true = output
    return y_pred, y_true


def _transform_remove_loss_and_round(output):
    """Simple transform to remove the loss from the output."""
    _, y_pred, y_true = output
    return torch.round(y_pred), y_true


def attach_metrics_classification(engine, labels, multilabel=True):
    """Attach classification metrics to an engine.
    
    Note: most multilabel metrics are treated as binary,
        i.e. the metrics are computed separately for each label.
    """
    loss = RunningAverage(output_transform=lambda x: x[0], alpha=1)
    loss.attach(engine, 'loss')
    
    if multilabel:
        acc = MultilabelAccuracy(output_transform=_transform_remove_loss_and_round)
        acc.attach(engine, 'acc')

        ham = Hamming(output_transform=_transform_remove_loss_and_round)
        ham.attach(engine, 'hamming')

        bce_loss = Loss(binary_cross_entropy,
                        output_transform=_transform_remove_loss_and_round)
        bce_loss.attach(engine, 'bce')

        _attach_binary_metrics(engine, labels, 'acc', Accuracy, True)
        _attach_binary_metrics(engine, labels, 'prec', Precision, True)
        _attach_binary_metrics(engine, labels, 'recall', Recall, True)
        _attach_binary_metrics(engine, labels, 'spec', Specificity, True)
        _attach_binary_metrics(engine, labels, 'roc_auc', RocAucMetric, False)
        _attach_binary_metrics(engine, labels, 'cm', ConfusionMatrix,
                        get_transform_fn=_get_transform_cm, metric_args=(2,))
    else:
        acc = Accuracy(output_transform=_transform_remove_loss)
        acc.attach(engine, 'acc')

        _attach_binary_metrics(engine, labels, 'prec', Precision,
                           get_transform_fn=_get_transform_one_class)
        _attach_binary_metrics(engine, labels, 'recall', Recall,
                           get_transform_fn=_get_transform_one_class)
        _attach_binary_metrics(engine, labels, 'spec', Specificity,
                           get_transform_fn=_get_transform_one_class)