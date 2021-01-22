from functools import reduce
import operator
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
    MetricsLambda,
)
from ignite.utils import to_onehot

from medai.metrics.classification.accuracy import MultilabelAccuracy
from medai.metrics.classification.hamming import Hamming
from medai.metrics.classification.roc_auc import RocAucMetric
from medai.metrics.classification.pr_auc import PRAucMetric
from medai.metrics.classification.specificity import Specificity


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
        y_pred = output['pred_labels']
        y_true = output['gt_labels']

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
        y_pred = output['pred_labels']
        y_true = output['gt_labels']

        _, y_pred = y_pred.max(dim=1)
        y_pred = (y_pred == label_index).long()
        y_true = (y_true == label_index).long()

        return y_pred, y_true
    return transform_fn


def _get_transform_cm_multilabel(label_index):
    """Creates a transform function to prepare the input for the ConfusionMatrix metric.

    Works for multilabel outputs only.
    The metric will have a 2x2 confusion-matrix, with positive and negative for the
        selected label.
    In this context, n_labels refers to the existing labels in the multilabel classification
        (such as different diseases); n_classes refers to 2, as for each label it can be
        present or absent (2 classes).
    """
    n_classes = 2

    def transform_fn(output):
        """Transform multilabel arrays into one-hot and indices array, respectively.

        Args:
            y_pred shape: batch_size, n_labels
            y_true shape: batch_size, n_labels
        Returns:
            y_pred shape: batch_size, n_classes (one-hot)
            y_true shape: batch_size
        """
        y_pred = output['pred_labels']
        y_true = output['gt_labels']

        y_pred = torch.round(y_pred[:, label_index]).long()
        y_pred = to_onehot(y_pred, n_classes)
        y_true = y_true[:, label_index].long()

        return y_pred, y_true

    return transform_fn


def _attach_binary_metrics(engine, labels, metric_name, MetricClass,
                           use_round=True,
                           get_transform_fn=None,
                           include_macro=False,
                           metric_args=()):
    """Attaches one metric per label to an engine."""
    metrics = []
    for index, disease in enumerate(labels):
        if get_transform_fn:
            transform_fn = get_transform_fn(index)
        else:
            transform_fn = _get_transform_one_label(index, use_round=use_round)

        metric = MetricClass(*metric_args, output_transform=transform_fn)
        metric.attach(engine, f'{metric_name}_{disease}')
        metrics.append(metric)

    if include_macro:
        def _calc_macro_avg(*metrics):
            return reduce(lambda x, y: x+y, metrics) / len(metrics)

        macro_avg = MetricsLambda(_calc_macro_avg, *metrics)
        macro_avg.attach(engine, metric_name)


def _transform_remove_loss(output):
    """Simple transform to remove the loss from the output."""
    y_pred = output['pred_labels']
    y_true = output['gt_labels']
    return y_pred, y_true


def _transform_remove_loss_and_round(output):
    """Simple transform to remove the loss from the output."""
    y_pred = output['pred_labels']
    y_true = output['gt_labels']
    return torch.round(y_pred), y_true


def attach_metrics_classification(engine, labels, multilabel=True, hint=False):
    """Attach classification metrics to an engine, to use during training.

    Note: most multilabel metrics are treated as binary,
        i.e. the metrics are computed separately for each label.
    """
    losses = ['loss']
    if hint:
        losses.extend(['cl_loss', 'hint_loss'])
    for loss_name in losses:
        loss_metric = RunningAverage(
            output_transform=operator.itemgetter(loss_name), alpha=1,
        )
        loss_metric.attach(engine, loss_name)

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
        _attach_binary_metrics(engine, labels, 'roc_auc', RocAucMetric, False,
                               include_macro=True)
        _attach_binary_metrics(engine, labels, 'pr_auc', PRAucMetric, False,
                               include_macro=True)
    else:
        acc = Accuracy(output_transform=_transform_remove_loss)
        acc.attach(engine, 'acc')

        _attach_binary_metrics(engine, labels, 'prec', Precision,
                           get_transform_fn=_get_transform_one_class)
        _attach_binary_metrics(engine, labels, 'recall', Recall,
                           get_transform_fn=_get_transform_one_class)
        _attach_binary_metrics(engine, labels, 'spec', Specificity,
                           get_transform_fn=_get_transform_one_class)


def attach_metric_cm(engine, labels, multilabel=True):
    """Attach ConfusionMatrix metrics to an engine.

    Note that CMs are not attached during training, since they are not easily visualized (e.g. TB).
    """
    if multilabel:
        _attach_binary_metrics(engine, labels, 'cm', ConfusionMatrix,
                               get_transform_fn=_get_transform_cm_multilabel,
                               metric_args=(2,),
                               )
    else:
        cm = ConfusionMatrix(len(labels), output_transform=_transform_remove_loss_and_round)
        cm.attach(engine, 'cm')
