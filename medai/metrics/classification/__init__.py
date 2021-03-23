from functools import reduce, partial
import operator
import torch
from torch.nn.functional import binary_cross_entropy
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    RunningAverage,
    ConfusionMatrix,
    Loss,
    MetricsLambda,
)
from ignite.utils import to_onehot

from medai.metrics.classification.accuracy import MultilabelAccuracy
from medai.metrics.classification.hamming import Hamming
from medai.metrics.classification.roc_auc import RocAucMetric
from medai.metrics.classification.pr_auc import PRAucMetric
from medai.metrics.classification.specificity import Specificity
from medai.metrics.segmentation.iou import IoU
from medai.metrics.segmentation.iobb import IoBB
from medai.utils.metrics import attach_metric_for_labels


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
                           include_macro=True,
                           metric_args=(),
                           device='cuda'):
    """Attaches one metric per label to an engine."""
    metrics = []
    for index, disease in enumerate(labels):
        if get_transform_fn:
            transform_fn = get_transform_fn(index)
        else:
            transform_fn = _get_transform_one_label(index, use_round=use_round)

        metric = MetricClass(*metric_args, output_transform=transform_fn, device=device)
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


def attach_metrics_classification(engine, labels, multilabel=True, hint=False, device='cuda'):
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
        # acc = MultilabelAccuracy(output_transform=_transform_remove_loss_and_round, device=device)
        # acc.attach(engine, 'acc')

        ham = Hamming(output_transform=_transform_remove_loss_and_round, device=device)
        ham.attach(engine, 'hamming')

        bce_loss = Loss(binary_cross_entropy,
                        output_transform=_transform_remove_loss_and_round,
                        device=device)
        bce_loss.attach(engine, 'bce')

        # _attach_binary_metrics(engine, labels, 'acc', Accuracy, True,
        #                        include_macro=False, device=device)
        # FIXME: Precision and Recall are failing with
        # "Metric  must have at least one example before it can be computed"
        # (After pytorch-ignite upgrade to 0.4.3)
        # FIXME: Accuracy, MultilabelAccuracy, Specificity, Precision and Recall
        # use round-in-0.5, instead of optimizing the threshold.
        # FIXME: two accuracies were attached before??
        # _attach_binary_metrics(engine, labels, 'prec', Precision, True, device=device)
        # _attach_binary_metrics(engine, labels, 'recall', Recall, True, device=device)
        # _attach_binary_metrics(engine, labels, 'spec', Specificity, True, device=device)
        _attach_binary_metrics(engine, labels, 'roc_auc', RocAucMetric, False, device=device)
        _attach_binary_metrics(engine, labels, 'pr_auc', PRAucMetric, False, device=device)
    else:
        acc = Accuracy(output_transform=_transform_remove_loss, device=device)
        acc.attach(engine, 'acc')

        _attach_binary_metrics(engine, labels, 'prec', Precision,
                               get_transform_fn=_get_transform_one_class,
                               device=device)
        _attach_binary_metrics(engine, labels, 'recall', Recall,
                               get_transform_fn=_get_transform_one_class,
                               device=device)
        _attach_binary_metrics(engine, labels, 'spec', Specificity,
                               get_transform_fn=_get_transform_one_class,
                               device=device)


def attach_hint_saliency(engine, labels, multilabel=True, device='cuda'):
    keys = [
        ('grad-cam', 'gt_activations', None),
    ]
    attach_metrics_image_saliency(engine, labels, keys, multilabel=multilabel, device=device)


def attach_metric_cm(engine, labels, multilabel=True, device='cuda'):
    """Attach ConfusionMatrix metrics to an engine.

    Note that CMs are not attached during training, since they are not easily visualized (e.g. TB).
    """
    if multilabel:
        _attach_binary_metrics(engine, labels, 'cm', ConfusionMatrix,
                               get_transform_fn=_get_transform_cm_multilabel,
                               metric_args=(2,), include_macro=False,
                               )
    else:
        cm = ConfusionMatrix(
            len(labels),
            output_transform=_transform_remove_loss_and_round,
            device=device,
        )
        cm.attach(engine, 'cm')


def attach_metrics_image_saliency(engine, labels, keys, multilabel=True, device='cuda'):
    """Wrapper to attach segmentation metrics (IoU, IoBB).

    FIXME: Clarify the semantics, or try not to use it!
    Confusing stuff: use of keys, metric is attached as "iou-<key>", and so on.
    """
    if not multilabel:
        raise NotImplementedError()

    def _extract_maps(output, key_gt, key_valid=None):
        activations = output['activations']

        gt_map = output[key_gt]
        gt_valid = output.get(key_valid, None)

        return activations, gt_map, gt_valid

    for (name, key_gt, key_valid) in keys:
        iou = IoU(
            reduce_sum=False,
            output_transform=partial(_extract_maps, key_gt=key_gt, key_valid=key_valid),
            device=device,
            )
        attach_metric_for_labels(engine, labels, iou, f'iou-{name}')

        iobb = IoBB(
            reduce_sum=False,
            output_transform=partial(_extract_maps, key_gt=key_gt, key_valid=key_valid),
            device=device)
        attach_metric_for_labels(engine, labels, iobb, f'iobb-{name}')
