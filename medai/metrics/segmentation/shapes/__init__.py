from functools import partial
import operator
import numpy as np
from ignite.metrics import MetricsLambda

from medai.metrics.segmentation.shapes.metric import OrganShapesAndHolesMetric


_IGNORE_ORGANS_IDX = (0,) # Background


def _extract_maps(output, key):
    selected_map = output[key]
    ndim = selected_map.dim()

    if ndim == 4:
        # Assume shape batch_size, n_labels, height, width (for generated case)
        selected_map = selected_map.argmax(dim=1)
        # shape: batch_size, height, width

    # Assume shape batch_size, height, width
    return selected_map


def _attach_shapes_and_holes(engine, organs, out_key='activations', name='act'):
    master_metric = OrganShapesAndHolesMetric(
        output_transform=partial(_extract_maps, key=out_key),
        ignore=_IGNORE_ORGANS_IDX,
    )

    def _extract_metric_for_organ(result, metric_key, organ_idx):
        values_by_organ = result[metric_key]
        return values_by_organ.get(organ_idx, 0)

    def _extract_macro_avg(result, metric_key):
        values_by_organ = result[metric_key]
        return np.mean(values_by_organ.values())

    for metric_key in ('n-shapes', 'n-holes'):
        for organ_idx, organ in enumerate(organs):
            if organ_idx in _IGNORE_ORGANS_IDX:
                continue

            metric_for_organ_i = MetricsLambda(
                partial(_extract_metric_for_organ, metric_key=metric_key, organ_idx=organ_idx),
                master_metric,
            )
            metric_for_organ_i.attach(engine, f'{metric_key}-{name}-{organ}')

        macro_avg = MetricsLambda(
            partial(_extract_macro_avg, metric_key=metric_key),
            master_metric,
        )
        macro_avg.attach(engine, f'{metric_key}-{name}')


def attach_organ_shapes_metric(engine, organs, gt=False):
    _attach_shapes_and_holes(engine, organs, out_key='activations', name='gen')

    if gt:
        _attach_shapes_and_holes(engine, organs, out_key='gt_map', name='gt')
