from collections import defaultdict, Counter
import logging
import numpy as np
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced

import rasterio.features
from shapely.geometry import Polygon

LOGGER = logging.getLogger(__name__)


def _divide_dict_by(items, divide_by):
    return {
        key: value / divide_by
        for key, value in items.items()
    }


class OrganShapesAndHolesMetric(Metric):
    """Computes shapes and holes in an organ segmentation output."""

    def __init__(self, ignore=[], **kwargs):
        super().__init__(**kwargs)

        if not isinstance(ignore, (tuple, list)):
            ignore = (ignore,)

        self._ignore_organs = set(ignore)

    @reinit__is_reduced
    def reset(self):
        self._n_holes_by_organ = Counter()
        self._n_shapes_by_organ = Counter()

        self._n_samples = 0

        super().reset()

    def _calculate_shapes_by_organ(self, arr_tensor):
        """Calculate the shapes for an array.

        Args:
            arr_tensor -- tensor of shape (batch_size, height, width)
        """
        arr_np = arr_tensor.detach().cpu().numpy().astype(np.int16)

        batch_shapes = []

        for sample_arr in arr_np:
            shapes_by_organ = defaultdict(list)

            for shape in rasterio.features.shapes(sample_arr):
                polygon, value = shape
                if value in self._ignore_organs:
                    continue

                organ_idx = int(value)
                shapes_by_organ[organ_idx].append(polygon['coordinates'])

            batch_shapes.append(shapes_by_organ)

        return batch_shapes

    @reinit__is_reduced
    def update(self, output):
        """Updates its internal count.

        Args:
            output: tensor of shape (batch_size, height, width)
        """
        self._n_samples += output.size(0)

        batch_shapes = self._calculate_shapes_by_organ(output)

        for shapes in batch_shapes:
            # shapes type: dict of list of polygons

            for organ_idx, polygons in shapes.items():
                # Count n_shapes
                self._n_shapes_by_organ[organ_idx] += len(polygons)

                if len(polygons) == 0:
                    continue

                # Count holes of the largest polygon
                # each polygon is a list of boundaries
                # first is the outside boundary, the rest are holes
                polygons = [
                    (
                        Polygon(poly[0]),
                        len(poly) - 1, # n_holes
                    )
                    for poly in polygons
                ]
                sorted_polygons = sorted(
                    polygons,
                    key=lambda tup: tup[0].area,
                    reverse=True,
                )
                unused_largest_polygon, n_holes = sorted_polygons[0]

                self._n_holes_by_organ[organ_idx] += n_holes


    @sync_all_reduce('_n_shapes_by_organ', '_n_samples')
    def compute(self):
        if self._n_samples == 0:
            LOGGER.error('_n_samples is 0')
            return None

        avg_n_shapes = _divide_dict_by(self._n_shapes_by_organ, self._n_samples)
        avg_n_holes = _divide_dict_by(self._n_holes_by_organ, self._n_samples)

        return {
            'n-shapes': avg_n_shapes,
            'n-holes': avg_n_holes,
        }
