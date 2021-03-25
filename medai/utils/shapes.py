from collections import defaultdict
import logging
import torch
import numpy as np
import mahotas.polygon
import rasterio.features
from shapely.geometry import Polygon

LOGGER = logging.getLogger(__name__)


#################### Functions used with organ-masks ####################
# TODO: rename functions as "for_organs", or alike
def calculate_polygons(mask, ignore_idx=0):
    """Finds the polygons in a mask.

    Args:
        mask -- tensor or array of shape (heigh, width)
    Returns:
        list of (coordinates, value), where:
        - coordinates: matches a Geo-JSON polygon shape, as returned per
            rasterio. I.e. is a list of ring lists, where each ring list
            is a list of points; the first ring is exterior, whilst the
            others are holes.
            See here: https://tools.ietf.org/html/rfc7946#section-3.1.6
        - value: value of the polygon in the mask
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    if not isinstance(ignore_idx, (list, tuple)):
        ignore_idx = (ignore_idx,)

    if mask.dtype not in (np.uint8, np.int16, np.int32):
        mask = mask.astype(np.uint8)

    shapes = rasterio.features.shapes(mask)
    polygons = [
        (shape['coordinates'], value)
        for shape, value in shapes
        if value not in ignore_idx
    ]
    return polygons


def get_largest_shapes(polygons, name=None):
    """Returns the largest shapes in a list of polygons.

    Args:
        polygons -- list of (coordinates, value), as described by calculate_polygons.
            The `value` corresponds to the idx depicted in the original array
            (e.g. organ_idx).
    Returns:
        list of (exterior-coordinates, value), where `exterior-coordinates` is a
            list of (x,y) coordinates of the exterior of the polygon.
    """
    polygons_by_organ = defaultdict(list)

    for boundaries, organ_idx in polygons:
        exterior = boundaries[0]
        polygons_by_organ[organ_idx].append(Polygon(exterior))

    largest_polygons = []

    for organ_idx in range(1, 4):
        organ_polys = polygons_by_organ[organ_idx]
        if len(organ_polys) == 0:
            largest_polygons.append(([], organ_idx))
            LOGGER.warning('Empty polygon for organ=%d, name=%s', organ_idx, name)
            continue

        organ_polys = sorted(organ_polys, key=lambda p: p.area, reverse=True)
        largest_poly = organ_polys[0]

        largest_polygons.append((largest_poly.exterior.coords, organ_idx))

    return largest_polygons


def _safe_to_int(num):
    if isinstance(num, float):
        assert num.is_integer(), f'Number is not integer: {num}'
        num = int(num)
    return num


def polygons_to_array(polygons, size):
    """Transforms a list of polygons to an array.

    Args:
        polygons -- list of (exterior-coordinates, value)
        size -- tuple describing the shape of the array to fill
    Returns:
        np.array of shape `size`, filled with the polygon with their respective values
    """
    arr = np.zeros(size, dtype=np.uint8)

    for coords, color in polygons:
        if len(coords) == 0:
            continue

        coords = [
            (_safe_to_int(y), _safe_to_int(x))
            for x, y in coords
        ]

        mahotas.polygon.fill_polygon(coords, arr, color=color)

    return arr

################################################################################



def calc_area(bbox):
    """Calculates the area of a bounding-box.

    Args:
        bbox -- tuple/list of four elements: x_min, y_min, x_max, y_max
    """
    x1, y1, x2, y2 = bbox
    dy = max(0, y2 - y1)
    dx = max(0, x2 - x1)
    return dx * dy

def calc_iou(bbox1, bbox2):
    """Calculates IoU for two bboxes.

    Args:
        bbox1 -- bbox as tuple
        bbox2 -- bbox as tuple
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = calc_area((x1, y1, x2, y2))

    area1 = calc_area(bbox1)
    area2 = calc_area(bbox2)
    union = area1 + area2 - intersection

    return intersection / union