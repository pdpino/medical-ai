from itertools import islice

import numpy as np
import rasterio.features
from shapely.geometry import Polygon

_MAX_BBS = 3

def _get_score(mask, bb):
    x_min, y_min, x_max, y_max = map(int, bb)

    box = mask[y_min:y_max, x_min:x_max]
    max_value = box.max().item()

    return max_value


def _h2bb_thresh_multiple(mask, heat_thresh=0.5):
    """Receives a heatmap, threshold it, and return all BBs found, sorting by max values.

    Args:
        mask -- tensor of shape (height, width), with values between 0 and 1.
        heat_thresh -- float to find polygons in the heatmap
    Returns:
        tuple of (xmin, ymin, xmax, ymax) representing the Bounding-box.
        None if no polygons are found
    """
    # Apply threshold
    thresholded_mask = mask >= heat_thresh

    # To numpy
    thresholded_mask = thresholded_mask.detach().cpu().numpy().astype(np.uint8)

    # Obtain shapes
    shapes = rasterio.features.shapes(thresholded_mask)

    # Get boundaries of the polygons found
    boundaries = [
        shape['coordinates']
        for shape, value in islice(shapes, 0, _MAX_BBS)
        if value == 1
    ]

    if len(boundaries) == 0:
        return []

    # Transform to shapely.Polygons
    polygons = [
        Polygon(bounds[0])
        for bounds in boundaries
    ]

    bbs_with_score = [
        (_get_score(mask, poly.bounds), poly.bounds)
        for poly in polygons
    ]

    # Sort by score
    bbs_with_score = sorted(bbs_with_score, key=lambda p: p[0], reverse=True)

    return bbs_with_score
