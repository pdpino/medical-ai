from itertools import islice

import numpy as np
import rasterio.features
from shapely.geometry import Polygon

_MAX_BBS = 3


def _h2bb_thresh_largest_area(mask, heat_thresh=0.5):
    """Receives a heatmap, threshold it, and return the BB with the largest area.

    Thresholds the mask (i.e. makes it a binary mask), and find the largest polygon
    with value=1.

    Args:
        mask -- tensor of shape (height, width), with values between 0 and 1.
        heat_thresh -- float to find polygons in the heatmap
    Returns:
        tuple of (xmin, ymin, xmax, ymax) representing the Bounding-box.
        None if no polygons are found
    """
    # Apply threshold
    mask = mask >= heat_thresh

    # To numpy
    mask = mask.detach().cpu().numpy().astype(np.uint8)

    # Obtain shapes
    shapes = rasterio.features.shapes(mask)

    # Get boundaries of the polygons found
    boundaries = [
        shape['coordinates']
        for shape, value in islice(shapes, 0, _MAX_BBS)
        if value == 1
    ]
    if len(boundaries) == 0:
        return None

    # Transform to shapely.Polygons
    polygons = [
        Polygon(bounds[0])
        for bounds in boundaries
    ]

    # Sort and keep the largest
    polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
    largest_polygon = polygons[0]

    # Return the BB
    return largest_polygon.bounds
