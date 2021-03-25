import numpy as np
import rasterio.features
from shapely.geometry import Polygon


def _h2bb_thresh_largest_area(mask, thresh=0.5):
    """Receives a heatmap, threshold it, and return the BB with the largest area.

    Thresholds the mask (i.e. makes it a binary mask), and find the largest polygon
    with value=1.

    Args:
        mask -- tensor of shape (height, width), with values between 0 and 1.
        thresh -- float to find polygons in the heatmap
    Returns:
        tuple of (xmin, ymin, xmax, ymax) representing the Bounding-box.
        None if no polygons are found

    TODO: maybe do not keep the largest one, but the strongest one (i.e. larger values)
    (new function??)
    """
    # Apply threshold
    mask = mask >= thresh

    # To numpy
    mask = mask.detach().cpu().numpy().astype(np.uint8)

    # Obtain shapes
    shapes = rasterio.features.shapes(mask)

    # Get boundaries of the polygons found
    boundaries = [
        shape['coordinates']
        for shape, value in shapes
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


# FIXME: delete _iter version??
def _iter_h2bb_thresh_largest_area(batch_preds, batch_heatmaps,
                                   cls_thresh=0.3, heat_thresh=0.8):
    """Method to convert heatmaps to BBs.

    See _h2bb_thresh_largest_area for a description.

    Args:
        pred_labels: tensor of predictions (sigmoided), shape (batch_size, n_diseases)
        heatmaps: tensor of shape (batch_size, n_diseases, height, width)
    Yield:
        tuple (image_name, predictions)
    """
    for preds, heatmaps in zip(batch_preds, batch_heatmaps):
        # preds shape: (n_diseases,)
        # heatmaps shape: (n_diseases, height, width)

        predictions = []
        for disease_idx, (pred, heatmap) in enumerate(zip(preds, heatmaps)):
            # pred shape: 1
            # heatmap shape: (height, width)
            score = pred.item()
            if score >= cls_thresh:
                bb = _h2bb_thresh_largest_area(heatmap, heat_thresh)
                if bb is not None:
                    predictions.append((disease_idx, score, *bb))
                else:
                    # TODO: what to do here?
                    pass

        if len(predictions) == 0:
            predictions = [
                (14, 1, 0, 0, 1, 1), # Predicts no-finding
            ]

        yield predictions

def h2bb_thresh_largest_area(*args, **kwargs):
    return list(_iter_h2bb_thresh_largest_area(*args, **kwargs))
