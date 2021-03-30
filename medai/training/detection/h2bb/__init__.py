from functools import partial

from medai.training.detection.h2bb.thresh_largest_area import _h2bb_thresh_largest_area
from medai.training.detection.h2bb.thresh_multiple import _h2bb_thresh_multiple
from medai.utils import tensor_to_range01


def _iter_h2bb(batch_preds, batch_heatmaps,
               cls_thresh=0.3, norm_heat=False,
               h2bb_fn=None, **kwargs):
    """Method to convert heatmaps to BBs.

    Args:
        batch_preds: tensor of predictions (sigmoided), shape (batch_size, n_diseases)
        batch_heatmaps: tensor of shape (batch_size, n_diseases, height, width)
        cls_thresh: Threshold to consider a prediction as positive
        norm_heat: whether or not to normalize heatmap scores
    Yield:
        tuple (image_name, predictions)
    """
    if norm_heat:
        # Notice this function rescales per image
        # (i.e. each heatmap of height x width will be normalized)
        batch_heatmaps = tensor_to_range01(batch_heatmaps)

    for preds, heatmaps in zip(batch_preds, batch_heatmaps):
        # preds shape: (n_diseases,)
        # heatmaps shape: (n_diseases, height, width)

        predictions = []
        for disease_idx, (pred, heatmap) in enumerate(zip(preds, heatmaps)):
            # pred shape: 1
            # heatmap shape: (height, width)

            score = pred.item()
            if score >= cls_thresh:
                bbs = h2bb_fn(heatmap, **kwargs)
                if isinstance(bbs, tuple):
                    bb = map(int, bbs)
                    predictions.append((disease_idx, score, *bb))
                elif isinstance(bbs, list):
                    for bb_score, bb in bbs:
                        bb = map(int, bb)
                        predictions.append((disease_idx, bb_score, *bb))
                else:
                    # no bb was found
                    pass

        if len(predictions) == 0:
            predictions = [
                (14, 1, 0, 0, 1, 1), # Predicts no-finding
            ]

        yield predictions


def _h2bb(*args, **kwargs):
    return list(_iter_h2bb(*args, **kwargs))


_H2BB_METHODS = {
    'thr-largarea': partial(_h2bb, h2bb_fn=_h2bb_thresh_largest_area),
    'thr-multiple': partial(_h2bb, h2bb_fn=_h2bb_thresh_multiple),
}

AVAILABLE_H2BB_METHODS = list(_H2BB_METHODS)

def get_h2bb_method(name, kwargs):
    if name not in _H2BB_METHODS:
        raise Exception(f'Method not found: {name}')

    method = _H2BB_METHODS[name]

    return partial(method, **kwargs)
