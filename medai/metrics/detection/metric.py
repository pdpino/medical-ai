import logging
import os
import pandas as pd
from ignite.metrics import Metric

from medai.datasets.vinbig import DATASET_DIR
from medai.metrics.detection.writer import CocoResultsWriter
from medai.metrics.detection.coco_wrapper import VinBigDataEval
from medai.utils.shapes import heatmap_to_bb

LOGGER = logging.getLogger(__name__)

_ASSERT_SAME_IMAGES = False

class MAPCocoMetric(Metric):
    """Mean Average-precision (COCO-like) metric."""
    def __init__(self, gt_df, temp_fpath, cls_thresh=0.3, **kwargs):
        self.csv_writer = CocoResultsWriter(temp_fpath)

        # TODO: pass this as param??
        self.cls_thresh = cls_thresh
        self.heat_thresh = 0.5

        # Create eval-object
        self.vineval = VinBigDataEval(gt_df)

        super().__init__(**kwargs)

    def reset(self):
        super().reset()

        self.csv_writer.reset()

    def _iterate_predictions(self, output):
        for image_name, preds, heatmaps in zip(*output):
            # preds shape: (n_diseases,)
            # heatmaps shape: (n_diseases, height, width)

            predictions = []
            for disease_idx, (pred, heatmap) in enumerate(zip(preds, heatmaps)):
                # pred shape: 1
                # heatmap shape: (height, width)
                score = pred.item()
                if score >= self.cls_thresh:
                    bb = heatmap_to_bb(heatmap, self.heat_thresh)
                    if bb is not None:
                        predictions.append((disease_idx, score, *bb))
                    else:
                        # TODO: what to do here?
                        pass

            if len(predictions) == 0:
                predictions = [
                    (14, 1, 0, 0, 1, 1), # Predicts no-finding
                ]

            yield (image_name, predictions)


    def update(self, output):
        """Updates its internal count.

        Args:
            output: tuple of (image_names, pred_labels, heatmaps),

            image_names: list of str, shape (batch_size,)
            pred_labels: tensor of predictions (sigmoided), shape (batch_size, n_diseases)
            heatmaps: tensor of shape (batch_size, n_diseases, height, width)
        """
        self.csv_writer.write(self._iterate_predictions(output))


    def compute(self):
        self.csv_writer.close()

        # Load predictions
        pred_df = pd.read_csv(self.csv_writer.filepath)

        if _ASSERT_SAME_IMAGES:
            gt_images = set(self.vineval.image_ids)
            pred_images = set(pred_df['image_id'])
            if gt_images != pred_images:
                LOGGER.error(
                    'Images from GT and pred do not match: %d %d (%s)',
                    len(gt_images), len(pred_images), self.csv_writer.filepath,
                )

        # Evaluate
        cocoEvalRes = self.vineval.evaluate(pred_df)

        return cocoEvalRes.stats[0]
