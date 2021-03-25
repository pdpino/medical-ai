import logging
import pandas as pd
from ignite.metrics import Metric

from medai.metrics.detection.coco_map.coco_wrapper import VinBigDataEval

LOGGER = logging.getLogger(__name__)

_ASSERT_SAME_IMAGES = False

class MAPCocoMetric(Metric):
    """Mean Average-precision (COCO-like) metric."""
    def __init__(self, gt_df, writer, donotcompute=False, **kwargs):
        self.csv_writer = writer

        # FIXME: using this for test subset, to avoid calculation of the metric
        self.donotcompute = donotcompute

        # Create eval-object
        self.vineval = VinBigDataEval(gt_df)

        super().__init__(**kwargs)

    def reset(self):
        super().reset()

        self.csv_writer.reset()


    def update(self, output):
        """Updates its internal count.

        Args:
            output: tuple of (image_names, coco_predictions),
        """
        image_names, predictions = output

        self.csv_writer.write(zip(image_names, predictions))


    def compute(self):
        self.csv_writer.close()

        if self.donotcompute:
            return 0

        # Load predictions
        pred_df = pd.read_csv(self.csv_writer.filepath)

        if _ASSERT_SAME_IMAGES:
            gt_images = set(self.vineval.image_name_to_idx.keys())
            pred_images = set(pred_df['image_id'])
            if gt_images != pred_images:
                LOGGER.error(
                    'Images from GT and pred do not match: %d %d (%s)',
                    len(gt_images), len(pred_images), self.csv_writer.filepath,
                )

        # Evaluate
        cocoEvalRes = self.vineval.evaluate(pred_df)

        return cocoEvalRes.stats[0]
