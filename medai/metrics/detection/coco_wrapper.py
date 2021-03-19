"""Wrapper to call pycocotools.

Adapted from: https://www.kaggle.com/pestipeti/competition-metric-map-0-4
"""
import logging
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from medai.datasets.common.constants import VINBIG_DISEASES

LOGGER = logging.getLogger(__name__)


def _gen_categories():
    return [
        {
            'id': index,
            'name': disease_name,
            'supercategory': 'none',
        }
        for index, disease_name in enumerate(VINBIG_DISEASES)
    ]

def _gen_images_list(image_ids):
    return [
        {
            'id': idx,
        }
        for idx in range(len(image_ids))
    ]


class VinBigDataEval:
    """Helper class for calculating the competition metric.

    You should remove the duplicated annoatations from the `true_df` dataframe
    before using this script. Otherwise it may give incorrect results.

        >>> vineval = VinBigDataEval(valid_df)
        >>> cocoEvalResults = vineval.evaluate(pred_df)

    Arguments:
        true_df: pd.DataFrame Clean (no duplication) Training/Validating dataframe.

    Authors:
        Peter (https://kaggle.com/pestipeti)

    See:
        https://www.kaggle.com/pestipeti/competition-metric-map-0-4

    Returns: None

    """
    def __init__(self, true_df):

        self.true_df = true_df

        self.image_ids = true_df["image_id"].unique()
        self.annotations = {
            "type": "instances",
            "images": _gen_images_list(self.image_ids),
            "categories": _gen_categories(),
            "annotations": self.__gen_annotations(self.true_df, self.image_ids)
        }

        self.predictions = {
            "images": self.annotations["images"].copy(),
            "categories": _gen_categories(),
            "annotations": None,
        }

    def __gen_annotations(self, df, image_ids):
        LOGGER.debug("Generating annotation data...")
        # TODO: optimize this function!!
        k = 0
        results = []

        for idx, image_id in enumerate(image_ids):

            # Add image annotations
            for _, row in df[df["image_id"] == image_id].iterrows():

                results.append({
                    "id": k,
                    "image_id": idx,
                    "category_id": row["class_id"],
                    "bbox": np.array([
                        row["x_min"],
                        row["y_min"],
                        row["x_max"],
                        row["y_max"]]
                    ),
                    "segmentation": [],
                    "ignore": 0,
                    "area":(row["x_max"] - row["x_min"]) * (row["y_max"] - row["y_min"]),
                    "iscrowd": 0,
                })

                k += 1

        return results

    def __decode_prediction_string(self, pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)

    def __gen_predictions(self, df, image_ids):
        LOGGER.debug("Generating prediction data...")
        k = 0
        results = []

        for _, row in df.iterrows():

            image_id = row["image_id"]
            preds = self.__decode_prediction_string(row["PredictionString"])

            for _, pred in enumerate(preds):

                results.append({
                    "id": k,
                    "image_id": int(np.where(image_ids == image_id)[0]),
                    "category_id": int(pred[0]),
                    "bbox": np.array([
                        pred[2], pred[3], pred[4], pred[5]
                    ]),
                    "segmentation": [],
                    "ignore": 0,
                    "area": (pred[4] - pred[2]) * (pred[5] - pred[3]),
                    "iscrowd": 0,
                    "score": pred[1]
                })

                k += 1

        return results

    def evaluate(self, pred_df, n_imgs = -1):
        """Evaluating your results

        Arguments:
            pred_df: pd.DataFrame your predicted results in the
                     competition output format.

            n_imgs:  int Number of images use for calculating the
                     result.All of the images if `n_imgs` <= 0

        Returns:
            COCOEval object
        """

        if pred_df is not None:
            self.predictions["annotations"] = self.__gen_predictions(pred_df, self.image_ids)

        coco_ds = COCO()
        coco_ds.dataset = self.annotations
        coco_ds.createIndex()

        coco_dt = COCO()
        coco_dt.dataset = self.predictions
        coco_dt.createIndex()

        imgIds = sorted(coco_ds.getImgIds())

        if n_imgs > 0:
            imgIds = np.random.choice(imgIds, n_imgs)

        cocoEval = COCOeval(coco_ds, coco_dt, 'bbox')
        cocoEval.params.imgIds  = imgIds
        cocoEval.params.useCats = True
        cocoEval.params.iouType = "bbox"
        cocoEval.params.iouThrs = np.array([0.4])

        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval
