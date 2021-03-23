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


class VinBigDataEval:
    """Helper class for calculating the competition metric.

    You should remove the duplicated annotations from the `true_df` dataframe
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

        image_names = true_df["image_id"].unique()
        self.image_name_to_idx = {
            name: idx
            for idx, name in enumerate(image_names)
        }

        self.ground_truth = {
            "type": "instances",
            "categories": _gen_categories(),
            "annotations": self._gen_gt_annotations()
        }

        self.predictions = {
            "categories": _gen_categories(),
        }


    def _gen_images_list(self, image_names):
        return [
            { 'id': self.image_name_to_idx[name] }
            for name in image_names
        ]


    def _gen_gt_annotations(self):
        LOGGER.debug("Generating annotation data...")
        k = 0
        results = []

        cols = ['class_id', 'x_min', 'y_min', 'x_max', 'y_max']
        bboxes_by_image_name = self.true_df.groupby('image_id')[cols].apply(
            lambda x: list(x.values),
        ).to_dict()

        for image_name, rows in bboxes_by_image_name.items():
            idx = self.image_name_to_idx[image_name]

            for row in rows:
                class_id, x_min, y_min, x_max, y_max = row

                results.append({
                    "id": k,
                    "image_id": idx,
                    "category_id": class_id,
                    "bbox": row[1:],
                    "segmentation": [],
                    "ignore": 0,
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                })

                k += 1

        return results

    def __decode_prediction_string(self, pred_str):
        data = list(map(float, pred_str.split(" ")))
        data = np.array(data)

        return data.reshape(-1, 6)

    def _gen_pred_annotations(self, df):
        LOGGER.debug("Generating prediction data...")
        k = 0
        results = []

        for _, row in df.iterrows():
            image_name = row["image_id"]
            preds = self.__decode_prediction_string(row["PredictionString"])

            image_idx = self.image_name_to_idx[image_name]

            for pred in preds:

                results.append({
                    "id": k,
                    "image_id": image_idx,
                    "category_id": int(pred[0]),
                    "bbox": pred[2:6],
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
            # Create predictions data
            self.predictions["annotations"] = self._gen_pred_annotations(pred_df)

            pred_images = pred_df['image_id'].unique()
            self.predictions["images"] = self._gen_images_list(pred_images)

            # Set GT data to use only the predicted images
            self.ground_truth["images"] = self._gen_images_list(pred_images)


        coco_ds = COCO()
        coco_ds.dataset = self.ground_truth
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
