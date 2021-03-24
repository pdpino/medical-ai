import os
import logging
from functools import partial

import pandas as pd
from ignite.engine import Events
from ignite.metrics import MetricsLambda

from medai.metrics.detection.coco_map.metric import MAPCocoMetric
from medai.metrics.detection.coco_writer.writer import CocoResultsWriter, get_outputs_fpath
from medai.metrics.detection.mse import HeatmapMSE
from medai.utils.files import get_results_folder
from medai.utils.heatmaps import threshold_attributions
from medai.utils.metrics import attach_metric_for_labels


LOGGER = logging.getLogger(__name__)


def attach_vinbig_writer(engine, dataloader, run_name, debug=True,
                         task='det', assert_samples=True):
    # FIXME: this function is not needed for now, as the mAP metric generates a csv as an output

    dataset_type = dataloader.dataset.dataset_type

    fpath = get_outputs_fpath(run_name, dataset_type, debug=debug,
                              task=task, prefix='submission')

    writer = CocoResultsWriter(fpath)

    @engine.on(Events.STARTED)
    def _open_writer(engine):
        writer.open()

        engine.state.line_counter = 0

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_prediction(unused_engine):
        # epoch = engine.state.epoch

        # output = engine.state.output
        # filenames = engine.state.batch.image_fname
        # batch_pred = output['pred_labels'] # shape: bs, n_labels
        # batch_gt = output['gt_labels']
        # shape(multilabel=True): bs, n_labels
        # shape(multilabel=False): bs


        # image_id: str
        # predictions: list of (class_id, score, x_min, y_min, x_max, y_max)

        images_preds_iter = []
        writer.write(images_preds_iter)


    @engine.on(Events.COMPLETED)
    def _close_writer():
        writer.close()

        if assert_samples:
            n_samples = len(dataloader.dataset)

            df = pd.read_csv(writer.filepath)
            if len(df) == n_samples:
                LOGGER.info(
                    'Correct amount of samples: %d, written to %s',
                    n_samples, writer.filepath,
                )
            else:
                LOGGER.error(
                    'Incorrect amount of samples: written=%d vs should=%d, written to: %s',
                    len(df),n_samples, writer.filepath,
                )
