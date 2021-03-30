import logging

import pandas as pd
from ignite.engine import Events

from medai.metrics.detection.coco_writer.writer import CocoResultsWriter, get_outputs_fpath

LOGGER = logging.getLogger(__name__)


def attach_vinbig_writer(engine, dataloader, run_name, debug=True,
                         task='det', suffix=None, assert_samples=True):
    dataset_type = dataloader.dataset.dataset_type

    fpath = get_outputs_fpath(run_name, dataset_type, debug=debug,
                              task=task, prefix='submission', suffix=suffix)

    writer = CocoResultsWriter(fpath)

    coco_key = 'coco_predictions'
    if suffix:
        coco_key += f'_{suffix}'

    @engine.on(Events.STARTED)
    def _open_writer(unused_engine):
        writer.reset()

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_prediction(engine):
        output = engine.state.output
        image_names = output['image_fnames']
        predictions = output[coco_key]

        # Re-scale predictions
        # REVIEW: mAP needs to re-scale as well? coco_df_gt is re-scaled???
        batch = engine.state.batch
        height, width = batch.image.size()[-2:]
        rescaled_predictions = []

        for original_size, image_preds in zip(batch.original_size, predictions):
            original_height, original_width = original_size
            horizontal_scale = original_width / width
            vertical_scale = original_height / height

            rescaled_image_preds = []
            for disease_idx, score, xmin, ymin, xmax, ymax in image_preds:
                if disease_idx != 14:
                    xmin = int(xmin * horizontal_scale)
                    xmax = int(xmax * horizontal_scale)
                    ymin = int(ymin * vertical_scale)
                    ymax = int(ymax * vertical_scale)

                rescaled_image_preds.append((disease_idx, score, xmin, ymin, xmax, ymax))

            rescaled_predictions.append(rescaled_image_preds)

        writer.write(zip(image_names, rescaled_predictions))

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
