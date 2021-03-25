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

    @engine.on(Events.STARTED)
    def _open_writer(engine):
        writer.open()

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_prediction(engine):
        output = engine.state.output
        image_names = output['image_fnames']
        predictions = output['coco_predictions']

        writer.write(zip(image_names, predictions))


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