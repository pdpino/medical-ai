import os
import logging
from ignite.engine import Events
from ignite.utils import to_onehot

from medai.utils.csv import CSVWriter
from medai.utils.files import get_results_folder


LOGGER = logging.getLogger(__name__)


def _get_outputs_fpath(run_id, save_mode=False):
    folder = get_results_folder(run_id, save_mode=save_mode)
    path = os.path.join(folder, 'outputs.csv')

    return path


def add_suffix_to_diseases(diseases, suffix):
    return [
        f'{disease}-{suffix}'
        for disease in diseases
    ]


def attach_prediction_writer(engine, run_id, diseases, assert_n_samples=None):
    """Attach a prediction-writer to an engine.

    For each example in the dataset writes to a CSV the ground truth and generated prediction.
    """
    fpath = _get_outputs_fpath(run_id, save_mode=True)
    writer = CSVWriter(fpath, columns=[
        'filename',
        'epoch',
        'dataset_type',
        *add_suffix_to_diseases(diseases, 'gt'),
        *add_suffix_to_diseases(diseases, 'pred'),
    ])

    @engine.on(Events.STARTED)
    def _open_writer(engine):
        writer.open()

        engine.state.line_counter = 0

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_prediction(engine):
        epoch = engine.state.epoch

        output = engine.state.output
        filenames = engine.state.batch.image_fname
        batch_pred = output['pred_labels'] # shape: bs, n_labels
        batch_gt = output['gt_labels']
        # shape(multilabel=True): bs, n_labels
        # shape(multilabel=False): bs

        dataset = engine.state.dataloader.dataset
        dataset_type = dataset.dataset_type
        assert diseases == dataset.labels, f'Diseases do not match: {diseases} vs {dataset.labels}'
        multilabel = dataset.multilabel

        if not multilabel:
            batch_gt = to_onehot(batch_gt, len(diseases))
            # shape: bs, n_labels

        # Save result
        for sample_gt, sample_pred, filename in zip(
            batch_gt,
            batch_pred,
            filenames,
        ):
            # gt shape: n_labels
            # pred shape: n_labels

            writer.write(
                filename,
                epoch,
                dataset_type,
                *sample_gt.tolist(),
                *sample_pred.tolist(),
            )

            engine.state.line_counter += 1

    @engine.on(Events.COMPLETED)
    def _close_writer():
        writer.close()

        if assert_n_samples is not None:
            sample_counter = engine.state.line_counter

            if sample_counter == assert_n_samples:
                LOGGER.info(
                    'Correct amount of samples: %d, written to %s',
                    sample_counter, fpath,
                )
            else:
                LOGGER.error(
                    'Incorrect amount of samples: written=%d vs should=%d, written to: %s',
                    sample_counter, assert_n_samples, fpath,
                )


def delete_previous_outputs(run_id):
    fpath = _get_outputs_fpath(run_id, save_mode=False)

    if os.path.isfile(fpath):
        os.remove(fpath)
        LOGGER.info('Deleted previous outputs file at %s', fpath)
