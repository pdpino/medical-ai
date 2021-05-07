import os
import logging
from ignite.engine import Events

from medai.utils.csv import CSVWriter
from medai.utils.files import get_results_folder
from medai.utils.nlp import ReportReader


LOGGER = logging.getLogger(__name__)


def _get_outputs_fpath(run_id, free=False):
    assert run_id.task == 'rg'

    folder = get_results_folder(run_id, save_mode=True)
    suffix = 'free' if free else 'notfree'
    path = os.path.join(folder, f'outputs-{suffix}.csv')

    return path


def attach_report_writer(engine, run_id, vocab, assert_n_samples=None, free=False):
    """Attach a report-writer to an engine.

    For each example in the dataset writes to a CSV the generated report and ground truth.
    """
    report_reader = ReportReader(vocab)

    fpath = _get_outputs_fpath(run_id, free=free)
    writer = CSVWriter(fpath, columns=[
        'filename',
        'epoch',
        'dataset_type',
        'ground_truth',
        'generated',
    ])

    @engine.on(Events.STARTED)
    def _open_writer(engine):
        writer.open()

        engine.state.line_counter = 0

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_text(engine):
        output = engine.state.output
        filenames = engine.state.batch.report_fnames
        gt_reports = output['flat_clean_reports_gt']
        gen_reports = output['flat_clean_reports_gen']

        epoch = engine.state.epoch
        dataset_type = engine.state.dataloader.dataset.dataset_type

        # Save result
        for report_idxs, generated_idxs, filename in zip(
            gt_reports,
            gen_reports,
            filenames,
            ):
            # Convert to text
            report = report_reader.idx_to_text(report_idxs)
            generated = report_reader.idx_to_text(generated_idxs)

            # HOTFIX: If text is empty, may be loaded as NaN and produce errors
            if generated == '':
                generated = '--'
            if report == '':
                LOGGER.warning(
                    'Empty GT report (%s): %s',
                    filename,
                    report_idxs,
                )
                report = '--'

            # Add quotes to avoid issues with commas
            report = f'"{report}"'
            generated = f'"{generated}"'

            writer.write(
                filename,
                epoch,
                dataset_type,
                report,
                generated,
            )

            engine.state.line_counter += 1

    @engine.on(Events.COMPLETED)
    def _close_writer():
        writer.close()

        sample_counter = engine.state.line_counter

        if assert_n_samples is not None:
            if sample_counter == assert_n_samples:
                LOGGER.debug(
                    'Correct amount of samples: %d, written to %s',
                    sample_counter, os.path.basename(fpath),
                )
            else:
                LOGGER.error(
                    'Incorrect amount of samples: written=%d vs should=%d, written to: %s',
                    sample_counter, assert_n_samples, fpath,
                )


def delete_previous_outputs(run_id, free=False):
    fpath = _get_outputs_fpath(run_id, free=free)

    if os.path.isfile(fpath):
        os.remove(fpath)
        LOGGER.info('Deleted previous outputs file at %s', fpath)
