import os
import re
import logging
import pandas as pd
from ignite.engine import Events

from medai.metrics.report_generation import build_suffix
from medai.utils.csv import CSVWriter
from medai.utils.files import get_results_folder
from medai.utils.nlp import ReportReader


LOGGER = logging.getLogger(__name__)


def _get_outputs_fpath(run_id, free=False, best=None, beam_size=0):
    assert run_id.task == 'rg'

    folder = get_results_folder(run_id, save_mode=True)
    suffix = build_suffix(free, best, beam_size)
    path = os.path.join(folder, f'outputs-{suffix}.csv')

    return path


def attach_report_writer(engine, run_id, vocab, assert_n_samples=None,
                         free=False, best=None, beam_size=0):
    """Attach a report-writer to an engine.

    For each example in the dataset writes to a CSV the generated report and ground truth.
    """
    report_reader = ReportReader(vocab)

    fpath = _get_outputs_fpath(run_id, free=free, best=best, beam_size=beam_size)
    writer = CSVWriter(fpath, columns=[
        'filename',
        'epoch',
        'dataset_type',
        'ground_truth',
        'generated',
        'image_fname',
    ])

    @engine.on(Events.STARTED)
    def _open_writer(engine):
        writer.open()

        engine.state.line_counter = 0

    @engine.on(Events.ITERATION_COMPLETED)
    def _save_text(engine):
        output = engine.state.output
        filenames = engine.state.batch.report_fnames
        image_fnames = engine.state.batch.image_fnames
        gt_reports = output['flat_clean_reports_gt']
        gen_reports = output['flat_clean_reports_gen']

        epoch = engine.state.epoch
        dataset_type = engine.state.dataloader.dataset.dataset_type

        # Save result
        for report_idxs, generated_idxs, filename, image_fname in zip(
            gt_reports,
            gen_reports,
            filenames,
            image_fnames,
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
                image_fname,
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


def delete_previous_outputs(run_id, free=False, best=None, beam_size=0):
    fpath = _get_outputs_fpath(run_id, free=free, best=best, beam_size=beam_size)

    if os.path.isfile(fpath):
        os.remove(fpath)
        LOGGER.info('Deleted previous outputs file at %s', fpath)


def load_rg_outputs(run_id, free=False, best=None, beam_size=0, labeled=False):
    """Load report-generation output dataframe.

    Returns a DataFrame with columns:
    filename,epoch,dataset_type,ground_truth,generated
    """
    assert run_id.task == 'rg'

    results_folder = get_results_folder(run_id)
    suffix = build_suffix(free, best, beam_size)

    if labeled:
        name = f'outputs-labeled-{suffix}.csv'
    else:
        name = f'outputs-{suffix}.csv'

    outputs_path = os.path.join(results_folder, name)

    LOGGER.info('Loading RG outputs from: %s', name)

    if not os.path.isfile(outputs_path):
        LOGGER.error('Outputs file not found: %s', outputs_path)
        return None

    return pd.read_csv(
        outputs_path,
        keep_default_na=False, # Do not treat the empty-string as NaN value
    )


def get_best_outputs_info(run_id, free_values=None, only_best=None, only_beam=None):
    """Get the info of the outputs.csv saved for a run."""
    outputs_with_suffix = re.compile(
        r'outputs-(?P<free>free|notfree)(-(?P<suffix>[\w\-]+))?(\.bs(?P<beam>\d+))?\.csv',
    )

    # Grab all infos
    infos = []
    for filename in os.listdir(get_results_folder(run_id)):
        match = outputs_with_suffix.match(filename)
        if match:
            free, suffix, beam = match.group('free'), match.group('suffix'), match.group('beam')
            free = bool(free == 'free')
            beam = int(beam or 0)
            infos.append((filename, free, suffix, beam))

    # Choose by free_values and only_best
    chosen, leftout = [], []
    for _, free, best, beam_size in infos:
        free_chosen = free_values is None or free in free_values
        best_chosen = only_best is None or best in only_best
        beam_chosen = only_beam is None or beam_size in only_beam

        if free_chosen and best_chosen and beam_chosen:
            chosen.append((free, best, beam_size))
        else:
            leftout.append((free, best, beam_size))

    return chosen, leftout
