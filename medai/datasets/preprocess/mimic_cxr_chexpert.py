"""Compute chexpert labels for MIMIC-CXR dataset."""
import os
import json
import logging
import pandas as pd
import numpy as np

from medai.datasets.mimic_cxr import DATASET_DIR
from medai.metrics.report_generation.chexpert import ChexpertLabeler
from medai.metrics.report_generation.labeler import (
    NBatchesLabeler,
    AvoidDuplicatedLabeler,
)
from medai.utils import config_logging, timeit_main

LOGGER = logging.getLogger(__name__)

def _load_study_to_filepath():
    master_df = pd.read_csv(os.path.join(DATASET_DIR, 'master_metadata.csv'))
    master_df = master_df[['report_fpath', 'study_id']].set_index('study_id')['report_fpath']
    study_id_to_filepath = master_df.to_dict()

    # key: study(int), value: filepath
    return study_id_to_filepath


def _load_clean_reports():
    fname = os.path.join(DATASET_DIR, 'reports', 'reports.clean.v4-2.json')
    with open(fname, 'r') as f:
        clean_reports = json.load(f)
    return clean_reports


def _save_numpy_backup(labels, out_fpath):
    fpath = out_fpath.replace('.csv', '.backup.npy')
    np.save(fpath, labels)
    LOGGER.info('Saved backup NPY file to: %s', fpath)


@timeit_main(LOGGER)
def label_reports_and_save(max_samples=None):
    out_fpath = os.path.join(DATASET_DIR, 'reports', 'reports_with_chexpert_labels.csv')

    if os.path.isfile(out_fpath):
        LOGGER.error('File %s already exists, backup first!', out_fpath)
        return

    study_id_to_filepath = _load_study_to_filepath()

    clean_reports = _load_clean_reports()
    studies = list(clean_reports)

    if max_samples is not None:
        LOGGER.warning('Using only %d max_samples', max_samples)
        studies = studies[:max_samples]

    # Get reports text ant their filepaths
    reports = [
        clean_reports[study_id]['clean_text']
        for study_id in studies
    ]
    filepaths = [
        study_id_to_filepath[int(study_id)]
        for study_id in studies
    ]

    # Apply labeler
    labeler = ChexpertLabeler(
        fill_empty=-2, fill_uncertain=-1, caller_id='mimic-preprocess',
    )
    labeler = NBatchesLabeler(labeler)
    labeler = AvoidDuplicatedLabeler(labeler)

    labels = labeler(reports)

    # Save labels as npy array, in case something fails later
    _save_numpy_backup(labels, out_fpath)

    assert labels.shape == (len(reports), 14)

    # Save as DF
    df = pd.DataFrame(labels, columns=labeler.diseases)
    df['Reports'] = reports
    df['filename'] = filepaths
    df = df[['filename', 'Reports', *labeler.diseases]]

    df.to_csv(out_fpath, index=False)



if __name__ == '__main__':
    config_logging()

    label_reports_and_save()
