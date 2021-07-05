import os
import logging
import pandas as pd

from medai.datasets.mimic_cxr import DATASET_DIR
from medai.metrics.report_generation.mirqi import MIRQILabeler
from medai.metrics.report_generation.labeler import (
    NBatchesLabeler,
    AvoidDuplicatedLabeler,
)
from medai.utils import timeit_main, config_logging

LOGGER = logging.getLogger('medai.preprocess.mimic.mirqi')

def _load_reports_df():
    fpath = os.path.join(DATASET_DIR, 'reports', 'reports_with_chexpert_labels.csv')
    df = pd.read_csv(fpath)
    df = df[['filename', 'Reports']]

    return df


@timeit_main(LOGGER)
def label_reports_and_save(max_samples=None):
    out_fpath = os.path.join(DATASET_DIR, 'reports', 'reports_with_mirqi_labels.csv')

    if os.path.isfile(out_fpath):
        LOGGER.error('File %s already exists, backup first!', out_fpath)
        return

    df = _load_reports_df()

    if max_samples is not None:
        LOGGER.warning('Using only %d max_samples', max_samples)
        df = df[:max_samples]


    reports = list(df['Reports'])

    labeler = MIRQILabeler(caller_id='mimic-preprocess-mirqi')
    labeler = NBatchesLabeler(labeler)
    labeler = AvoidDuplicatedLabeler(labeler)

    attributes = labeler(reports)

    if attributes.shape != (len(reports), 1):
        LOGGER.error(
            'Wrong shape: expected=%s, got=%s (saving anyway)',
            (len(reports), 1), attributes.shape,
        )

    df['attributes'] = attributes.squeeze()

    df.to_csv(out_fpath, index=False)
    LOGGER.info('Saved MIRQI labels to to %s', out_fpath)


if __name__ == '__main__':
    config_logging()

    label_reports_and_save()
