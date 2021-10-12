import csv
import logging
import os
import subprocess
import numpy as np
import pandas as pd

from medai.utils.lock import with_lock
from medai.datasets.iu_xray import DATASET_DIR as IU_DIR
from medai.datasets.mimic_cxr import DATASET_DIR as MIMIC_DIR
from medai.metrics.report_generation.labeler.utils import (
    HolisticLabeler,
    CacheLookupLabeler,
    NBatchesLabeler,
    AvoidDuplicatedLabeler,
)
from medai.utils import TMP_DIR

LOGGER = logging.getLogger(__name__)

TMP_FOLDER = os.path.join(TMP_DIR, 'mirqi')
MIRQI_FOLDER = '~/software/MIRQI'
CHEXPERT_PYTHON = '~/software/miniconda3/envs/chexpert-label/bin/python'

# Medical Image Reporting Quality Indexing
def MIRQI(gt_list, cand_list, pos_weight=0.8, attribute_weight=0.3):
    """Compute the score of matching keyword and associated attributes
    between gt list and candidate list.

    Copied from https://github.com/xiaosongwang/MIRQI, minor changes:
        - return format
        - linter fixes
        - docstring improvements

    Args:
        gt_list -- list of entities. Each entity is a tuple
            (mention, abnormality, valoration, adjectives)
        cand_list -- list of entities
    Returns:
        dict with MIRQI-f, MIRQI-r and MIRQI-p
    """

    MIRQI_r = []
    MIRQI_p = []
    MIRQI_f = []

    for gt_report_entry, cand_report_entry in zip(gt_list, cand_list):
        # attribute_cand_all = []

        pos_count_in_gt = 0
        pos_count_in_cand = 0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        for gt_entity in gt_report_entry:
            if gt_entity[2] == 'NEGATIVE':
                continue
            pos_count_in_gt = pos_count_in_gt + 1
        # neg_count_in_gt = len(gt_report_entry) - pos_count_in_gt

        for _, cand_entity in enumerate(cand_report_entry):
            if cand_entity[2] == 'NEGATIVE':
                for _, gt_entity in enumerate(gt_report_entry):
                    if  gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            tn = tn + 1     # true negative hits
                            break
                        else:
                            fn = fn + 1     # false negative hits
                            break
            else:
                pos_count_in_cand = pos_count_in_cand + 1
                for _, gt_entity in enumerate(gt_report_entry):
                    if gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            fp = fp + 1     # false positive hits
                            break
                        else:
                            # true positive hits (key words part)
                            tp = tp + 1.0 - attribute_weight
                            # count attribute hits
                            if gt_entity[3] == '':
                                break
                            attributes_all_gt = gt_entity[3].split('/')
                            attribute_hit_count = 0
                            for attribute in attributes_all_gt:
                                if attribute in cand_entity[3]:
                                    attribute_hit_count = attribute_hit_count + 1
                            # true positive hits (attributes part)
                            tp = tp + attribute_hit_count/len(attributes_all_gt)*attribute_weight
                            break
        # neg_count_in_cand = len(cand_report_entry) - pos_count_in_cand
        #
        # calculate score for positive/uncertain mentions
        if pos_count_in_gt == 0 and pos_count_in_cand == 0:
            score_r = 1.0
            score_p = 1.0
        elif pos_count_in_gt == 0 and pos_count_in_cand != 0:
            score_r = 0.0
            score_p = 0.0
        elif pos_count_in_gt != 0 and pos_count_in_cand == 0:
            score_r = 0.0
            score_p = 0.0
        else:
            score_r = tp / (tp + fn + 0.000001)
            score_p = tp / (tp + fp + 0.000001)

        # calculate score for negative mentions
        # if neg_count_in_cand != 0 and neg_count_in_gt != 0:
        if tn != 0:
            score_r = score_r * pos_weight + tn / (tn + fp + 0.000001) * (1.0 - pos_weight)
            score_p = score_p * pos_weight + tn / (tn + fn + 0.000001) * (1.0 - pos_weight)

        MIRQI_r.append(score_r)
        MIRQI_p.append(score_p)
        rec_prec = (score_r + score_p)
        MIRQI_f.append(2*(score_r * score_p) / rec_prec if rec_prec != 0.0 else 0.0)

    scores = {
        'MIRQI-r': MIRQI_r,
        'MIRQI-p': MIRQI_p,
        'MIRQI-f': MIRQI_f,
    }

    return scores


def MIRQI_v2(gt_list, cand_list, epsilon=1e-6):
    """Compute a v2 of the MIRQI metric.

    It returns scores: MIRQI-r, MIRQI-p, MIRQI_sp, MIRQI_np, MIRQI_f, MIRQI_attr_p, MIRQI_attr_r
    """

    MIRQI_r = [] # Recall
    MIRQI_p = [] # Precision (Positive predictive value)
    MIRQI_sp = [] # Specificity = "Negative recall"
    MIRQI_np = [] # Negative predictive value
    MIRQI_f = [] # F1

    MIRQI_attr_p = [] # Precision in attributes
    MIRQI_attr_r = [] # Recall in attributes

    for gt_report_entry, cand_report_entry in zip(gt_list, cand_list):
        pos_count_in_gt = 0
        pos_count_in_cand = 0
        tp = 0.0
        fp = 0.0
        tn = 0.0
        fn = 0.0

        tp_attr = 0
        fp_attr = 0
        fn_attr = 0

        for gt_entity in gt_report_entry:
            if len(gt_entity) < 3:
                LOGGER.debug('Len less than 3: %s %s', gt_report_entry, cand_report_entry)
            if gt_entity[2] == 'NEGATIVE':
                continue
            pos_count_in_gt = pos_count_in_gt + 1
        neg_count_in_gt = len(gt_report_entry) - pos_count_in_gt

        for _, cand_entity in enumerate(cand_report_entry):
            if cand_entity[2] == 'NEGATIVE':
                for _, gt_entity in enumerate(gt_report_entry):
                    if  gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            tn = tn + 1     # true negative hits
                            break
                        else:
                            fn = fn + 1     # false negative hits
                            break
            else:
                pos_count_in_cand = pos_count_in_cand + 1
                for _, gt_entity in enumerate(gt_report_entry):
                    if gt_entity[1] == cand_entity[1]:
                        if gt_entity[2] == 'NEGATIVE':
                            fp = fp + 1     # false positive hits
                            break
                        else:
                            tp = tp + 1

                            # count attribute hits
                            if gt_entity[3] == '':
                                break
                            gt_attrs = set(gt_entity[3].split('/'))
                            cand_attrs = set(cand_entity[3].split('/'))

                            tp_attr += len(gt_attrs.intersection(cand_attrs))
                            fp_attr += len(cand_attrs - gt_attrs)
                            fn_attr += len(gt_attrs - cand_attrs)

                            break
        neg_count_in_cand = len(cand_report_entry) - pos_count_in_cand

        # Compute recall and precision
        if pos_count_in_gt == 0 and pos_count_in_cand == 0:
            score_r = 1.0
            score_p = 1.0

            score_attr_r = 1.0
            score_attr_p = 1.0
        elif pos_count_in_gt == 0 or pos_count_in_cand == 0:
            score_r = 0.0
            score_p = 0.0

            score_attr_r = 0.0
            score_attr_p = 0.0
        else:
            score_r = tp / (tp + fn + epsilon)
            score_p = tp / (tp + fp + epsilon)

            score_attr_r = tp_attr / (tp_attr + fn_attr + epsilon)
            score_attr_p = tp_attr / (tp_attr + fp_attr + epsilon)

        # Compute spec and negative predictive value
        if neg_count_in_gt == 0 and neg_count_in_cand == 0:
            score_sp = 1.0
            score_np = 1.0
        elif neg_count_in_gt == 0 or neg_count_in_cand == 0:
            score_sp = 0.0
            score_np = 0.0
        else:
            score_sp = tn / (tn + fp + epsilon)
            score_np = tn / (tn + fn + epsilon)

        MIRQI_r.append(score_r)
        MIRQI_p.append(score_p)
        MIRQI_sp.append(score_sp)
        MIRQI_np.append(score_np)

        rec_prec = (score_r + score_p)
        MIRQI_f.append(2*(score_r * score_p) / rec_prec if rec_prec != 0.0 else 0.0)

        MIRQI_attr_p.append(score_attr_p)
        MIRQI_attr_r.append(score_attr_r)

    scores = {
        'MIRQI-v2-r': MIRQI_r,
        'MIRQI-v2-p': MIRQI_p,
        'MIRQI-v2-sp': MIRQI_sp,
        'MIRQI-v2-np': MIRQI_np,
        'MIRQI-v2-f': MIRQI_f,
        'MIRQI-v2-attr-p': MIRQI_attr_p,
        'MIRQI-v2-attr-r': MIRQI_attr_r,
    }

    return scores


@with_lock(TMP_FOLDER, 'caller_id', raise_error=True)
def _call_mirqi_for_reports(reports, caller_id='main'):
    """Applies MIRQI scorer for a set of GT and generated reports."""
    # Tmp folder can be removed afterwards
    os.makedirs(TMP_FOLDER, exist_ok=True)

    # Temp filenames
    INPUT_PATH = os.path.join(TMP_FOLDER, f'input_{caller_id}.csv')
    OUTPUT_PATH = os.path.join(TMP_FOLDER, f'output_{caller_id}.csv')

    # Create input files
    pd.DataFrame(reports).to_csv(INPUT_PATH, header=False, index=False, quoting=csv.QUOTE_ALL)

    # Call MIRQI
    cmd_cd = f'cd {MIRQI_FOLDER}'
    cmd_call = f'{CHEXPERT_PYTHON} evaluate.py \
        --reports_path_cand {INPUT_PATH} --output_path {OUTPUT_PATH}'
    cmd = f'{cmd_cd} && {cmd_call}'

    try:
        LOGGER.info('Evaluating %s reports with MIRQI...', f'{len(reports):,}')
        LOGGER.debug('Calling %s', cmd_call)
        subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        LOGGER.error('MIRQI script failed: %s', e.stderr)
        raise

    # Read MIRQI output # contains reports, "graph" and scores
    out_df = pd.read_csv(OUTPUT_PATH)
    attributes = out_df['attributes'].to_numpy()
    # shape: n_reports

    return np.expand_dims(attributes, 1) # shape: (n_reports, 1)


class MIRQILabeler(HolisticLabeler):
    """MIRQI labeler.

    This labeler is a bit different to chexpert:
        - chexpert returns a matrix of (n_samples, n_diseases) with 0/1
        - MIRQI returns tuples with detailed information about each disease
        - To avoid changing too much code, this labeler has only one disease: "attributes"
        - the attributes are returned as a list/ndarray of str with the detailed information
    """
    diseases = ['attributes']

    def __init__(self, **kwargs):
        super().__init__(None)

        self.kwargs = kwargs

    def forward(self, reports):
        labels = _call_mirqi_for_reports(reports, **self.kwargs)

        return labels


def _load_gt_df(dataset_name):
    gt_attributes_fpath = os.path.join(
        MIMIC_DIR if 'mimic' in dataset_name else IU_DIR,
        'reports', 'reports_with_mirqi_labels.csv',
    )
    if not os.path.isfile(gt_attributes_fpath):
        raise Exception(f'Ground truth labels not found: {gt_attributes_fpath}')

    # Load CSV
    gt_with_attributes = pd.read_csv(gt_attributes_fpath)
    return gt_with_attributes


def _fetch_gt_attributes(target_df, gt_with_labels):
    """Given a target_df and a gt_df, get the chexpert-labels for the target reports.

    Makes sure the order of the reports is the same.
    """
    # Assure it has all necessary reports
    target_reports = set(target_df['filename'])
    saved_reports = set(gt_with_labels['filename'])
    if not target_reports.issubset(saved_reports):
        missing = saved_reports.difference(target_reports)
        raise Exception(f'GT missing {len(missing)} reports')

    # Merge on filenames
    merged = target_df.merge(gt_with_labels, how='left', on='filename')

    assert len(merged) == len(target_df), \
        f'Size mismatch: {len(merged)} vs {len(target_df)}'

    # Return only np.array with labels
    cols = ['attributes']
    return merged[cols].to_numpy() # shape: (n_reports, 1)


def apply_mirqi_to_df(df, timestamp, batches=None, dataset_name='iu-x-ray'):
    # Load attributes for GT
    gt_attributes_df = _load_gt_df(dataset_name)

    gt_attributes = _fetch_gt_attributes(df, gt_attributes_df)

    # Calculate attributes for Generated
    labeler = MIRQILabeler(caller_id=timestamp)
    labeler = CacheLookupLabeler(labeler, gt_attributes_df)
    labeler = NBatchesLabeler(labeler, batches)
    labeler = AvoidDuplicatedLabeler(labeler)

    gen_attributes = labeler(list(df['generated']))

    df = df.assign(**{
        'attributes-gt': gt_attributes.squeeze(),
        'attributes-gen': gen_attributes.squeeze(),
    })

    return df
