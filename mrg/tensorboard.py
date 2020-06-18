"""Tensorboard util functions."""
import torch
import numpy as np
import os
from tensorboardX import SummaryWriter

from mrg import utils


CLASSIFICATION_METRICS = ['roc_auc', 'prec', 'recall', 'acc']

def _get_log_dir(run_name, classification=True,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True):
    debug_folder = 'debug' if debug else ''
    mode_folder = 'classification' if classification else 'report_generation'
    return os.path.join(workspace_dir, mode_folder, 'runs', debug_folder, run_name)


def _write_histogram(writer, model, epoch, wall_time):
    for name, params in model.named_parameters():
        params = params.cpu().detach().numpy()
        writer.add_histogram(name, params, global_step=epoch, walltime=wall_time)


class TBClassificationWriter:
    def __init__(self, run_name, labels=None,
                 log_metrics=CLASSIFICATION_METRICS,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True, **kwargs):
        self.log_dir = _get_log_dir(run_name, classification=True,
                                    workspace_dir=workspace_dir, debug=debug)

        self.writer = SummaryWriter(self.log_dir, **kwargs)

        self.labels = list(labels)
        self.log_metrics = list(log_metrics)


    def write_histogram(self, model, epoch, wall_time):
        _write_histogram(self.writer, model, epoch, wall_time)


    def write_metrics(self, metrics, run_type, epoch, wall_time):
        if self.labels is None:
            raise Exception('Writer does not have labels to write classification metrics')

        loss = metrics.get('loss', -1)
        self.writer.add_scalar(f'Loss/{run_type}', loss, epoch, wall_time)

        for metric_base_name in self.log_metrics:
            for label in self.labels:
                metric_name = f'{metric_base_name}_{label}'
                metric_value = metrics.get(metric_name, -1)

                self.writer.add_scalar(f'{metric_name}/{run_type}',
                                       metric_value,
                                       epoch,
                                       wall_time,
                                       )

    def close(self):
        self.writer.close()


class TBReportGenerationWriter:
    def __init__(self, run_name,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True, **kwargs):
        self.log_dir = _get_log_dir(run_name, classification=False,
                                    workspace_dir=workspace_dir, debug=debug)

        self.writer = SummaryWriter(self.log_dir, **kwargs)


    def write_histogram(self, model, epoch, wall_time):
        _write_histogram(self.writer, model, epoch, wall_time)


    def write_metrics(self, metrics, run_type, epoch, wall_time):
        for metric_name in ['loss', 'word_acc']:
            metric_value = metrics.get(metric_name, -1)
            self.writer.add_scalar(f'{metric_name}/{run_type}', metric_value, epoch, wall_time)


    def close(self):
        self.writer.close()
