"""Tensorboard util functions."""
import os
import re
import numbers
from tensorboardX import SummaryWriter
from torch import nn

from medai import utils
from medai.utils.files import get_tb_log_folder, get_tb_large_log_folder

IGNORE_METRICS = [
    r'\Acm', # Confusion-matrix, no sense to put it in TB
    '_timer', # Medical correctness timers, return strings
]

class TBWriter:
    def __init__(self, run_name, task,
                 ignore_metrics=IGNORE_METRICS,
                 dryrun=False,
                 histogram=False,
                 large=False,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True, **kwargs):
        if large:
            _get_tb_folder = get_tb_large_log_folder
        else:
            _get_tb_folder = get_tb_log_folder

        self.log_dir = _get_tb_folder(
            run_name, task=task, workspace_dir=workspace_dir, debug=debug,
        )

        self.writer = SummaryWriter(self.log_dir,
                                    write_to_disk=not dryrun,
                                    **kwargs)

        self.ignore_regex = '|'.join(ignore_metrics)
        self.ignore_regex = re.compile(f'({self.ignore_regex})')

        self._histogram = histogram

        # Capitalize so in TB appears first
        self._name_mappings = {
            'loss': 'Loss',
            'acc': 'Acc',
            'hamming': 'Hamming',
            'bce': 'BCELoss',
            'roc_auc': 'Roc_auc',
            'iou': 'IoU',
            'iobb': 'IoBB',
            'mAP': 'MAP', # mean Average-precision (coco-challenge)
        }

        self._loss_regex = re.compile(r'(\w+)_loss')

    def _map_metric_name(self, name):
        found = self._loss_regex.search(name)
        if found:
            captured = found.group(1)
            return f'Loss_{captured}'

        return self._name_mappings.get(name, name)


    def write_histogram(self, model, epoch, wall_time):
        if not self._histogram:
            return

        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        for name, params in model.named_parameters():
            if params.numel() == 0:
                continue

            self.writer.add_histogram(name, params, global_step=epoch, walltime=wall_time)

            if params.grad is not None:
                self.writer.add_histogram(
                    f'{name}/grad', params.grad, global_step=epoch, walltime=wall_time,
                )


    def write_metrics(self, metrics, run_type, epoch, wall_time):
        for name, value in metrics.items():
            if self.ignore_regex.search(name) or not isinstance(value, numbers.Number):
                continue

            if value == -1:
                continue

            name = self._map_metric_name(name)
            self.writer.add_scalar(f'{name}/{run_type}', value, epoch, wall_time)


    def write_graph(self, model, inputs, verbose=False):
        self.writer.add_graph(model, inputs, verbose=verbose)


    def close(self):
        self.writer.close()
