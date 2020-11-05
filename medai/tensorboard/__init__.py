"""Tensorboard util functions."""
import os
import re
from tensorboardX import SummaryWriter
from torch import nn

from medai import utils
from medai.utils.files import get_tb_log_folder

IGNORE_METRICS = [
    'cm', # Confusion-matrix, no sense to put it in TB
]

class TBWriter:
    def __init__(self, run_name, task,
                 ignore_metrics=IGNORE_METRICS,
                 dryrun=False,
                 histogram=False,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True, **kwargs):
        self.log_dir = get_tb_log_folder(run_name, task=task,
                                    workspace_dir=workspace_dir, debug=debug)

        self.writer = SummaryWriter(self.log_dir,
                                    write_to_disk=not dryrun,
                                    **kwargs)

        self.ignore_regex = '|'.join(ignore_metrics)
        self.ignore_regex = re.compile(f'\A({self.ignore_regex})')
        # NOTE: Consider starting patterns, as metrics are written as <metric>_<disease>


        # Capitalize so in TB appears first
        self._name_mappings = {
            'loss': 'Loss',
            'acc': 'Acc',
            'hamming': 'Hamming',
            'bce': 'BCELoss',
            'roc_auc': 'Roc_auc', # Macro averaged
            'word_loss': 'Loss_word',
            'stop_loss': 'Loss_stop',
        }

        self._histogram = histogram


    def write_histogram(self, model, epoch, wall_time):
        if not self._histogram:
            return

        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        for name, params in model.named_parameters():
            if params.numel() == 0:
                continue

            params = params.cpu().detach().numpy()
            self.writer.add_histogram(name, params, global_step=epoch, walltime=wall_time)


    def write_metrics(self, metrics, run_type, epoch, wall_time):
        for name, value in metrics.items():
            if self.ignore_regex.search(name):
                continue

            name = self._name_mappings.get(name, name)
            self.writer.add_scalar(f'{name}/{run_type}', value, epoch, wall_time)

    def close(self):
        self.writer.close()
