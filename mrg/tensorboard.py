"""Tensorboard util functions."""
import os
import re
from tensorboardX import SummaryWriter
from torch import nn

from mrg import utils

IGNORE_METRICS = [
    'cm', # Confusion-matrix, no sense to put it in TB
]

def _get_log_dir(run_name, classification=True,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True):
    debug_folder = 'debug' if debug else ''
    mode_folder = 'classification' if classification else 'report_generation'
    return os.path.join(workspace_dir, mode_folder, 'runs', debug_folder, run_name)


class TBWriter:
    def __init__(self, run_name, classification=True,
                 ignore_metrics=IGNORE_METRICS,
                 workspace_dir=utils.WORKSPACE_DIR, debug=True, **kwargs):
        self.log_dir = _get_log_dir(run_name, classification=classification,
                                    workspace_dir=workspace_dir, debug=debug)

        self.writer = SummaryWriter(self.log_dir, **kwargs)

        self.ignore_regex = '|'.join(ignore_metrics)
        self.ignore_regex = re.compile(f'\A({self.ignore_regex})')
        # NOTE: Consider starting patterns, as metrics are written as <metric>_<disease>


        # Capitalize so in TB appears first
        self._name_mappings = {
            'loss': 'Loss',
            'acc': 'Acc',
            'hamming': 'Hamming',
            'bce': 'BCELoss'
        }

    def write_histogram(self, model, epoch, wall_time):
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
