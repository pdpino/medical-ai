"""Tensorboard util functions."""
import os
import re
import logging
import numbers
from tensorboardX import SummaryWriter
from torch import nn

from medai import utils
from medai.utils.files import get_tb_log_folder, get_tb_large_log_folder

IGNORE_METRICS = [
    r'\Acm', # Confusion-matrix, no sense to put it in TB
    # '_timer', # Medical correctness timers, return strings
]

LOGGER = logging.getLogger(__name__)

class TBWriter:
    def __init__(self, run_id,
                 ignore_metrics=IGNORE_METRICS,
                 dryrun=False,
                 scalars=True,
                 histogram=False,
                 histogram_filter=None,
                 histogram_freq=None,
                 graph=False,
                 workspace_dir=utils.WORKSPACE_DIR, **kwargs):

        self._histogram = histogram
        self._hist_filter = histogram_filter
        self._hist_freq = histogram_freq or 1
        self._graph = graph
        self._scalars = scalars

        if scalars:
            self.writer = SummaryWriter(get_tb_log_folder(run_id, workspace_dir=workspace_dir),
                                        write_to_disk=not dryrun,
                                        **kwargs)
            _info = {
                'dryrun': dryrun,
                **kwargs,
            }
            _info_str = ' '.join(f"{k}={v}" for k, v in _info.items())
            LOGGER.info('Creating TB: %s', _info_str)
        else:
            LOGGER.warning('Not saving scalars to TB')
            self.writer = None

        if histogram or graph:
            self.large_writer = SummaryWriter(
                get_tb_large_log_folder(run_id, workspace_dir=workspace_dir),
                write_to_disk=not dryrun,
                **kwargs,
            )
            _info = {
                'graph': graph,
                'histogram': histogram,
                'hist-freq': histogram_freq,
                'hist-filter': histogram_filter,
            }
            _info_str = ' '.join(f"{k}={v}" for k, v in _info.items())
            LOGGER.info('Creating TB-large: %s', _info_str)
        else:
            self.large_writer = None

        self.ignore_regex = '|'.join(ignore_metrics)
        self.ignore_regex = re.compile(f'({self.ignore_regex})')

        # Capitalize so in TB appears first
        self._name_mappings = {
            'loss': 'Loss',
            'acc': 'Acc',
            'hamming': 'Hamming',
            'bce': 'BCELoss',
            'roc_auc': 'Roc_auc',
            'pr_auc': 'PR_auc',
            'recall': 'Recall',
            'prec': 'Prec',
            'spec': 'Spec',
            'f1': 'F1',
            'iou': 'IoU',
            'iobb': 'IoBB',
            'ioo': 'IoO',
            'iou-grad-cam': 'IoU-grad-cam',
            'iobb-grad-cam': 'IoBB-grad-cam',
            'mAP': 'MAP', # mean Average-precision (coco-challenge)
            'mse-total': 'MSE-total',
            'mse-pos': 'MSE-pos',
            'mse-neg': 'MSE-neg',
            'chex_acc': 'Chex_acc',
            'chex_f1': 'Chex_f1',
            'chex_npv': 'Chex_npv',
            'chex_prec': 'Chex_prec',
            'chex_recall': 'Chex_recall',
            'chex_spec': 'Chex_spec',
            'lighter-chex_acc': 'Lighter-chex_acc',
            'lighter-chex_f1': 'Lighter-chex_f1',
            'lighter-chex_npv': 'Lighter-chex_npv',
            'lighter-chex_prec': 'Lighter-chex_prec',
            'lighter-chex_recall': 'Lighter-chex_recall',
            'lighter-chex_spec': 'Lighter-chex_spec',
            'bleu': 'Bleu',
            'ciderD': 'Cider-D',
            'rougeL': 'Rouge-L',
            'organ-acc': 'Organ-acc',
        }

        self._loss_regex = re.compile(r'(\w+)_loss')
        self._chex_macro_avg_regex = re.compile(r'chex_(\w+_woNF)\Z')

    def _map_metric_name(self, name):
        found = self._loss_regex.search(name)
        if found:
            captured = found.group(1)
            return f'Loss_{captured}'

        found = self._chex_macro_avg_regex.search(name)
        if found:
            captured = found.group(1)
            return f'Chex_{captured}'

        return self._name_mappings.get(name, name)


    def write_histogram(self, model, epoch, wall_time):
        if not self._histogram:
            return

        if epoch != 1 and epoch % self._hist_freq != 0:
            # Write always on the first epoch
            # Write only every _hist_freq epochs
            return

        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model = model.module

        for name, params in model.named_parameters():
            if params.numel() == 0:
                continue

            if self._hist_filter is not None:
                if self._hist_filter not in name:
                    continue

            self.large_writer.add_histogram(name, params, global_step=epoch, walltime=wall_time)

            if params.grad is not None:
                self.large_writer.add_histogram(
                    f'{name}/grad', params.grad, global_step=epoch, walltime=wall_time,
                )


    def write_metrics(self, metrics, run_type, epoch, wall_time):
        if not self._scalars:
            return

        for name, value in metrics.items():
            if self.ignore_regex.search(name) or not isinstance(value, numbers.Number):
                continue

            if value == -1:
                continue

            name = self._map_metric_name(name)
            self.writer.add_scalar(f'{name}/{run_type}', value, epoch, wall_time)


    def write_graph(self, model, inputs, verbose=False):
        if not self._graph:
            return

        self.large_writer.add_graph(model, inputs, verbose=verbose)


    def close(self):
        if self.writer:
            self.writer.close()

        if self.large_writer:
            self.large_writer.close()
