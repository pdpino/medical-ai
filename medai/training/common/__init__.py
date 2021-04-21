import abc
import argparse
import logging
import time

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import prepare_data_classification, AVAILABLE_CLASSIFICATION_DATASETS
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    save_metadata,
    load_compiled_model,
)
from medai.tensorboard import TBWriter
from medai.utils import (
    get_timestamp,
    duration_to_str,
    print_hw_options,
    parsers,
    config_logging,
    set_seed,
    RunId,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_lr_scheduler_handler,
)


class TrainingProcess(abc.ABC):
    LOGGER_NAME = 'medai.train'
    base_print_metrics = []
    task = 'none'
    key_metric = 'roc_auc'
    default_es_metric = None
    default_lr_metric = None
    checkpoint_metric = None
    default_num_workers = 2
    prepare_data_fn = staticmethod(prepare_data_classification)

    available_datasets = AVAILABLE_CLASSIFICATION_DATASETS
    default_dataset = 'cxr14'

    allow_augmenting = True
    allow_sampling = True

    default_image_format = 'RGB'


    def __init__(self):
        self.logger = logging.getLogger(self.LOGGER_NAME)

        # Attributes that will be filled later (and can be accessed)
        self.run_id = None
        self.device = None
        self.args = None
        self.dataset_kwargs = None
        self.dataset_train_kwargs = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.model_kwargs = None
        self.model = None
        self.opt_kwargs = None
        self.optimizer = None
        self.lr_scheduler = None
        self.compiled_model = None
        self.trainer = None
        self.validator = None
        self.tb_writer = None
        self.metadata = None
        self.other_train_kwargs = None

    def _add_additional_args(self, parser):
        """Call additional add_args functions."""

    def _build_additional_args(self, parser, args):
        """Call additional build_args functions."""

    def _parse_args(self):
        parser = argparse.ArgumentParser(usage='%(prog)s [options]')

        parser.add_argument('-exp', '--experiment', type=str, default='',
                            help='Custom experiment name')
        parser.add_argument('--seed', type=int, default=1234,
                            help='Set a seed (initial run only)')
        parser.add_argument('--resume', type=str, default=None,
                            help='If present, resume a previous run')
        parser.add_argument('-d', '--dataset', type=str, default=self.default_dataset,
                            choices=self.available_datasets,
                            help='Choose dataset to train on')
        parser.add_argument('--labels', type=str, nargs='*', default=None,
                            help='Subset of labels')
        parser.add_argument('-e', '--epochs', type=int, default=1,
                            help='Number of epochs')
        parser.add_argument('--no-debug', action='store_true',
                            help='If is a non-debugging run')
        parser.add_argument('--dryrun', action='store_true',
                            help='If present, do not store model or TB run')
        parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                            help='Additional metrics to print to stdout')
        parser.add_argument('--max-samples', type=int, default=None,
                            help='Max samples to load (debugging)')
        parser.add_argument('-bs', '--batch-size', type=int, default=10,
                            help='Batch size')
        parser.add_argument('--shuffle', action='store_true', default=None,
                            help='Whether to shuffle or not the samples when training')

        parsers.add_args_images(parser, image_format=self.default_image_format)
        parsers.add_args_lr_sch(
            parser,
            lr=0.0001,
            metric=self.default_lr_metric or self.key_metric,
            patience=3,
        )
        parsers.add_args_early_stopping(
            parser,
            metric=self.default_es_metric or self.key_metric,
        )
        parsers.add_args_tb(parser)

        if self.allow_augmenting:
            parsers.add_args_augment(parser)

        if self.allow_sampling:
            parsers.add_args_sampling(parser)

        parsers.add_args_hw(parser, num_workers=self.default_num_workers)

        self._add_additional_args(parser)

        args = parser.parse_args()

        # Shortcuts
        args.debug = not args.no_debug

        # Build params
        parsers.build_args_early_stopping_(args)
        parsers.build_args_lr_sch_(args)
        parsers.build_args_tb_(args)

        if self.allow_augmenting:
            parsers.build_args_augment_(args)
        if self.allow_sampling:
            parsers.build_args_sampling_(args)

        self._build_additional_args(parser, args)

        self.args = args

    def run(self):
        self._parse_args()

        config_logging()

        if self.args.num_threads > 0:
            torch.set_num_threads(self.args.num_threads)

        self.device = torch.device(
            'cuda'
            if not self.args.cpu and torch.cuda.is_available() else 'cpu'
        )

        print_hw_options(self.device, self.args)

        start_time = time.time()

        if self.args.resume:
            self.resume_training()
        else:
            self.train_from_scratch()

        total_time = time.time() - start_time
        self.logger.info('Total time: %s', duration_to_str(total_time))
        self.logger.info('=' * 50)

    def _fill_run_name_sampling(self, run_name):
        if self.args.oversample:
            run_name += '_os'
            if self.args.oversample_ratio is not None:
                run_name += f'-r{self.args.oversample_ratio}'
            elif self.args.oversample_max_ratio is not None:
                run_name += f'-max{self.args.oversample_max_ratio}'
            if self.args.oversample_class is not None:
                run_name += f'-cl{self.args.oversample_class}'
        elif self.args.undersample:
            run_name += '_us'
        elif self.args.balanced_sampler:
            run_name += '_balance'

        return run_name

    def _fill_run_name_augmenting(self, run_name):
        if self.args.augment:
            run_name += f'_aug{self.args.augment_times}'
            if self.args.augment_mode != 'single':
                run_name += f'-{self.args.augment_mode}'
            if self.args.augment_label is not None:
                run_name += f'-{self.args.augment_label}'
                if self.args.augment_class is not None:
                    run_name += f'-cls{self.args.augment_class}'

        return run_name

    def _fill_run_name_lr(self, run_name):
        run_name += f'_lr{self.args.learning_rate}'
        if self.args.weight_decay != 0:
            run_name += f'_wd{self.args.weight_decay}'
        if self.args.lr_metric:
            factor = self.args.lr_sch_kwargs['factor']
            patience = self.args.lr_sch_kwargs['patience']
            run_name += f'_sch-{self.args.lr_metric.replace("_", "-")}-p{patience}-f{factor}'

            cooldown = self.args.lr_sch_kwargs.get('cooldown', 0)
            if cooldown != 0:
                run_name += f'-c{cooldown}'

        return run_name

    def _fill_run_name_data(self, run_name):
        if self.args.norm_by_sample:
            run_name += '_normS'
        else:
            run_name += '_normD'
        if self.args.image_size != 256:
            run_name += f'_size{self.args.image_size}'

        if isinstance(self.args.labels, (tuple, list)):
            if len(self.args.labels) == 1:
                run_name += f'_{self.args.labels[0]}'
            else:
                run_name += f'_labels{len(self.args.labels)}'

        if not self.args.shuffle:
            run_name += '_no-shuffle'

        return run_name

    def _fill_run_name_checkpoint(self, run_name):
        if self.checkpoint_metric is None:
            run_name += '_notbest'

        return run_name

    @abc.abstractmethod
    def _fill_run_name_model(self, run_name):
        """Fill run_name with descriptive model information."""

    def _fill_run_name_other(self, run_name):
        """Fill run_name with extra info."""
        return run_name

    def _create_run_name(self):
        run_name = f'{get_timestamp()}_{self.args.dataset}'

        run_name = self._fill_run_name_model(run_name)

        run_name = self._fill_run_name_data(run_name)

        run_name = self._fill_run_name_lr(run_name)

        if self.allow_sampling:
            run_name = self._fill_run_name_sampling(run_name)

        if self.allow_augmenting:
            run_name = self._fill_run_name_augmenting(run_name)

        run_name = self._fill_run_name_checkpoint(run_name)

        run_name = self._fill_run_name_other(run_name)

        run_name = run_name.replace(' ', '-')

        self.run_id = RunId(
            run_name, self.args.debug, self.task, self.args.experiment,
        )


    def train_from_scratch(self):
        """Trains a model from scratch."""
        self._create_run_name()

        set_seed(self.args.seed)

        self._prepare_dataset_kwargs()
        self._create_dataloaders()

        self._create_model()
        self._load_state_dict_pretrained_model()
        self._move_model_to_device()

        self._create_optimizer()

        self._create_lr_scheduler()

        self._prepare_other_train_kwargs()

        self._prepare_and_save_metadata()

        self.compiled_model = CompiledModel(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.metadata,
        )

        return self.train_model(**self.other_train_kwargs)

    def _load_resumed_model(self):
        self.compiled_model = load_compiled_model(
            self.run_id,
            device=self.device,
            multiple_gpu=self.args.multiple_gpu,
        )

        self.metadata = self.compiled_model.metadata
        self.model, self.optimizer, self.lr_scheduler = self.compiled_model.get_elements()

    def resume_training(self):
        # Create run_name attributes
        self.run_id = RunId(
            self.args.resume, self.args.debug, self.task, self.args.experiment,
        )

        self._load_resumed_model()

        # Set seed from metadata
        seed = self.metadata.get('seed', None)
        if seed is not None:
            set_seed(seed)
        else:
            self.logger.warning('Seed not found in metadata')

        # Retrieve dataset kwargs
        self.dataset_kwargs = self.metadata.get('dataset_kwargs', {})
        self.dataset_train_kwargs = self.metadata.get('dataset_train_kwargs', {})

        # Override dataset_kwargs, if present
        self._create_dataloaders()

        # Retrieve other_train_kwargs
        self.other_train_kwargs = self.metadata.get('other_train_kwargs', {})

        # Train model
        self.train_model(**self.other_train_kwargs)

    def _prepare_dataset_kwargs(self):
        """Create dataset_kwargs and dataset_train_kwargs."""
        image_size = self.args.image_size
        if not isinstance(image_size, (tuple, list)):
            image_size = (image_size, image_size)

        self.dataset_kwargs = {
            'dataset_name': self.args.dataset,
            'labels': self.args.labels,
            'max_samples': self.args.max_samples,
            'batch_size': self.args.batch_size,
            'image_size': image_size,
            'frontal_only': self.args.frontal_only,
            'num_workers': self.args.num_workers,
            'norm_by_sample': self.args.norm_by_sample,
            'image_format': self.args.image_format,
        }
        self.dataset_train_kwargs = {
            'shuffle': self.args.shuffle,
        }
        if self.allow_augmenting:
            self.dataset_train_kwargs.update({
                'augment': self.args.augment,
                'augment_mode': self.args.augment_mode,
                'augment_label': self.args.augment_label,
                'augment_class': self.args.augment_class,
                'augment_times': self.args.augment_times,
                'augment_kwargs': self.args.augment_kwargs,
            })
        if self.allow_sampling:
            self.dataset_train_kwargs.update({
                'oversample': self.args.oversample,
                'oversample_label': self.args.oversample_label,
                'oversample_class': self.args.oversample_class,
                'oversample_ratio': self.args.oversample_ratio,
                'oversample_max_ratio': self.args.oversample_max_ratio,
                'undersample': self.args.undersample,
                'undersample_label': self.args.undersample_label,
                'balanced_sampler': self.args.balanced_sampler,
            })

        self._fill_dataset_kwargs()

    def _fill_dataset_kwargs(self):
        """Fill more key-value pairs in dataset_kwargs."""

    def _create_dataloaders(self):
        self.train_dataloader = self.prepare_data_fn(
            dataset_type='train',
            **self.dataset_kwargs,
            **self.dataset_train_kwargs,
        )
        self.val_dataloader = self.prepare_data_fn(
            dataset_type='val',
            **self.dataset_kwargs,
        )

    @abc.abstractmethod
    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        self.model_kwargs = {}
        self.model = None

    def _load_state_dict_pretrained_model(self):
        """Load state_dict from a pretrained model to self.model."""

    def _move_model_to_device(self):
        if self.args.multiple_gpu:
            # TODO: use DistributedDataParallel instead ??
            self.model = nn.DataParallel(self.model)

    def _create_optimizer(self):
        self.opt_kwargs = {
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }
        self.optimizer = optim.Adam(self.model.parameters(), **self.opt_kwargs)

    def _create_lr_scheduler(self):
        if self.args.lr_metric:
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, **self.args.lr_sch_kwargs)

            self.logger.info('Using LR-scheduler with metric=%s', self.args.lr_metric)
        else:
            self.lr_scheduler = None
            self.logger.warning('Not using a LR-scheduler')

    def _fill_other_train_kwargs(self):
        """Fill more key-value pairs in self.other_train_kwargs.

        This dict will keep any other argument used during training,
        that will be necessary to recreate the run in the future (e.g. resuming training).
        """

    def _prepare_other_train_kwargs(self):
        self.other_train_kwargs = {
            'early_stopping': self.args.early_stopping,
            'early_stopping_kwargs': self.args.early_stopping_kwargs,
            'lr_metric': self.args.lr_metric,
        }

        self._fill_other_train_kwargs()

    def _fill_metadata(self):
        """Fill self.metadata with more key-value pairs."""

    def _prepare_and_save_metadata(self):
        """Create self.metadata."""
        self.metadata = {
            'model_kwargs': self.model_kwargs,
            'opt_kwargs': self.opt_kwargs,
            'lr_sch_kwargs': self.args.lr_sch_kwargs if self.lr_scheduler is not None else None,
            'hparams': {
                'batch_size': self.args.batch_size,
            },
            'other_train_kwargs': self.other_train_kwargs,
            'dataset_kwargs': self.dataset_kwargs,
            'dataset_train_kwargs': self.dataset_train_kwargs,
            'seed': self.args.seed,
        }

        self._fill_metadata()

        save_metadata(self.metadata, self.run_id)

    def train_model(self, early_stopping=True,
                    early_stopping_kwargs={},
                    lr_metric='loss',
                    **other_train_kwargs):
        # Log initial info
        self.logger.info('Training run: %s', self.run_id)
        initial_epoch = self.compiled_model.get_current_epoch()
        if initial_epoch > 0:
            self.logger.info('Resuming from epoch: %s', initial_epoch)

        self._create_tb()

        self._create_engines(**other_train_kwargs)

        # Create Timer to measure wall time between epochs
        timer = Timer(average=True)
        timer.attach(self.trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

        attach_log_metrics(
            self.trainer,
            self.validator,
            self.compiled_model,
            self.val_dataloader,
            self.tb_writer,
            timer,
            logger=self.logger,
            initial_epoch=initial_epoch,
            print_metrics=self._get_print_metrics(),
        )

        # Attach checkpoint
        attach_checkpoint_saver(
            self.run_id,
            self.compiled_model,
            self.trainer,
            self.validator,
            metric=self.checkpoint_metric,
            dryrun=self.args.dryrun,
        )

        if early_stopping:
            attach_early_stopping(self.trainer, self.validator, **early_stopping_kwargs)

        if self.lr_scheduler is not None:
            attach_lr_scheduler_handler(
                self.lr_scheduler,
                self.trainer,
                self.validator,
                lr_metric,
            )

        # Train!
        self.logger.info('-' * 50)
        self.logger.info('Training...')
        self.trainer.run(self.train_dataloader, self.args.epochs)

        # Capture time per epoch
        secs_per_epoch = timer.value()
        self.logger.info('Average time per epoch: %s', duration_to_str(secs_per_epoch))
        self.logger.info('-' * 50)

        self.tb_writer.close()

        self.logger.info('Finished training: %s', self.run_id)

        return self.trainer.state.metrics, self.validator.state.metrics

    def _create_tb(self):
        self.tb_writer = TBWriter(
            self.run_id,
            dryrun=self.args.dryrun,
            **self.args.tb_kwargs,
        )

    def _get_print_metrics(self):
        print_metrics = self.base_print_metrics

        if self.args.print_metrics is not None:
            for metric in self.args.print_metrics:
                if metric not in print_metrics:
                    print_metrics.append(metric)

        return print_metrics

    def _build_common_for_engines(self, **unused_kwargs):
        """Build common vars to create the engines later."""
        return []

    @abc.abstractmethod
    def _create_engine(self, training, *args, **other_train_kwargs):
        """Create an ignite Engine and attach metric handlers."""

    def _create_engines(self, **other_train_kwargs):
        """Create self.trainer and self.validator."""

        extra_args = self._build_common_for_engines(**other_train_kwargs)

        self.trainer = self._create_engine(True, *extra_args, **other_train_kwargs)
        self.validator = self._create_engine(False, *extra_args, **other_train_kwargs)
