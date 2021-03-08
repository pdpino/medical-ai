import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.losses import get_loss_function, AVAILABLE_LOSSES, POS_WEIGHTS_BY_DATASET
from medai.metrics.classification import attach_metrics_classification
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.models.classification import (
    create_cnn,
    AVAILABLE_CLASSIFICATION_MODELS,
    DEPRECATED_CNNS,
)
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    save_metadata,
    load_compiled_model_classification,
)
from medai.training.classification import get_step_fn
from medai.training.classification.grad_cam import create_grad_cam_evaluator
from medai.tensorboard import TBWriter
from medai.utils import (
    get_timestamp,
    duration_to_str,
    parse_str_or_int,
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    set_seed,
    set_seed_from_metadata,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_lr_scheduler_handler,
)

LOGGER = logging.getLogger('medai.cl.train')


def _choose_print_metrics(dataset_name, additional=None):
    if dataset_name == 'cxr14':
        print_metrics = ['loss', 'roc_auc', 'hamming']
    elif 'covid' in dataset_name:
        print_metrics = ['loss', 'roc_auc', 'prec_covid', 'recall_covid']
    else:
        print_metrics = ['loss', 'roc_auc']

    if additional is not None:
        print_metrics += [m for m in additional if m not in print_metrics]

    return print_metrics


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                loss_name='wbce',
                loss_kwargs={},
                grad_cam=True,
                grad_cam_thresh=0.5,
                early_stopping=True,
                early_stopping_kwargs={},
                hint=False,
                hint_lambda=1,
                lr_sch_metric='loss',
                debug=True,
                dryrun=False,
                tb_kwargs={},
                print_metrics=['loss', 'acc'],
                device='cuda',
                ):
    # Prepare run
    LOGGER.info('Training run: %s (debug=%s)', run_name, debug)
    tb_writer = TBWriter(run_name, task='cls', debug=debug, dryrun=dryrun, **tb_kwargs)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info('Resuming from epoch: %s', initial_epoch)

    # Unwrap stuff
    model, optimizer, lr_scheduler = compiled_model.get_elements()

    # Classification description
    labels = train_dataloader.dataset.labels
    multilabel = train_dataloader.dataset.multilabel

    # Prepare loss
    if loss_name == 'focal':
        loss_kwargs['multilabel'] = multilabel
    loss = get_loss_function(loss_name, **loss_kwargs)
    loss = loss.to(device)
    LOGGER.info('Using loss: %s, %s', loss_name, loss_kwargs)

    # Create validator engine
    validator = Engine(get_step_fn(model,
                                   loss,
                                   training=False,
                                   multilabel=multilabel,
                                   hint=hint,
                                   hint_lambda=hint_lambda,
                                   diseases=labels,
                                   device=device,
                                   ))
    attach_metrics_classification(validator, labels,
                                  multilabel=multilabel, hint=hint,
                                  device=device)

    # Create trainer engine
    trainer = Engine(get_step_fn(model,
                                 loss,
                                 optimizer=optimizer,
                                 training=True,
                                 multilabel=multilabel,
                                 hint=hint,
                                 hint_lambda=hint_lambda,
                                 diseases=labels,
                                 device=device,
                                 ))
    attach_metrics_classification(trainer, labels,
                                  multilabel=multilabel, hint=hint,
                                  device=device)

    if grad_cam:
        create_grad_cam_evaluator(
            trainer,
            compiled_model,
            [train_dataloader, val_dataloader],
            tb_writer,
            thresh=grad_cam_thresh,
            device=device,
            multiple_gpu=False,
            )

    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    attach_log_metrics(trainer,
                       validator,
                       compiled_model,
                       val_dataloader,
                       tb_writer,
                       timer,
                       logger=LOGGER,
                       initial_epoch=initial_epoch,
                       print_metrics=print_metrics,
                       )

    # Attach checkpoint
    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            validator,
                            task='cls',
                            metric=early_stopping_kwargs['metric'] if early_stopping else None,
                            debug=debug,
                            dryrun=dryrun,
                            )

    if early_stopping:
        attach_early_stopping(trainer, validator, **early_stopping_kwargs)

    if lr_scheduler is not None:
        attach_lr_scheduler_handler(lr_scheduler, trainer, validator, lr_sch_metric)

    # Train!
    LOGGER.info('-' * 50)
    LOGGER.info('Training...')
    trainer.run(train_dataloader, n_epochs)

    # Capture time per epoch
    secs_per_epoch = timer.value()
    LOGGER.info('Average time per epoch: %s', duration_to_str(secs_per_epoch))
    LOGGER.info('-'*50)

    tb_writer.close()

    LOGGER.info('Finished training: %s (debug=%s)', run_name, debug)

    return trainer.state.metrics, validator.state.metrics


@timeit_main(LOGGER)
def resume_training(run_name,
                    max_samples=None,
                    n_epochs=10,
                    lr=None,
                    batch_size=None,
                    print_metrics=None,
                    tb_kwargs={},
                    debug=True,
                    multiple_gpu=False,
                    device='cuda',
                    ):
    """Resume training from a previous run."""
    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        device=device,
                                                        multiple_gpu=multiple_gpu)


    # Metadata (contains all configuration)
    metadata = compiled_model.metadata
    set_seed_from_metadata(metadata)

    # Dataset kwargs
    dataset_kwargs = metadata['dataset_kwargs']
    dataset_kwargs['max_samples'] = max_samples
    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size


    # HACK(backward compatibility): dataset_name may not be saved in metadata
    # --> find it in run_name
    if 'dataset_name' not in dataset_kwargs:
        for d in AVAILABLE_CLASSIFICATION_DATASETS:
            if d in run_name:
                dataset_name = d
                break
        dataset_kwargs['dataset_name'] = dataset_name

    # Load data
    train_dataloader = prepare_data_classification(dataset_type='train',
                                                   **metadata['dataset_train_kwargs'],
                                                   **dataset_kwargs,
                                                   )
    val_dataloader = prepare_data_classification(dataset_type='val', **dataset_kwargs)

    # Select metadata
    loss_name = metadata['hparams']['loss_name']
    dataset_name = metadata['dataset_kwargs']['dataset_name']
    loss_kwargs = metadata['hparams'].get('loss_kwargs', {})

    # Override previous LR
    if lr is not None:
        # FIXME: delete this hack??
        old_lr = metadata['opt_kwargs']['lr']
        LOGGER.info('Changing learning rate to %s, was %s', lr, old_lr)
        for param_group in compiled_model.optimizer.param_groups:
            param_group['lr'] = lr

    # Train
    other_train_kwargs = metadata.get('other_train_kwargs', {})
    train_model(run_name, compiled_model, train_dataloader, val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                print_metrics=_choose_print_metrics(dataset_name, print_metrics),
                tb_kwargs=tb_kwargs,
                debug=debug,
                device=device,
                **other_train_kwargs,
                )


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       dataset_name,
                       shuffle=False,
                       clahe=False,
                       image_format='RGB',
                       cnn_name='resnet-50',
                       dropout=0,
                       imagenet=True,
                       freeze=False,
                       cnn_pooling='max',
                       fc_layers=(),
                       max_samples=None,
                       image_size=512,
                       loss_name=None,
                       loss_kwargs={},
                       print_metrics=None,
                       lr=None,
                       weight_decay=0,
                       labels=None,
                       batch_size=None,
                       norm_by_sample=False,
                       n_epochs=10,
                       hint=False,
                       hint_lambda=1,
                       grad_cam=True,
                       grad_cam_thresh=0.5,
                       early_stopping=True,
                       early_stopping_kwargs={},
                       lr_sch_metric='loss',
                       lr_sch_kwargs={},
                       frontal_only=False,
                       oversample=False,
                       oversample_label=None,
                       oversample_class=None,
                       oversample_ratio=None,
                       oversample_max_ratio=None,
                       undersample=False,
                       undersample_label=None,
                       balanced_sampler=False,
                       augment=False,
                       augment_label=None,
                       augment_class=None,
                       augment_times=1,
                       augment_kwargs={},
                       tb_kwargs={},
                       debug=True,
                       multiple_gpu=False,
                       num_workers=2,
                       device='cuda',
                       seed=None,
                       ):
    """Train a model from scratch."""
    # Default values
    lr = lr or 1e-6
    batch_size = batch_size or 10

    # Create run name
    run_name = f'{run_name}_{dataset_name}'

    if clahe:
        run_name += '-clahe'

    run_name += f'_{cnn_name}_lr{lr}'

    if hint:
        run_name += '_hint'
        if hint_lambda != 1:
            run_name += f'-{hint_lambda}'
        if UP_TO_DATE_MASKS_VERSION != 'v0':
            run_name += f'_masks-{UP_TO_DATE_MASKS_VERSION}'
    if not imagenet:
        run_name += '_noig'
    if cnn_pooling != 'max':
        run_name += f'_g{cnn_pooling}'
    if dropout != 0:
        run_name += f'_drop{dropout}'
    if freeze:
        run_name += '_frz'
    if fc_layers and len(fc_layers) > 0:
        run_name += '_fc' + '-'.join(str(l) for l in fc_layers)
    if oversample:
        run_name += '_os'
        if oversample_ratio is not None:
            run_name += f'-r{oversample_ratio}'
        elif oversample_max_ratio is not None:
            run_name += f'-max{oversample_max_ratio}'
        if oversample_class is not None:
            run_name += f'-cl{oversample_class}'
    elif undersample:
        run_name += '_us'
    elif balanced_sampler:
        run_name += '_balance'
    if augment:
        run_name += '_aug'
        if augment_label is not None:
            run_name += f'-{augment_label}'
            if augment_class is not None:
                run_name += f'-cls{augment_class}'
    if loss_name:
        run_name += f'_{loss_name}'
        if loss_name == 'focal':
            _kwargs_str = '-'.join(f'{k[0]}{v}' for k, v in loss_kwargs.items())
            run_name += f'-{_kwargs_str}' if _kwargs_str else ''
        elif loss_name == 'bce':
            if 'pos_weight' in loss_kwargs:
                run_name += '-w'
    if labels and dataset_name == 'cxr14':
        # labels only works in CXR-14, for now
        labels_str = '-'.join(labels)
        run_name += f'_{labels_str}'
    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if weight_decay != 0:
        run_name += f'_wd{weight_decay}'
    if image_size != 512:
        run_name += f'_size{image_size}'
    if n_epochs == 0:
        run_name += '_e0'
    if lr_sch_metric:
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        run_name += f'_sch-{lr_sch_metric}-p{patience}-f{factor}'
    if not early_stopping:
        run_name += '_noes'
    # else:
    #     metric = early_stopping_kwargs['metric']
    #     patience = early_stopping_kwargs['patience']
    #     run_name += f'_es-{metric}-p{patience}'
    run_name = run_name.replace(' ', '-')

    set_seed(seed)

    # Load data
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'labels': labels,
        'max_samples': max_samples,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
        'frontal_only': frontal_only,
        'num_workers': num_workers,
        'norm_by_sample': norm_by_sample,
        'masks': grad_cam or hint,
        'masks_version': UP_TO_DATE_MASKS_VERSION,
        'images_version': 'clahe' if clahe else None,
        'image_format': image_format,
    }
    dataset_train_kwargs = {
        'shuffle': shuffle,
        'oversample': oversample,
        'oversample_label': oversample_label,
        'oversample_class': oversample_class,
        'oversample_ratio': oversample_ratio,
        'oversample_max_ratio': oversample_max_ratio,
        'augment': augment,
        'augment_label': augment_label,
        'augment_class': augment_class,
        'augment_times': augment_times,
        'augment_kwargs': augment_kwargs,
        'undersample': undersample,
        'undersample_label': undersample_label,
        'balanced_sampler': balanced_sampler,
    }

    train_dataloader = prepare_data_classification(dataset_type='train',
                                                   **dataset_train_kwargs,
                                                   **dataset_kwargs,
                                                   )
    val_dataloader = prepare_data_classification(dataset_type='val', **dataset_kwargs)


    # Set default loss
    if not loss_name:
        if train_dataloader.dataset.multilabel:
            loss_name = 'wbce'
        else:
            loss_name = 'cross-entropy'
    else:
        # TODO: ensure correct loss functions are used for both cases
        pass


    # Create model
    if cnn_name in DEPRECATED_CNNS:
        raise Exception(f'CNN is deprecated: {cnn_name}')
    model_kwargs = {
        'model_name': cnn_name,
        'labels': train_dataloader.dataset.labels,
        'imagenet': imagenet,
        'freeze': freeze,
        'gpool': cnn_pooling,
        'fc_layers': fc_layers,
        'dropout': dropout,
    }
    model = create_cnn(**model_kwargs).to(device)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)


    # Create optimizer
    opt_kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
    }
    optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    # Create lr_scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, **lr_sch_kwargs) if lr_sch_metric else None
    LOGGER.info('Using LR-scheduler=%s', lr_scheduler is not None)

    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
        'grad_cam': grad_cam,
        'grad_cam_thresh': grad_cam_thresh,
        'hint': hint,
        'hint_lambda': hint_lambda,
    }


    # Save model metadata
    metadata = {
        'model_kwargs': model_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs if lr_sch_metric else None,
        'hparams': {
            'loss_name': loss_name,
            'loss_kwargs': loss_kwargs,
            'batch_size': batch_size,
        },
        'other_train_kwargs': other_train_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_name, task='cls', debug=debug)


    # Create compiled_model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Train!
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                print_metrics=_choose_print_metrics(dataset_name, print_metrics),
                tb_kwargs=tb_kwargs,
                debug=debug,
                device=device,
                **other_train_kwargs,
                )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', type=str, default=None,
                        help='If present, resume a previous run')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_DATASETS,
                        help='Choose dataset to train on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--shuffle', action='store_true', default=None,
                        help='Whether to shuffle or not the samples when training')
    parser.add_argument('--labels', type=str, nargs='*', default=None,
                        help='Subset of labels')
    parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                        help='Additional metrics to print to stdout')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--hint', action='store_true',
                        help='Use HINT training')
    parser.add_argument('--hint-lambda', type=float, default=1,
                        help='Factor to multiply hint-loss')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')

    cnn_group = parser.add_argument_group('CNN params')
    cnn_group.add_argument('-m', '--model', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN to use')
    cnn_group.add_argument('-drop', '--dropout', type=float, default=0,
                        help='dropout-rate to use (only available for some models)')
    cnn_group.add_argument('-noig', '--no-imagenet', action='store_true',
                        help='If present, dont use imagenet pretrained weights')
    cnn_group.add_argument('-frz', '--freeze', action='store_true',
                        help='If present, freeze base cnn parameters (only train FC layers)')
    cnn_group.add_argument('--cnn-pooling', type=str, default='max',
                        choices=AVAILABLE_POOLING_REDUCTIONS,
                        help='Choose reduction for global-pooling layer')
    cnn_group.add_argument('--fc-layers', nargs='+', type=int, default=(),
                        help='Choose sizes for FC layers at the end')

    loss_group = parser.add_argument_group('Loss params')
    loss_group.add_argument('-l', '--loss-name', type=str, default=None,
                            choices=AVAILABLE_LOSSES,
                            help='Loss to use')
    loss_group.add_argument('--weight-decay', type=float, default=0,
                            help='Weight decay passed to the optimizer')
    loss_group.add_argument('--focal-alpha', type=float, default=0.75, help='Focal alpha param')
    loss_group.add_argument('--focal-gamma', type=float, default=2, help='Focal gamma param')
    loss_group.add_argument('--bce-pos-weight', action='store_true', help='Use pos weights in BCE')


    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')
    images_group.add_argument('--frontal-only', action='store_true',
                              help='Use only frontal images')
    images_group.add_argument('--norm-by-sample', action='store_true',
                              help='If present, normalize each sample \
                                    (instead of using dataset stats)')
    images_group.add_argument('--clahe', action='store_true',
                              help='Use CLAHE normalized images')
    images_group.add_argument('--image-format', type=str, default='RGB', choices=['RGB', 'L'],
                              help='Image format to use')

    grad_cam_group = parser.add_argument_group('Grad-CAM evaluation params')
    grad_cam_group.add_argument('--grad-cam', action='store_true',
                              help='Evaluate grad-cam vs masks')
    grad_cam_group.add_argument('--grad-cam-thresh', type=float, default=0.5,
                              help='Threshold to apply to activations')

    parsers.add_args_augment(parser)

    sampl_group = parser.add_argument_group('Data sampling params')
    sampl_group.add_argument('-os', '--oversample', default=None,
                             help='Oversample samples with a given label present (str/int)')
    sampl_group.add_argument('--os-ratio', type=int, default=None,
                             help='Specify oversample ratio. If none, chooses ratio \
                                   to level positive and negative samples')
    sampl_group.add_argument('--os-class', type=int, choices=[0,1], default=None,
                             help='Force class value to oversample (0=neg, 1=pos). \
                                   If none, chooses least represented')
    sampl_group.add_argument('--os-max-ratio', type=int, default=None,
                             help='Max ratio to oversample by')

    sampl_group.add_argument('-us', '--undersample', default=None,
                             help='Undersample from the majority class \
                                   with a given label (str/int)')

    sampl_group.add_argument('--balanced-sampler', action='store_true',
                             help='Use a multilabel balanced sampler')

    parsers.add_args_tb(parser)
    parsers.add_args_early_stopping(parser, metric='roc_auc')
    parsers.add_args_lr_sch(parser, lr=None, metric='roc_auc', patience=3)

    parsers.add_args_hw(parser, num_workers=2)

    args = parser.parse_args()

    # If training from scratch, require dataset and model
    if not args.resume:
        if args.dataset is None:
            parser.error('A dataset must be selected')
        if args.model is None:
            parser.error('A model must be selected')

    # Require frontal_only
    if args.grad_cam or args.hint:
        if not args.frontal_only:
            parser.error('If grad_cam or hint, frontal_only must be True')

    # Require correct image-format
    if args.model in ('tiny-res-scan',):
        if not args.image_format == 'L':
            parser.error(f'For model {args.model}, image-format must be "L"')

    # Build loss params
    if args.loss_name == 'focal':
        args.loss_kwargs = {
            'alpha': args.focal_alpha,
            'gamma': args.focal_gamma,
        }
    elif args.loss_name == 'bce' and args.bce_pos_weight:
        if args.dataset not in POS_WEIGHTS_BY_DATASET:
            parser.error(f'bce-pos-weights not available for dataset {args.dataset}')
        args.loss_kwargs = {
            'pos_weight': POS_WEIGHTS_BY_DATASET[args.dataset],
        }
    else:
        args.loss_kwargs = {}

    # Enable passing str or int for oversample labels
    if args.oversample is not None:
        args.oversample = parse_str_or_int(args.oversample)
    if args.undersample is not None:
        args.undersample = parse_str_or_int(args.undersample)

    # Shortcuts
    args.debug = not args.no_debug

    # Build params
    parsers.build_args_early_stopping_(args)
    parsers.build_args_lr_sch_(args)
    parsers.build_args_tb_(args)
    parsers.build_args_augment_(args)

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    if ARGS.num_threads > 0:
        torch.set_num_threads(ARGS.num_threads)

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    if ARGS.resume:
        resume_training(ARGS.resume,
                        max_samples=ARGS.max_samples,
                        n_epochs=ARGS.epochs,
                        lr=ARGS.learning_rate,
                        batch_size=ARGS.batch_size,
                        print_metrics=ARGS.print_metrics,
                        tb_kwargs=ARGS.tb_kwargs,
                        debug=ARGS.debug,
                        multiple_gpu=ARGS.multiple_gpu,
                        device=DEVICE)
    else:
        train_from_scratch(
            get_timestamp(),
            ARGS.dataset,
            shuffle=ARGS.shuffle,
            clahe=ARGS.clahe,
            image_format=ARGS.image_format,
            cnn_name=ARGS.model,
            dropout=ARGS.dropout,
            imagenet=not ARGS.no_imagenet,
            freeze=ARGS.freeze,
            cnn_pooling=ARGS.cnn_pooling,
            fc_layers=ARGS.fc_layers,
            max_samples=ARGS.max_samples,
            image_size=ARGS.image_size,
            loss_name=ARGS.loss_name,
            loss_kwargs=ARGS.loss_kwargs,
            print_metrics=ARGS.print_metrics,
            labels=ARGS.labels,
            lr=ARGS.learning_rate,
            weight_decay=ARGS.weight_decay,
            batch_size=ARGS.batch_size,
            norm_by_sample=ARGS.norm_by_sample,
            n_epochs=ARGS.epochs,
            hint=ARGS.hint,
            hint_lambda=ARGS.hint_lambda,
            grad_cam=ARGS.grad_cam,
            grad_cam_thresh=ARGS.grad_cam_thresh,
            early_stopping=ARGS.early_stopping,
            early_stopping_kwargs=ARGS.early_stopping_kwargs,
            lr_sch_metric=ARGS.lr_metric,
            lr_sch_kwargs=ARGS.lr_sch_kwargs,
            frontal_only=ARGS.frontal_only,
            oversample=ARGS.oversample is not None,
            oversample_label=ARGS.oversample,
            oversample_class=ARGS.os_class,
            oversample_ratio=ARGS.os_ratio,
            oversample_max_ratio=ARGS.os_max_ratio,
            augment=ARGS.augment,
            augment_label=ARGS.augment_label,
            augment_class=ARGS.augment_class,
            augment_times=ARGS.augment_times,
            augment_kwargs=ARGS.augment_kwargs,
            undersample=ARGS.undersample is not None,
            undersample_label=ARGS.undersample,
            balanced_sampler=ARGS.balanced_sampler,
            debug=ARGS.debug,
            tb_kwargs=ARGS.tb_kwargs,
            multiple_gpu=ARGS.multiple_gpu,
            num_workers=ARGS.num_workers,
            device=DEVICE,
            seed=ARGS.seed,
            )
