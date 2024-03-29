import argparse
import logging
import os

import torch
from torch import nn
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
    UP_TO_DATE_MASKS_VERSION,
)
from medai.losses import get_loss_function, AVAILABLE_LOSSES, POS_WEIGHTS_BY_DATASET
from medai.losses.schedulers import create_lr_sch_handler
from medai.models import load_pretrained_weights_cnn_
from medai.metrics import attach_losses
from medai.metrics.classification import (
    attach_metrics_classification,
    attach_hint_saliency,
)
from medai.models.classification import (
    create_cnn,
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
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    set_seed,
    set_seed_from_metadata,
    RunId,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_save_training_stats,
)

LOGGER = logging.getLogger('medai.cl.train')

_DEFAULT_TARGET_METRIC = 'pr_auc'

def _choose_print_metrics(dataset_name, additional=None):
    if dataset_name == 'cxr14':
        print_metrics = ['loss', 'roc_auc', 'pr_auc', 'hamming']
    elif 'covid' in dataset_name:
        print_metrics = ['loss', 'prec_covid', 'recall_covid']
    elif 'imagenet' in dataset_name :
        print_metrics = ['loss', 'acc', 'f1', 'prec', 'recall']
    else:
        print_metrics = ['loss', 'roc_auc', 'pr_auc']

    if additional is not None:
        print_metrics += [m for m in additional if m not in print_metrics]

    return print_metrics


def train_model(run_id,
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
                dryrun=False,
                tb_kwargs={},
                checkpoint_metric=_DEFAULT_TARGET_METRIC,
                print_metrics=['loss'],
                device='cuda',
                hw_options={},
                ):
    # Prepare run
    LOGGER.info('Training run: %s', run_id)
    tb_writer = TBWriter(run_id, dryrun=dryrun, **tb_kwargs)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info('Resuming from epoch: %s', initial_epoch)

    # Unwrap stuff
    model, optimizer, lr_sch_handler = compiled_model.get_elements()

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
    losses = ['cl_loss', 'hint_loss'] if hint else []
    attach_losses(validator, losses, device=device)
    attach_metrics_classification(validator, labels,
                                  multilabel=multilabel,
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
    attach_losses(trainer, losses, device=device)
    attach_metrics_classification(trainer, labels,
                                  multilabel=multilabel,
                                  device=device)

    if hint:
        attach_hint_saliency(validator, labels, multilabel=multilabel, device=device)
        attach_hint_saliency(trainer, labels, multilabel=multilabel, device=device)

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
    attach_checkpoint_saver(run_id,
                            compiled_model,
                            trainer,
                            validator,
                            metric=checkpoint_metric,
                            dryrun=dryrun,
                            )

    attach_save_training_stats(
        trainer,
        run_id,
        timer,
        n_epochs,
        hw_options,
        initial_epoch=initial_epoch,
        dryrun=dryrun,
    )

    attach_early_stopping(trainer, validator, attach=early_stopping, **early_stopping_kwargs)

    lr_sch_handler.attach(trainer, validator)

    # Train!
    LOGGER.info('-' * 50)
    LOGGER.info('Training...')
    trainer.run(train_dataloader, n_epochs)

    # Capture time per epoch
    secs_per_epoch = timer.value()
    LOGGER.info('Average time per epoch: %s', duration_to_str(secs_per_epoch))
    LOGGER.info('-'*50)

    tb_writer.close()

    LOGGER.info('Finished training: %s', run_id)

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
                    hw_options={},
                    ):
    """Resume training from a previous run."""
    run_id = RunId(run_name, debug, 'cls')

    # Load model
    compiled_model = load_compiled_model_classification(run_id,
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
    train_model(run_id, compiled_model, train_dataloader, val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                print_metrics=_choose_print_metrics(dataset_name, print_metrics),
                tb_kwargs=tb_kwargs,
                device=device,
                hw_options=hw_options,
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
                       dropout_features=0,
                       imagenet=True,
                       freeze=False,
                       cnn_pooling='avg',
                       fc_layers=(),
                       max_samples=None,
                       image_size=512,
                       crop_center=None,
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
                       lr_sch_kwargs={},
                       frontal_only=False,
                       pretrained_run_id=None,
                       pretrained_cls=True,
                       oversample=False,
                       oversample_label=None,
                       oversample_class=None,
                       oversample_ratio=None,
                       oversample_max_ratio=None,
                       undersample=False,
                       undersample_label=None,
                       balanced_sampler=False,
                       augment=False,
                       augment_mode='single',
                       augment_label=None,
                       augment_class=None,
                       augment_times=1,
                       augment_kwargs={},
                       tb_kwargs={},
                       debug=True,
                       experiment=None,
                       multiple_gpu=False,
                       num_workers=2,
                       device='cuda',
                       seed=None,
                       hw_options={},
                       dryrun=False,
                       ):
    """Train a model from scratch."""
    # Default values
    lr = lr or 1e-4
    batch_size = batch_size or 10

    # Create run name
    run_name = f'{run_name}_{dataset_name}'

    if clahe:
        run_name += '-clahe'

    run_name += f'_{cnn_name}'

    if hint:
        run_name += '_hint'
        if hint_lambda != 1:
            run_name += f'-{hint_lambda}'
        if UP_TO_DATE_MASKS_VERSION != 'v0':
            run_name += f'_masks-{UP_TO_DATE_MASKS_VERSION}'
    if not imagenet:
        run_name += '_noig'
    if cnn_pooling not in ('avg', 'mean'):
        run_name += f'_g{cnn_pooling}'
    if dropout != 0:
        run_name += f'_drop{dropout}'
    if dropout_features != 0:
        run_name += f'_dropf{dropout_features}'
    if freeze:
        run_name += '_frz'
    if fc_layers and len(fc_layers) > 0:
        run_name += '_fc' + '-'.join(str(l) for l in fc_layers)
    if pretrained_run_id:
        run_name += f'_pre{pretrained_run_id.short_clean_name}'
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
        run_name += f'_aug{augment_times}'
        if augment_mode != 'single':
            run_name += f'-{augment_mode}'
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
    if isinstance(labels, (tuple, list)):
        if len(labels) == 1:
            run_name += f'_{labels[0]}'
        else:
            run_name += f'_labels{len(labels)}'
    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if not shuffle:
        run_name += '_noshuffle'
    if image_size != 256:
        run_name += f'_size{image_size}'
    if frontal_only:
        run_name += '_front'
    if crop_center is not None:
        run_name += f'_crop{crop_center}'
    if n_epochs == 0:
        run_name += '_e0'
    run_name += f'_lr{lr}'
    if weight_decay != 0:
        run_name += f'_wd{weight_decay}'

    lr_sch_name = lr_sch_kwargs['name']
    if lr_sch_name == 'plateau':
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        metric = lr_sch_kwargs['metric'].replace('_', '-')
        run_name += f'_sch-{metric}-p{patience}-f{factor}'

        cooldown = lr_sch_kwargs.get('cooldown', 0)
        if cooldown != 0:
            run_name += f'-c{cooldown}'
    elif lr_sch_name == 'step':
        step = lr_sch_kwargs['step_size']
        factor = lr_sch_kwargs['gamma']
        run_name += f'_sch-step{step}-f{factor}'

    # if not early_stopping:
    #     run_name += '_noes'
    # else:
    #     metric = early_stopping_kwargs['metric']
    #     patience = early_stopping_kwargs['patience']
    #     run_name += f'_es-{metric}-p{patience}'
    run_name = run_name.replace(' ', '-')

    run_id = RunId(run_name, debug, 'cls', experiment)

    set_seed(seed)

    # Load data
    enable_masks = grad_cam or hint
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'labels': labels,
        'max_samples': max_samples,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
        'crop_center': crop_center,
        'frontal_only': frontal_only,
        'num_workers': num_workers,
        'norm_by_sample': norm_by_sample,
        'masks': enable_masks,
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
        'augment_mode': augment_mode,
        'augment_label': augment_label,
        'augment_class': augment_class,
        'augment_times': augment_times,
        'augment_kwargs': augment_kwargs,
        'augment_seg_mask': enable_masks,
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
    model_kwargs = {
        'model_name': cnn_name,
        'labels': train_dataloader.dataset.labels,
        'imagenet': imagenet,
        'freeze': freeze,
        'gpool': cnn_pooling,
        'fc_layers': fc_layers,
        'dropout': dropout,
        'dropout_features': dropout_features,
    }
    model = create_cnn(**model_kwargs).to(device)

    # Load features from pretrained CNN
    if pretrained_run_id:
        load_pretrained_weights_cnn_(
            model, pretrained_run_id,
            cls_weights=pretrained_cls, seg_weights=False,
            device=device,
        )


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
    lr_sch_handler = create_lr_sch_handler(optimizer, **lr_sch_kwargs)

    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'grad_cam': grad_cam,
        'grad_cam_thresh': grad_cam_thresh,
        'hint': hint,
        'hint_lambda': hint_lambda,
        'checkpoint_metric': 'f1' if 'imagenet' in dataset_name else _DEFAULT_TARGET_METRIC,
    }

    # Some additional hparams
    hparams = {
        'loss_name': loss_name,
        'loss_kwargs': loss_kwargs,
        'batch_size': batch_size,
    }
    if pretrained_run_id:
        hparams.update({
            'pretrained': pretrained_run_id.to_dict(),
            'pretrained-cls': pretrained_cls,
            'pretrained-seg': False,
        })

    # Save model metadata
    metadata = {
        'model_kwargs': model_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs,
        'hparams': hparams,
        'other_train_kwargs': other_train_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_id)


    # Create compiled_model
    compiled_model = CompiledModel(run_id, model, optimizer, lr_sch_handler, metadata)

    # Train!
    train_metrics, val_metrics = train_model(
        run_id,
        compiled_model,
        train_dataloader,
        val_dataloader,
        n_epochs=n_epochs,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        print_metrics=_choose_print_metrics(dataset_name, print_metrics),
        tb_kwargs=tb_kwargs,
        device=device,
        hw_options=hw_options,
        dryrun=dryrun,
        **other_train_kwargs,
    )

    return run_id, compiled_model, train_metrics, val_metrics


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

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
    parser.add_argument('-exp', '--experiment', type=str, default='',
                        help='Custom experiment name')
    parser.add_argument('--hint', action='store_true',
                        help='Use HINT training')
    parser.add_argument('--hint-lambda', type=float, default=1,
                        help='Factor to multiply hint-loss')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')
    parser.add_argument('--dont-save', action='store_true', help='Do not save stuff')

    parsers.add_args_cnn(parser)

    loss_group = parser.add_argument_group('Loss params')
    loss_group.add_argument('-l', '--loss-name', type=str, default=None,
                            choices=AVAILABLE_LOSSES,
                            help='Loss to use')
    loss_group.add_argument('--focal-alpha', type=float, default=0.75, help='Focal alpha param')
    loss_group.add_argument('--focal-gamma', type=float, default=2, help='Focal gamma param')
    loss_group.add_argument('--bce-pos-weight', action='store_true', help='Use pos weights in BCE')


    images_group = parsers.add_args_images(parser)
    images_group.add_argument('--clahe', action='store_true',
                              help='Use CLAHE normalized images')

    grad_cam_group = parser.add_argument_group('Grad-CAM evaluation params')
    grad_cam_group.add_argument('--grad-cam', action='store_true',
                              help='Evaluate grad-cam vs masks')
    grad_cam_group.add_argument('--grad-cam-thresh', type=float, default=0.5,
                              help='Threshold to apply to activations')

    parser.add_argument('--pretrained', type=str, default=None,
                        help='Run name of a pretrained CNN')
    parser.add_argument('--pretrained-task', type=str, default='cls',
                        choices=('cls', 'cls-seg'), help='Task to choose the CNN from')
    parser.add_argument('--pretrained-cls', action='store_true',
                        help='Copy classifier weights also')
    parser.add_argument('--pretrained-seg', action='store_true',
                        help='Dummy parameter!! Allows using the same API as train_cls_seg, \
                              but is useless')

    parsers.add_args_augment(parser)
    parsers.add_args_sampling(parser)

    parsers.add_args_tb(parser)
    parsers.add_args_early_stopping(parser, metric=_DEFAULT_TARGET_METRIC)
    parsers.add_args_lr(parser, lr=None)
    parsers.add_args_lr_sch(parser, metric=_DEFAULT_TARGET_METRIC, factor=0.5, patience=3)

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
        pos_weight = POS_WEIGHTS_BY_DATASET[args.dataset]

        if args.dataset == 'chexpert' and args.labels is not None:
            if len(args.labels) == 13:
                pos_weight = pos_weight[1:]
            else:
                # FIXME: this should be handled elsewhere??
                parser.error('Pos-weight case not handled')
        args.loss_kwargs = {
            'pos_weight': pos_weight,
        }
    else:
        args.loss_kwargs = {}

    # Shortcuts
    args.debug = not args.no_debug

    # Build params
    parsers.build_args_sampling_(args)
    parsers.build_args_early_stopping_(args)
    parsers.build_args_lr_sch_(args, parser)
    parsers.build_args_tb_(args)
    parsers.build_args_augment_(args)

    if args.pretrained:
        args.pretrained_run_id = RunId(args.pretrained, debug=False, task=args.pretrained_task)
    else:
        args.pretrained_run_id = None

    return args


if __name__ == '__main__':
    config_logging()

    ARGS = parse_args()

    if ARGS.num_threads > 0:
        torch.set_num_threads(ARGS.num_threads)

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    HW_OPTIONS = {
        'device': str(DEVICE),
        'visible': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'multiple': ARGS.multiple_gpu,
        'num_threads': ARGS.num_threads,
    }

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
                        device=DEVICE,
                        hw_options=HW_OPTIONS,
                        )
    else:
        train_from_scratch(
            get_timestamp(),
            ARGS.dataset,
            shuffle=ARGS.shuffle,
            clahe=ARGS.clahe,
            image_format=ARGS.image_format,
            cnn_name=ARGS.model,
            dropout=ARGS.dropout,
            dropout_features=ARGS.dropout_features,
            imagenet=not ARGS.no_imagenet,
            freeze=ARGS.freeze,
            cnn_pooling=ARGS.cnn_pooling,
            fc_layers=ARGS.fc_layers,
            max_samples=ARGS.max_samples,
            image_size=ARGS.image_size,
            crop_center=ARGS.crop_center,
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
            lr_sch_kwargs=ARGS.lr_sch_kwargs,
            frontal_only=ARGS.frontal_only,
            pretrained_run_id=ARGS.pretrained_run_id,
            pretrained_cls=ARGS.pretrained_cls,
            oversample=ARGS.oversample is not None,
            oversample_label=ARGS.oversample,
            oversample_class=ARGS.os_class,
            oversample_ratio=ARGS.os_ratio,
            oversample_max_ratio=ARGS.os_max_ratio,
            augment=ARGS.augment,
            augment_mode=ARGS.augment_mode,
            augment_label=ARGS.augment_label,
            augment_class=ARGS.augment_class,
            augment_times=ARGS.augment_times,
            augment_kwargs=ARGS.augment_kwargs,
            undersample=ARGS.undersample is not None,
            undersample_label=ARGS.undersample,
            balanced_sampler=ARGS.balanced_sampler,
            debug=ARGS.debug,
            experiment=ARGS.experiment,
            tb_kwargs=ARGS.tb_kwargs,
            multiple_gpu=ARGS.multiple_gpu,
            num_workers=ARGS.num_workers,
            device=DEVICE,
            seed=ARGS.seed,
            hw_options=HW_OPTIONS,
            dryrun=ARGS.dont_save,
            )
