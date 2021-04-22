import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import prepare_data_classification
from medai.losses import (
    get_loss_function,
    get_detection_hint_loss,
    AVAILABLE_HINT_LOSSES,
)
from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import (
    attach_mAP_coco,
    attach_metrics_iox,
    attach_mse,
)
from medai.models.classification import create_cnn, find_cnn_name_in_run_name
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    load_compiled_model_classification,
    save_metadata,
)
from medai.training.detection.hint import get_step_fn_hint
from medai.training.detection.h2bb import get_h2bb_method
from medai.tensorboard import TBWriter
from medai.utils import (
    get_timestamp,
    duration_to_str,
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    set_seed,
    RunId,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_lr_scheduler_handler,
)

LOGGER = logging.getLogger('medai.det.train')

_DEFAULT_PRINT_METRICS = [
    'cl_loss', 'hint_loss', 'roc_auc',
    'mse-total',
    'mAP', 'iou',
]


def _choose_print_metrics(additional=None):
    print_metrics = list(_DEFAULT_PRINT_METRICS)

    if additional is not None:
        print_metrics += [m for m in additional if m not in print_metrics]

    return print_metrics


def train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                hint_loss_name='oot',
                early_stopping=True,
                early_stopping_kwargs={},
                hint_lambda=1,
                cl_lambda=1,
                h2bb_method_name=None,
                h2bb_method_kwargs={},
                lr_sch_metric='loss',
                dryrun=False,
                tb_kwargs={},
                print_metrics=_DEFAULT_PRINT_METRICS,
                device='cuda',
                ):
    # Prepare run
    LOGGER.info('Training run: %s', run_id)
    tb_writer = TBWriter(run_id, dryrun=dryrun, **tb_kwargs)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info('Resuming from epoch: %s', initial_epoch)

    # Unwrap stuff
    model, optimizer, lr_scheduler = compiled_model.get_elements()

    # Classification description
    labels = train_dataloader.dataset.labels
    multilabel = True

    # Prepare loss
    cl_loss_fn = get_loss_function('wbce').to(device)
    hint_loss_fn = get_detection_hint_loss(hint_loss_name).to(device)

    # Choose step_fn
    get_step_fn = get_step_fn_hint

    # Choose h2bb method
    h2bb_method = get_h2bb_method(h2bb_method_name, h2bb_method_kwargs)

    # Create validator engine
    validator = Engine(get_step_fn(model,
                                   cl_loss_fn,
                                   hint_loss_fn,
                                   h2bb_method=h2bb_method,
                                   training=False,
                                   hint_lambda=hint_lambda,
                                   cl_lambda=cl_lambda,
                                   device=device,
                                   ))
    losses = ['cl_loss', 'hint_loss']
    attach_losses(validator, losses=losses, device=device)
    attach_metrics_classification(validator, labels,
                                  multilabel=multilabel,
                                  device=device)
    attach_mAP_coco(validator, val_dataloader, run_id, device=device)
    attach_metrics_iox(validator, labels, multilabel=multilabel, device=device)
    attach_mse(validator, labels, multilabel=multilabel, device=device)


    # Create trainer engine
    trainer = Engine(get_step_fn(model,
                                 cl_loss_fn,
                                 hint_loss_fn,
                                 h2bb_method=h2bb_method,
                                 optimizer=optimizer,
                                 training=True,
                                 hint_lambda=hint_lambda,
                                 cl_lambda=cl_lambda,
                                 device=device,
                                 ))
    attach_losses(trainer, losses=losses, device=device)
    attach_metrics_classification(trainer, labels,
                                  multilabel=multilabel,
                                  device=device)
    attach_mAP_coco(trainer, train_dataloader, run_id, device=device)
    attach_metrics_iox(trainer, labels, multilabel=multilabel, device=device)
    attach_mse(trainer, labels, multilabel=multilabel, device=device)

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
                            metric='mAP',
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

    LOGGER.info('Finished training: %s', run_id)

    return trainer.state.metrics, validator.state.metrics


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       shuffle=False,
                       image_format='RGB',
                       pretrained_run_id=None,
                       cnn_name='resnet-50',
                       dropout=0,
                       imagenet=True,
                       freeze=False,
                       cnn_pooling='max',
                       fc_layers=(),
                       max_samples=None,
                       image_size=512,
                       print_metrics=None,
                       hint_loss_name='oot',
                       lr=1e-4,
                       weight_decay=0,
                       labels=None,
                       batch_size=10,
                       norm_by_sample=False,
                       n_epochs=10,
                       hint_lambda=1,
                       cl_lambda=1,
                       early_stopping=True,
                       early_stopping_kwargs={},
                       lr_sch_metric='loss',
                       lr_sch_kwargs={},
                       h2bb_method_name=None,
                       h2bb_method_kwargs={},
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
                       augment_mode='single',
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
    dataset_name = 'vinbig'

    # Create run name
    run_name = f'{run_name}_{dataset_name}'

    if pretrained_run_id:
        run_name += f'_precnn-{find_cnn_name_in_run_name(pretrained_run_id.name)}'
    else:
        run_name += f'_{cnn_name}'

    run_name += f'_hint-lmb{hint_lambda}'
    if cl_lambda != 1:
        run_name += f'-{cl_lambda}'
    run_name += f'-{hint_loss_name}'

    if not pretrained_run_id:
        if not imagenet:
            run_name += '_noig'
        if cnn_pooling not in ('avg', 'mean'):
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
        run_name += f'_aug{augment_times}'
        if augment_mode != 'single':
            run_name += f'-{augment_mode}'
        if augment_label is not None:
            run_name += f'-{augment_label}'
            if augment_class is not None:
                run_name += f'-cls{augment_class}'
    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if weight_decay != 0:
        run_name += f'_wd{weight_decay}'
    if image_size != 512:
        run_name += f'_size{image_size}'
    run_name += f'_lr{lr}'
    if lr_sch_metric:
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        run_name += f'_sch-{lr_sch_metric}-p{patience}-f{factor}'
    if shuffle:
        run_name += '_shuffle'
    run_name = run_name.replace(' ', '-')

    run_id = RunId(run_name, debug, 'det')

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
        'masks': True,
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
        'augment_seg_mask': True,
        'undersample': undersample,
        'undersample_label': undersample_label,
        'balanced_sampler': balanced_sampler,
    }

    train_dataloader = prepare_data_classification(dataset_type='train',
                                                   **dataset_train_kwargs,
                                                   **dataset_kwargs,
                                                   )
    val_dataloader = prepare_data_classification(dataset_type='val', **dataset_kwargs)

    # Create CNN
    if pretrained_run_id:
        # Load pretrained
        compiled_cnn = load_compiled_model_classification(pretrained_run_id,
                                                          device=device,
                                                          multiple_gpu=multiple_gpu,
                                                          )
        model = compiled_cnn.model
        cnn_kwargs = compiled_cnn.metadata.get('model_kwargs', {})
    else:
        cnn_kwargs = {
            'model_name': cnn_name,
            'labels': train_dataloader.dataset.labels,
            'imagenet': imagenet,
            'freeze': freeze,
            'gpool': cnn_pooling,
            'fc_layers': fc_layers,
            'dropout': dropout,
        }
        model = create_cnn(**cnn_kwargs).to(device)

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
    LOGGER.info(
        'Using LR-scheduler=%s, metric=%s',
        lr_scheduler is not None, lr_sch_metric,
    )

    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
        'cl_lambda': cl_lambda,
        'hint_lambda': hint_lambda,
        'hint_loss_name': hint_loss_name,
        'h2bb_method_name': h2bb_method_name,
        'h2bb_method_kwargs': h2bb_method_kwargs,
    }


    # Save model metadata
    metadata = {
        'model_kwargs': cnn_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs if lr_sch_metric else None,
        'hparams': {
            'pretrained_cnn': pretrained_run_id.to_dict(),
            'batch_size': batch_size,
        },
        'other_train_kwargs': other_train_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_id)


    # Create compiled_model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Train!
    train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                print_metrics=_choose_print_metrics(print_metrics),
                device=device,
                **other_train_kwargs,
                )


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    # parser.add_argument('--resume', type=str, default=None,
    #                     help='If present, resume a previous run')
    # parser.add_argument('-d', '--dataset', type=str, default=None,
    #                     choices=AVAILABLE_CLASSIFICATION_DATASETS,
    #                     help='Choose dataset to train on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--shuffle', action='store_true', default=None,
                        help='Whether to shuffle or not the samples when training')
    # parser.add_argument('--labels', type=str, nargs='*', default=None,
    #                     help='Subset of labels')
    parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                        help='Additional metrics to print to stdout')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    # parser.add_argument('--hint', action='store_true',
    #                     help='Use HINT training')
    parser.add_argument('--hint-lambda', type=float, default=1,
                        help='Factor to multiply hint-loss')
    parser.add_argument('--cl-lambda', type=float, default=1,
                        help='Factor to multiply CL-loss')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')
    parser.add_argument('-cp', '--cnn-pretrained', type=str, default=None,
                        help='Run name of a pretrained CNN')

    loss_group = parser.add_argument_group('Loss params')
    loss_group.add_argument('--hint-loss-name', type=str, default='oot',
                            choices=AVAILABLE_HINT_LOSSES, help='Loss to use with HINT')


    parsers.add_args_cnn(parser)
    parsers.add_args_images(parser)
    parsers.add_args_augment(parser)
    parsers.add_args_sampling(parser)
    parsers.add_args_tb(parser)
    parsers.add_args_early_stopping(parser, metric='roc_auc')
    parsers.add_args_lr_sch(parser, lr=0.0001, metric='roc_auc', patience=3)
    # REVIEW: use mAP as metric??
    parsers.add_args_h2bb(parser)

    parsers.add_args_hw(parser, num_workers=2)

    args = parser.parse_args()

    # Shortcuts
    args.debug = not args.no_debug

    if ARGS.cnn_pretrained:
        args.pretrained_run_id = RunId(ARGS.cnn_pretrained, False, 'cls')
    else:
        args.pretrained_run_id = None

    # Build params
    parsers.build_args_h2bb_(args)
    parsers.build_args_sampling_(args)
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

    train_from_scratch(
        get_timestamp(),
        shuffle=ARGS.shuffle,
        image_format=ARGS.image_format,
        cnn_name=ARGS.model,
        pretrained_run_id=ARGS.pretrained_run_id,
        dropout=ARGS.dropout,
        imagenet=not ARGS.no_imagenet,
        freeze=ARGS.freeze,
        cnn_pooling=ARGS.cnn_pooling,
        fc_layers=ARGS.fc_layers,
        max_samples=ARGS.max_samples,
        image_size=ARGS.image_size,
        print_metrics=ARGS.print_metrics,
        hint_loss_name=ARGS.hint_loss_name,
        lr=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        batch_size=ARGS.batch_size,
        norm_by_sample=ARGS.norm_by_sample,
        n_epochs=ARGS.epochs,
        hint_lambda=ARGS.hint_lambda,
        cl_lambda=ARGS.cl_lambda,
        early_stopping=ARGS.early_stopping,
        early_stopping_kwargs=ARGS.early_stopping_kwargs,
        lr_sch_metric=ARGS.lr_metric,
        lr_sch_kwargs=ARGS.lr_sch_kwargs,
        h2bb_method_name=ARGS.h2bb_method_name,
        h2bb_method_kwargs=ARGS.h2bb_method_kwargs,
        frontal_only=ARGS.frontal_only,
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
        tb_kwargs=ARGS.tb_kwargs,
        multiple_gpu=ARGS.multiple_gpu,
        num_workers=ARGS.num_workers,
        device=DEVICE,
        seed=ARGS.seed,
        )
