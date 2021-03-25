import argparse
import logging
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import (
    prepare_data_segmentation,
    AVAILABLE_SEGMENTATION_DATASETS,
)
from medai.metrics.segmentation import attach_metrics_segmentation
from medai.models.segmentation import (
    create_fcn,
    # AVAILABLE_SEGMENTATION_MODELS,
)
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    save_metadata,
    load_compiled_model_segmentation,
)
from medai.tensorboard import TBWriter
from medai.training.segmentation import get_step_fn
from medai.utils import (
    get_timestamp,
    duration_to_str,
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


LOGGER = logging.getLogger('medai.seg.train')


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                loss_weights=None,
                early_stopping=True,
                early_stopping_kwargs={},
                lr_sch_metric='loss',
                n_epochs=1,
                print_metrics=None,
                tb_kwargs={},
                debug=True,
                device='cuda',
                ):
    LOGGER.info('Training run: %s (debug=%s)', run_name, debug)

    tb_writer = TBWriter(run_name,
                         task='seg',
                         debug=debug,
                         **tb_kwargs,
                         )

    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info('Resuming from epoch: %s', initial_epoch)

    model, optimizer, lr_scheduler = compiled_model.get_elements()

    labels = train_dataloader.dataset.seg_labels
    multilabel = train_dataloader.dataset.multilabel


    validator = Engine(get_step_fn(model,
                                   training=False,
                                   multilabel=multilabel,
                                   loss_weights=loss_weights,
                                   device=device,
                                   ))
    attach_metrics_segmentation(validator, labels, multilabel=multilabel)


    trainer = Engine(get_step_fn(model,
                                 optimizer,
                                 training=True,
                                 multilabel=multilabel,
                                 loss_weights=loss_weights,
                                 device=device,
                                 ))
    attach_metrics_segmentation(trainer, labels, multilabel=multilabel)

    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    base_metrics = ['loss', 'iou', 'dice', 'iobb']
    if print_metrics:
        print_metrics = base_metrics + print_metrics
    else:
        print_metrics = base_metrics

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

    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            validator,
                            task='seg',
                            metric='iou',
                            debug=debug,
                            )

    if early_stopping:
        attach_early_stopping(trainer, validator, **early_stopping_kwargs)

    if lr_scheduler is not None:
        attach_lr_scheduler_handler(lr_scheduler, trainer, validator, lr_sch_metric)


    # Train
    LOGGER.info('-'*51)
    LOGGER.info('Training...')

    trainer.run(train_dataloader, n_epochs)

    tb_writer.close()

    # Capture time per epoch
    secs_per_epoch = timer.value()
    duration_per_epoch = duration_to_str(secs_per_epoch)
    LOGGER.info('Average time per epoch: %s', duration_per_epoch)

    LOGGER.info('Finished training: %s (debug=%s)', run_name, debug)

    return trainer.state.metrics, validator.state.metrics


@timeit_main(LOGGER)
def resume_training(run_name,
                    n_epochs=10,
                    print_metrics=None,
                    tb_kwargs={},
                    debug=True,
                    multiple_gpu=False,
                    device='cuda',
                    ):
    """Resume training from a previous run."""
    # Load model
    compiled_model = load_compiled_model_segmentation(run_name,
                                                      debug=debug,
                                                      device=device,
                                                      multiple_gpu=multiple_gpu)


    # Metadata (contains all configuration)
    metadata = compiled_model.metadata
    set_seed_from_metadata(metadata)

    dataset_kwargs = metadata['dataset_kwargs']

    # Load data
    train_dataloader = prepare_data_segmentation(dataset_type='train',
                                                   **metadata['dataset_train_kwargs'],
                                                   **dataset_kwargs,
                                                   )
    val_dataloader = prepare_data_segmentation(dataset_type='val', **dataset_kwargs)

    # Train
    other_hparams = metadata.get('hparams', {})

    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                print_metrics=print_metrics,
                tb_kwargs=tb_kwargs,
                debug=debug,
                device=device,
                **other_hparams,
                )


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       dataset_name='jsrt',
                       shuffle=False,
                       fcn_name='scan',
                       image_size=512,
                       print_metrics=None,
                       lr=None,
                       loss_weights=None,
                       early_stopping=True,
                       early_stopping_kwargs={},
                       lr_sch_metric='loss',
                       lr_sch_kwargs={},
                       batch_size=10,
                       max_samples=None,
                       norm_by_sample=False,
                       n_epochs=10,
                       augment=False,
                       augment_label=None,
                       augment_class=None,
                       augment_times=1,
                       augment_kwargs={},
                       debug=True,
                       tb_kwargs={},
                       num_workers=2,
                       multiple_gpu=False,
                       device='cuda',
                       seed=None,
                       ):
    run_name = f'{run_name}_{dataset_name}_{fcn_name}_lr{lr}'

    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if image_size != 512:
        run_name += f'_size{image_size}'
    if loss_weights:
        ws = '-'.join(str(int(w*10)) for w in loss_weights)
        run_name += f'_wce{ws}'
    if augment:
        run_name += f'_aug{augment_times}'
    if lr_sch_metric:
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        run_name += f'_sch-{lr_sch_metric}-p{patience}-f{factor}'

    set_seed(seed)

    # Load data
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
        'num_workers': num_workers,
        'norm_by_sample': norm_by_sample,
        'image_format': 'L', # SCAN model needs 1-channel
        'max_samples': max_samples,
        'masks': True,
    }
    dataset_train_kwargs = {
        'shuffle': shuffle,
        'augment': augment,
        'augment_label': augment_label,
        'augment_class': augment_class,
        'augment_times': augment_times,
        'augment_kwargs': augment_kwargs,
    }
    train_dataloader = prepare_data_segmentation(
        dataset_type='train',
        **dataset_kwargs,
        **dataset_train_kwargs,
        )

    val_dataloader = prepare_data_segmentation(dataset_type='val', **dataset_kwargs)

    # Create model
    model_kwargs = {
        'model_name': fcn_name,
        'n_classes': len(train_dataloader.dataset.seg_labels)
    }
    model = create_fcn(**model_kwargs).to(device)

    if multiple_gpu:
        model = nn.DataParallel(model)

    # Create optimizer
    opt_kwargs = {
        'lr': lr,
    }
    optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    # Create lr_scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, **lr_sch_kwargs) if lr_sch_metric else None

    # Other training params
    other_train_kwargs = {
        'loss_weights': loss_weights,
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
    }

    # Save metadata
    metadata = {
        'model_kwargs': model_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs if lr_sch_metric else None,
        'hparams': other_train_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_name, task='seg', debug=debug)

    # Create compiled model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Train
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                print_metrics=print_metrics,
                tb_kwargs=tb_kwargs,
                debug=debug,
                device=device,
                **other_train_kwargs,
                )


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    parser.add_argument('--resume', type=str, default=None,
                        help='If present, resume a previous run')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        choices=AVAILABLE_SEGMENTATION_DATASETS,
                        help='Choose dataset to train on')
    parser.add_argument('-wce', '--weight-ce', action='store_true',
                        help='If present add weights to the cross-entropy')
    parser.add_argument('-bs', '--batch-size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--shuffle', action='store_true', default=None,
                        help='Whether to shuffle or not the samples when training')
    parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                        help='Additional metrics to print to stdout')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')

    # fcn_group = parser.add_argument_group('FCN params')
    # fcn_group.add_argument('-m', '--model', type=str, default=None,
    #                     choices=AVAILABLE_SEGMENTATION_MODELS,
    #                     help='Choose FCN to use')

    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')
    images_group.add_argument('--norm-by-sample', action='store_true',
                              help='If present, normalize each sample \
                                   (instead of using dataset stats)')

    parsers.add_args_early_stopping(parser, metric='iou')

    parsers.add_args_lr_sch(parser, lr=0.0001, metric='iou')

    parsers.add_args_augment(parser)

    parsers.add_args_tb(parser)

    parsers.add_args_hw(parser, num_workers=2)

    args = parser.parse_args()

    # Shortcuts
    args.debug = not args.no_debug

    if args.weight_ce and args.dataset == 'jsrt':
        # args.weight_ce = [0.1, 0.3, 0.3, 0.3]
        # args.weight_ce = [0.1, 0.4, 0.3, 0.3]
        args.weight_ce = [0.1, 0.6, 0.3, 0.3]
    else:
        args.weight_ce = None

    # Build parameters
    parsers.build_args_early_stopping_(args)
    parsers.build_args_lr_sch_(args)
    parsers.build_args_augment_(args)
    parsers.build_args_tb_(args)

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
                    n_epochs=ARGS.epochs,
                    print_metrics=ARGS.print_metrics,
                    tb_kwargs=ARGS.tb_kwargs,
                    debug=ARGS.debug,
                    multiple_gpu=ARGS.multiple_gpu,
                    device=DEVICE,
                    )
    else:
        train_from_scratch(
            get_timestamp(),
            dataset_name=ARGS.dataset,
            shuffle=ARGS.shuffle,
            image_size=ARGS.image_size,
            print_metrics=ARGS.print_metrics,
            lr=ARGS.learning_rate,
            loss_weights=ARGS.weight_ce,
            early_stopping=ARGS.early_stopping,
            early_stopping_kwargs=ARGS.early_stopping_kwargs,
            lr_sch_metric=ARGS.lr_metric,
            lr_sch_kwargs=ARGS.lr_sch_kwargs,
            batch_size=ARGS.batch_size,
            max_samples=ARGS.max_samples,
            norm_by_sample=ARGS.norm_by_sample,
            n_epochs=ARGS.epochs,
            augment=ARGS.augment,
            augment_label=ARGS.augment_label,
            augment_class=ARGS.augment_class,
            augment_times=ARGS.augment_times,
            augment_kwargs=ARGS.augment_kwargs,
            tb_kwargs=ARGS.tb_kwargs,
            debug=ARGS.debug,
            multiple_gpu=ARGS.multiple_gpu,
            num_workers=ARGS.num_workers,
            device=DEVICE,
            seed=ARGS.seed,
            )
