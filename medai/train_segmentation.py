import argparse
import logging
import time
import os
import torch
from torch import nn
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import (
    prepare_data_segmentation,
    AVAILABLE_SEGMENTATION_DATASETS,
)
from medai.metrics import save_results
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
from medai.utils import (
    get_timestamp,
    duration_to_str,
    parse_str_or_int,
    print_hw_options,
)
from medai.utils.handlers import attach_log_metrics, attach_early_stopping


logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s(%(asctime)s) %(message)s',
    datefmt='%m-%d %H:%M',
)
LOGGER = logging.getLogger('seg')
LOGGER.setLevel(logging.INFO)


def _get_step_fn(model, optimizer=None, training=False,
                 loss_weights=None,
                 device='cuda'):
    if isinstance(loss_weights, (list, tuple)):
        loss_weights = torch.tensor(loss_weights).to(device)
    elif isinstance(loss_weights, torch.Tensor):
        loss_weights = loss_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    def step_fn(engine, batch):
        images = batch.image.to(device) # shape: batch_size, 1, height, width
        masks = batch.masks.to(device) # shape: batch_size, height, width

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        if training:
            optimizer.zero_grad()

        # Pass thru model
        output = model(images)
        # shape: batch_size, n_labels, height, width

        loss = criterion(output, masks)
        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        return {
            'loss': batch_loss,
            'activations': output,
            'gt_map': masks,
        }

    return step_fn


def evaluate_model(model,
                   dataloader,
                   n_epochs=1,
                   device='cuda'):
    """Evaluate a segmentation model on a dataloader."""
    if dataloader is None:
        return {}

    LOGGER.info(f'Evaluating model in {dataloader.dataset.dataset_type}...')

    labels = dataloader.dataset.seg_labels

    engine = Engine(_get_step_fn(model,
                                 training=False,
                                 device=device,
                                 ))
    attach_metrics_segmentation(engine, labels, multilabel=False)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_and_save(run_name,
                      model,
                      dataloaders,
                      debug=True,
                      device='cuda'):
    """Evaluates a model on multiple dataloaders."""
    kwargs = {
        'device': device,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        name = dataloader.dataset.dataset_type
        metrics[name] = evaluate_model(model, dataloader, **kwargs)

    save_results(metrics, run_name, task='seg', debug=debug)

    return metrics


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                loss_weights=None,
                early_stopping=True,
                early_stopping_kwargs={},
                n_epochs=1,
                print_metrics=None,
                debug=True,
                device='cuda',
                ):
    LOGGER.info(f'Training run: {run_name}')

    tb_writer = TBWriter(run_name,
                         task='seg',
                         debug=debug,
                         )

    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info(f'Resuming from epoch: {initial_epoch}')

    model, optimizer = compiled_model.get_model_optimizer()

    labels = train_dataloader.dataset.seg_labels


    validator = Engine(_get_step_fn(model,
                                    training=False,
                                    loss_weights=loss_weights,
                                    device=device,
                                    ))
    attach_metrics_segmentation(validator, labels, multilabel=False)


    trainer = Engine(_get_step_fn(model,
                                  optimizer,
                                  training=True,
                                  loss_weights=loss_weights,
                                  device=device,
                                  ))
    attach_metrics_segmentation(trainer, labels, multilabel=False)

    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    base_metrics = ['loss', 'iou', 'dice']
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
                            metric=early_stopping_kwargs['metric'] if early_stopping else None,
                            debug=debug,
                            )

    if early_stopping:
        attach_early_stopping(trainer, validator, **early_stopping_kwargs)

    # Train
    LOGGER.info('-'*51)
    LOGGER.info('Training...')

    trainer.run(train_dataloader, n_epochs)

    tb_writer.close()

    # Capture time per epoch
    secs_per_epoch = timer.value()
    duration_per_epoch = duration_to_str(secs_per_epoch)
    LOGGER.info(f'Average time per epoch: {duration_per_epoch}')

    LOGGER.info(f'Finished training: {run_name}')
    LOGGER.info('-'*50)

    return trainer.state.metrics, validator.state.metrics


def resume_training(run_name,
                    n_epochs=10,
                    post_evaluation=True,
                    print_metrics=None,
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
                debug=debug,
                device=device,
                **other_hparams,
                )

    if post_evaluation:
        test_dataloader = prepare_data_segmentation(dataset_type='test',
                                                      **metadata['dataset_kwargs'])

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          debug=debug,
                          device=device)


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
                       batch_size=10,
                       norm_by_sample=False,
                       n_epochs=10,
                       debug=True,
                       num_workers=2,
                       multiple_gpu=False,
                       device='cuda',
                       post_evaluation=True,
                       ):
    run_name = f'{run_name}_{dataset_name}_{fcn_name}_lr{lr}'

    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if image_size != 512:
        run_name += f'_size{image_size}'
    if loss_weights is not None:
        ws = '-'.join(str(int(w*10)) for w in loss_weights)
        run_name += f'_wce{ws}'

    # Load data
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
        'num_workers': num_workers,
        'norm_by_sample': norm_by_sample,
    }
    dataset_train_kwargs = {
        'shuffle': shuffle,
    }
    train_dataloader = prepare_data_segmentation(dataset_type='train', **dataset_kwargs, **dataset_train_kwargs)

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

    # Other training params
    other_train_kwargs = {
        'loss_weights': loss_weights,
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
    }

    # Save metadata
    metadata = {
        'model_kwargs': model_kwargs,
        'opt_kwargs': opt_kwargs,
        'hparams': other_train_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
    }
    save_metadata(metadata, run_name, task='seg', debug=debug)

    # Create compiled model
    compiled_model = CompiledModel(model, optimizer)

    # Train
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                print_metrics=print_metrics,
                debug=debug,
                device=device,
                **other_train_kwargs,
                )

    if post_evaluation:
        test_dataloader = prepare_data_segmentation(dataset_type='test', **dataset_kwargs)

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          debug=debug,
                          device=device)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', type=str, default=None,
                        help='If present, resume a previous run')
    # parser.add_argument('-d', '--dataset', type=str, default=None,
    #                     choices=AVAILABLE_SEGMENTATION_DATASETS,
    #                     help='Choose dataset to train on')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
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
    parser.add_argument('--no-eval', action='store_true',
                        help='If present, dont run post-evaluation')

    # fcn_group = parser.add_argument_group('FCN params')
    # fcn_group.add_argument('-m', '--model', type=str, default=None,
    #                     choices=AVAILABLE_SEGMENTATION_MODELS,
    #                     help='Choose FCN to use')

    es_group = parser.add_argument_group('Early stopping params')
    es_group.add_argument('--no-early-stopping', action='store_true',
                          help='If present, dont early stop the training')
    es_group.add_argument('--es-patience', type=int, default=10,
                          help='Patience value for early-stopping')
    es_group.add_argument('--es-metric', type=str, default='iou',
                          help='Metric to monitor for early-stopping')

    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')
    images_group.add_argument('--norm-by-sample', action='store_true',
                              help='If present, normalize each sample (instead of using dataset stats)')

    hw_group = parser.add_argument_group('Hardware params')
    hw_group.add_argument('--multiple-gpu', action='store_true',
                          help='Use multiple gpus')
    hw_group.add_argument('--cpu', action='store_true',
                          help='Use CPU only')
    hw_group.add_argument('--num-workers', type=int, default=2,
                          help='Number of workers for dataloader')
    hw_group.add_argument('--num-threads', type=int, default=1,
                          help='Number of threads for pytorch')

    args = parser.parse_args()

    # Shortcuts
    args.debug = not args.no_debug
    args.post_evaluation = not args.no_eval

    if args.weight_ce:
        # args.weight_ce = [0.1, 0.3, 0.3, 0.3]
        args.weight_ce = [0.1, 0.4, 0.3, 0.3]
        # args.weight_ce = [0.1, 0.6, 0.3, 0.3]

    # Build early-stopping parameters
    args.early_stopping = not args.no_early_stopping
    args.early_stopping_kwargs = {
        'patience': args.es_patience,
        'metric': args.es_metric,
    }

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(device, args)

    start_time = time.time()

    if args.resume:
        resume_training(args.resume,
                    n_epochs=args.epochs,
                    post_evaluation=args.post_evaluation,
                    print_metrics=args.print_metrics,
                    debug=args.debug,
                    multiple_gpu=args.multiple_gpu,
                    device=device,
                    )
    else:
        run_name = get_timestamp()

        train_from_scratch(run_name,
            shuffle=args.shuffle,
            image_size=args.image_size,
            print_metrics=args.print_metrics,
            lr=args.learning_rate,
            loss_weights=args.weight_ce,
            early_stopping=args.early_stopping,
            early_stopping_kwargs=args.early_stopping_kwargs,
            batch_size=args.batch_size,
            norm_by_sample=args.norm_by_sample,
            n_epochs=args.epochs,
            post_evaluation=args.post_evaluation,
            debug=args.debug,
            multiple_gpu=args.multiple_gpu,
            num_workers=args.num_workers,
            device=device,
            )

    total_time = time.time() - start_time
    LOGGER.info(f'Total time: {duration_to_str(total_time)}')
    LOGGER.info('='*50)