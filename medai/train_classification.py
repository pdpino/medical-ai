import time
import argparse
import os

import torch
from torch import nn
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
)
from medai.losses import get_loss_function, AVAILABLE_LOSSES
from medai.metrics import save_results
from medai.metrics.classification import (
    attach_metrics_classification,
    attach_metric_cm,
)
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.models.classification import (
    create_cnn,
    AVAILABLE_CLASSIFICATION_MODELS,
)
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    save_metadata,
    # load_metadata,
    load_compiled_model_classification,
)
from medai.tensorboard import TBWriter
from medai.utils import (
    get_timestamp,
    duration_to_str,
    parse_str_or_int,
    print_hw_options,
)
from medai.utils.handlers import attach_log_metrics


def _choose_print_metrics(dataset_name, additional=None):
    if dataset_name == 'cxr14':
        print_metrics = ['loss', 'acc', 'hamming']
    elif 'covid' in dataset_name:
        print_metrics = ['loss', 'acc', 'prec_covid', 'recall_covid']
    else:
        print_metrics = ['loss', 'acc']

    if additional is not None:
        print_metrics += [m for m in additional if m not in print_metrics]

    return print_metrics


def get_step_fn(model, loss_fn, optimizer=None, training=True, multilabel=True, device='cuda'):
    """Creates a step function for an Engine."""
    def step_fn(engine, data_batch):
        # Move inputs to GPU
        images = data_batch.image.to(device)
        # shape: batch_size, channels=3, height, width

        labels = data_batch.labels.to(device)
        # shape(multilabel=True): batch_size, n_labels
        # shape(multilabel=False): batch_size

        # Enable training
        model.train(training)
        torch.set_grad_enabled(training)

        # zero the parameter gradients
        if training:
            optimizer.zero_grad()

        # Forward
        output_tuple = model(images)
        outputs = output_tuple[0]
        # shape: batch_size, n_labels

        if multilabel:
            labels = labels.float()
        else:
            labels = labels.long()

        # Compute classification loss
        loss = loss_fn(outputs, labels)

        batch_loss = loss.item()

        if training:
            loss.backward()
            optimizer.step()

        if multilabel:
            # NOTE: multilabel metrics assume output is sigmoided
            outputs = torch.sigmoid(outputs)

        return batch_loss, outputs, labels

    return step_fn


def evaluate_model(model,
                   dataloader,
                   loss_name='wbce',
                   loss_kwargs={},
                   n_epochs=1,
                   device='cuda'):
    """Evaluate a classification model on a dataloader."""
    if dataloader is None:
        return {}

    print(f'Evaluating model in {dataloader.dataset.dataset_type}...')
    loss = get_loss_function(loss_name, **loss_kwargs)

    labels = dataloader.dataset.labels
    multilabel = dataloader.dataset.multilabel

    engine = Engine(get_step_fn(model,
                                loss,
                                training=False,
                                multilabel=multilabel,
                                device=device,
                               ))
    attach_metrics_classification(engine, labels, multilabel=multilabel)
    attach_metric_cm(engine, labels, multilabel=multilabel)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                loss_name='wbce',
                loss_kwargs={},
                debug=True,
                dryrun=False,
                print_metrics=['loss', 'acc'],
                device='cuda',
                ):
    # Prepare run
    print('Training run: ', run_name)
    tb_writer = TBWriter(run_name, task='cls', debug=debug, dryrun=dryrun)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        print('Resuming from epoch: ', initial_epoch)

    # Unwrap stuff
    model, optimizer = compiled_model.get_model_optimizer()

    # Classification description
    labels = train_dataloader.dataset.labels
    multilabel = train_dataloader.dataset.multilabel

    # Prepare loss
    if loss_name == 'focal':
        loss_kwargs['multilabel'] = multilabel
    loss = get_loss_function(loss_name, **loss_kwargs)
    print('Using loss: ', loss_name, loss_kwargs)

    # Create validator engine
    validator = Engine(get_step_fn(model,
                                   loss,
                                   training=False,
                                   multilabel=multilabel,
                                   device=device,
                                   ))
    attach_metrics_classification(validator, labels, multilabel=multilabel)

    # Create trainer engine
    trainer = Engine(get_step_fn(model,
                                 loss,
                                 optimizer=optimizer,
                                 training=True,
                                 multilabel=multilabel,
                                 device=device,
                                 ))
    attach_metrics_classification(trainer, labels, multilabel=multilabel)

    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    attach_log_metrics(trainer,
                       validator,
                       compiled_model,
                       val_dataloader,
                       tb_writer,
                       timer,
                       # logger=LOGGER,
                       initial_epoch=initial_epoch,
                       print_metrics=print_metrics,
                       )

    # Attach checkpoint
    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            validator,
                            task='cls',
                            debug=debug,
                            dryrun=dryrun,
                            )

    # Train!
    print('-' * 50)
    print('Training...')
    trainer.run(train_dataloader, n_epochs)

    # Capture time per epoch
    secs_per_epoch = timer.value()
    duration_per_epoch = duration_to_str(secs_per_epoch)
    print('Average time per epoch: ', duration_per_epoch)
    print('-'*50)

    tb_writer.close()

    return trainer.state.metrics, validator.state.metrics


def evaluate_and_save(run_name,
                      model,
                      dataloaders,
                      loss_name,
                      loss_kwargs={},
                      suffix='',
                      debug=True,
                      device='cuda',
                      ):
    """Evaluates a model on multiple dataloaders."""
    kwargs = {
        'loss_name': loss_name,
        'loss_kwargs': loss_kwargs,
        'device': device,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        name = dataloader.dataset.dataset_type
        metrics[name] = evaluate_model(model, dataloader, **kwargs)

    save_results(metrics, run_name, task='cls', debug=debug, suffix=suffix)

    return metrics


def resume_training(run_name,
                    max_samples=None,
                    n_epochs=10,
                    lr=None,
                    batch_size=None,
                    post_evaluation=True,
                    print_metrics=None,
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
    dataset_kwargs = metadata['dataset_kwargs']

    # Override values
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
        old_lr = metadata['opt_kwargs']['lr']
        print(f'Changing learning rate to {lr}, was {old_lr}')
        for param_group in compiled_model.optimizer.param_groups:
            param_group['lr'] = lr

    # Train
    train_model(run_name, compiled_model, train_dataloader, val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                print_metrics=_choose_print_metrics(dataset_name, print_metrics),
                debug=debug,
                device=device,
                )

    print('Finished training: ', run_name)

    if post_evaluation:
        test_dataloader = prepare_data_classification(dataset_type='test',
                                                      **metadata['dataset_kwargs'])

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          loss_name,
                          loss_kwargs=loss_kwargs,
                          debug=debug,
                          device=device)


def train_from_scratch(run_name,
                       dataset_name,
                       shuffle=False,
                       cnn_name='resnet-50',
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
                       labels=None,
                       batch_size=None,
                       norm_by_sample=False,
                       n_epochs=10,
                       frontal_only=False,
                       oversample=False,
                       oversample_label=None,
                       oversample_class=None,
                       oversample_ratio=None,
                       oversample_max_ratio=None,
                       undersample=False,
                       undersample_label=None,
                       augment=False,
                       augment_label=None,
                       augment_class=None,
                       augment_kwargs={},
                       post_evaluation=True,
                       debug=True,
                       multiple_gpu=False,
                       num_workers=2,
                       device='cuda',
                       ):
    """Train a model from scratch."""
    # Default values
    lr = lr or 1e-6
    batch_size = batch_size or 10

    # Create run name
    run_name = f'{run_name}_{dataset_name}_{cnn_name}_lr{lr}'

    if not imagenet:
        run_name += '_noig'
    if freeze:
        run_name += '_frz'
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
    if labels and dataset_name == 'cxr14':
        # labels only works in CXR-14, for now
        labels_str = '_'.join(labels)
        run_name += f'_{labels_str}'
    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if image_size != 512:
        run_name += f'_size{image_size}'
    if n_epochs == 0:
        run_name += '_e0'
    if fc_layers and len(fc_layers) > 0:
        run_name += '_fc' + '-'.join(str(l) for l in fc_layers)


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
        'augment_kwargs': augment_kwargs,
        'undersample': undersample,
        'undersample_label': undersample_label,
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
    }
    model = create_cnn(**model_kwargs).to(device)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)


    # Create optimizer
    opt_kwargs = {
        'lr': lr,
    }
    optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    # Save model metadata
    metadata = {
        'model_kwargs': model_kwargs,
        'opt_kwargs': opt_kwargs,
        'hparams': {
            'loss_name': loss_name,
            'loss_kwargs': loss_kwargs,
            'batch_size': batch_size,
        },
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
    }
    save_metadata(metadata, run_name, task='cls', debug=debug)


    # Create compiled_model
    compiled_model = CompiledModel(model, optimizer, metadata)

    # Train!
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                loss_kwargs=loss_kwargs,
                print_metrics=_choose_print_metrics(dataset_name, print_metrics),
                debug=debug,
                device=device,
                )

    print('Finished training: ', run_name)

    if post_evaluation:
        test_dataloader = prepare_data_classification(dataset_type='test', **dataset_kwargs)

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          loss_name,
                          loss_kwargs=loss_kwargs,
                          debug=debug,
                          device=device)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--resume', type=str, default=None,
                        help='If present, resume a previous run')
    parser.add_argument('-d', '--dataset', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_DATASETS,
                        help='Choose dataset to train on')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=None,
                        help='Learning rate')
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
    parser.add_argument('--no-eval', action='store_true',
                        help='If present, dont run post-evaluation')

    cnn_group = parser.add_argument_group('CNN params')
    cnn_group.add_argument('-m', '--model', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN to use')
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
    loss_group.add_argument('--focal-alpha', type=float, default=0.75, help='Focal alpha param')
    loss_group.add_argument('--focal-gamma', type=float, default=2, help='Focal gamma param')


    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')
    images_group.add_argument('--frontal-only', action='store_true',
                              help='Use only frontal images')
    images_group.add_argument('--norm-by-sample', action='store_true',
                              help='If present, normalize each sample (instead of using dataset stats)')

    aug_group = parser.add_argument_group('Data-augmentation params')
    aug_group.add_argument('--augment', action='store_true',
                        help='If present, augment dataset')
    aug_group.add_argument('--augment-label', default=None,
                        help='Augment only samples with a given label present (str/int)')
    aug_group.add_argument('--augment-class', type=int, choices=[0,1], default=None,
                        help='If --augment-label is provided, choose if augmenting \
                              positive (1) or negative (0) samples')
    aug_group.add_argument('--aug-crop', type=float, default=0.8,
                        help='Augment samples by cropping a random fraction')
    aug_group.add_argument('--aug-translate', type=float, default=0.1,
                        help='Augment samples by translating a random fraction')
    aug_group.add_argument('--aug-rotation', type=int, default=15,
                        help='Augment samples by rotating a random amount of degrees')
    aug_group.add_argument('--aug-contrast', type=float, default=0.5,
                        help='Augment samples by changing the contrast randomly')
    aug_group.add_argument('--aug-brightness', type=float, default=0.5,
                        help='Augment samples by changing the brightness randomly')

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
                             help='Undersample from the majority class with a given label (str/int)')

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

    # If training from scratch, require dataset and model
    if not args.resume:
        if args.dataset is None: parser.error('A dataset must be selected')
        if args.model is None: parser.error('A model must be selected')


    # Build loss params
    if args.loss_name == 'focal':
        args.loss_kwargs = {
            'alpha': args.focal_alpha,
            'gamma': args.focal_gamma,
        }
    else:
        args.loss_kwargs = {}

    # Build augment params
    if args.augment:
        args.augment_kwargs = {
            'crop': args.aug_crop,
            'translate': args.aug_translate,
            'rotation': args.aug_rotation,
            'contrast': args.aug_contrast,
            'brightness': args.aug_brightness,
        }
    else:
        args.augment_kwargs = {}

    # Enable passing str or int for augment/oversample labels
    if args.augment_label is not None:
        args.augment_label = parse_str_or_int(args.augment_label)
    if args.oversample is not None:
        args.oversample = parse_str_or_int(args.oversample)
    if args.undersample is not None:
        args.undersample = parse_str_or_int(args.undersample)

    # Shortcuts
    args.debug = not args.no_debug
    args.post_evaluation = not args.no_eval

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
                        max_samples=args.max_samples,
                        n_epochs=args.epochs,
                        lr=args.learning_rate,
                        batch_size=args.batch_size,
                        print_metrics=args.print_metrics,
                        post_evaluation=args.post_evaluation,
                        debug=args.debug,
                        multiple_gpu=args.multiple_gpu,
                        device=device)
    else:
        run_name = get_timestamp()

        train_from_scratch(run_name,
            args.dataset,
            shuffle=args.shuffle,
            cnn_name=args.model,
            imagenet=not args.no_imagenet,
            freeze=args.freeze,
            cnn_pooling=args.cnn_pooling,
            fc_layers=args.fc_layers,
            max_samples=args.max_samples,
            image_size=args.image_size,
            loss_name=args.loss_name,
            loss_kwargs=args.loss_kwargs,
            print_metrics=args.print_metrics,
            labels=args.labels,
            lr=args.learning_rate,
            batch_size=args.batch_size,
            norm_by_sample=args.norm_by_sample,
            n_epochs=args.epochs,
            oversample=args.oversample is not None,
            oversample_label=args.oversample,
            oversample_class=args.os_class,
            oversample_ratio=args.os_ratio,
            oversample_max_ratio=args.os_max_ratio,
            augment=args.augment,
            augment_label=args.augment_label,
            augment_class=args.augment_class,
            augment_kwargs=args.augment_kwargs,
            undersample=args.undersample is not None,
            undersample_label=args.undersample,
            post_evaluation=args.post_evaluation,
            debug=args.debug,
            multiple_gpu=args.multiple_gpu,
            num_workers=args.num_workers,
            device=device,
            )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
    print('=' * 80)
