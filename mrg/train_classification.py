import time
import argparse

import torch
from torch import nn
from torch import optim
from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver

from mrg.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
)
from mrg.losses import get_loss_function
from mrg.metrics.classification import attach_metrics_classification
from mrg.models.classification import (
    init_empty_model,
    AVAILABLE_CLASSIFICATION_MODELS,
)
from mrg.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    save_metadata,
)
from mrg.tensorboard import TBWriter
from mrg.utils import get_timestamp, duration_to_str


def get_step_fn(model, loss_fn, optimizer=None, training=True, multilabel=True, device='cuda'):
    """Creates a step function for an Engine."""
    def step_fn(engine, data_batch):
        # Move inputs to GPU
        images = data_batch[0].to(device)
        # shape: batch_size, channels=3, height, width
        
        labels = data_batch[1].to(device)
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

        return batch_loss, outputs, labels

    return step_fn


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                loss_name='wbce',
                debug=True,
                print_metrics=['loss', 'acc'],
                device='cuda',
                ):
    # Prepare run
    tb_writer = TBWriter(run_name, classification=True, debug=debug)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        print('Resuming from epoch: ', initial_epoch)
    
    # Unwrap stuff
    model, optimizer = compiled_model.get_model_optimizer()
    
    # Prepare loss
    loss = get_loss_function(loss_name)
    
    # Classification labels
    labels = train_dataloader.dataset.labels
    multilabel = train_dataloader.dataset.multilabel

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

    # Attach checkpoint
    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            classification=True,
                            debug=debug,
                            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_metrics(trainer):
        # Run on evaluation
        validator.run(val_dataloader, 1)

        # State
        epoch = trainer.state.epoch + initial_epoch
        max_epochs = trainer.state.max_epochs + initial_epoch
        train_metrics = trainer.state.metrics
        val_metrics = validator.state.metrics

        # Save state
        compiled_model.save_current_epoch(epoch)
        
        # Walltime
        wall_time = time.time()

        # Log to TB
        tb_writer.write_histogram(model, epoch, wall_time)
        tb_writer.write_metrics(train_metrics, 'train', epoch, wall_time)
        tb_writer.write_metrics(val_metrics, 'val', epoch, wall_time)
        
        # Print results
        print_str = f'Finished epoch {epoch}/{max_epochs}'
        for metric in print_metrics:
            train_value = train_metrics.get(metric, -1)
            val_value = val_metrics.get(metric, -1)
            metric_str = f' {metric} {train_value:.4f} {val_value:.4f},'
            print_str += metric_str

        print_str += f' {duration_to_str(timer._elapsed())}'
        print(print_str)

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

    return


def main(run_name,
         dataset_name,
         cnn_name='resnet',
         imagenet=True,
         freeze=False,
         max_samples=None,
         lr = 0.000001,
         batch_size=10,
         n_epochs=10,
         oversample=False,
         oversample_label=None,
         oversample_max_ratio=None,
         debug=True,
         multiple_gpu=False,
         device='cuda',
         ):
    # Create run name
    run_name = f'{run_name}_{dataset_name}_{cnn_name}_lr{lr}'

    if not imagenet:
        run_name += '_noig'
    if freeze:
        run_name += '_frz'
    if oversample:
        run_name += '_os'

    print('Run: ', run_name)


    # Load data
    train_dataloader = prepare_data_classification(dataset_name,
                                                   dataset_type='train',
                                                   max_samples=max_samples,
                                                   batch_size=batch_size,
                                                   oversample=oversample,
                                                   oversample_label=oversample_label,
                                                   oversample_max_ratio=oversample_max_ratio,
                                                   )
    val_dataloader = prepare_data_classification(dataset_name,
                                                 dataset_type='val',
                                                 max_samples=max_samples,
                                                 batch_size=batch_size)

    # Create model
    model_kwargs = {
        'model_name': cnn_name,
        'labels': train_dataloader.dataset.labels,
        'multilabel': train_dataloader.dataset.multilabel,
        'imagenet': imagenet,
        'freeze': freeze,
    }
    model = init_empty_model(**model_kwargs).to(device)

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
        # TODO: add hparams here?
    }
    save_metadata(metadata, run_name, classification=True, debug=debug)


    # Create compiled_model
    compiled_model = CompiledModel(model, optimizer)


    # Print metrics (hardcoded)
    if dataset_name == 'cxr14':
        print_metrics = ['loss', 'acc', 'hamming']
    elif 'covid' in dataset_name:
        print_metrics = ['loss', 'acc', 'spec_covid', 'recall_covid']
    else:
        print_metrics = ['loss', 'acc']

    # Decide loss
    if train_dataloader.dataset.multilabel:
        loss_name = 'wbce'
        # REVIEW: enable choosing a different loss?
    else:
        loss_name = 'cross-entropy'

    # Train!
    train_model(run_name, compiled_model, train_dataloader, val_dataloader,
                n_epochs=n_epochs,
                loss_name=loss_name,
                print_metrics=print_metrics,
                debug=debug,
                device=device,
                )

    print('Finished run: ', run_name)


def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--name', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default=None, required=True,
                        choices=AVAILABLE_CLASSIFICATION_DATASETS,
                        help='Choose dataset to train on')
    parser.add_argument('-m', '--model', type=str, default=None, required=True,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN to use')
    parser.add_argument('-noig', '--no-imagenet', action='store_true',
                        help='If present, dont use imagenet pretrained weights')
    parser.add_argument('-frz', '--freeze', action='store_true',
                        help='If present, freeze base cnn parameters (only train FC layers)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('-os', '--oversample', default=None,
                        help='Oversample samples with a given label')
    parser.add_argument('--os-max-ratio', default=None,
                        help='Max ratio ')
    parser.add_argument('--multiple-gpu', action='store_true',
                        help='Use multiple gpus')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    # TODO: once model resuming is implemented, enable names other than timestamp
    run_name = get_timestamp()

    start_time = time.time()

    main(run_name,
         args.dataset,
         cnn_name=args.model,
         imagenet=not args.no_imagenet,
         freeze=args.freeze,
         max_samples=args.max_samples,
         lr=args.learning_rate,
         batch_size=args.batch_size,
         n_epochs=args.epochs,
         oversample=args.oversample is not None,
         oversample_label=args.oversample,
         oversample_max_ratio=args.os_max_ratio,
         debug=not args.no_debug,
         multiple_gpu=args.multiple_gpu,
         device=device,
         )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
