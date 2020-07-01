import time
import argparse

import torch
from torch import nn
from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver
from torch import optim

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
    get_checkpoint_folder,
)
from mrg.tensorboard import TBWriter
from mrg.utils import get_timestamp, duration_to_str


DEVICE = torch.device('cuda')


def get_step_fn(model, loss_fn, optimizer=None, training=True, multilabel=True, device=DEVICE):
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


def train_model(run_name, compiled_model, train_dataloader, val_dataloader, n_epochs=1,
               loss_name='wbce', debug=True, print_metrics=['loss', 'acc']):
    # Prepare run
    tb_writer = TBWriter(run_name, classification=True, debug=debug)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        print('Resuming from epoch: ', initial_epoch)
    
    # Unwrap stuff
    model, optimizer = compiled_model.state()
    
    # Prepare loss
    loss = get_loss_function(loss_name)
    
    # Classification labels
    labels = train_dataloader.dataset.labels
    multilabel = train_dataloader.dataset.multilabel

    # Create validator engine
    validator = Engine(get_step_fn(model, loss, training=False, multilabel=multilabel))
    attach_metrics_classification(validator, labels, multilabel=multilabel)
    
    # Create trainer engine
    trainer = Engine(get_step_fn(model, loss, optimizer=optimizer,
                                 training=True, multilabel=multilabel))
    attach_metrics_classification(trainer, labels, multilabel=multilabel)
    
    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    # Attach checkpoint
    folderpath = get_checkpoint_folder(run_name, classification=True, debug=debug)
    checkpoint = Checkpoint(
        compiled_model.to_save_checkpoint(),
        DiskSaver(folderpath, require_empty=False, atomic=False),
        # global_step_transform=lambda trainer, _: trainer.state.epoch + initial_epoch,
    )
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

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
            if not (metric in train_metrics and metric in val_metrics):
                continue
            train_value = train_metrics[metric]
            val_value = val_metrics[metric]
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
         debug=True,
         multiple_gpu=False,
         ):
    # Create run name
    run_name = f'{run_name}_{dataset_name}_{cnn_name}_lr{lr}'

    if not imagenet:
        run_name += '_noig'
    if freeze:
        run_name += '_frz'

    print('Run: ', run_name)


    # Load data
    train_dataloader = prepare_data_classification(dataset_name,
                                                   dataset_type='train',
                                                   max_samples=max_samples,
                                                   batch_size=batch_size)
    val_dataloader = prepare_data_classification(dataset_name,
                                                 dataset_type='val',
                                                 max_samples=max_samples,
                                                 batch_size=batch_size)

    # Create model
    model = init_empty_model(cnn_name,
                             train_dataloader.dataset.labels,
                             multilabel=train_dataloader.dataset.multilabel,
                             imagenet=imagenet,
                             freeze=freeze,
                            ).to(DEVICE)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)

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
    parser.add_argument('--multiple-gpu', action='store_true',
                        help='Use multiple gpus')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
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
         debug=not args.no_debug,
         multiple_gpu=args.multiple_gpu,
         )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
