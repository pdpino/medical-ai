import time
import argparse

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver

from mrg.datasets.iu_xray import IUXRayDataset
from mrg.metrics import save_results
from mrg.metrics.report_generation import attach_metrics_report_generation
from mrg.models.classification import (
    AVAILABLE_CLASSIFICATION_MODELS,
    init_empty_model,
)
from mrg.models.report_generation import (
    is_decoder_hierarchical,
    create_decoder,
    AVAILABLE_DECODERS,
)
from mrg.models.report_generation.cnn_to_seq import CNN2Seq
from mrg.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    load_compiled_model_classification,
    save_metadata,
)
from mrg.tensorboard import TBWriter
from mrg.training.report_generation.flat import (
    create_flat_dataloader,
    get_step_fn_flat,
)
from mrg.training.report_generation.hierarchical import (
    create_hierarchical_dataloader,
    get_step_fn_hierarchical,
)
from mrg.utils import get_timestamp, duration_to_str


def evaluate_model(model,
                   dataloader,
                   n_epochs=1,
                   hierarchical=False,
                   device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat

    engine = Engine(get_step_fn(model, training=False, device=device))
    attach_metrics_report_generation(engine, hierarchical=hierarchical)

    engine.run(dataloader, n_epochs)
    
    return engine.state.metrics


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                hierarchical=False,
                debug=True,
                save_model=True,
                dryrun=False,
                print_metrics=['loss', 'bleu'],
                device='cuda',
               ):
    # Prepare run stuff
    tb_writer = TBWriter(run_name, classification=False, debug=debug, dryrun=dryrun)
    initial_epoch = compiled_model.get_current_epoch()

    # Unwrap stuff
    model, optimizer = compiled_model.get_model_optimizer()

    # Flat vs hierarchical step_fn
    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat


    # Create validator engine
    validator = Engine(get_step_fn(model, training=False, device=device))
    attach_metrics_report_generation(validator, hierarchical=hierarchical)
    
    # Create trainer engine
    trainer = Engine(get_step_fn(model, optimizer=optimizer, training=True, device=device))
    attach_metrics_report_generation(trainer, hierarchical=hierarchical)
    
    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def tb_write_metrics(trainer):
        # Run on evaluation
        validator.run(val_dataloader, 1)

        # State
        epoch = trainer.state.epoch + initial_epoch
        max_epochs = trainer.state.max_epochs + initial_epoch
        train_metrics = trainer.state.metrics
        val_metrics = validator.state.metrics
        
        # Save state
        compiled_model.save_current_epoch(epoch)

        # Common time
        wall_time = time.time()
        
        # Log to TB
        tb_writer.write_histogram(model, epoch, wall_time)
        tb_writer.write_metrics(train_metrics, 'train', epoch, wall_time)
        tb_writer.write_metrics(val_metrics, 'val', epoch, wall_time)
        
        # Print metrics
        print_str = f'Finished epoch {epoch}/{max_epochs}'
        for metric in print_metrics:
            train_value = train_metrics.get(metric, -1)
            val_value = val_metrics.get(metric, -1)
            metric_str = f' {metric} {train_value:.4f} {val_value:.4f},'
            print_str += metric_str

        print_str += f' {duration_to_str(timer._elapsed())}'
        print(print_str)

    # Attach checkpoint
    attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            classification=False,
                            debug=debug,
                            dryrun=dryrun or (not save_model),
                           )

    # Train!
    print('-' * 50)
    print('Training...')
    trainer.run(train_dataloader, n_epochs)

    # Capture time
    secs_per_epoch = timer.value()
    duration_per_epoch = duration_to_str(secs_per_epoch)
    print('Average time per epoch: ', duration_per_epoch)
    print('-'*50)
    
    # Close stuff
    tb_writer.close()

    return trainer.state.metrics, validator.state.metrics


def main(run_name,
         decoder_name='lstm',
         batch_size=15,
         teacher_forcing=True,
         embedding_size=100,
         hidden_size=100,
         lr=0.0001,
         n_epochs=10,
         cnn_run_name=None,
         cnn_model_name='resnet-50',
         cnn_imagenet=True,
         cnn_freeze=False,
         max_samples=None,
         debug=True,
         multiple_gpu=False,
         post_evaluation=True,
         device='cuda',
         ):

    # Create run name
    run_name = f'{run_name}_{decoder_name}_lr{lr}'
    if cnn_run_name:
        run_name += '_precnn'
    else:
        run_name += f'_{cnn_model_name}'

    print('Run: ', run_name)

    # Decide hierarchical
    hierarchical = is_decoder_hierarchical(decoder_name)
    if hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader


    # Load data
    train_dataset = IUXRayDataset(dataset_type='train', max_samples=max_samples)
    val_dataset = IUXRayDataset(dataset_type='val',
                                vocab=train_dataset.get_vocab(),
                                max_samples=max_samples,
                                )

    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size)


    # Create CNN
    if cnn_run_name:
        # Load pretrained
        compiled_cnn = load_compiled_model_classification(cnn_run_name,
                                                          debug=debug,
                                                          device=device)
        cnn = compiled_cnn.model
        cnn_kwargs = compiled_model.metadata.get('model_kwargs', {})
    else:
        # Create new
        cnn_kwargs = {
            'model_name': cnn_model_name,
            'labels': [], # headless
            'imagenet': cnn_imagenet,
            'freeze': cnn_freeze,
        }
        cnn = init_empty_model(**cnn_kwargs).to(device)

    # Create decoder
    decoder_kwargs = {
        'decoder_name': decoder_name,
        'vocab_size': len(train_dataset.word_to_idx),
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'features_size': cnn.features_size,
        'teacher_forcing': teacher_forcing,
    }
    decoder = create_decoder(**decoder_kwargs).to(device)

    # Full model
    model = CNN2Seq(cnn, decoder).to(device)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    # Optimizer
    opt_kwargs = {
        'lr': lr,
    }
    optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    # Save metadata
    metadata = {
        'cnn_kwargs': cnn_kwargs,
        'decoder_kwargs': decoder_kwargs,
        'opt_kwargs': opt_kwargs,
        'hparams': {
            'pretrained_cnn': cnn_run_name,
        },
    }
    save_metadata(metadata, run_name, classification=False, debug=debug)

    # Compiled model
    compiled_model = CompiledModel(model, optimizer, metadata)

    # Train
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                device=device,
                debug=debug)


    print(f'Finished run: {run_name}')


    if post_evaluation:
        test_dataset = IUXRayDataset(dataset_type='test',
                                     vocab=train_dataset.get_vocab(),
                                     max_samples=max_samples,
                                    )
        test_dataloader = create_dataloader(test_dataset, batch_size=batch_size)

        kwargs = {
            'hierarchical': hierarchical,
            'device': device
        }
        train_metrics = evaluate_model(model, train_dataloader, **kwargs)
        val_metrics = evaluate_model(model, val_dataloader, **kwargs)
        test_metrics = evaluate_model(model, test_dataloader, **kwargs)

        metrics = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
        }
        save_results(metrics, run_name, classification=False, debug=debug)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-dec', '--decoder', type=str, required=True,
                        choices=AVAILABLE_DECODERS, help='Choose Decoder')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-6,
                        help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('-emb', '--embedding-size', type=int, default=100,
                        help='Embedding size of the decoder')
    parser.add_argument('-hs', '--hidden-size', type=int, default=100,
                        help='Hidden size of the decoder')
    parser.add_argument('-notf', '--no-teacher-forcing', action='store_true',
                        help='If present, does not use teacher forcing')
    parser.add_argument('--multiple-gpu', action='store_true',
                        help='Use multiple gpus')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')

    cnn_group = parser.add_argument_group('CNN')
    cnn_group.add_argument('-c', '--cnn', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN class (create new)')
    cnn_group.add_argument('-noig', '--no-imagenet', action='store_true',
                        help='If present, dont use imagenet pretrained weights')
    cnn_group.add_argument('-frz', '--freeze', action='store_true',
                        help='If present, freeze base cnn parameters (only train FC layers)')
    cnn_group.add_argument('-cp', '--cnn-pretrained', type=str, default=None,
                        help='Run name of a pretrained CNN')

    args = parser.parse_args()

    if args.cnn is None and args.cnn_pretrained is None:
        raise Exception('Must choose one of cnn or cnn_pretrained')

    return args


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    # TODO: once model resuming is implemented, enable names other than timestamp
    run_name = get_timestamp()

    start_time = time.time()

    main(run_name,
         decoder_name=args.decoder,
         batch_size=args.batch_size,
         teacher_forcing=not args.no_teacher_forcing,
         embedding_size=args.embedding_size,
         hidden_size=args.hidden_size,
         lr=args.learning_rate,
         n_epochs=args.epochs,
         cnn_run_name=args.cnn_pretrained,
         cnn_model_name=args.cnn,
         cnn_imagenet=not args.no_imagenet,
         cnn_freeze=args.freeze,
         max_samples=args.max_samples,
         debug=not args.no_debug,
         multiple_gpu=args.multiple_gpu,
         device=device,
         )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
