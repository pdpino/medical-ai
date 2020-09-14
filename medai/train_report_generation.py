import time
import argparse
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from ignite.engine import Engine, Events
from ignite.handlers import Timer, Checkpoint, DiskSaver

from medai.datasets.iu_xray import IUXRayDataset
from medai.metrics import save_results
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_report_writer,
)
from medai.models.classification import (
    AVAILABLE_CLASSIFICATION_MODELS,
    create_cnn,
)
from medai.models.report_generation import (
    is_decoder_hierarchical,
    create_decoder,
    AVAILABLE_DECODERS,
)
from medai.models.report_generation.cnn_to_seq import CNN2Seq
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    load_compiled_model_classification,
    load_compiled_model_report_generation,
    save_metadata,
    load_metadata,
)
from medai.tensorboard import TBWriter
from medai.training.report_generation.flat import (
    create_flat_dataloader,
    get_step_fn_flat,
)
from medai.training.report_generation.hierarchical import (
    create_hierarchical_dataloader,
    get_step_fn_hierarchical,
)
from medai.utils import get_timestamp, duration_to_str


def evaluate_model(run_name,
                   model,
                   dataloader,
                   n_epochs=1,
                   hierarchical=False,
                   free=False,
                   debug=True,
                   device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset): dataset = dataset.dataset
    print(f'Evaluating model in {dataset.dataset_type}...')

    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat

    engine = Engine(get_step_fn(model, training=False, free=free, device=device))
    attach_metrics_report_generation(engine, hierarchical=hierarchical, free=free)
    attach_report_writer(engine, dataset.get_vocab(), run_name, debug=debug)

    engine.run(dataloader, n_epochs)
    
    return engine.state.metrics


def evaluate_and_save(run_name,
                      model,
                      dataloaders,
                      hierarchical=False,
                      free=False,
                      debug=True,
                      device='cuda',
                      suffix='',
                      ):
    kwargs = {
        'hierarchical': hierarchical,
        'free': free,
        'device': device,
        'debug': debug,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        name = dataloader.dataset.dataset_type
        metrics[name] = evaluate_model(run_name, model, dataloader, **kwargs)

    save_results(metrics, run_name, classification=False, debug=debug, suffix=suffix)


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                hierarchical=False,
                debug=True,
                save_model=True,
                dryrun=False,
                print_metrics=['loss', 'bleu', 'ciderD'],
                device='cuda',
               ):
    # Prepare run stuff
    print('Run: ', run_name)
    tb_writer = TBWriter(run_name, classification=False, debug=debug, dryrun=dryrun)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        print(f'Resuming from epoch {initial_epoch}')

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


def resume_training(run_name,
                    n_epochs=10,
                    max_samples=None,
                    debug=True,
                    multiple_gpu=False,
                    post_evaluation=True,
                    device='cuda',
                    ):
    """Resume training."""
    # Load model
    compiled_model = load_compiled_model_report_generation(run_name,
                                                           debug=debug,
                                                           device=device,
                                                           multiple_gpu=multiple_gpu)

    # Load metadata (contains all configuration)
    metadata = compiled_model.metadata

    # Decide hierarchical
    decoder_name = metadata['decoder_kwargs']['decoder_name']
    hierarchical = is_decoder_hierarchical(decoder_name)
    if hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader


    # Load data
    vocab = metadata['vocab']
    image_size = metadata.get('image_size', (512, 512))
    dataset_kwargs = {
        'vocab': vocab,
        'image_size': image_size,
        'max_samples': max_samples,
    }
    train_dataset = IUXRayDataset(dataset_type='train', **dataset_kwargs)
    val_dataset = IUXRayDataset(dataset_type='val', **dataset_kwargs)

    batch_size = metadata['hparams'].get('batch_size', 24) # backward compatibility
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size)

    # Train
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                device=device,
                debug=debug)

    print(f'Finished training: {run_name}')


    if post_evaluation:
        test_dataset = IUXRayDataset(dataset_type='test', **dataset_kwargs)
        test_dataloader = create_dataloader(test_dataset, batch_size=batch_size)

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          hierarchical=hierarchical,
                          debug=debug,
                          device=device,
                          )


def train_from_scratch(run_name,
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
                       image_size=512,
                       debug=True,
                       multiple_gpu=False,
                       post_evaluation=True,
                       device='cuda',
                       ):
    """Train a model from scratch."""
    # Create run name
    run_name = f'{run_name}_{decoder_name}_lr{lr}'
    if cnn_run_name:
        run_name += '_precnn'
    else:
        run_name += f'_{cnn_model_name}'
    if image_size != 512:
        run_name += f'_size{image_size}'

    # Decide hierarchical
    hierarchical = is_decoder_hierarchical(decoder_name)
    if hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader

    # Load data
    image_size = (image_size, image_size)
    train_dataset = IUXRayDataset(dataset_type='train',
                                  max_samples=max_samples,
                                  image_size=image_size,
                                  )
    val_dataset = IUXRayDataset(dataset_type='val',
                                image_size=image_size,
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
                                                          device=device,
                                                          multiple_gpu=False,
                                                          # gpus are handled in CNN2Seq!
                                                          )
        cnn = compiled_cnn.model
        cnn_kwargs = compiled_cnn.metadata.get('model_kwargs', {})
    else:
        # Create new
        cnn_kwargs = {
            'model_name': cnn_model_name,
            'labels': [], # headless
            'imagenet': cnn_imagenet,
            'freeze': cnn_freeze,
        }
        cnn = create_cnn(**cnn_kwargs).to(device)

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
        'vocab': train_dataset.get_vocab(),
        'image_size': image_size,
        'hparams': {
            'pretrained_cnn': cnn_run_name,
            'batch_size': batch_size,
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
                                     image_size=image_size,
                                    )
        test_dataloader = create_dataloader(test_dataset, batch_size=batch_size)

        dataloaders = [
            train_dataloader,
            val_dataloader,
            test_dataloader,
        ]

        evaluate_and_save(run_name,
                          compiled_model.model,
                          dataloaders,
                          hierarchical=hierarchical,
                          debug=debug,
                          device=device,
                          )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run')
    parser.add_argument('-dec', '--decoder', type=str,
                        choices=AVAILABLE_DECODERS, help='Choose Decoder')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('-emb', '--embedding-size', type=int, default=100,
                        help='Embedding size of the decoder')
    parser.add_argument('-hs', '--hidden-size', type=int, default=100,
                        help='Hidden size of the decoder')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Input image sizes')
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

    if not args.resume:
        if not args.decoder:
            parser.error('Must choose a decoder')
        if args.cnn is None and args.cnn_pretrained is None:
            parser.error('Must choose one of cnn or cnn_pretrained')

    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _CUDA_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print(f'Using device={device} visible={_CUDA_VISIBLE} multiple={args.multiple_gpu}')

    run_name = get_timestamp()

    start_time = time.time()

    if args.resume:
        resume_training(args.resume,
                        n_epochs=args.epochs,
                        max_samples=args.max_samples,
                        debug=not args.no_debug,
                        multiple_gpu=args.multiple_gpu,
                        device=device,
                        )
    else:
        train_from_scratch(run_name,
                           decoder_name=args.decoder,
                           batch_size=args.batch_size,
                           teacher_forcing=not args.no_teacher_forcing,
                           embedding_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           lr=args.learning_rate,
                           n_epochs=args.epochs,
                           image_size=args.image_size,
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
