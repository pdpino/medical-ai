import time
import argparse
import os
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import prepare_data_report_generation
from medai.metrics import save_results
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_medical_correctness,
    attach_report_writer,
)
from medai.models.classification import (
    create_cnn,
    AVAILABLE_CLASSIFICATION_MODELS,
    DEPRECATED_CNNS,
)
from medai.models.report_generation import (
    is_decoder_hierarchical,
    create_decoder,
    AVAILABLE_DECODERS,
    DEPRECATED_DECODERS,
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
from medai.utils import (
    get_timestamp,
    duration_to_str,
    print_hw_options,
    parsers,
    config_logging,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_lr_scheduler_handler,
)


config_logging()
LOGGER = logging.getLogger('rg')
LOGGER.setLevel(logging.INFO)


def evaluate_model(run_name,
                   model,
                   dataloader,
                   n_epochs=1,
                   hierarchical=False,
                   supervise_attention=False,
                   free=False,
                   debug=True,
                   device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset): dataset = dataset.dataset # HACK
    print(f'Evaluating model in {dataset.dataset_type}, free={free}...')

    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat

    engine = Engine(get_step_fn(model,
                                training=False,
                                supervise_attention=supervise_attention,
                                free=free,
                                device=device))
    attach_metrics_report_generation(engine,
                                     hierarchical=hierarchical,
                                     free=free,
                                     supervise_attention=supervise_attention,
                                     )
    attach_report_writer(engine, dataset.get_vocab(), run_name, free=free,
                         debug=debug)
    attach_medical_correctness(engine, None, dataset.get_vocab())

    # Catch errors, specially for free=True case
    engine.add_event_handler(Events.EXCEPTION_RAISED, lambda _, err: print(err))

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_and_save(run_name,
                      model,
                      dataloaders,
                      hierarchical=False,
                      supervise_attention=False,
                      free='both',
                      debug=True,
                      device='cuda',
                      suffix='',
                      ):
    kwargs = {
        'hierarchical': hierarchical,
        'device': device,
        'supervise_attention': supervise_attention,
        'debug': debug,
    }

    if free == 'both':
        free_values = [False, True]
    elif free:
        free_values = [True]
    else:
        free_values = [False]

    for free in free_values:
        # Add a suffix
        more_suffix = 'free' if free else 'notfree'
        if suffix:
            used_suffix = f'{suffix}-{more_suffix}'
        else:
            used_suffix = more_suffix

        metrics = {}

        kwargs['free'] = free

        for dataloader in dataloaders:
            if dataloader is None:
                continue
            name = dataloader.dataset.dataset_type
            metrics[name] = evaluate_model(run_name, model, dataloader, **kwargs)

        save_results(metrics,
                     run_name,
                     task='rg',
                     debug=debug,
                     suffix=used_suffix,
                     )


def train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                supervise_attention=False,
                hierarchical=False,
                debug=True,
                save_model=True,
                dryrun=False,
                tb_kwargs={},
                medical_correctness=True,
                early_stopping=True,
                early_stopping_kwargs={},
                lr_sch_metric='loss',
                print_metrics=['loss', 'bleu', 'ciderD', 'chex_timer'],
                device='cuda',
               ):
    # Prepare run stuff
    print('Run: ', run_name)
    tb_writer = TBWriter(run_name, task='rg', debug=debug, dryrun=dryrun, **tb_kwargs)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        print(f'Resuming from epoch {initial_epoch}')

    # Unwrap stuff
    model, optimizer, lr_scheduler = compiled_model.get_elements()

    # Flat vs hierarchical step_fn
    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat


    # Create validator engine
    validator = Engine(get_step_fn(model,
                                   training=False,
                                   supervise_attention=supervise_attention,
                                   device=device))
    attach_metrics_report_generation(validator,
                                     hierarchical=hierarchical,
                                     supervise_attention=supervise_attention,
                                     )

    # Create trainer engine
    trainer = Engine(get_step_fn(model,
                                 optimizer=optimizer,
                                 training=True,
                                 supervise_attention=supervise_attention,
                                 device=device))
    attach_metrics_report_generation(trainer,
                                     hierarchical=hierarchical,
                                     supervise_attention=supervise_attention,
                                     )

    # Attach medical correctness metrics
    if medical_correctness:
        vocab = train_dataloader.dataset.get_vocab()
        attach_medical_correctness(trainer, validator, vocab)

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
                            task='rg',
                            metric=early_stopping_kwargs.get('metric') if early_stopping else None,
                            debug=debug,
                            dryrun=dryrun or (not save_model),
                           )

    if early_stopping:
        attach_early_stopping(trainer, validator, **early_stopping_kwargs)

    if lr_scheduler is not None:
        attach_lr_scheduler_handler(lr_scheduler, trainer, validator, lr_sch_metric)

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

    return trainer, validator


def resume_training(run_name,
                    n_epochs=10,
                    max_samples=None,
                    tb_kwargs={},
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
    dataset_kwargs = metadata.get('dataset_kwargs', None)
    if dataset_kwargs is None:
        # HACK: backward compatibility
        dataset_kwargs = {
            'vocab': metadata['vocab'],
            'image_size': metadata.get('image_size', (512, 512)),
            'max_samples': max_samples,
            'batch_size': metadata['hparams'].get('batch_size', 24),
        }
    dataset_train_kwargs = metadata.get('dataset_train_kwargs', {})

    train_dataloader = prepare_data_report_generation(
        create_dataloader,
        dataset_type='train',
        **dataset_kwargs,
        **dataset_train_kwargs,
    )
    val_dataloader = prepare_data_report_generation(
        create_dataloader,
        dataset_type='val',
        **dataset_kwargs,
    )

    # Train
    other_train_kwargs = metadata.get('other_train_kwargs', {})
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                device=device,
                debug=debug,
                **other_train_kwargs,
                )

    print(f'Finished training: {run_name}')


    # TODO: move evaluation to a different script
    return

    if post_evaluation:
        test_dataloader = prepare_data_report_generation(
            create_dataloader,
            dataset_type='test',
            **dataset_kwargs,
        )

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
                       supervise_attention=False,
                       batch_size=15,
                       sort_samples=True,
                       shuffle=False,
                       frontal_only=False,
                       teacher_forcing=True,
                       embedding_size=100,
                       hidden_size=100,
                       lr=0.0001,
                       n_epochs=10,
                       medical_correctness=True,
                       cnn_run_name=None,
                       cnn_model_name='resnet-50',
                       cnn_imagenet=True,
                       cnn_freeze=False,
                       max_samples=None,
                       image_size=512,
                       early_stopping=True,
                       early_stopping_kwargs={},
                       lr_sch_metric='loss',
                       lr_sch_kwargs={},
                       augment=False,
                       augment_label=None,
                       augment_class=None,
                       augment_times=1,
                       augment_kwargs={},
                       tb_kwargs={},
                       debug=True,
                       multiple_gpu=False,
                       post_evaluation=True,
                       num_workers=2,
                       device='cuda',
                       ):
    """Train a model from scratch."""
    # Create run name
    run_name = f'{run_name}_{decoder_name}_lr{lr}'
    if supervise_attention:
        run_name += '_satt'
    if cnn_run_name:
        run_name += '_precnn'
    else:
        run_name += f'_{cnn_model_name}'
    if image_size != 512:
        run_name += f'_size{image_size}'
    if not sort_samples:
        run_name += '_nosort'
    if shuffle:
        run_name += '_shf'
    if augment:
        run_name += '_aug'
        if augment_label is not None:
            run_name += f'-{augment_label}'
            if augment_class is not None:
                run_name += f'-cls{augment_class}'
    if lr_sch_metric:
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        run_name += f'_sch-{lr_sch_metric}-p{patience}-f{factor}'
    if not early_stopping:
        run_name += '_noes'
    if frontal_only and not supervise_attention: # If supervise attention, frontal_only is implied
        run_name += '_front'

    # Is deprecated
    if decoder_name in DEPRECATED_DECODERS:
        raise Exception(f'RG model is deprecated: {decoder_name}')

    # Decide hierarchical
    hierarchical = is_decoder_hierarchical(decoder_name)
    if hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader

    if supervise_attention:
        if not hierarchical:
            raise Exception('Attention supervision is only available for hierarchical decoders')
        if not frontal_only:
            raise Exception('Attention supervision is only available with frontal_only images')


    # Load data
    image_size = (image_size, image_size)
    dataset_kwargs = {
        'dataset_name': 'iu-x-ray',
        'max_samples': max_samples,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'masks': hierarchical,
        'frontal_only': frontal_only,
    }
    dataset_train_kwargs = {
        'sort_samples': sort_samples,
        'shuffle': shuffle,
        'augment': augment,
        'augment_label': augment_label,
        'augment_class': augment_class,
        'augment_times': augment_times,
        'augment_kwargs': augment_kwargs,
    }
    train_dataloader = prepare_data_report_generation(
        create_dataloader,
        dataset_type='train',
        **dataset_kwargs,
        **dataset_train_kwargs,
    )
    vocab = train_dataloader.dataset.get_vocab()
    dataset_kwargs['vocab'] = vocab

    val_dataloader = prepare_data_report_generation(
        create_dataloader,
        dataset_type='val',
        **dataset_kwargs,
    )


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
        if cnn_model_name in DEPRECATED_CNNS:
            raise Exception(f'CNN is deprecated: {cnn_model_name}')
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
        'vocab_size': len(vocab),
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

    # Create lr_scheduler
    if lr_sch_metric:
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_sch_kwargs)
        LOGGER.info('Using ReduceLROnPlateau')
    else:
        LOGGER.info('Not using a LR scheduler')
        lr_scheduler = None

    # Other training params
    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
        'supervise_attention': supervise_attention,
        'medical_correctness': medical_correctness,
    }

    # Save metadata
    metadata = {
        'cnn_kwargs': cnn_kwargs,
        'decoder_kwargs': decoder_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs if lr_sch_metric else None,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        'vocab': vocab,
        'image_size': image_size,
        'hparams': {
            'pretrained_cnn': cnn_run_name,
            'batch_size': batch_size,
        },
        'other_train_kwargs': other_train_kwargs,
    }
    save_metadata(metadata, run_name, task='rg', debug=debug)

    # Compiled model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Train
    train_model(run_name,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                device=device,
                debug=debug,
                **other_train_kwargs,
                )


    print(f'Finished run: {run_name}')


    # TODO: move evaluation to a different script
    return

    if post_evaluation:
        test_dataloader = prepare_data_report_generation(
            create_dataloader,
            dataset_type='test',
            **dataset_kwargs,
        )

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

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run')
    parser.add_argument('-dec', '--decoder', type=str,
                        choices=AVAILABLE_DECODERS, help='Choose Decoder')
    parser.add_argument('--superv-att', action='store_true',
                        help='If present, supervise the attention')
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
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--no-med', action='store_true',
                        help='If present, do not use medical-correctness metrics')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--image-size', type=int, default=512,
                            help='Input image sizes')
    data_group.add_argument('--max-samples', type=int, default=None,
                            help='Max samples to load (debugging)')
    data_group.add_argument('--no-sort', action='store_true',
                            help='Do not sort samples')
    data_group.add_argument('--shuffle', action='store_true',
                            help='Shuffle samples on training')
    data_group.add_argument('--frontal-only', action='store_true',
                            help='Use only frontal images')

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

    parsers.add_args_early_stopping(parser)
    parsers.add_args_lr_sch(parser, lr=0.0001, metric=None)
    parsers.add_args_tb(parser)
    parsers.add_args_augment(parser)

    parsers.add_args_hw(parser, num_workers=4)

    args = parser.parse_args()

    if not args.resume:
        if not args.decoder:
            parser.error('Must choose a decoder')
        if args.cnn is None and args.cnn_pretrained is None:
            parser.error('Must choose one of cnn or cnn_pretrained')

    # Build params
    parsers.build_args_early_stopping_(args)
    parsers.build_args_lr_sch_(args)
    parsers.build_args_augment_(args)
    parsers.build_args_tb_(args)

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(device, args)

    run_name = get_timestamp()

    start_time = time.time()

    if args.resume:
        resume_training(args.resume,
                        n_epochs=args.epochs,
                        max_samples=args.max_samples,
                        tb_kwargs=args.tb_kwargs,
                        debug=not args.no_debug,
                        multiple_gpu=args.multiple_gpu,
                        device=device,
                        )
    else:
        train_from_scratch(run_name,
                           decoder_name=args.decoder,
                           supervise_attention=args.superv_att,
                           batch_size=args.batch_size,
                           sort_samples=not args.no_sort,
                           shuffle=args.shuffle,
                           frontal_only=args.frontal_only,
                           teacher_forcing=not args.no_teacher_forcing,
                           embedding_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           lr=args.learning_rate,
                           n_epochs=args.epochs,
                           medical_correctness=not args.no_med,
                           image_size=args.image_size,
                           cnn_run_name=args.cnn_pretrained,
                           cnn_model_name=args.cnn,
                           cnn_imagenet=not args.no_imagenet,
                           cnn_freeze=args.freeze,
                           max_samples=args.max_samples,
                           early_stopping=args.early_stopping,
                           early_stopping_kwargs=args.early_stopping_kwargs,
                           lr_sch_metric=args.lr_metric,
                           lr_sch_kwargs=args.lr_sch_kwargs,
                           augment=args.augment,
                           augment_label=args.augment_label,
                           augment_class=args.augment_class,
                           augment_times=args.augment_times,
                           augment_kwargs=args.augment_kwargs,
                           tb_kwargs=args.tb_kwargs,
                           debug=not args.no_debug,
                           multiple_gpu=args.multiple_gpu,
                           num_workers=args.num_workers,
                           device=device,
                           )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
    print('=' * 80)
