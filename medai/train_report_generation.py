import argparse
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import prepare_data_report_generation, AVAILABLE_REPORT_DATASETS
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_attention_vs_masks,
    attach_losses_rg,
)
from medai.metrics.report_generation.labeler_correctness import attach_medical_correctness
from medai.models.classification import (
    create_cnn,
    AVAILABLE_CLASSIFICATION_MODELS,
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
    load_compiled_model,
    load_compiled_model_report_generation,
    save_metadata,
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
    set_seed,
    set_seed_from_metadata,
    timeit_main,
    RunId,
)
from medai.utils.handlers import (
    attach_log_metrics,
    attach_early_stopping,
    attach_lr_scheduler_handler,
)


LOGGER = logging.getLogger('medai.rg.train')


_CORRECTNESS_TARGET_METRIC = 'chex_f1_woNF'

def _get_print_metrics(additional_metrics):
    print_metrics = ['loss', 'bleu', 'ciderD', _CORRECTNESS_TARGET_METRIC]

    for m in (additional_metrics or []):
        if m not in print_metrics:
            print_metrics.append(m)

    return print_metrics



def train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                n_epochs=1,
                supervise_attention=False,
                hierarchical=False,
                save_model=True,
                dryrun=False,
                tb_kwargs={},
                medical_correctness=True,
                med_kwargs={},
                att_vs_masks=False,
                early_stopping=True,
                early_stopping_kwargs={},
                lr_sch_metric='loss',
                print_metrics=None,
                device='cuda',
               ):
    # Prepare run stuff
    LOGGER.info('Training run: %s', run_id)
    tb_writer = TBWriter(run_id, dryrun=dryrun, **tb_kwargs)
    initial_epoch = compiled_model.get_current_epoch()
    if initial_epoch > 0:
        LOGGER.info('Resuming from epoch %s', initial_epoch)

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
    attach_losses_rg(
        validator,
        hierarchical=hierarchical,
        supervise_attention=supervise_attention,
        device=device,
    )
    attach_metrics_report_generation(validator, device=device)

    # Create trainer engine
    trainer = Engine(get_step_fn(model,
                                 optimizer=optimizer,
                                 training=True,
                                 supervise_attention=supervise_attention,
                                 device=device))
    # Set state dict
    # Since the state is explictly set here:
    #  - n_epochs must not be passed to `trainer.run()`, or this state will be overriden
    #  - initial_epoch must not be passed to attach_log_metrics
    #    (the trainer knows the correct current epoch)
    trainer.load_state_dict({
        'epoch': initial_epoch,
        'max_epochs': initial_epoch + n_epochs,
        'epoch_length': len(train_dataloader),
    })
    attach_losses_rg(
        trainer,
        hierarchical=hierarchical,
        supervise_attention=supervise_attention,
        device=device,
    )
    attach_metrics_report_generation(trainer, device=device)

    # Attach medical correctness metrics
    if medical_correctness:
        vocab = train_dataloader.dataset.get_vocab()
        attach_medical_correctness(trainer, validator, vocab, device=device, **med_kwargs)

    if att_vs_masks:
        attach_attention_vs_masks(trainer, device=device)
        attach_attention_vs_masks(validator, device=device)

        if not train_dataloader.dataset.enable_masks:
            raise Exception('Att-vs-masks attached, but masks are not enabled!')

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
                       print_metrics=_get_print_metrics(print_metrics),
                       )

    # Attach checkpoint
    checkpoint_metric = _CORRECTNESS_TARGET_METRIC if medical_correctness else None
    attach_checkpoint_saver(run_id,
                            compiled_model,
                            trainer,
                            validator,
                            metric=checkpoint_metric,
                            dryrun=dryrun or (not save_model),
                           )

    if early_stopping:
        attach_early_stopping(trainer, validator, **early_stopping_kwargs)

    if lr_scheduler is not None:
        attach_lr_scheduler_handler(lr_scheduler, trainer, validator, lr_sch_metric)

    # Train!
    LOGGER.info('-' * 51)
    LOGGER.info('Training...')
    trainer.run(train_dataloader)

    # Capture time
    secs_per_epoch = timer.value()
    LOGGER.info('Average time per epoch: %s', duration_to_str(secs_per_epoch))
    LOGGER.info('-' * 50)

    # Close stuff
    tb_writer.close()

    LOGGER.info('Finished training: %s', run_id)

    return trainer, validator


@timeit_main(LOGGER)
def resume_training(run_id,
                    n_epochs=10,
                    max_samples=None,
                    tb_kwargs={},
                    print_metrics=None,
                    multiple_gpu=False,
                    device='cuda',
                    ):
    """Resume training."""
    assert run_id.task == 'rg'
    # Load model
    compiled_model = load_compiled_model_report_generation(run_id,
                                                           device=device,
                                                           multiple_gpu=multiple_gpu)

    # Load metadata (contains all configuration)
    metadata = compiled_model.metadata
    set_seed_from_metadata(metadata)

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
    train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                print_metrics=print_metrics,
                device=device,
                **other_train_kwargs,
                )


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       dataset_name='iu-x-ray',
                       decoder_name='lstm',
                       supervise_attention=False,
                       batch_size=15,
                       sort_samples=True,
                       shuffle=False,
                       frontal_only=False,
                       norm_by_sample=False,
                       teacher_forcing=True,
                       embedding_size=100,
                       hidden_size=100,
                       lr=0.0001,
                       n_epochs=10,
                       medical_correctness=True,
                       med_kwargs={},
                       cnn_run_id=None,
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
                       print_metrics=None,
                       ):
    """Train a model from scratch."""
    # Create run name
    run_name = f'{run_name}_{dataset_name}_{decoder_name}'
    if supervise_attention:
        run_name += '_satt'
    if embedding_size != 100:
        run_name += f'_embs-{embedding_size}'
    if hidden_size != 100:
        run_name += f'_hs-{hidden_size}'
    if cnn_run_id:
        run_name += f'_precnn-{cnn_run_id.short_clean_name}'
    else:
        run_name += f'_{cnn_model_name}'
    if norm_by_sample:
        run_name += '_normS'
    else:
        run_name += '_normD'
    if image_size != 256:
        run_name += f'_size{image_size}'
    if not shuffle:
        if sort_samples:
            run_name += '_sorted'
        else:
            run_name += '_notshuffle'
    if augment:
        run_name += f'_aug{augment_times}'
        if augment_mode != 'single':
            run_name += f'-{augment_mode}'
        if augment_label is not None:
            run_name += f'-{augment_label}'
            if augment_class is not None:
                run_name += f'-cls{augment_class}'
    run_name += f'_lr{lr}'
    if lr_sch_metric:
        factor = lr_sch_kwargs['factor']
        patience = lr_sch_kwargs['patience']
        run_name += f'_sch-{lr_sch_metric.replace("_", "-")}-p{patience}-f{factor}'
    if frontal_only and not supervise_attention: # If supervise attention, frontal_only is implied
        run_name += '_front'

    # Is deprecated
    if decoder_name in DEPRECATED_DECODERS:
        raise Exception(f'RG model is deprecated: {decoder_name}')

    run_id = RunId(run_name, debug, 'rg', experiment)

    set_seed(seed)

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
    enable_masks = supervise_attention
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'max_samples': max_samples,
        'norm_by_sample': norm_by_sample,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'masks': enable_masks,
        'frontal_only': frontal_only,
    }
    dataset_train_kwargs = {
        'sort_samples': sort_samples,
        'shuffle': shuffle,
        'augment': augment,
        'augment_mode': augment_mode,
        'augment_label': augment_label,
        'augment_class': augment_class,
        'augment_times': augment_times,
        'augment_kwargs': augment_kwargs,
        'augment_seg_mask': enable_masks,
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
    if cnn_run_id:
        # Load pretrained
        compiled_cnn = load_compiled_model(
            cnn_run_id, device=device, multiple_gpu=False, # gpus are handled in CNN2Seq!
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
        LOGGER.info('Using ReduceLROnPlateau (with %s})', lr_sch_metric)
    else:
        LOGGER.warning('Not using a LR scheduler')
        lr_scheduler = None

    # Other training params
    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
        'supervise_attention': supervise_attention,
        'medical_correctness': medical_correctness,
        'med_kwargs': med_kwargs,
        'att_vs_masks': enable_masks,
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
            'pretrained_cnn': cnn_run_id.to_dict() if cnn_run_id else None,
            'batch_size': batch_size,
        },
        'other_train_kwargs': other_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_id)

    # Compiled model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Train
    train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                device=device,
                print_metrics=print_metrics,
                **other_train_kwargs,
                )


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    parser.add_argument('-d', '--dataset', type=str, default='iu-x-ray',
                        help='Batch size', choices=AVAILABLE_REPORT_DATASETS)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run')
    parser.add_argument('-dec', '--decoder', type=str,
                        choices=AVAILABLE_DECODERS, help='Choose Decoder')
    parser.add_argument('--superv-att', action='store_true',
                        help='If present, supervise the attention')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
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
    parser.add_argument('-exp', '--experiment', type=str, default='',
                        help='Custom experiment name')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')
    parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                        help='Additional metrics to print to stdout')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--image-size', type=int, default=512,
                            help='Input image sizes')
    data_group.add_argument('--no-sort', action='store_true',
                            help='Do not sort samples')
    data_group.add_argument('--shuffle', action='store_true',
                            help='Shuffle samples on training')
    data_group.add_argument('--frontal-only', action='store_true',
                            help='Use only frontal images')
    data_group.add_argument('--norm-by-sample', action='store_true',
                            help='Normalize each sample (instead of by dataset stats)')

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
    cnn_group.add_argument('-cp-task', '--cnn-pretrained-task', type=str, default='cls',
                        choices=('cls', 'cls-seg'), help='Task to choose the CNN from')

    parsers.add_args_early_stopping(parser, metric=_CORRECTNESS_TARGET_METRIC)
    parsers.add_args_lr_sch(parser, lr=0.0001, metric=None)
    parsers.add_args_tb(parser)
    parsers.add_args_augment(parser)

    parsers.add_args_hw(parser, num_workers=4)
    parsers.add_args_med(parser)

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
    parsers.build_args_med_(args)

    if not args.no_med and args.early_stopping and \
        (args.med_after is not None and args.es_patience <= args.med_after):
        LOGGER.warning(
            'ES-patience (%d) is less than med-after (%d), run may get preempted',
            args.es_patience, args.med_after,
        )

    if args.cnn_pretrained is not None:
        args.precnn_run_id = RunId(
            args.cnn_pretrained,
            debug=False, # NOTE: Even when debugging, --no-debug pre-cnns are more common
            task=args.cnn_pretrained_task,
        )
    else:
        args.precnn_run_id = None

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    if ARGS.num_threads > 0:
        torch.set_num_threads(ARGS.num_threads)

    config_logging()

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    if ARGS.resume:
        resume_training(RunId(ARGS.resume, not ARGS.no_debug, 'rg'),
                        n_epochs=ARGS.epochs,
                        max_samples=ARGS.max_samples,
                        tb_kwargs=ARGS.tb_kwargs,
                        multiple_gpu=ARGS.multiple_gpu,
                        device=DEVICE,
                        print_metrics=ARGS.print_metrics,
                        )
    else:
        train_from_scratch(get_timestamp(),
                           dataset_name=ARGS.dataset,
                           decoder_name=ARGS.decoder,
                           supervise_attention=ARGS.superv_att,
                           batch_size=ARGS.batch_size,
                           sort_samples=not ARGS.no_sort,
                           shuffle=ARGS.shuffle,
                           frontal_only=ARGS.frontal_only,
                           norm_by_sample=ARGS.norm_by_sample,
                           teacher_forcing=not ARGS.no_teacher_forcing,
                           embedding_size=ARGS.embedding_size,
                           hidden_size=ARGS.hidden_size,
                           lr=ARGS.learning_rate,
                           n_epochs=ARGS.epochs,
                           medical_correctness=not ARGS.no_med,
                           med_kwargs=ARGS.med_kwargs,
                           image_size=ARGS.image_size,
                           cnn_run_id=ARGS.precnn_run_id,
                           cnn_model_name=ARGS.cnn,
                           cnn_imagenet=not ARGS.no_imagenet,
                           cnn_freeze=ARGS.freeze,
                           max_samples=ARGS.max_samples,
                           early_stopping=ARGS.early_stopping,
                           early_stopping_kwargs=ARGS.early_stopping_kwargs,
                           lr_sch_metric=ARGS.lr_metric,
                           lr_sch_kwargs=ARGS.lr_sch_kwargs,
                           augment=ARGS.augment,
                           augment_mode=ARGS.augment_mode,
                           augment_label=ARGS.augment_label,
                           augment_class=ARGS.augment_class,
                           augment_times=ARGS.augment_times,
                           augment_kwargs=ARGS.augment_kwargs,
                           tb_kwargs=ARGS.tb_kwargs,
                           debug=not ARGS.no_debug,
                           experiment=ARGS.experiment,
                           multiple_gpu=ARGS.multiple_gpu,
                           num_workers=ARGS.num_workers,
                           device=DEVICE,
                           seed=ARGS.seed,
                           print_metrics=ARGS.print_metrics,
                           )
