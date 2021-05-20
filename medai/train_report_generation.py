import argparse
import logging
import os

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Engine, Events
from ignite.handlers import Timer

from medai.datasets import prepare_data_report_generation, AVAILABLE_REPORT_DATASETS
from medai.datasets.common import LATEST_REPORTS_VERSION
from medai.models import save_training_stats
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_attention_vs_masks,
    attach_losses_rg,
    attach_organ_by_sentence,
)
from medai.metrics.report_generation.labeler_correctness import attach_medical_correctness
from medai.models.classification import (
    AVAILABLE_CLASSIFICATION_MODELS,
)
from medai.models.report_generation import (
    is_decoder_hierarchical,
    create_decoder,
    AVAILABLE_DECODERS,
    DEPRECATED_DECODERS,
)
from medai.models.report_generation.word_embedding import AVAILABLE_PRETRAINED_EMBEDDINGS
from medai.models.report_generation.cnn_to_seq import CNN2Seq
from medai.models.checkpoint import (
    CompiledModel,
    attach_checkpoint_saver,
    load_compiled_model,
    load_compiled_model_report_generation,
    save_metadata,
    create_cnn_rg,
)
from medai.losses.optimizers import create_optimizer
from medai.tensorboard import TBWriter
from medai.training.report_generation.flat import get_step_fn_flat
from medai.training.report_generation.hierarchical import get_step_fn_hierarchical
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
from medai.utils.nlp import attach_unclean_report_checker


LOGGER = logging.getLogger('medai.rg.train')


_CORRECTNESS_TARGET_METRIC = 'chex_f1_woNF'

def _get_print_metrics(additional_metrics,
                       hierarchical=False,
                       supervise_attention=False, supervise_sentences=False):
    if hierarchical:
        print_metrics = ['word_loss', 'stop_loss']
    else:
        print_metrics = ['loss']
    if supervise_attention:
        print_metrics.append('att_loss')
    if supervise_sentences:
        print_metrics.append('sentence_loss')

    print_metrics += ['bleu', _CORRECTNESS_TARGET_METRIC]

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
                supervise_sentences=False,
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
                lambda_word=1,
                lambda_stop=1,
                lambda_att=1,
                lambda_sent=1,
                organ_by_sentence=True,
                print_metrics=None,
                check_unclean=True,
                checkpoint_metric=None,
                device='cuda',
                hw_options={},
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

    step_kwargs = {
        'supervise_attention': supervise_attention,
        'supervise_sentences': supervise_sentences,
        'lambda_word': lambda_word,
        'lambda_stop': lambda_stop,
        'lambda_att': lambda_att,
        'lambda_sent': lambda_sent,
        'device': device,
    }
    loss_attacher_kwargs = {
        'hierarchical': hierarchical,
        'supervise_attention': supervise_attention,
        'supervise_sentences': supervise_sentences,
        'device': device,
    }

    vocab = train_dataloader.dataset.get_vocab()

    # Create validator engine
    validator = Engine(get_step_fn(model, training=False, **step_kwargs))
    attach_losses_rg(validator, **loss_attacher_kwargs)
    attach_unclean_report_checker(validator, check=check_unclean)
    attach_metrics_report_generation(validator, device=device)
    attach_organ_by_sentence(validator, vocab, organ_by_sentence, device=device)

    # Create trainer engine
    trainer = Engine(get_step_fn(model, optimizer=optimizer, training=True, **step_kwargs))
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
    attach_unclean_report_checker(trainer, check=check_unclean)
    attach_losses_rg(trainer, **loss_attacher_kwargs)
    attach_metrics_report_generation(trainer, device=device)
    attach_organ_by_sentence(trainer, vocab, organ_by_sentence, device=device)

    # Attach medical correctness metrics
    if medical_correctness:
        attach_medical_correctness(trainer, validator, vocab, device=device, **med_kwargs)

    if att_vs_masks:
        attach_attention_vs_masks(trainer, device=device)
        attach_attention_vs_masks(validator, device=device)

        if not train_dataloader.dataset.enable_masks:
            raise Exception('Att-vs-masks attached, but masks are not enabled!')

    # Create Timer to measure wall time between epochs
    timer = Timer(average=True)
    timer.attach(trainer, start=Events.EPOCH_STARTED, step=Events.EPOCH_COMPLETED)

    attach_log_metrics(
        trainer,
        validator,
        compiled_model,
        val_dataloader,
        tb_writer,
        timer,
        logger=LOGGER,
        print_metrics=_get_print_metrics(
            print_metrics,
            hierarchical=hierarchical,
            supervise_attention=supervise_attention,
            supervise_sentences=supervise_sentences,
        ),
    )

    # Attach checkpoint
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

    save_training_stats(
        run_id,
        train_dataloader,
        n_epochs,
        secs_per_epoch,
        hw_options,
        initial_epoch,
        dryrun=(not save_model),
    )

    LOGGER.info('Finished training: %s', run_id)

    return trainer, validator


@timeit_main(LOGGER)
def resume_training(run_id,
                    n_epochs=10,
                    max_samples=None,
                    tb_kwargs={},
                    med_kwargs={},
                    early_stopping=None,
                    print_metrics=None,
                    multiple_gpu=False,
                    check_unclean=True,
                    device='cuda',
                    hw_options={},
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

    # Load data
    dataset_kwargs = metadata.get('dataset_kwargs', None)
    if dataset_kwargs is None:
        raise NotImplementedError('Fully deprecated...')
        # # HACK: backward compatibility
        # dataset_kwargs = {
        #     'vocab': metadata['vocab'],
        #     'image_size': metadata.get('image_size', (512, 512)),
        #     'max_samples': max_samples,
        #     'batch_size': metadata['hparams'].get('batch_size', 24),
        # }
    if max_samples is not None:
        dataset_kwargs['max_samples'] = max_samples
    if 'hierarchical' not in dataset_kwargs:
        # Backward compatibility
        dataset_kwargs['hierarchical'] = hierarchical

    dataset_train_kwargs = metadata.get('dataset_train_kwargs', {})

    train_dataloader = prepare_data_report_generation(
        dataset_type='train',
        **dataset_kwargs,
        **dataset_train_kwargs,
    )
    val_dataloader = prepare_data_report_generation(
        dataset_type='val',
        **dataset_kwargs,
    )

    # Override train_kwargs
    other_train_kwargs = metadata.get('other_train_kwargs', {})
    if other_train_kwargs.get('early_stopping', True) and not early_stopping:
        # FIXME: when resuming, you can only change this from false to true,
        other_train_kwargs['early_stopping'] = False

    if other_train_kwargs.get('medical_correctness', False):
        # FIXME: you can only change med_kwargs
        for key in other_train_kwargs['med_kwargs']:
            new_value = med_kwargs.get(key, None)
            if new_value is not None:
                other_train_kwargs['med_kwargs'][key] = new_value


    # Train
    train_model(run_id,
                compiled_model,
                train_dataloader,
                val_dataloader,
                hierarchical=hierarchical,
                n_epochs=n_epochs,
                tb_kwargs=tb_kwargs,
                print_metrics=print_metrics,
                device=device,
                check_unclean=check_unclean,
                hw_options=hw_options,
                **other_train_kwargs,
                )


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       dataset_name='iu-x-ray',
                       decoder_name='lstm',
                       dropout_recursive=0,
                       dropout_out=0,
                       att_double_bias=False,
                       supervise_attention=False,
                       supervise_sentences=False,
                       batch_size=15,
                       sort_samples=True,
                       shuffle=False,
                       frontal_only=False,
                       norm_by_sample=False,
                       vocab_greater=None,
                       teacher_forcing=True,
                       embedding_size=100,
                       embedding_kwargs={},
                       hidden_size=100,
                       lr=0.0001,
                       weight_decay=0,
                       custom_lr=None,
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
                       lambda_word=1,
                       lambda_stop=1,
                       lambda_att=1,
                       lambda_sent=1,
                       checkpoint_metric=None,
                       organ_by_sentence=True,
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
                       save_model=True,
                       seed=None,
                       check_unclean=True,
                       print_metrics=None,
                       hw_options={},
                       ):
    """Train a model from scratch."""
    # Decide hierarchical
    hierarchical = is_decoder_hierarchical(decoder_name)

    # Create run name
    run_name = f'{run_name}_{dataset_name}_{decoder_name}'
    if lambda_word != 1:
        run_name += f'_word-{lambda_word}'
    if hierarchical:
        if lambda_stop != 1:
            run_name += f'_stop-{lambda_stop}'
    if supervise_attention:
        run_name += '_satt'
        if lambda_att != 1:
            run_name += f'-{lambda_att}'
    if supervise_sentences:
        run_name += '_ssent'
        if lambda_sent != 1:
            run_name += f'-{lambda_sent}'
    if dropout_recursive != 0:
        run_name += f'_dropr{dropout_recursive}'
    if dropout_out != 0:
        run_name += f'_dropo{dropout_out}'
    if embedding_size != 100:
        run_name += f'_embsize-{embedding_size}'
    if embedding_kwargs is not None:
        # Collect all embedding options
        options = []
        _pretrained = embedding_kwargs.get('pretrained')
        if _pretrained is not None:
            if _pretrained != 'radglove':
                options.append(_pretrained)
            _freeze = embedding_kwargs.get('freeze')
            if _freeze:
                options.append('frz')
        else:
            options.append('rand')
        if embedding_kwargs.get('scale_grad_by_freq'):
            options.append('scale')
        if embedding_kwargs.get('batch_normalization'):
            options.append('bn')
        # Add options to run_name
        if len(options) > 0:
            run_name += f'_emb-{"-".join(options)}'
    if hidden_size != 100:
        run_name += f'_hs-{hidden_size}'
    if 'att' in decoder_name and att_double_bias:
        run_name += '_att-bias2'
    if cnn_run_id:
        run_name += f'_precnn-{cnn_run_id.short_clean_name}'
    else:
        run_name += f'_{cnn_model_name}'
    if not norm_by_sample:
        run_name += '_normD'
    if image_size != 256:
        run_name += f'_size{image_size}'
    if not shuffle:
        if sort_samples:
            run_name += '_sorted'
        else:
            run_name += '_notshuffle'
    if vocab_greater is not None:
        run_name += f'_voc{vocab_greater}'
    if augment:
        run_name += f'_aug{augment_times}'
        if augment_mode != 'single':
            run_name += f'-{augment_mode}'
        if augment_label is not None:
            run_name += f'-{augment_label}'
            if augment_class is not None:
                run_name += f'-cls{augment_class}'
    run_name += f'_lr{lr}'
    if custom_lr is not None:
        lr_emb = custom_lr.get('word_embeddings')
        if lr_emb is not None:
            run_name += f'_lr-emb{lr_emb}'
        lr_att = custom_lr.get('attention')
        if lr_att is not None:
            run_name += f'_lr-att{lr_att}'
    if weight_decay != 0:
        run_name += f'_wd{weight_decay}'
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

    if supervise_attention:
        if not hierarchical:
            raise Exception('Attention supervision is only available for hierarchical decoders')
        if not frontal_only:
            raise Exception('Attention supervision is only available with frontal_only images')
        if 'h-lstm-att' not in decoder_name:
            raise Exception('Attention supervision is only available for h-lstm-att decoders')

    if supervise_sentences:
        # TODO: move this to a more appropiate place?
        if hidden_size != 100:
            raise Exception('Hidden size must be 100 if supervise_sentences=True')


    # Load data
    image_size = (image_size, image_size)
    enable_masks = supervise_attention
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'hierarchical': hierarchical,
        'max_samples': max_samples,
        'norm_by_sample': norm_by_sample,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'masks': enable_masks,
        'frontal_only': frontal_only,
        'vocab_greater': vocab_greater,
        'reports_version': LATEST_REPORTS_VERSION,
        'sentence_embeddings': supervise_sentences,
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
        dataset_type='train',
        **dataset_kwargs,
        **dataset_train_kwargs,
    )
    vocab = train_dataloader.dataset.get_vocab()
    dataset_kwargs['vocab'] = vocab

    val_dataloader = prepare_data_report_generation(
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
        # HACK: kind of hacky solution to support both CLS and CLS-SEG tasks
        cnn_kwargs['task'] = cnn_run_id.task
    else:
        # Create new
        cnn_kwargs = {
            'model_name': cnn_model_name,
            'labels': [], # headless
            'imagenet': cnn_imagenet,
            'freeze': cnn_freeze,
            'task': 'cls',
        }
        cnn = create_cnn_rg(**cnn_kwargs).to(device)

    # Create decoder
    decoder_kwargs = {
        'decoder_name': decoder_name,
        'vocab': vocab,
        'embedding_size': embedding_size,
        'embedding_kwargs': embedding_kwargs,
        'hidden_size': hidden_size,
        'features_size': cnn.features_size,
        'teacher_forcing': teacher_forcing,
        'dropout_recursive': dropout_recursive,
        'dropout_out': dropout_out,
        'double_bias': att_double_bias,
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
        'weight_decay': weight_decay,
        'custom_lr': custom_lr,
    }
    optimizer = create_optimizer(model, **opt_kwargs)

    # Create lr_scheduler
    if lr_sch_metric:
        LOGGER.info('Using ReduceLROnPlateau metric=%s, %s', lr_sch_metric, lr_sch_kwargs)
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_sch_kwargs)
    else:
        LOGGER.warning('Not using a LR scheduler')
        lr_scheduler = None

    # Other training params
    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'lr_sch_metric': lr_sch_metric,
        'supervise_attention': supervise_attention,
        'supervise_sentences': supervise_sentences,
        'medical_correctness': medical_correctness,
        'med_kwargs': med_kwargs,
        'att_vs_masks': enable_masks,
        'lambda_word': lambda_word,
        'lambda_stop': lambda_stop,
        'lambda_att': lambda_att,
        'lambda_sent': lambda_sent,
        'organ_by_sentence': organ_by_sentence,
        'checkpoint_metric': checkpoint_metric,
        # 'checkpoint_metric': _CORRECTNESS_TARGET_METRIC if medical_correctness else None,
    }

    # Save metadata
    metadata = {
        'cnn_kwargs': cnn_kwargs,
        'decoder_kwargs': decoder_kwargs,
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs if lr_sch_metric else None,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        # 'vocab': vocab, # save space, is already saved in the dataset_kwargs
        'image_size': image_size,
        'hparams': {
            'pretrained_cnn': cnn_run_id.to_dict() if cnn_run_id else None,
            # 'batch_size': batch_size,
        },
        'other_train_kwargs': other_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_id, dryrun=not save_model)

    # Compiled model
    compiled_model = CompiledModel(run_id, model, optimizer, lr_scheduler, metadata)

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
                check_unclean=check_unclean,
                save_model=save_model,
                hw_options=hw_options,
                **other_train_kwargs,
                )


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    parser.add_argument('-d', '--dataset', type=str, default='iu-x-ray',
                        help='Batch size', choices=AVAILABLE_REPORT_DATASETS)
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from a previous run')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('-exp', '--experiment', type=str, default='',
                        help='Custom experiment name')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Set a seed (initial run only)')
    parser.add_argument('--print-metrics', type=str, nargs='*', default=None,
                        help='Additional metrics to print to stdout')
    parser.add_argument('--check-unclean', action='store_true',
                        help='If present, check for unclean reports in the outputs')
    parser.add_argument('--dont-save', action='store_true',
                        help='If present, do not save model checkpoints to disk')
    parser.add_argument('--lambda-word', type=float, default=1,
                        help='Lambda for word loss')
    parser.add_argument('--lambda-stop', type=float, default=1,
                        help='Lambda for stop loss')
    parser.add_argument('--lambda-att', type=float, default=1,
                        help='Lambda for att loss')
    parser.add_argument('--lambda-sent', type=float, default=1,
                        help='Lambda for sent loss')
    parser.add_argument('--skip-organ-by-sentence', action='store_true',
                        help='If present, do not attach organ-by-sentence metrics')
    parser.add_argument('--checkpoint-metric', type=str, default=None,
                        help='If present, save checkpoints with best value')

    decoder_group = parser.add_argument_group('Decoder')
    decoder_group.add_argument('-dec', '--decoder', type=str,
                               choices=AVAILABLE_DECODERS, help='Choose Decoder')
    decoder_group.add_argument('--supervise-att', action='store_true',
                               help='If present, supervise the attention')
    decoder_group.add_argument('--supervise-sent', action='store_true',
                               help='If present, supervise the sentence embeddings')
    decoder_group.add_argument('-emb', '--embedding-size', type=int, default=100,
                               help='Embedding size of the decoder')
    decoder_group.add_argument('-hs', '--hidden-size', type=int, default=100,
                               help='Hidden size of the decoder')
    decoder_group.add_argument('-drop-r', '--dropout-recursive', type=float, default=0,
                               help='Recursive dropout (for LSTM models)')
    decoder_group.add_argument('-drop-o', '--dropout-out', type=float, default=0,
                               help='Out dropout')
    decoder_group.add_argument('-notf', '--no-teacher-forcing', action='store_true',
                               help='If present, does not use teacher forcing')
    decoder_group.add_argument('--att-double-bias', action='store_true',
                               help='Use double bias in the attention layer')

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
    data_group.add_argument('--vocab-greater', type=int, default=None,
                            help='Only keep tokens with more than k appearances')


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

    emb_group = parser.add_argument_group('Word embedding')
    emb_group.add_argument('--emb-pretrained', type=str, default=None,
                          choices=AVAILABLE_PRETRAINED_EMBEDDINGS,
                          help='Choose pretrained embedding')
    emb_group.add_argument('--emb-freeze', action='store_true',
                          help='Freeze the pretrained embedding')
    emb_group.add_argument('--emb-scaled', action='store_true',
                          help='embedding param: scale_grad_by_freq')
    emb_group.add_argument('--emb-bn', action='store_true',
                          help='Use a batch-normalization after the word-embedding')

    lr_group = parsers.add_args_lr_sch(parser, lr=0.0001, metric=None)
    lr_group.add_argument('--custom-lr-word-embedding', type=float, default=None,
                          help='Custom LR for the word_embedding params')
    lr_group.add_argument('--custom-lr-attention', type=float, default=None,
                          help='Custom LR for the attention params')

    parsers.add_args_early_stopping(parser, metric=_CORRECTNESS_TARGET_METRIC)
    parsers.add_args_tb(parser, tb_hist_filter='decoder', tb_hist_freq=10)
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
        (args.med_after is not None and args.es_patience <= args.med_after) and \
        'chex' in args.es_metric:
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


    # Build word-embedding kwargs
    args.embedding_kwargs = {
        'pretrained': args.emb_pretrained,
        'freeze': args.emb_freeze,
        'scale_grad_by_freq': args.emb_scaled,
        'batch_normalization': args.emb_bn,
    }

    # Build custom lr kwargs
    args.custom_lr = {}
    min_lr = []
    if args.custom_lr_word_embedding is not None:
        args.custom_lr['word_embeddings'] = args.custom_lr_word_embedding
        min_lr.append(args.custom_lr_word_embedding)

    if args.custom_lr_attention is not None:
        args.custom_lr['attention'] = args.custom_lr_attention
        min_lr.append(args.custom_lr_attention)

    if len(args.custom_lr) > 0:
        # Do not reduce the customly set LR
        min_lr.append(args.lr_sch_kwargs['min_lr'])
        args.lr_sch_kwargs.update({
            'min_lr': min_lr,
        })
    else:
        args.custom_lr = None


    return args


if __name__ == '__main__':
    config_logging()

    ARGS = parse_args()

    if ARGS.num_threads > 0:
        torch.set_num_threads(ARGS.num_threads)

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    HW_OPTIONS = {
        'device': str(DEVICE),
        'visible': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
        'multiple': ARGS.multiple_gpu,
        'num_threads': ARGS.num_threads,
    }

    if ARGS.resume:
        resume_training(RunId(ARGS.resume, not ARGS.no_debug, 'rg'),
                        n_epochs=ARGS.epochs,
                        max_samples=ARGS.max_samples,
                        tb_kwargs=ARGS.tb_kwargs,
                        multiple_gpu=ARGS.multiple_gpu,
                        device=DEVICE,
                        print_metrics=ARGS.print_metrics,
                        med_kwargs=ARGS.med_kwargs,
                        early_stopping=ARGS.early_stopping,
                        check_unclean=ARGS.check_unclean,
                        hw_options=HW_OPTIONS,
                        )
    else:
        train_from_scratch(get_timestamp(),
                           dataset_name=ARGS.dataset,
                           decoder_name=ARGS.decoder,
                           supervise_attention=ARGS.supervise_att,
                           supervise_sentences=ARGS.supervise_sent,
                           dropout_recursive=ARGS.dropout_recursive,
                           dropout_out=ARGS.dropout_out,
                           att_double_bias=ARGS.att_double_bias,
                           batch_size=ARGS.batch_size,
                           sort_samples=not ARGS.no_sort,
                           shuffle=ARGS.shuffle,
                           frontal_only=ARGS.frontal_only,
                           norm_by_sample=ARGS.norm_by_sample,
                           vocab_greater=ARGS.vocab_greater,
                           teacher_forcing=not ARGS.no_teacher_forcing,
                           embedding_size=ARGS.embedding_size,
                           embedding_kwargs=ARGS.embedding_kwargs,
                           hidden_size=ARGS.hidden_size,
                           lr=ARGS.learning_rate,
                           weight_decay=ARGS.weight_decay,
                           custom_lr=ARGS.custom_lr,
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
                           lambda_word=ARGS.lambda_word,
                           lambda_stop=ARGS.lambda_stop,
                           lambda_att=ARGS.lambda_att,
                           lambda_sent=ARGS.lambda_sent,
                           organ_by_sentence=not ARGS.skip_organ_by_sentence,
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
                           checkpoint_metric=ARGS.checkpoint_metric,
                           save_model=not ARGS.dont_save,
                           check_unclean=ARGS.check_unclean,
                           print_metrics=ARGS.print_metrics,
                           hw_options=HW_OPTIONS,
                           )
