"""TODO: merge this script with train_report_generation??.

Script to train coatt model (not h-coatt model!)
Most up-to-date is "h-coatt" version.
"""
import argparse
import logging
import os

import torch
from torch import nn

from medai.datasets import prepare_data_report_generation, AVAILABLE_REPORT_DATASETS
from medai.datasets.common import LATEST_REPORTS_VERSION
from medai.models.report_generation.coatt import CoAttModel
from medai.models.checkpoint import (
    CompiledModel,
    save_metadata,
    load_compiled_model,
)
from medai.models import freeze_cnn as freeze_cnn_fn, load_pretrained_weights_cnn_
from medai.losses.schedulers import create_lr_sch_handler
from medai.losses.optimizers import create_optimizer
from medai.utils import (
    get_timestamp,
    print_hw_options,
    parsers,
    config_logging,
    set_seed,
    timeit_main,
    RunId,
)
from medai.train_report_generation import train_model


LOGGER = logging.getLogger('medai.rg.train.coatt')


@timeit_main(LOGGER)
def train_from_scratch(run_name,
                       dataset_name='iu-x-ray',
                       reports_version=LATEST_REPORTS_VERSION,
                       batch_size=15,
                       sort_samples=True,
                       shuffle=False,
                       frontal_only=False,
                       norm_by_sample=False,
                       vocab_greater=None,
                       embedding_size=100,
                       hidden_size=100,
                       lr=0.0001,
                       weight_decay=0,
                       n_epochs=10,
                       medical_correctness=True,
                       med_kwargs={},
                       cnn_model_name='resnet-50',
                       cnn_imagenet=True,
                       # cnn_freeze=False,
                       max_samples=None,
                       image_size=512,
                       early_stopping=True,
                       early_stopping_kwargs={},
                       lr_sch_kwargs={},
                       lambda_word=1,
                       lambda_stop=1,
                       lambda_tag=1,
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
                       cnn_run_id=None,
                       pretrained_cls=False,
                       freeze_cnn=False,
                       freeze_mlc=False,
                       ):
    """Train a model from scratch."""
    # Create run name
    run_name = f'{run_name}_{dataset_name}_coatt'
    if lambda_word != 1:
        run_name += f'_word-{lambda_word}'
    if lambda_stop != 1:
        run_name += f'_stop-{lambda_stop}'
    if lambda_tag != 1:
        run_name += f'-{lambda_tag}'
    if embedding_size != 512:
        run_name += f'_embsize-{embedding_size}'
    if hidden_size != 512:
        run_name += f'_hs-{hidden_size}'
    if cnn_run_id:
        run_name += f'_precnn-{cnn_run_id.short_clean_name}'
    if freeze_mlc:
        run_name += '_freeze-mlc'
    elif freeze_cnn:
        run_name += '_freeze-cnn'
    if reports_version != LATEST_REPORTS_VERSION:
        run_name += f'_reports-{reports_version}'
    if not norm_by_sample:
        run_name += '_normD'
    if image_size != 224:
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
    # if custom_lr is not None:
    #     lr_emb = custom_lr.get('word_embeddings')
    #     if lr_emb is not None:
    #         run_name += f'_lr-emb{lr_emb}'
    #     lr_att = custom_lr.get('attention')
    #     if lr_att is not None:
    #         run_name += f'_lr-att{lr_att}'
    if weight_decay != 0:
        run_name += f'_wd{weight_decay}'

    # lr_sch_name = lr_sch_kwargs['name']
    # if lr_sch_name == 'plateau':
    #     factor = lr_sch_kwargs['factor']
    #     patience = lr_sch_kwargs['patience']
    #     metric = lr_sch_kwargs['metric'].replace('_', '-')
    #     run_name += f'_sch-{metric}-p{patience}-f{factor}'

    #     cooldown = lr_sch_kwargs.get('cooldown', 0)
    #     if cooldown != 0:
    #         run_name += f'-c{cooldown}'
    # elif lr_sch_name == 'step':
    #     step = lr_sch_kwargs['step_size']
    #     factor = lr_sch_kwargs['gamma']
    #     run_name += f'_sch-step{step}-f{factor}'

    if frontal_only:
        run_name += '_front'

    run_id = RunId(run_name, debug, 'rg', experiment)

    set_seed(seed)

    # Load data
    image_size = (image_size, image_size)
    enable_masks = False
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'hierarchical': True,
        'max_samples': max_samples,
        'norm_by_sample': norm_by_sample,
        'image_size': image_size,
        'batch_size': batch_size,
        'num_workers': num_workers,
        'masks': enable_masks,
        'frontal_only': frontal_only,
        'vocab_greater': vocab_greater,
        'reports_version': reports_version,
        'sentence_embeddings': False,
        'labels': 'coatt-labels',
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

    # Create model
    model_kwargs = {
        'vocab': vocab,
        'labels': train_dataloader.dataset.labels,
        'hidden_size': hidden_size,
        'embedding_size': embedding_size,
        'bn_momentum': 0.1,
        'sentence_num_layers': 1,
        'word_num_layers': 1,
        'max_words': 100,
        'imagenet': cnn_imagenet,
        'cnn_model_name': cnn_model_name,
    }
    # Full model
    model = CoAttModel(**model_kwargs).to(device)

    if cnn_run_id:
        # Load pretrained
        compiled_cnn = load_compiled_model(
            cnn_run_id, device=device, multiple_gpu=False,
        )
        cnn = compiled_cnn.model

        load_pretrained_weights_cnn_(model.extractor, cnn)
        if pretrained_cls:
            load_pretrained_weights_cnn_(model.mlc, cnn, features=False, cls_weights=True)

    if freeze_mlc:
        freeze_cnn_fn(model.mlc)
        freeze_cnn_fn(model.extractor)
    elif freeze_cnn:
        freeze_cnn_fn(model.extractor)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    # Optimizer
    opt_kwargs = {
        'lr': lr,
        'weight_decay': weight_decay,
        # 'custom_lr': custom_lr,
    }
    optimizer = create_optimizer(model, **opt_kwargs)

    # Create lr_scheduler
    lr_sch_handler = create_lr_sch_handler(optimizer, **lr_sch_kwargs)

    # Other training params
    other_train_kwargs = {
        'early_stopping': early_stopping,
        'early_stopping_kwargs': early_stopping_kwargs,
        'supervise_attention': False,
        'supervise_sentences': False,
        'medical_correctness': medical_correctness,
        'med_kwargs': med_kwargs,
        'att_vs_masks': enable_masks,
        'lambda_word': lambda_word,
        'lambda_stop': lambda_stop,
        'lambda_tag': lambda_tag,
        'organ_by_sentence': organ_by_sentence,
        'checkpoint_metric': checkpoint_metric,
    }

    # Save metadata
    metadata = {
        'model_kwargs': {
            'name': 'coatt',
            **model_kwargs,
        },
        'opt_kwargs': opt_kwargs,
        'lr_sch_kwargs': lr_sch_kwargs,
        'dataset_kwargs': dataset_kwargs,
        'dataset_train_kwargs': dataset_train_kwargs,
        # 'vocab': vocab, # save space, is already saved in the dataset_kwargs
        'image_size': image_size,
        # 'hparams': {
        #     # 'pretrained_cnn': cnn_run_id.to_dict() if cnn_run_id else None,
        #     # 'batch_size': batch_size,
        # },
        'other_train_kwargs': other_train_kwargs,
        'seed': seed,
    }
    save_metadata(metadata, run_id, dryrun=not save_model)

    # Compiled model
    compiled_model = CompiledModel(run_id, model, optimizer, lr_sch_handler, metadata)

    # Train
    train_model(
        run_id,
        compiled_model,
        train_dataloader,
        val_dataloader,
        hierarchical=True,
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
    parser.add_argument('--lambda-tag', type=float, default=1,
                        help='Lambda for tag loss')
    parser.add_argument('--skip-organ-by-sentence', action='store_true',
                        help='If present, do not attach organ-by-sentence metrics')

    model_group = parser.add_argument_group('Model')
    model_group.add_argument('-cnn', '--cnn-model-name', type=str, default='resnet152',
                             help='Choose CNN feature extractor')
    model_group.add_argument('-noig', '--no-imagenet', action='store_true',
                             help='If present, dont use imagenet pretrained weights')
    model_group.add_argument('-emb', '--embedding-size', type=int, default=512,
                             help='Embedding size of the decoder')
    model_group.add_argument('-hs', '--hidden-size', type=int, default=512,
                             help='Hidden size of the decoder')
    model_group.add_argument('-cp', '--cnn-pretrained', type=str, default=None,
                             help='Run name of a pretrained CNN')
    model_group.add_argument('--pretrained-cls', action='store_true',
                             help='Also copy MLC weights')
    model_group.add_argument('--freeze-cnn', action='store_true',
                             help='If present, freeze extractor parameters')
    model_group.add_argument('--freeze-mlc', action='store_true',
                             help='If present, freeze MLC parameters')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--image-size', type=int, default=224,
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
    data_group.add_argument('--reports-version', type=str, default=LATEST_REPORTS_VERSION,
                            help='Specify an reports-version')


    # cnn_group = parser.add_argument_group('CNN')
    # cnn_group.add_argument('-c', '--cnn', type=str, default=None,
    #                       choices=AVAILABLE_CLASSIFICATION_MODELS,
    #                       help='Choose base CNN class (create new)')
    # cnn_group.add_argument('-cp', '--cnn-pretrained', type=str, default=None,
    #                       help='Run name of a pretrained CNN')
    # cnn_group.add_argument('-cp-task', '--cnn-pretrained-task', type=str, default='cls',
    #                       choices=('cls', 'cls-seg'), help='Task to choose the CNN from')

    parsers.add_args_lr(parser, lr=0.0001)
    # lr_group.add_argument('--custom-lr-word-embedding', type=float, default=None,
    #                       help='Custom LR for the word_embedding params')
    # lr_group.add_argument('--custom-lr-attention', type=float, default=None,
    #                       help='Custom LR for the attention params')
    parsers.add_args_lr_sch(parser, scheduler=None, metric=None)

    parsers.add_args_early_stopping(parser, metric=None)
    parsers.add_args_tb(parser)
    parsers.add_args_augment(parser)

    parsers.add_args_hw(parser, num_workers=4)
    parsers.add_args_med(parser)
    parsers.add_args_checkpoint_metric(parser)

    args = parser.parse_args()

    # Build params
    parsers.build_args_early_stopping_(args)
    parsers.build_args_lr_sch_(args, parser)
    parsers.build_args_augment_(args)
    parsers.build_args_tb_(args)
    parsers.build_args_med_(args)
    parsers.build_args_checkpoint_metric_(args)

    def _assert_med_metric_is_present(metric, argname):
        if metric is None or 'chex' not in metric:
            # Is not a medical-correctness metric
            return
        if not args.medical_correctness:
            parser.error(f'Cannot use {argname} {metric} and --no-med')
        med_metric = args.med_kwargs['metric']
        if metric.startswith('chex') and med_metric != 'light-chexpert':
            parser.error(f'Cannot use {argname} "chex-" without --med-metric light-chexpert')
        elif metric.startswith('lighter') and med_metric != 'lighter-chexpert':
            parser.error(f'Cannot use {argname} "lighter-" without --med-metric lighter-chexpert')

    if args.lr_metric is not None:
        _assert_med_metric_is_present(args.lr_metric, '--lr-metric')

    if args.early_stopping is None:
        _assert_med_metric_is_present(args.es_metric, '--es-metric')

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
            task='cls',
        )
    else:
        args.precnn_run_id = None

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
    train_from_scratch(
        get_timestamp(),
        dataset_name=ARGS.dataset,
        reports_version=ARGS.reports_version,
        batch_size=ARGS.batch_size,
        sort_samples=not ARGS.no_sort,
        shuffle=ARGS.shuffle,
        frontal_only=ARGS.frontal_only,
        norm_by_sample=ARGS.norm_by_sample,
        vocab_greater=ARGS.vocab_greater,
        embedding_size=ARGS.embedding_size,
        hidden_size=ARGS.hidden_size,
        lr=ARGS.learning_rate,
        weight_decay=ARGS.weight_decay,
        n_epochs=ARGS.epochs,
        medical_correctness=ARGS.medical_correctness,
        med_kwargs=ARGS.med_kwargs,
        image_size=ARGS.image_size,
        cnn_model_name=ARGS.cnn_model_name,
        cnn_imagenet=not ARGS.no_imagenet,
        max_samples=ARGS.max_samples,
        early_stopping=ARGS.early_stopping,
        early_stopping_kwargs=ARGS.early_stopping_kwargs,
        lr_sch_kwargs=ARGS.lr_sch_kwargs,
        lambda_word=ARGS.lambda_word,
        lambda_stop=ARGS.lambda_stop,
        lambda_tag=ARGS.lambda_tag,
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
        cnn_run_id=ARGS.precnn_run_id,
        pretrained_cls=ARGS.pretrained_cls,
        freeze_cnn=ARGS.freeze_cnn,
        freeze_mlc=ARGS.freeze_mlc,
    )
