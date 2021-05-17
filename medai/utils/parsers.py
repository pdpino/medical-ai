"""Parser utilities."""
from medai.datasets.tools.augmentation import AVAILABLE_AUG_MODES
from medai.models.classification import AVAILABLE_CLASSIFICATION_MODELS
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.utils import parse_str_or_int
from medai.training.detection.h2bb import AVAILABLE_H2BB_METHODS

# FIXME: this module (medai.utils) imports from medai.models


def add_args_augment(parser):
    aug_group = parser.add_argument_group('Data-augmentation params')
    aug_group.add_argument('--augment', action='store_true',
                           help='If present, augment dataset')
    aug_group.add_argument('--augment-mode', type=str, choices=AVAILABLE_AUG_MODES,
                           default='single', help='Method to augment the images')
    aug_group.add_argument('--augment-label', default=None,
                           help='Augment only samples with a given label present (str/int)')
    aug_group.add_argument('--augment-class', type=int, choices=[0,1], default=None,
                           help='If --augment-label is provided, choose if augmenting \
                              positive (1) or negative (0) samples')
    aug_group.add_argument('--augment-times', type=int, default=1,
                        help='Number of times to randomly augment by each method')

    aug_group.add_argument('--aug-crop', type=float, default=0.8,
                           help='Augment samples by cropping a random fraction')
    aug_group.add_argument('--aug-translate', type=float, default=0.1,
                           help='Augment samples by translating a random fraction')
    aug_group.add_argument('--aug-rotation', type=int, default=30,
                           help='Augment samples by rotating a random amount of degrees')
    aug_group.add_argument('--aug-contrast', type=float, default=0.8,
                           help='Augment samples by changing the contrast randomly')
    aug_group.add_argument('--aug-brightness', type=float, default=0.8,
                           help='Augment samples by changing the brightness randomly')
    aug_group.add_argument('--aug-shear', nargs=2, default=(10, 10),
                           help='Augment samples by applying a shear transformation.')
    aug_group.add_argument('--aug-gaussian', type=float, default=0.1,
                           help='Augment samples by adding N(0,1)*value noise.')


def build_args_augment_(args):
    # Build kwargs
    if args.augment:
        args.augment_kwargs = {
            'crop': args.aug_crop,
            'translate': args.aug_translate,
            'rotation': args.aug_rotation,
            'contrast': args.aug_contrast,
            'brightness': args.aug_brightness,
            'shear': args.aug_shear,
            'noise_gaussian': args.aug_gaussian
        }
    else:
        args.augment_kwargs = {}

    # Allow passing str or int
    if args.augment_label is not None:
        args.augment_label = parse_str_or_int(args.augment_label)


def add_args_tb(parser, tb_hist_filter=None, tb_hist_freq=1):
    tb_group = parser.add_argument_group('Tensorboard params')
    tb_group.add_argument('--tb-no-scalars', action='store_true',
                           help='If present, do not save scalars to TB')
    tb_group.add_argument('--tb-hist', action='store_true',
                           help='If present, save histograms to TB')
    tb_group.add_argument('--tb-hist-freq', type=int, default=tb_hist_freq,
                           help='In add_hist, only write every n epochs')
    tb_group.add_argument('--tb-hist-filter', type=str, default=tb_hist_filter,
                           help='In add_hist, include only layers matching this string')


def build_args_tb_(args):
    args.tb_kwargs = {
        'scalars': not args.tb_no_scalars,
        'histogram': args.tb_hist,
        'histogram_filter': args.tb_hist_filter,
        'histogram_freq': args.tb_hist_freq,
    }


def add_args_hw(parser, num_workers=4):
    hw_group = parser.add_argument_group('Hardware params')
    hw_group.add_argument('--multiple-gpu', action='store_true',
                          help='Use multiple gpus')
    hw_group.add_argument('--cpu', action='store_true',
                          help='Use CPU only')
    hw_group.add_argument('--num-workers', type=int, default=num_workers,
                          help='Number of workers for dataloader')
    hw_group.add_argument('--num-threads', type=int, default=1,
                          help='Number of threads for pytorch')


def add_args_lr_sch(parser, lr=0.0001, patience=5, factor=0.5, cooldown=0, metric='loss'):
    lr_group = parser.add_argument_group('LR scheduler params')
    lr_group.add_argument('-lr', '--learning-rate', type=float, default=lr,
                          help='Initial learning rate')
    lr_group.add_argument('--lr-metric', type=str, default=metric,
                          help='Select the metric to regulate the LR')
    lr_group.add_argument('--lr-patience', type=int, default=patience,
                          help='Patience value for LR-scheduler')
    lr_group.add_argument('--lr-factor', type=float, default=factor,
                          help='Factor to multiply the LR on each update')
    lr_group.add_argument('--lr-cooldown', type=int, default=cooldown,
                          help='Cooldown for lr-scheduler')
    lr_group.add_argument('--lr-min', type=float, default=0,
                          help='Min LR to reach if using scheduler')
    lr_group.add_argument('-wd', '--weight-decay', type=float, default=0,
                          help='Weight decay passed to the optimizer')

    return lr_group

def build_args_lr_sch_(args):
    args.lr_sch_kwargs = {
        'mode': 'min' if args.lr_metric == 'loss' else 'max',
        'threshold_mode': 'abs',
        'factor': args.lr_factor,
        'patience': args.lr_patience,
        'cooldown': args.lr_cooldown,
        'verbose': True,
        'min_lr': args.lr_min,
    }


def add_args_early_stopping(parser, metric='loss'):
    es_group = parser.add_argument_group('Early stopping params')
    es_group.add_argument('-noes', '--no-early-stopping', action='store_true',
                          help='If present, dont early stop the training')
    es_group.add_argument('--es-patience', type=int, default=10,
                          help='Patience value for early-stopping')
    es_group.add_argument('--es-metric', type=str, default=metric,
                          help='Metric to monitor for early-stopping')
    es_group.add_argument('--es-min-delta', type=float, default=0.001,
                          help='Min delta to use for early-stopping')


def build_args_early_stopping_(args):
    args.early_stopping = not args.no_early_stopping
    args.early_stopping_kwargs = {
        'patience': args.es_patience,
        'metric': args.es_metric,
        'min_delta': args.es_min_delta,
    }


def add_args_free_values(parser):
    parser.add_argument('--skip-free', action='store_true',
                        help='If present, do not run in free mode')
    parser.add_argument('--skip-notfree', action='store_true',
                        help='If present, do not run in not-free mode')


def build_args_free_values_(args, parser):
    use_free = not args.skip_free
    use_notfree = not args.skip_notfree

    if use_free and use_notfree:
        args.free_values = [False, True]
    elif use_free:
        args.free_values = [True]
    elif use_notfree:
        args.free_values = [False]
    else:
        parser.error('Cannot skip both free and not free')


def add_args_med(parser):
    med_group = parser.add_argument_group('Medical correctness metrics')
    med_group.add_argument('--no-med', action='store_true',
                           help='If present, do not use medical-correctness metrics')
    med_group.add_argument('--med-after', type=int, default=None,
                           help='Start using med-metrics after N epochs (in training)')
    med_group.add_argument('--med-steps', type=int, default=None,
                           help='Run med-metrics every N epochs (in training)')
    med_group.add_argument('--med-val-after', type=int, default=None,
                           help='Start using med-metrics after N epochs (in validation)')
    med_group.add_argument('--med-val-steps', type=int, default=None,
                           help='Run med-metrics every N epochs (in validation)')
    med_group.add_argument('--med-dummy', action='store_true',
                           help='Use a dummy labeler (debug faster!)')



def build_args_med_(args):
    args.med_kwargs = {
        'after': args.med_after,
        'steps': args.med_steps,
        'val_after': args.med_val_after,
        'val_steps': args.med_val_steps,
        'dummy': args.med_dummy,
    }


def add_args_cnn(parser):
    cnn_group = parser.add_argument_group('CNN params')
    cnn_group.add_argument('-m', '--model', type=str, default=None,
                        choices=AVAILABLE_CLASSIFICATION_MODELS,
                        help='Choose base CNN to use')
    cnn_group.add_argument('-drop', '--dropout', type=float, default=0,
                        help='dropout-rate to use (only available for some models)')
    cnn_group.add_argument('-noig', '--no-imagenet', action='store_true',
                        help='If present, dont use imagenet pretrained weights')
    cnn_group.add_argument('-frz', '--freeze', action='store_true',
                        help='If present, freeze base cnn parameters (only train FC layers)')
    cnn_group.add_argument('--cnn-pooling', type=str, default='avg',
                        choices=AVAILABLE_POOLING_REDUCTIONS,
                        help='Choose reduction for global-pooling layer')
    cnn_group.add_argument('--fc-layers', nargs='+', type=int, default=(),
                        help='Choose sizes for FC layers at the end')


def add_args_images(parser, image_format='RGB'):
    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')
    images_group.add_argument('--frontal-only', action='store_true',
                              help='Use only frontal images')
    images_group.add_argument('--norm-by-sample', action='store_true',
                              help='Normalize each sample (instead of by dataset stats)')
    images_group.add_argument('--image-format', type=str, default=image_format,
                              choices=['RGB', 'L'], help='Image format to use')

    return images_group


def add_args_sampling(parser):
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
                             help='Undersample from the majority class \
                                   with a given label (str/int)')

    sampl_group.add_argument('--balanced-sampler', action='store_true',
                             help='Use a multilabel balanced sampler')


def build_args_sampling_(args):
    # Enable passing str or int for oversample labels
    if args.oversample is not None:
        args.oversample = parse_str_or_int(args.oversample)
    if args.undersample is not None:
        args.undersample = parse_str_or_int(args.undersample)


def add_args_h2bb(parser):
    h2bb_group = parser.add_argument_group('H2BB params')
    h2bb_group.add_argument('--h2bb-method', type=str, default='thr-largarea',
                            choices=AVAILABLE_H2BB_METHODS, help='Heatmap to BB method')
    h2bb_group.add_argument('--cls-thresh', type=float, default=0.3,
                            help='CLS threshold')
    h2bb_group.add_argument('--heat-thresh', type=float, default=0.8,
                            help='Heatmap threshold')

def build_args_h2bb_(args):
    args.h2bb_method_name = args.h2bb_method
    args.h2bb_method_kwargs = {
        'cls_thresh': args.cls_thresh,
        'heat_thresh': args.heat_thresh,
    }
    args.h2bb_suffix = f'{args.h2bb_method_name}-{args.cls_thresh}-{args.heat_thresh}'
