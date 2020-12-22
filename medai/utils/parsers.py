"""Parser utilities."""

def add_args_augment(parser):
    aug_group = parser.add_argument_group('Data-augmentation params')
    aug_group.add_argument('--augment', action='store_true',
                           help='If present, augment dataset')
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
    aug_group.add_argument('--aug-rotation', type=int, default=15,
                           help='Augment samples by rotating a random amount of degrees')
    aug_group.add_argument('--aug-contrast', type=float, default=0.8,
                           help='Augment samples by changing the contrast randomly')
    aug_group.add_argument('--aug-brightness', type=float, default=0.8,
                           help='Augment samples by changing the brightness randomly')
    aug_group.add_argument('--aug-shear', nargs=2, default=(10, 10),
                           help='Augment samples by applying a shear transformation.')


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
        }
    else:
        args.augment_kwargs = {}

    # Allow passing str or int
    if args.augment_label is not None:
        args.augment_label = parse_str_or_int(args.augment_label)


def add_args_tb(parser):
    tb_group = parser.add_argument_group('Tensorboard params')
    tb_group.add_argument('--tb-hist', action='store_true',
                           help='If present, save histograms to TB')

def build_args_tb_(args):
    args.tb_kwargs = {
        'histogram': args.tb_hist,
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


def add_args_lr_sch(parser, lr=0.0001, metric='loss'):
    lr_group = parser.add_argument_group('LR scheduler params')
    lr_group.add_argument('-lr', '--learning-rate', type=float, default=lr,
                          help='Initial learning rate')
    lr_group.add_argument('--lr-metric', type=str, default=metric,
                          help='Select the metric to regulate the LR')
    lr_group.add_argument('--lr-patience', type=int, default=5,
                          help='Patience value for LR-scheduler')
    lr_group.add_argument('--lr-factor', type=float, default=0.1,
                          help='Factor to multiply the LR on each update')

def build_args_lr_sch_(args):
    args.lr_sch_kwargs = {
        'mode': 'min' if args.lr_metric == 'loss' else 'max',
        'threshold_mode': 'abs',
        'factor': args.lr_factor,
        'patience': args.lr_patience,
        'verbose': True,
    }

def add_args_early_stopping(parser, metric='loss'):
    es_group = parser.add_argument_group('Early stopping params')
    es_group.add_argument('-noes', '--no-early-stopping', action='store_true',
                          help='If present, dont early stop the training')
    es_group.add_argument('--es-patience', type=int, default=10,
                          help='Patience value for early-stopping')
    es_group.add_argument('--es-metric', type=str, default=metric,
                          help='Metric to monitor for early-stopping')
    es_group.add_argument('--es-min-delta', type=float, default=0,
                          help='Min delta to use for early-stopping')


def build_args_early_stopping_(args):
    args.early_stopping = not args.no_early_stopping
    args.early_stopping_kwargs = {
        'patience': args.es_patience,
        'metric': args.es_metric,
        'min_delta': args.es_min_delta,
    }