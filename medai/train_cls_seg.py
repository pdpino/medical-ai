from ignite.engine import Engine

from medai.losses import AVAILABLE_LOSSES
from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import attach_metrics_iox
from medai.models.checkpoint import load_compiled_model_cls_seg
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.models.cls_seg import create_cls_seg_model, AVAILABLE_CLS_SEG_MODELS
from medai.training.common import TrainingProcess
from medai.training.detection.cls_seg import get_step_fn_cls_seg
from medai.utils import parsers

class TrainingClsSeg(TrainingProcess):
    LOGGER_NAME = 'medai.cls-seg.train'
    base_print_metrics = ['cl_loss', 'seg_loss', 'roc_auc', 'iou', 'iobb']
    task = 'cls-seg'

    load_compiled_model_fn = staticmethod(load_compiled_model_cls_seg)

    allow_augmenting = True
    allow_sampling = False

    default_dataset = 'cxr14'
    default_image_format = 'L'

    # TODO: use better metric? that captures both cls and seg
    key_metric = 'roc_auc'
    default_es_metric = None
    default_lr_metric = None
    checkpoint_metric = 'roc_auc'


    def _add_additional_args(self, parser):
        parser.add_argument('-m', '--model', type=str, default=None,
                            choices=AVAILABLE_CLS_SEG_MODELS,
                            help='Choose model to use')
        parser.add_argument('--cnn-pooling', type=str, default='avg',
                            choices=AVAILABLE_POOLING_REDUCTIONS,
                            help='Choose reduction for global-pooling layer')
        parser.add_argument('-noig', '--no-imagenet', action='store_true',
                            help='If present, dont use imagenet pretrained weights')
        parser.add_argument('-drop', '--dropout', type=float, default=0,
                            help='dropout-rate to use (only available for some models)')

        parser.add_argument('--seg-lambda', type=float, default=1,
                            help='Factor to multiply seg-loss')
        parser.add_argument('--cl-lambda', type=float, default=1,
                            help='Factor to multiply CL-loss')
        parser.add_argument('--cl-loss-name', type=str, choices=AVAILABLE_LOSSES,
                            default='bce', help='CL Loss to use')
        parser.add_argument('--weight-organs', action='store_true',
                            help='Add weight to organs')

        parsers.add_args_h2bb(parser)

    def _build_additional_args(self, parser, args):
        if not args.resume:
            if 'scan' in args.model and args.image_format != 'L':
                parser.error('For scan model, image-format must be L')
            elif args.image_format != 'RGB':
                parser.error('Image-format must be RGB')

        if not args.resume and not args.model:
            parser.error('Model is required')

        if args.weight_organs:
            args.weight_organs = [0.1, 0.6, 0.3, 0.3]
        else:
            args.weight_organs = None

        parsers.build_args_h2bb_(args)

    def _fill_run_name_model(self):
        self.run_name += f'_{self.args.model}'

        if self.args.dropout != 0:
            self.run_name += f'_drop{self.args.dropout}'

        if self.args.cnn_pooling not in ('avg', 'mean'):
            self.run_name += f'_g{self.args.cnn_pooling}'

        if self.args.no_imagenet:
            self.run_name += '_noig'

        cl_lambda = self.args.cl_lambda
        seg_lambda = self.args.seg_lambda
        if cl_lambda != 1 or seg_lambda != 1:
            self.run_name += f'_lmb-{cl_lambda}-{seg_lambda}'

    def _fill_dataset_kwargs(self):
        # self.dataset_kwargs['fallback_organs'] = False
        self.dataset_kwargs['masks'] = True
        self.dataset_kwargs['masks_version'] = 'v1'
        self.dataset_kwargs['seg_multilabel'] = False

        if self.args.augment:
            self.dataset_train_kwargs['augment_kwargs'].update({
                'seg_mask': True,
            })

    def _fill_run_name_other(self):
        if self.args.cl_loss_name != 'bce':
            self.run_name += f'_cl-{self.args.cl_loss_name}'

        if self.args.weight_organs:
            self.run_name += '_seg-w'

    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        labels = self.train_dataloader.dataset.labels
        organs = self.train_dataloader.dataset.organs

        self.model_kwargs = {
            'model_name': self.args.model,
            'cl_labels': labels,
            'seg_labels': organs,
            'gpool': self.args.cnn_pooling,
            'dropout': self.args.dropout,
            'imagenet': not self.args.no_imagenet,
        }
        self.model = create_cls_seg_model(**self.model_kwargs).to(self.device)

    def _fill_other_train_kwargs(self):
        d = dict()
        d['cl_lambda'] = self.args.cl_lambda
        d['seg_lambda'] = self.args.seg_lambda
        d['cl_loss_name'] = self.args.cl_loss_name
        if self.args.weight_organs:
            d['weight_organs'] = self.args.weight_organs

        info_str = ' '.join(
            f"{k}={v}"
            for k, v in d.items()
        )
        self.logger.info('Using params: %s', info_str)

        self.other_train_kwargs.update(d)

    # pylint: disable=arguments-differ
    def _create_engine(self, training,
                       cl_lambda=1,
                       seg_lambda=1,
                       weight_organs=None,
                       cl_loss_name='bce',
                       **unused_kwargs,
                       ):
        cl_labels = self.train_dataloader.dataset.labels
        seg_labels = self.train_dataloader.dataset.organs
        cl_multilabel = self.train_dataloader.dataset.multilabel
        seg_multilabel = self.train_dataloader.dataset.seg_multilabel
        assert cl_multilabel, 'Dataset is not cl_multilabel'
        assert not seg_multilabel, 'Dataset is seg_multilabel'

        # Create engine
        engine = Engine(get_step_fn_cls_seg(
            self.model,
            optimizer=self.optimizer if training else None,
            training=training,
            cl_lambda=cl_lambda,
            seg_lambda=seg_lambda,
            seg_weights=weight_organs,
            cl_loss_name=cl_loss_name,
            device=self.device,
        ))
        attach_losses(engine, ['cl_loss', 'seg_loss'], device=self.device)
        attach_metrics_classification(
            engine,
            cl_labels,
            multilabel=cl_multilabel,
            device=self.device,
            extra_bce=cl_loss_name != 'bce',
        )

        attach_metrics_iox(
            engine,
            seg_labels,
            multilabel=seg_multilabel,
            device=self.device,
        )

        return engine


if __name__ == '__main__':
    training_process = TrainingClsSeg()

    training_process.run()
