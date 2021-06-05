from ignite.engine import Engine

from medai.datasets import UP_TO_DATE_MASKS_VERSION
from medai.losses import AVAILABLE_LOSSES
from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import attach_metrics_iox
from medai.models import load_pretrained_weights_cnn_
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.models.cls_spatial import create_cls_spatial_model, AVAILABLE_CLS_SPATIAL_MODELS
from medai.training.common import TrainingProcess
from medai.training.detection.cls_spatial import get_step_fn_cls_spatial
from medai.utils import RunId


class TrainingClsSpatial(TrainingProcess):
    LOGGER_NAME = 'medai.cls-spatial.train'
    base_print_metrics = ['cl_loss', 'spatial_loss', 'roc_auc', 'pr_auc', 'ioo']
    task = 'cls-spatial'

    allow_augmenting = True
    allow_sampling = False

    default_dataset = 'cxr14'
    default_image_format = 'RGB'

    key_metric = 'pr_auc'
    default_es_metric = None
    default_lr_metric = None
    checkpoint_metric = 'pr_auc'


    def _add_additional_args(self, parser):
        parser.add_argument('-m', '--model', type=str, default=None,
                            choices=AVAILABLE_CLS_SPATIAL_MODELS,
                            help='Choose model to use')
        parser.add_argument('--cnn-pooling', type=str, default='max',
                            choices=AVAILABLE_POOLING_REDUCTIONS,
                            help='Choose reduction for global-pooling layer')
        parser.add_argument('-noig', '--no-imagenet', action='store_true',
                            help='If present, dont use imagenet pretrained weights')
        parser.add_argument('-drop', '--dropout', type=float, default=0,
                            help='dropout-rate to use (only available for some models)')
        parser.add_argument('-dropf', '--dropout-features', type=float, default=0,
                            help='dropout-rate to use after model features')

        parser.add_argument('--spatial-lambda', type=float, default=1,
                            help='Factor to multiply spatial-loss')
        parser.add_argument('--cl-lambda', type=float, default=1,
                            help='Factor to multiply CL-loss')
        parser.add_argument('--cl-loss-name', type=str, choices=AVAILABLE_LOSSES,
                            default='wbce', help='CL Loss to use')
        parser.add_argument('--spatial-loss-positives', action='store_true',
                            help='Include positive pixels in the spatial loss')

        parser.add_argument('--pretrained', type=str, default=None,
                            help='Run name of a pretrained CNN')
        parser.add_argument('--pretrained-task', type=str, default='cls',
                            choices=('cls', 'cls-seg', 'cls-spatial'),
                            help='Task to choose the CNN from')
        parser.add_argument('--pretrained-spatial', action='store_true',
                            help='Also copy spatial-classifier weights')

    def _build_additional_args(self, parser, args):
        if not args.resume:
            if args.image_format != 'RGB':
                parser.error('Image-format must be RGB')

        if not args.resume and not args.model:
            parser.error('Model is required')

        # Build pretrained args
        if args.pretrained and args.pretrained_spatial:
            if args.pretrained_task != 'cls-spatial':
                parser.error(
                    'To copy pretrained spatial-cls weights, pretrained-task must be cls-spatial')

        if args.pretrained:
            args.pretrained_run_id = RunId(args.pretrained, debug=False, task=args.pretrained_task)
        else:
            args.pretrained_run_id = None

    def _fill_run_name_model(self, run_name):
        run_name += f'_{self.args.model}'

        if self.args.pretrained_run_id:
            run_name += f'_pre{self.args.pretrained_run_id.short_clean_name}'

        if self.args.dropout != 0:
            run_name += f'_drop{self.args.dropout}'
        if self.args.dropout_features:
            run_name += f'_dropf{self.args.dropout_features}'

        if self.args.cnn_pooling != 'max':
            run_name += f'_g{self.args.cnn_pooling}'

        if self.args.no_imagenet:
            run_name += '_noig'

        def _pretty_float(v):
            if int(v) == v:
                return int(v)
            return v

        cl_lambda = self.args.cl_lambda
        spatial_lambda = self.args.spatial_lambda
        if cl_lambda != 1 or spatial_lambda != 1:
            run_name += f'_lmb-{_pretty_float(cl_lambda)}-{_pretty_float(spatial_lambda)}'

        return run_name

    def _fill_dataset_kwargs(self):
        self.dataset_kwargs['masks'] = True
        self.dataset_kwargs['masks_version'] = UP_TO_DATE_MASKS_VERSION
        self.dataset_kwargs['seg_multilabel'] = True
        self.dataset_kwargs['organ_masks_to_diseases'] = True

        if self.args.augment:
            self.dataset_train_kwargs['augment_seg_mask'] = True

    def _fill_run_name_other(self, run_name):
        if self.args.cl_loss_name != 'wbce':
            run_name += f'_cl-{self.args.cl_loss_name}'

        if self.args.spatial_loss_positives:
            run_name += '_spat-pos'

        return run_name

    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        labels = self.train_dataloader.dataset.labels

        self.model_kwargs = {
            'model_name': self.args.model,
            'cl_labels': labels,
            'gpool': self.args.cnn_pooling,
            'dropout': self.args.dropout,
            'dropout_features': self.args.dropout_features,
            'imagenet': not self.args.no_imagenet,
        }
        self.model = create_cls_spatial_model(**self.model_kwargs).to(self.device)

    def _load_state_dict_pretrained_model(self):
        if self.args.pretrained_run_id is None:
            return

        load_pretrained_weights_cnn_(
            self.model, self.args.pretrained_run_id,
            spatial_weights=self.args.pretrained_spatial,
        )

    def _fill_hparams(self):
        if self.args.pretrained_run_id:
            self.metadata['hparams'].update({
                'pretrained': self.args.pretrained_run_id.to_dict(),
                'pretrained-spatial': self.args.pretrained_spatial,
            })

    def _fill_other_train_kwargs(self):
        d = dict()
        d['cl_lambda'] = self.args.cl_lambda
        d['spatial_lambda'] = self.args.spatial_lambda
        d['cl_loss_name'] = self.args.cl_loss_name
        d['out_of_target_only'] = not self.args.spatial_loss_positives

        info_str = ' '.join(
            f"{k}={v}"
            for k, v in d.items()
        )
        self.logger.info('Using params: %s', info_str)

        self.other_train_kwargs.update(d)

    # pylint: disable=arguments-differ
    def _create_engine(self, training,
                       cl_lambda=1,
                       spatial_lambda=1,
                       out_of_target_only=None,
                       cl_loss_name='bce',
                       **unused_kwargs,
                       ):
        cl_labels = self.train_dataloader.dataset.labels
        cl_multilabel = self.train_dataloader.dataset.multilabel
        seg_multilabel = self.train_dataloader.dataset.seg_multilabel
        assert cl_multilabel, 'Dataset is not cl_multilabel'
        assert seg_multilabel, 'Dataset is not seg_multilabel'

        # Create engine
        engine = Engine(get_step_fn_cls_spatial(
            self.model,
            optimizer=self.optimizer if training else None,
            training=training,
            cl_loss_name=cl_loss_name,
            cl_lambda=cl_lambda,
            spatial_lambda=spatial_lambda,
            out_of_target_only=out_of_target_only,
            device=self.device,
        ))
        attach_losses(engine, ['cl_loss', 'spatial_loss'], device=self.device)
        attach_metrics_classification(
            engine,
            cl_labels,
            multilabel=cl_multilabel,
            device=self.device,
            extra_bce=cl_loss_name != 'bce',
        )

        attach_metrics_iox(
            engine,
            cl_labels,
            multilabel=seg_multilabel,
            device=self.device,
            ioo=True,
            cls_thresh=None,
            heat_thresh=None,
            only='T',
        )

        return engine


if __name__ == '__main__':
    training_process = TrainingClsSpatial()

    training_process.run()
