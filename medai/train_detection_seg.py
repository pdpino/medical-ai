from torch import nn
from ignite.engine import Engine

from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import (
    attach_mAP_coco,
    attach_metrics_iox,
)
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.training.common import TrainingProcess
from medai.training.detection.h2bb import get_h2bb_method
from medai.training.detection.det_seg import get_step_fn_det_seg
from medai.models.detection import create_detection_seg_model, AVAILABLE_DETECTION_SEG_MODELS
from medai.utils import parsers

class TrainingDetectionSeg(TrainingProcess):
    LOGGER_NAME = 'medai.det-seg.train'
    base_print_metrics = ['cl_loss', 'seg_loss', 'roc_auc', 'iou', 'iobb', 'mAP']
    task = 'det'

    allow_augmenting = False
    allow_sampling = False

    default_dataset = 'vinbig'

    # TODO: pick better metrics here?
    key_metric = 'roc_auc'
    default_es_metric = None
    default_lr_metric = None
    checkpoint_metric = 'mAP'


    def _add_additional_args(self, parser):
        parser.add_argument('-m', '--model', type=str, default=None, required=True,
                            choices=AVAILABLE_DETECTION_SEG_MODELS,
                            help='Choose model to use')
        parser.add_argument('--cnn-pooling', type=str, default='max',
                            choices=AVAILABLE_POOLING_REDUCTIONS,
                            help='Choose reduction for global-pooling layer')

        parser.add_argument('--seg-lambda', type=float, default=1,
                            help='Factor to multiply seg-loss')
        parser.add_argument('--cl-lambda', type=float, default=1,
                            help='Factor to multiply CL-loss')

        parsers.add_args_h2bb(parser)

    def _build_additional_args(self, parser, args):
        if args.image_format != 'L':
            parser.error('Image-format must be L')

        if args.dataset != 'vinbig':
            parser.error('Only works with vinbig dataset')


        parsers.build_args_h2bb_(args)

    def _fill_run_name_model(self):
        self.run_name += f'_{self.args.model}'

        if self.args.cnn_pooling not in ('avg', 'mean'):
            self.run_name += f'_g{self.args.cnn_pooling}'

    def _fill_dataset_kwargs(self):
        self.dataset_kwargs['fallback_organs'] = True
        self.dataset_kwargs['masks'] = True
        self.dataset_kwargs['masks_version'] = 'v1'

    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        labels = self.train_dataloader.dataset.labels

        self.model_kwargs = {
            'model_name': self.args.model,
            'labels': labels,
            'gpool': self.args.cnn_pooling,
        }
        self.model = create_detection_seg_model(**self.model_kwargs).to(self.device)

    def _get_additional_other_train_kwargs(self):
        d = dict()
        d['cl_lambda'] = self.args.cl_lambda
        d['seg_lambda'] = self.args.seg_lambda
        # d['seg_loss_name'] = self.args.seg_loss_name
        d['h2bb_method_name'] = self.args.h2bb_method_name
        d['h2bb_method_kwargs'] = self.args.h2bb_method_kwargs

        self.other_train_kwargs.update(d)

    def _build_common_for_engines(self):
        # Prepare loss
        cl_loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        seg_loss_fn = nn.BCEWithLogitsLoss().to(self.device)

        # Choose h2bb method
        h2bb_method = get_h2bb_method(self.args.h2bb_method_name, self.args.h2bb_method_kwargs)

        return cl_loss_fn, seg_loss_fn, h2bb_method

    def _create_engine(self, training, *args):
        labels = self.train_dataloader.dataset.labels
        multilabel = True
        assert self.train_dataloader.dataset.multilabel == multilabel, 'Dataset is not multilabel'

        cl_loss_fn, seg_loss_fn, h2bb_method = args

        # Choose step_fn
        get_step_fn = get_step_fn_det_seg

        # Create engine
        engine = Engine(get_step_fn(
            self.model,
            cl_loss_fn,
            seg_loss_fn,
            h2bb_method,
            optimizer=self.optimizer if training else None,
            training=training,
            cl_lambda=self.args.cl_lambda,
            seg_lambda=self.args.seg_lambda,
            device=self.device,
        ))
        attach_losses(engine, ['cl_loss', 'seg_loss'], device=self.device)
        attach_metrics_classification(
            engine,
            labels,
            multilabel=multilabel,
            device=self.device,
        )

        dataloader = self.train_dataloader if training else self.val_dataloader

        attach_mAP_coco(engine, dataloader, self.run_name, debug=self.debug, device=self.device)
        attach_metrics_iox(engine,
                           labels,
                           multilabel=multilabel,
                           device=self.device,
                           only_tp=False,
                           heat_thresh=None,
                           )

        return engine


if __name__ == '__main__':
    training_process = TrainingDetectionSeg()

    training_process.run()
