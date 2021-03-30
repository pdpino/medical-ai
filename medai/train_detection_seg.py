"""Script to train detection + segmentation models, used in the VinBig challenge.

For a more generalized version, use train_cls_seg.py
"""
from torch import nn
from ignite.engine import Engine

from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import (
    attach_mAP_coco,
    attach_metrics_iox,
)
from medai.models.checkpoint import load_compiled_model_detection_seg
from medai.models.common import AVAILABLE_POOLING_REDUCTIONS
from medai.models.detection import create_detection_seg_model, AVAILABLE_DETECTION_SEG_MODELS
from medai.training.common import TrainingProcess
from medai.training.detection.h2bb import get_h2bb_method
from medai.training.detection.det_seg import get_step_fn_det_seg
from medai.utils import parsers

class TrainingDetectionSeg(TrainingProcess):
    LOGGER_NAME = 'medai.det-seg.train'
    base_print_metrics = ['cl_loss', 'seg_loss', 'roc_auc', 'iou', 'iobb', 'mAP']
    task = 'det'

    load_compiled_model_fn = staticmethod(load_compiled_model_detection_seg)

    allow_augmenting = True
    allow_sampling = False

    default_dataset = 'vinbig'
    default_image_format = 'L'

    # TODO: pick better metrics here?
    key_metric = 'iou'
    default_es_metric = None
    default_lr_metric = None
    checkpoint_metric = 'iou'


    def _add_additional_args(self, parser):
        parser.add_argument('-m', '--model', type=str, default=None,
                            choices=AVAILABLE_DETECTION_SEG_MODELS,
                            help='Choose model to use')
        parser.add_argument('--cnn-pooling', type=str, default='avg',
                            choices=AVAILABLE_POOLING_REDUCTIONS,
                            help='Choose reduction for global-pooling layer')

        parser.add_argument('--seg-lambda', type=float, default=1,
                            help='Factor to multiply seg-loss')
        parser.add_argument('--cl-lambda', type=float, default=1,
                            help='Factor to multiply CL-loss')

        parser.add_argument('--seg-only-diseases', action='store_true',
                            help='If present, segment only diseases')

        parsers.add_args_h2bb(parser)

    def _build_additional_args(self, parser, args):
        if args.image_format != 'L':
            parser.error('Image-format must be L')

        if args.dataset != 'vinbig':
            # mAP metrics, h2bb stuff, etc
            parser.error('Only works with vinbig dataset')

        if not args.resume and not args.model:
            parser.error('Model is required')

        parsers.build_args_h2bb_(args)


    def _fill_run_name_model(self):
        self.run_name += f'_{self.args.model}'

        if self.args.cnn_pooling not in ('avg', 'mean'):
            self.run_name += f'_g{self.args.cnn_pooling}'

        if self.args.seg_only_diseases:
            self.run_name += '_seg-only-dis'

    def _fill_dataset_kwargs(self):
        self.dataset_kwargs['fallback_organs'] = not self.args.seg_only_diseases
        self.dataset_kwargs['masks'] = True
        self.dataset_kwargs['masks_version'] = 'v1'

        if self.args.augment:
            if 'augment_kwargs' not in self.dataset_train_kwargs:
                self.dataset_train_kwargs['augment_kwargs'] = {}
            self.dataset_train_kwargs['augment_kwargs'].update({
                'seg_mask': True,
            })

    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        labels = self.train_dataloader.dataset.labels

        self.model_kwargs = {
            'model_name': self.args.model,
            'labels': labels,
            'gpool': self.args.cnn_pooling,
        }
        self.model = create_detection_seg_model(**self.model_kwargs).to(self.device)

    def _fill_other_train_kwargs(self):
        d = dict()
        d['cl_lambda'] = self.args.cl_lambda
        d['seg_lambda'] = self.args.seg_lambda
        # d['seg_loss_name'] = self.args.seg_loss_name
        d['h2bb_method_name'] = self.args.h2bb_method_name
        d['h2bb_method_kwargs'] = self.args.h2bb_method_kwargs
        d['seg_only_diseases'] = self.args.seg_only_diseases

        self.other_train_kwargs.update(d)

    # pylint: disable=arguments-differ
    def _build_common_for_engines(self,
                                  seg_only_diseases=True,
                                  h2bb_method_name=None,
                                  h2bb_method_kwargs=None,
                                  **unused_kwargs,
                                  ):
        self.logger.info(
            'Using seg-only-dis=%s',
            seg_only_diseases,
        )

        # Prepare loss
        cl_loss_fn = nn.BCEWithLogitsLoss().to(self.device)
        seg_loss_fn = nn.BCEWithLogitsLoss(
            reduction='none' if seg_only_diseases else 'mean'
        ).to(self.device)

        # Choose h2bb method
        h2bb_method = get_h2bb_method(h2bb_method_name, h2bb_method_kwargs)

        return cl_loss_fn, seg_loss_fn, h2bb_method

    # pylint: disable=arguments-differ
    def _create_engine(self, training, *args,
                       seg_only_diseases=True,
                       cl_lambda=1,
                       seg_lambda=1,
                       **unused_kwargs,
                       ):
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
            seg_only_diseases=seg_only_diseases,
            cl_lambda=cl_lambda,
            seg_lambda=seg_lambda,
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
                           only='T',
                           heat_thresh=None,
                           )

        return engine


if __name__ == '__main__':
    training_process = TrainingDetectionSeg()

    training_process.run()
