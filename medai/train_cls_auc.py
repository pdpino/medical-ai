from ignite.engine import Engine
from libauc.optimizers import PESG

from medai.losses.aucm import AUCMLoss
from medai.metrics import attach_losses
from medai.metrics.classification import attach_metrics_classification
from medai.models import load_compiled_model, freeze_cnn
from medai.training.common import TrainingProcess
from medai.training.classification.auc import get_step_fn_cls_auc
from medai.utils import RunId


class TrainingClsAUC(TrainingProcess):
    LOGGER_NAME = 'medai.cls-auc.train'
    base_print_metrics = ['loss', 'roc_auc', 'pr_auc']
    # base_print_metrics = ['loss', 'roc_auc_Cardiomegaly', 'pr_auc_Cardiomegaly']
    task = 'cls'

    allow_augmenting = True
    allow_sampling = False

    allow_resume = False

    default_dataset = 'chexpert'
    default_image_format = 'RGB'
    default_image_size = 256

    # TODO: use better metric? that captures both cls and seg
    key_metric = 'pr_auc'
    default_es_metric = None
    default_lr_metric = None
    default_checkpoint_metric = 'pr_auc'

    def _add_additional_args(self, parser):
        parser.add_argument('--pretrained', '--run-name', type=str, default=None,
                            help='Run name of a pretrained CNN')
        parser.add_argument('--pretrained-task', type=str, default='cls',
                            choices=('cls', 'cls-seg'), help='Task to choose the CNN from')
        parser.add_argument('--cnn-freeze', action='store_true', help='Freeze features')

        parser.add_argument('--label', type=str, default='Cardiomegaly',
                            help='Label to use for training (with 1 label only)')

    def _build_additional_args(self, parser, args):
        if not args.pretrained:
            parser.error('--pretrained or --run-name is required')

        args.pretrained_run_id = RunId(args.pretrained, debug=False, task=args.pretrained_task)

        if not args.print_metrics:
            args.print_metrics = []
        disease = args.label
        args.print_metrics.extend([f'roc_auc_{disease}', f'pr_auc_{disease}'])

    def _fill_run_name_model(self, run_name):
        run_name += '_cls-auc'
        run_name += f'_{self.args.pretrained_run_id.short_clean_name}'
        run_name += f'_{self.args.label}'

        if self.args.cnn_freeze:
            run_name += '_cnn-freeze'

        return run_name

    def _create_model(self):
        """Create self.model and self.model_kwargs."""
        compiled_model = load_compiled_model(
            self.args.pretrained_run_id,
            device=self.device,
            multiple_gpu=self.args.multiple_gpu,
        )

        self.model = compiled_model.model

        self.model_kwargs = {
            'cnn_freeze': self.args.cnn_freeze,
        }
        if self.args.cnn_freeze:
            freeze_cnn(self.model.features)

        # TODO: move this calculation to the dataset??
        dataset = self.train_dataloader.dataset
        imratio = dataset.label_index[self.args.label].sum() / len(dataset)

        self.logger.info('imratio=%s for %s', imratio, self.args.label)

        # Must be defined before the optimizer
        # pylint: disable=attribute-defined-outside-init
        self.loss = AUCMLoss(imratio=imratio, device=self.device)

    def _create_optimizer(self):
        self.opt_kwargs = {
            'lr': self.args.learning_rate,
            'weight_decay': self.args.weight_decay,
        }
        self.optimizer = PESG(
            self.model,
            a=self.loss.a,
            b=self.loss.b,
            alpha=self.loss.alpha,
            imratio=self.loss.p,
            **self.opt_kwargs,
        )

    def _fill_hparams(self):
        self.metadata['hparams'].update({
            'pretrained': self.args.pretrained_run_id.to_dict(),
        })

    def _fill_other_train_kwargs(self):
        self.other_train_kwargs.update({
            'label': self.args.label,
        })

    # pylint: disable=arguments-differ
    def _create_engine(self, training, label='', **unused_kwargs):
        cl_labels = self.train_dataloader.dataset.labels
        cl_multilabel = self.train_dataloader.dataset.multilabel
        assert cl_multilabel, 'Dataset is not cl_multilabel'

        label_index = cl_labels.index(label)

        # Create engine
        engine = Engine(get_step_fn_cls_auc(
            self.model,
            self.loss,
            label_index,
            optimizer=self.optimizer if training else None,
            training=training,
            device=self.device,
        ))
        attach_losses(engine, device=self.device)
        attach_metrics_classification(
            engine,
            cl_labels,
            multilabel=cl_multilabel,
            device=self.device,
            extra_bce=True,
        )

        return engine


if __name__ == '__main__':
    training_process = TrainingClsAUC()

    training_process.run()
