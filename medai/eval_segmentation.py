import argparse
import logging
from pprint import pprint
import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_segmentation
from medai.metrics import save_results
from medai.metrics.segmentation import attach_metrics_segmentation
from medai.metrics.segmentation.shapes import attach_organ_shapes_metric
from medai.models.checkpoint import load_compiled_model_segmentation
from medai.training.segmentation import get_step_fn
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    RunId,
)


LOGGER = logging.getLogger('medai.seg.eval')



def evaluate_model(model,
                   dataloader,
                   n_epochs=1,
                   device='cuda'):
    """Evaluate a segmentation model on a dataloader."""
    if dataloader is None:
        return {}

    LOGGER.info('Evaluating model in %s...', dataloader.dataset.dataset_type)

    labels = dataloader.dataset.seg_labels

    engine = Engine(get_step_fn(model,
                                training=False,
                                device=device,
                                ))
    attach_metrics_segmentation(engine, labels, multilabel=False)
    attach_organ_shapes_metric(engine, labels, gt=True)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


@timeit_main(LOGGER)
def evaluate_run(run_name,
                 dataset_types=('train', 'val', 'test'),
                 batch_size=None,
                 debug=True,
                 device='cuda',
                 multiple_gpu=False,
                 quiet=False,
                 ignore_augment=False,
                 ):
    """Evaluates a saved run."""
    run_id = RunId(run_name, debug, 'seg')

    # Load model
    compiled_model = load_compiled_model_segmentation(run_id,
                                                      device=device,
                                                      multiple_gpu=multiple_gpu)

    # Metadata (contains all configuration)
    metadata = compiled_model.metadata
    dataset_kwargs = metadata['dataset_kwargs']
    dataset_train_kwargs = metadata['dataset_train_kwargs']
    if ignore_augment:
        dataset_train_kwargs = {}

    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size

    # Create dataloaders
    dataloaders = [
        prepare_data_segmentation(
            dataset_type=dataset_type,
            **dataset_kwargs,
            **(dataset_train_kwargs if dataset_type == 'train' else {}),
        )
        for dataset_type in dataset_types
    ]

    eval_kwargs = {
        'device': device,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        name = dataloader.dataset.dataset_type

        if name == 'train' and ignore_augment:
            name = 'train-no-aug'

        metrics[name] = evaluate_model(compiled_model.model, dataloader, **eval_kwargs)

    save_results(metrics, run_id)

    if not quiet:
        pprint(metrics)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run name to evaluate')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics to stdout')

    data_group = parser.add_argument_group('Data')
    data_group.add_argument('-bs', '--batch-size', type=int, default=None,
                            help='Batch size')
    data_group.add_argument('--ignore-aug', action='store_true',
                            help='Ignore augmentation and oversampling params in train')
    # data_group.add_argument('--image-size', type=int, default=512,
    #                           help='Image size in pixels')
    # data_group.add_argument('--norm-by-sample', action='store_true',
    #                           help='If present, normalize each sample \
    #                                (instead of using dataset stats)')

    parsers.add_args_hw(parser, num_workers=2)

    args = parser.parse_args()

    args.debug = not args.no_debug

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(ARGS.run_name,
                 dataset_types=ARGS.eval_in,
                 batch_size=ARGS.batch_size,
                 debug=not ARGS.no_debug,
                 device=DEVICE,
                 multiple_gpu=ARGS.multiple_gpu,
                 quiet=ARGS.quiet,
                 ignore_augment=ARGS.ignore_aug,
                 )
