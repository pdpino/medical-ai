"""FIXME: really similar to eval_classification.py"""
import argparse
import logging
from pprint import pprint

import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_classification
from medai.metrics import save_results, are_results_saved, attach_losses
from medai.metrics.detection import attach_metrics_iox
from medai.metrics.classification import (
    attach_metrics_classification,
)
from medai.metrics.classification.writer import (
    attach_prediction_writer,
    delete_previous_outputs,
)
from medai.models.checkpoint import load_compiled_model_cls_seg
from medai.training.detection.cls_seg import get_step_fn_cls_seg
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    RunId,
)


LOGGER = logging.getLogger('medai.cls-seg.eval')


def evaluate_model(run_id,
                   model,
                   dataloader,
                   n_epochs=1,
                   cl_lambda=1,
                   seg_lambda=1,
                   weight_organs=None,
                   cl_loss_name='bce',
                   device='cuda'):
    """Evaluate a classification model on a dataloader."""
    if dataloader is None:
        return {}

    LOGGER.info('Evaluating model in %s...', dataloader.dataset.dataset_type)

    cl_labels = dataloader.dataset.labels
    seg_labels = dataloader.dataset.organs

    engine = Engine(get_step_fn_cls_seg(
        model,
        training=False,
        cl_lambda=cl_lambda,
        seg_lambda=seg_lambda,
        seg_weights=weight_organs,
        cl_loss_name=cl_loss_name,
        device=device,
    ))
    attach_losses(engine, ['cl_loss', 'seg_loss'], device=device)
    attach_metrics_classification(
        engine,
        cl_labels,
        multilabel=True,
        device=device,
        extra_bce=cl_loss_name != 'bce',
    )
    attach_metrics_iox(
        engine,
        seg_labels,
        multilabel=False,
        device=device,
    )
    attach_prediction_writer(
        engine, run_id, cl_labels, assert_n_samples=len(dataloader.dataset),
    )

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


@timeit_main(LOGGER)
def evaluate_run(run_id,
                 dataset_types=['train', 'val', 'test'],
                 n_epochs=1,
                 batch_size=None,
                 max_samples=None,
                 multiple_gpu=False,
                 override=False,
                 quiet=False,
                 device='cuda',
                 ):
    """Evaluate a model."""
    if not override and are_results_saved(run_id):
        LOGGER.warning('Skipping run, already calculated')
        return {}

    # Delete previous CSV outputs
    delete_previous_outputs(run_id)

    # Load model
    compiled_model = load_compiled_model_cls_seg(
        run_id,
        device=device,
        multiple_gpu=multiple_gpu,
    )
    compiled_model.model.eval()

    # Metadata (contains all configuration)
    metadata = compiled_model.metadata

    # Load data
    dataset_kwargs = metadata['dataset_kwargs']
    # REVIEW: use dataset_train_kwargs??
    if max_samples is not None:
        dataset_kwargs['max_samples'] = max_samples
    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size

    dataloaders = [
        prepare_data_classification(dataset_type=dataset_type, **dataset_kwargs)
        for dataset_type in dataset_types
    ]

    # Load stuff from metadata
    other_train_kwargs = metadata.get('other_train_kwargs')

    # Evaluate
    eval_kwargs = {
        'cl_lambda': other_train_kwargs['cl_lambda'],
        'seg_lambda': other_train_kwargs['seg_lambda'],
        'cl_loss_name': other_train_kwargs.get('cl_loss_name', 'bce'),
        'weight_organs': other_train_kwargs.get('weight_organs'),
        'device': device,
        'n_epochs': n_epochs,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        dataset_type = dataloader.dataset.dataset_type
        metrics[dataset_type] = evaluate_model(
            run_id, compiled_model.model, dataloader, **eval_kwargs,
        )

    save_results(metrics, run_id)

    if not quiet:
        pprint(metrics)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(usage='%(prog)s [options]')

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run-name to load')
    parser.add_argument('--eval-in', nargs='*', default=['test', 'train', 'val'],
                        help='Eval in datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--override', action='store_true',
                        help='Override previous results')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print results to stdout')

    parsers.add_args_hw(parser)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(RunId(ARGS.run_name, not ARGS.no_debug, 'cls-seg'),
                 dataset_types=ARGS.eval_in,
                 max_samples=ARGS.max_samples,
                 batch_size=ARGS.batch_size,
                 n_epochs=ARGS.epochs,
                 override=ARGS.override,
                 quiet=ARGS.quiet,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                )
