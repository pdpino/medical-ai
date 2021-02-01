import argparse
import logging
from pprint import pprint

import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_classification
from medai.losses import get_loss_function
from medai.metrics import save_results, are_results_saved
from medai.metrics.classification import (
    attach_metrics_classification,
    attach_metric_cm,
)
from medai.models.checkpoint import load_compiled_model_classification
from medai.training.classification import get_step_fn
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
)


LOGGER = logging.getLogger('medai.cl.eval')


def evaluate_model(model,
                   dataloader,
                   loss_name='wbce',
                   loss_kwargs={},
                   n_epochs=1,
                   device='cuda'):
    """Evaluate a classification model on a dataloader."""
    if dataloader is None:
        return {}

    LOGGER.info('Evaluating model in %s...', dataloader.dataset.dataset_type)
    loss = get_loss_function(loss_name, **loss_kwargs)

    labels = dataloader.dataset.labels
    multilabel = dataloader.dataset.multilabel

    engine = Engine(get_step_fn(model,
                                loss,
                                training=False,
                                multilabel=multilabel,
                                device=device,
                               ))
    attach_metrics_classification(engine, labels, multilabel=multilabel)
    attach_metric_cm(engine, labels, multilabel=multilabel)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


@timeit_main(LOGGER)
def evaluate_run(run_name,
                 dataset_types=['train', 'val', 'test'],
                 n_epochs=1,
                 batch_size=None,
                 max_samples=None,
                 debug=True,
                 multiple_gpu=False,
                 override=False,
                 quiet=False,
                 device='cuda',
                 ):
    """Evaluate a model."""
    if not override and are_results_saved(run_name, task='cls', debug=debug):
        LOGGER.info('Skipping run, already calculated')
        return {}

    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        device=device,
                                                        multiple_gpu=multiple_gpu)
    compiled_model.model.eval()

    # Metadata (contains all configuration)
    metadata = compiled_model.metadata

    # Load data
    dataset_kwargs = metadata['dataset_kwargs']
    if max_samples is not None:
        dataset_kwargs['max_samples'] = max_samples
    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size

    dataloaders = [
        prepare_data_classification(dataset_type=dataset_type, **dataset_kwargs)
        for dataset_type in dataset_types
    ]

    # Load stuff from metadata
    hparams = metadata.get('hparams')
    loss_name = hparams.get('loss_name', 'cross-entropy')
    loss_kwargs = hparams.get('loss_kwargs', {})

    # Evaluate
    eval_kwargs = {
        'loss_name': loss_name,
        'loss_kwargs': loss_kwargs,
        'device': device,
        'n_epochs': n_epochs,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        dataset_type = dataloader.dataset.dataset_type
        metrics[dataset_type] = evaluate_model(compiled_model.model, dataloader, **eval_kwargs)

    save_results(metrics, run_name, task='cls', debug=debug)

    if not quiet:
        pprint(metrics)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser()

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

    evaluate_run(ARGS.run_name,
                 dataset_types=ARGS.eval_in,
                 max_samples=ARGS.max_samples,
                 batch_size=ARGS.batch_size,
                 n_epochs=ARGS.epochs,
                 debug=not ARGS.no_debug,
                 override=ARGS.override,
                 quiet=ARGS.quiet,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                )
