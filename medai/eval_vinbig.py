import argparse
import logging
from pprint import pprint

import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_classification
from medai.losses import get_loss_function
from medai.metrics import save_results
from medai.metrics.classification import attach_metrics_classification
from medai.metrics.detection import attach_mAP_coco
# from medai.metrics.detection.coco_writer import attach_vinbig_writer
from medai.models.checkpoint import load_compiled_model_classification
from medai.training.detection import get_step_fn_hint
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
)


LOGGER = logging.getLogger('medai.det.eval')


def evaluate_model(run_name,
                   model,
                   dataloader,
                   task='det',
                   hint_lambda=1,
                   n_epochs=1,
                   debug=True,
                   device='cuda'):
    """Evaluate a vinbig model on a dataloader."""
    if dataloader is None:
        return {}

    LOGGER.info('Evaluating model in %s...', dataloader.dataset.dataset_type)
    loss = get_loss_function('wbce')
    loss = loss.to(device)

    labels = dataloader.dataset.labels
    multilabel = dataloader.dataset.multilabel

    get_step_fn = get_step_fn_hint

    engine = Engine(get_step_fn(model,
                                loss,
                                training=False,
                                hint_lambda=hint_lambda,
                                device=device,
                               ))
    attach_metrics_classification(engine, labels, multilabel=multilabel, device=device)
    attach_mAP_coco(engine, dataloader, run_name, task=task, debug=debug, device=device)
    # attach_vinbig_writer(engine, dataloader, run_name, debug=debug, task=task)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


@timeit_main(LOGGER)
def evaluate_run(run_name,
                 dataset_types=['train', 'val', 'test'],
                 n_epochs=1,
                 task='det',
                 batch_size=None,
                 max_samples=None,
                 debug=True,
                 multiple_gpu=False,
                 # override=False,
                 quiet=False,
                 device='cuda',
                 ):
    """Evaluate a model."""
    # TODO: implement override
    # if not override and are_results_saved(run_name, task=task, debug=debug):
    #     LOGGER.warning('Skipping run, already calculated')
    #     return {}

    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        task=task,
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

    # Evaluate
    other_train_kwargs = metadata['other_train_kwargs']
    eval_kwargs = {
        'hint_lambda': other_train_kwargs['hint_lambda'],
        'task': task,
        'device': device,
        'n_epochs': n_epochs,
        'debug': debug,
    }

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        dataset_type = dataloader.dataset.dataset_type
        metrics[dataset_type] = evaluate_model(
            run_name, compiled_model.model, dataloader, **eval_kwargs,
        )

    save_results(metrics, run_name, task=task, debug=debug)

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
    parser.add_argument('--task', type=str, default='det', choices=['cls', 'det'],
                        help='Task to load the model from')
    parser.add_argument('-bs', '--batch-size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    # parser.add_argument('--override', action='store_true',
    #                     help='Override previous results')
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
                 task=ARGS.task,
                 batch_size=ARGS.batch_size,
                 max_samples=ARGS.max_samples,
                 n_epochs=ARGS.epochs,
                 debug=not ARGS.no_debug,
                 # override=ARGS.override,
                 quiet=ARGS.quiet,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                )
