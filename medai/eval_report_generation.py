import argparse
import logging
import torch
from torch.utils.data.dataset import Subset
from ignite.engine import Engine

from medai.datasets import prepare_data_report_generation
from medai.metrics import save_results, are_results_saved
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_attention_vs_masks,
    attach_losses_rg,
    attach_organ_by_sentence,
    build_suffix,
    print_rg_metrics,
)
from medai.metrics.report_generation.labeler import attach_medical_correctness
from medai.metrics.report_generation.writer import (
    attach_report_writer,
    delete_previous_outputs,
)
from medai.models.report_generation import is_decoder_hierarchical
from medai.models.checkpoint import (
    load_compiled_model_report_generation,
    get_best_checkpoints_metrics,
)
from medai.training.report_generation import get_step_fn_getter
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    RunId,
)
from medai.utils.nlp import attach_unclean_report_checker


LOGGER = logging.getLogger('medai.rg.eval')


def _evaluate_model_in_dataloader(
        run_id,
        model,
        dataloader,
        n_epochs=1,
        hierarchical=False,
        supervise_attention=False,
        medical_correctness=True,
        att_vs_masks=False,
        free=False,
        best=None,
        check_unclean=True,
        model_name='lstm',
        device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset # HACK

    vocab = dataset.get_vocab()

    get_step_fn = get_step_fn_getter(model_name)

    engine = Engine(get_step_fn(model,
                                training=False,
                                supervise_attention=supervise_attention,
                                free=free,
                                device=device))
    attach_unclean_report_checker(engine, check=check_unclean)
    attach_losses_rg(
        engine,
        free=free,
        hierarchical=hierarchical,
        supervise_attention=supervise_attention,
        model_name=model_name,
    )
    attach_metrics_report_generation(engine,
                                     free=free,
                                     device=device,
                                     )
    attach_organ_by_sentence(engine, vocab, device=device)
    attach_report_writer(engine, run_id,
                         vocab, assert_n_samples=len(dataset),
                         free=free, best=best)

    if medical_correctness:
        attach_medical_correctness(engine, None, vocab, device=device)

    if att_vs_masks and not free:
        attach_attention_vs_masks(engine, device=device)

    LOGGER.info('Evaluating model in %s, free=%s, best=%s', dataset.dataset_type, free, best)
    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_model_and_save(
        run_id,
        model,
        dataloaders,
        hierarchical=False,
        device='cuda',
        medical_correctness=True,
        supervise_attention=False,
        att_vs_masks=False,
        n_epochs=1,
        free_values=[False, True],
        best=None,
        check_unclean=True,
        model_name='lstm',
        quiet=False,
        ):
    """Evaluates a model in ."""
    evaluate_kwargs = {
        'hierarchical': hierarchical,
        'device': device,
        'medical_correctness': medical_correctness,
        'att_vs_masks': att_vs_masks,
        'supervise_attention': supervise_attention,
        'n_epochs': n_epochs,
        'check_unclean': check_unclean,
        'model_name': model_name,
        'best': best,
    }

    for free_value in free_values:
        delete_previous_outputs(run_id, free=free_value, best=best)

        metrics = {}

        evaluate_kwargs['free'] = free_value

        for dataloader in dataloaders:
            if dataloader is None:
                continue
            dataset_type = dataloader.dataset.dataset_type
            metrics[dataset_type] = _evaluate_model_in_dataloader(
                run_id,
                model,
                dataloader,
                **evaluate_kwargs,
            )

        # Add a suffix
        suffix = build_suffix(free_value, best)

        save_results(metrics, run_id, suffix=suffix)

        if not quiet:
            print_rg_metrics(metrics)

    return metrics


@timeit_main(LOGGER, sep='-', sep_times=50)
def evaluate_run_by_best(run_id,
                         best,
                         n_epochs=1,
                         free_values=[False, True],
                         device='cuda',
                         multiple_gpu=False,
                         batch_size=None,
                         dataset_types=('train','val','test'),
                         medical_correctness=True,
                         max_samples=None,
                         override=False,
                         check_unclean=True,
                         quiet=False,
                         ):
    """Evaluates a saved run."""
    LOGGER.info('Evaluating %s best=%s', run_id.short_name, best)

    # Check if overriding
    if not override:
        filtered_free_values = []
        for free_value in free_values:
            suffix = build_suffix(free_value, best)

            if are_results_saved(run_id, suffix=suffix):
                LOGGER.info('Already calculated for %s, skipping', suffix)
            else:
                filtered_free_values.append(free_value)

        free_values = filtered_free_values
        if len(free_values) == 0:
            LOGGER.info('Skipping run')
            return

    # Load model
    compiled_model = load_compiled_model_report_generation(
        run_id, device=device, multiple_gpu=multiple_gpu, mode=best,
    )

    # Metadata contains all configuration
    metadata = compiled_model.metadata

    # Decide hierarchical
    model_name = metadata['model_kwargs']['name']
    hierarchical = is_decoder_hierarchical(model_name)
    # Load data kwargs
    dataset_kwargs = metadata.get('dataset_kwargs', None)
    if dataset_kwargs is None:
        raise NotImplementedError('Fully deprecated')
        # # HACK: backward compatibility
        # dataset_kwargs = {
        #     'vocab': metadata['vocab'],
        #     'image_size': metadata.get('image_size', (512, 512)),
        #     'batch_size': metadata['hparams'].get('batch_size', 24),
        # }
    if 'hierarchical' not in dataset_kwargs:
        # backward compatibility
        dataset_kwargs['hierarchical'] = hierarchical
    dataset_train_kwargs = metadata.get('dataset_train_kwargs', {})

    if max_samples is not None:
        dataset_kwargs['max_samples'] = max_samples
    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size

    # Create dataloaders
    dataloaders = [
        prepare_data_report_generation(
            dataset_type=dataset_type,
            **dataset_kwargs,
            **(dataset_train_kwargs if dataset_type == 'train' else {}),
        )
        for dataset_type in dataset_types
    ]

    # Other evaluation kwargs
    other_train_kwargs = metadata.get('other_train_kwargs', {})
    supervise_attention = other_train_kwargs.get('supervise_attention', False)


    evaluate_model_and_save(
        run_id,
        compiled_model.model,
        dataloaders,
        hierarchical=hierarchical,
        device=device,
        medical_correctness=medical_correctness,
        supervise_attention=supervise_attention,
        att_vs_masks=supervise_attention,
        n_epochs=n_epochs,
        free_values=free_values,
        best=best,
        check_unclean=check_unclean,
        model_name=model_name,
        quiet=quiet,
    )

@timeit_main(LOGGER)
def evaluate_run(run_id, only_best=None, **kwargs):
    """Evaluates a run by all its saved checkpoints."""
    LOGGER.info('Evaluating %s', run_id)
    best_metrics = get_best_checkpoints_metrics(run_id)

    if only_best:
        best_chosen = [b for b in best_metrics if b in only_best]
        best_leftout = [b for b in best_metrics if b not in only_best]
    else:
        best_chosen = best_metrics
        best_leftout = []

    LOGGER.info('\t\tChosen metrics: %s, leftout: %s', best_chosen, best_leftout)

    for best in best_chosen:
        evaluate_run_by_best(run_id, best, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run-name to load')
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    # parser.add_argument('--image-size', type=int, default=512,
    #                     help='Image size in pixels')
    # parser.add_argument('--frontal-only', action='store_true',
    #                     help='Use only frontal images')
    # parser.add_argument('--norm-by-sample', action='store_true',
    #                     help='If present, normalize each sample (instead of using dataset stats)')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics to stdout')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override previous results')
    parser.add_argument('--use-med', action='store_true',
                        help='If present, use (runtime) medical-correctness metrics')
    parser.add_argument('--skip-check-unclean', action='store_true',
                        help='If present, do not check for unclean reports in the outputs')
    parser.add_argument('--only-best', type=str, nargs='*',
                        help='Only eval best by certain metrics')

    parsers.add_args_free_values(parser)
    parsers.add_args_hw(parser, num_workers=4)

    args = parser.parse_args()


    parsers.build_args_free_values_(args, parser)

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(
        run_id=RunId(ARGS.run_name, not ARGS.no_debug, 'rg'),
        free_values=ARGS.free_values,
        dataset_types=ARGS.eval_in,
        multiple_gpu=ARGS.multiple_gpu,
        device=DEVICE,
        medical_correctness=ARGS.use_med,
        n_epochs=ARGS.epochs,
        max_samples=ARGS.max_samples,
        override=ARGS.override,
        batch_size=ARGS.batch_size,
        check_unclean=not ARGS.skip_check_unclean,
        quiet=ARGS.quiet,
        only_best=ARGS.only_best,
    )
