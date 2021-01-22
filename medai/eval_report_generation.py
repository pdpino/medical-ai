import argparse
import logging
import torch
from torch.utils.data.dataset import Subset
from ignite.engine import Engine

from medai.datasets import prepare_data_report_generation
from medai.metrics import save_results, are_results_saved
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_medical_correctness,
    attach_attention_vs_masks,
)
from medai.metrics.report_generation.writer import (
    attach_report_writer,
    delete_previous_outputs,
)
from medai.models.report_generation import is_decoder_hierarchical
from medai.models.checkpoint import load_compiled_model_report_generation
from medai.training.report_generation.flat import (
    create_flat_dataloader,
    get_step_fn_flat,
)
from medai.training.report_generation.hierarchical import (
    create_hierarchical_dataloader,
    get_step_fn_hierarchical,
)
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
)

LOGGER = logging.getLogger('medai.rg.eval')


def _evaluate_model_in_dataloader(
        run_name,
        model,
        dataloader,
        n_epochs=1,
        hierarchical=False,
        supervise_attention=False,
        medical_correctness=True,
        att_vs_masks=False,
        free=False,
        debug=True,
        device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset # HACK
    LOGGER.info('Evaluating model in %s, free=%s', dataset.dataset_type, free)

    vocab = dataset.get_vocab()

    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat

    engine = Engine(get_step_fn(model,
                                training=False,
                                supervise_attention=supervise_attention,
                                free=free,
                                device=device))
    attach_metrics_report_generation(engine,
                                     hierarchical=hierarchical,
                                     free=free,
                                     supervise_attention=supervise_attention,
                                     )
    attach_report_writer(engine, run_name,
                         vocab, assert_n_samples=len(dataset),
                         free=free, debug=debug)

    if medical_correctness:
        attach_medical_correctness(engine, None, vocab)

    if att_vs_masks and not free:
        attach_attention_vs_masks(engine)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_model_and_save(
        run_name,
        model,
        dataloaders,
        debug=True,
        hierarchical=False,
        device='cuda',
        medical_correctness=True,
        supervise_attention=False,
        att_vs_masks=False,
        n_epochs=1,
        free_values=[False, True],
        ):
    """Evaluates a model in ."""
    evaluate_kwargs = {
        'hierarchical': hierarchical,
        'device': device,
        'debug': debug,
        'medical_correctness': medical_correctness,
        'att_vs_masks': att_vs_masks,
        'supervise_attention': supervise_attention,
        'n_epochs': n_epochs,
    }

    for free_value in free_values:
        delete_previous_outputs(run_name, debug=debug, free=free_value)

        metrics = {}

        evaluate_kwargs['free'] = free_value

        for dataloader in dataloaders:
            if dataloader is None:
                continue
            dataset_type = dataloader.dataset.dataset_type
            metrics[dataset_type] = _evaluate_model_in_dataloader(
                run_name,
                model,
                dataloader,
                **evaluate_kwargs,
            )

        # Add a suffix
        suffix = 'free' if free_value else 'notfree'

        save_results(metrics,
                     run_name,
                     task='rg',
                     debug=debug,
                     suffix=suffix,
                     )


@timeit_main(LOGGER)
def evaluate_run(run_name,
                 n_epochs=1,
                 free_values=[False, True],
                 debug=True,
                 device='cuda',
                 multiple_gpu=False,
                 batch_size=None,
                 dataset_types=('train','val','test'),
                 medical_correctness=True,
                 max_samples=None,
                 override=False,
                 ):
    """Evaluates a saved run."""
    # Check if overriding
    if not override:
        filtered_free_values = []
        for free_value in free_values:
            suffix = 'free' if free_value else 'notfree'

            if are_results_saved(run_name, task='rg', debug=debug, suffix=suffix):
                LOGGER.info('Already calculated for %s, skipping', suffix)
            else:
                filtered_free_values.append(free_value)

        free_values = filtered_free_values
        if len(free_values) == 0:
            LOGGER.info('Skipping run')
            return


    # Load model
    compiled_model = load_compiled_model_report_generation(run_name,
                                                           debug=debug,
                                                           device=device,
                                                           multiple_gpu=multiple_gpu)

    # Metadata contains all configuration
    metadata = compiled_model.metadata

    # Decide hierarchical
    decoder_name = metadata['decoder_kwargs']['decoder_name']
    hierarchical = is_decoder_hierarchical(decoder_name)
    if hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader

    # Load data kwargs
    dataset_kwargs = metadata.get('dataset_kwargs', None)
    if dataset_kwargs is None:
        # HACK: backward compatibility
        dataset_kwargs = {
            'vocab': metadata['vocab'],
            'image_size': metadata.get('image_size', (512, 512)),
            'batch_size': metadata['hparams'].get('batch_size', 24),
        }
    dataset_train_kwargs = metadata.get('dataset_train_kwargs', {})

    if max_samples is not None:
        dataset_kwargs['max_samples'] = max_samples
    if batch_size is not None:
        dataset_kwargs['batch_size'] = batch_size

    # Create dataloaders
    dataloaders = [
        prepare_data_report_generation(
            create_dataloader,
            dataset_type=dataset_type,
            **dataset_kwargs,
            **(dataset_train_kwargs if dataset_type == 'train' else {}),
        )
        for dataset_type in dataset_types
    ]

    # Other evaluation kwargs
    other_train_kwargs = metadata.get('other_train_kwargs', {})
    supervise_attention = other_train_kwargs.get('supervise_attention', False)
    att_vs_masks = decoder_name.startswith('h-lstm-att') # HACK: copied from train_model()


    evaluate_model_and_save(
        run_name,
        compiled_model.model,
        dataloaders,
        debug=debug,
        hierarchical=hierarchical,
        device=device,
        medical_correctness=medical_correctness,
        supervise_attention=supervise_attention,
        att_vs_masks=att_vs_masks,
        n_epochs=n_epochs,
        free_values=free_values,
        )


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
    parser.add_argument('--override', action='store_true',
                        help='Whether to override previous results')
    parser.add_argument('--no-med', action='store_true',
                        help='If present, do not use medical-correctness metrics')
    parsers.add_args_free_values(parser)
    parsers.add_args_hw(parser, num_workers=4)

    args = parser.parse_args()


    parsers.build_args_free_values_(args, parser)
    use_free = not args.skip_free
    use_notfree = not args.skip_notfree
    if use_free and use_notfree:
        args.free = 'both'
    elif use_free:
        args.free = True
    elif use_notfree:
        args.free = False
    else:
        parser.error('Cannot skip both free and not free')

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(ARGS.run_name,
                 free_values=ARGS.free_values,
                 dataset_types=ARGS.eval_in,
                 debug=not ARGS.no_debug,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                 medical_correctness=not ARGS.no_med,
                 n_epochs=ARGS.epochs,
                 max_samples=ARGS.max_samples,
                 override=ARGS.override,
                 batch_size=ARGS.batch_size,
                #  frontal_only=ARGS.frontal_only,
                #  image_size=ARGS.image_size,
                #  norm_by_sample=ARGS.norm_by_sample,
                 )