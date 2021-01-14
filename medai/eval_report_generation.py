import time
import argparse
import logging
import torch
from torch.utils.data.dataset import Subset
from ignite.engine import Engine, Events

from medai.datasets import prepare_data_report_generation
from medai.metrics import save_results
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    attach_medical_correctness,
    attach_attention_vs_masks,
    attach_report_writer,
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
    duration_to_str,
    print_hw_options,
    parsers,
    config_logging,
)


config_logging()
LOGGER = logging.getLogger('rg')
LOGGER.setLevel(logging.INFO)


def evaluate_model(run_name,
                   compiled_model,
                   dataloader,
                   n_epochs=1,
                   hierarchical=False,
                   supervise_attention=False,
                   medical_correctness=True,
                   free=False,
                   debug=True,
                   device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset # HACK
    LOGGER.info('Evaluating model in %s, free=%s', dataset.dataset_type, free)

    if hierarchical:
        get_step_fn = get_step_fn_hierarchical
    else:
        get_step_fn = get_step_fn_flat

    engine = Engine(get_step_fn(compiled_model.model,
                                training=False,
                                supervise_attention=supervise_attention,
                                free=free,
                                device=device))
    attach_metrics_report_generation(engine,
                                     hierarchical=hierarchical,
                                     free=free,
                                     supervise_attention=supervise_attention,
                                     )
    attach_report_writer(engine, dataset.get_vocab(), run_name, free=free,
                         debug=debug)

    if medical_correctness:
        attach_medical_correctness(engine, None, dataset.get_vocab())

    # Decide att-vs-masks # HACK: copied from train_model()
    decoder_name = compiled_model.metadata['decoder_kwargs']['decoder_name']
    if decoder_name.startswith('h-lstm-att'):
        attach_attention_vs_masks(engine)

    # Catch errors, specially for free=True case
    engine.add_event_handler(Events.EXCEPTION_RAISED, lambda _, err: LOGGER.error(err))

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_run(run_name,
                 n_epochs=1,
                 free='both',
                 debug=True,
                 device='cuda',
                 multiple_gpu=False,
                 dataset_types=('train','val','test'),
                 medical_correctness=True,
                 max_samples=None,
                #  batch_size=None,
                #  frontal_only=False,
                #  image_size=None,
                #  norm_by_sample=None,
                 ):
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
    evaluate_kwargs = {
        'hierarchical': hierarchical,
        'device': device,
        'debug': debug,
        'medical_correctness': medical_correctness,
        'supervise_attention': other_train_kwargs.get('supervise_attention', False),
        'n_epochs': n_epochs,
    }

    # Decide free
    if free == 'both':
        free_values = [False, True]
    else:
        free_values = [bool(free)]

    for free_value in free_values:
        # Add a suffix
        suffix = 'free' if free_value else 'notfree'

        metrics = {}

        evaluate_kwargs['free'] = free_value

        for dataloader in dataloaders:
            if dataloader is None:
                continue
            dataset_type = dataloader.dataset.dataset_type
            metrics[dataset_type] = evaluate_model(
                run_name,
                compiled_model,
                dataloader,
                **evaluate_kwargs,
            )

        save_results(metrics,
                     run_name,
                     task='rg',
                     debug=debug,
                     suffix=suffix,
                     )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run-name to load')
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    # parser.add_argument('-bs', '--batch_size', type=int, default=10,
    #                     help='Batch size')
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
    parser.add_argument('--no-med', action='store_true',
                        help='If present, do not use medical-correctness metrics')
    parser.add_argument('--skip-free', action='store_true',
                        help='If present, do not evaluate in free mode')
    parser.add_argument('--skip-notfree', action='store_true',
                        help='If present, do not evaluate in not-free mode')

    parsers.add_args_hw(parser, num_workers=4)

    args = parser.parse_args()


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

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    start_time = time.time()

    evaluate_run(ARGS.run_name,
                 free=ARGS.free,
                 dataset_types=ARGS.eval_in,
                 debug=not ARGS.no_debug,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                 medical_correctness=not ARGS.no_med,
                 n_epochs=ARGS.epochs,
                 max_samples=ARGS.max_samples,
                #  batch_size=ARGS.batch_size,
                #  frontal_only=ARGS.frontal_only,
                #  image_size=ARGS.image_size,
                #  norm_by_sample=ARGS.norm_by_sample,
                 )

    total_time = time.time() - start_time
    LOGGER.info('Total time: %s', duration_to_str(total_time))
