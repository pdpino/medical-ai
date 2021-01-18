import argparse
import logging
import torch

from medai.datasets.iu_xray import IUXRayDataset
from medai.models.checkpoint import load_compiled_model_classification
from medai.models.classification import create_cnn
from medai.models.report_generation.dummy.constant import ConstantReport
from medai.models.report_generation.dummy.common_words import MostCommonWords
from medai.models.report_generation.dummy.common_sentences import MostCommonSentences
from medai.models.report_generation.dummy.random import RandomReport
from medai.models.report_generation.dummy.most_similar_image import MostSimilarImage
from medai.training.report_generation.flat import create_flat_dataloader
from medai.training.report_generation.hierarchical import create_hierarchical_dataloader
from medai.eval_report_generation import evaluate_model_and_save
from medai.utils import (
    get_timestamp,
    config_logging,
    timeit_main,
    parsers,
    print_hw_options,
)


config_logging()
LOGGER = logging.getLogger('rg.eval.dummy')
LOGGER.setLevel(logging.INFO)


_AVAILABLE_DUMMY_MODELS = [
    'constant',
    'common-words',
    'common-sentences',
    'random',
    'most-similar-image',
]

def _is_hierarchical(model_name):
    return model_name == 'common-sentences'


@timeit_main(LOGGER)
def evaluate_dummy_model(model_name,
                         batch_size=20,
                         k_first=100,
                         similar_run_name=None,
                         similar_cnn_kwargs={},
                         free_values=[False, True],
                         debug=True,
                         device='cuda',
                         ):
    # Run name
    run_name = f'{get_timestamp()}_dummy-{model_name}'
    if 'common-' in model_name:
        run_name += f'-{str(k_first)}'
    if model_name == 'most-similar-image':
        if similar_run_name:
            run_name += f'_{similar_run_name}'
        else:
            cnn_name = similar_cnn_kwargs.get('model_name', None)
            run_name += f'_{cnn_name}'

    LOGGER.info('Evaluating %s', run_name)

    # Load datasets
    dataset_kwargs = {
        'frontal_only': True,
    }
    train_dataset = IUXRayDataset('train', **dataset_kwargs)
    val_dataset = IUXRayDataset('val', **dataset_kwargs)
    test_dataset = IUXRayDataset('test', **dataset_kwargs)

    vocab = train_dataset.get_vocab()

    # Decide hierarchical
    is_hierarchical = _is_hierarchical(model_name)
    if is_hierarchical:
        create_dataloader = create_hierarchical_dataloader
    else:
        create_dataloader = create_flat_dataloader


    # Create dataloaders
    train_dataloader = create_dataloader(train_dataset, batch_size=batch_size)
    val_dataloader = create_dataloader(val_dataset, batch_size=batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size=batch_size)


    # Choose model
    if model_name == 'constant':
        model = ConstantReport(vocab)

    elif model_name == 'common-words':
        model = MostCommonWords(train_dataset, k_first)

    elif model_name == 'common-sentences':
        model = MostCommonSentences(train_dataset, k_first)

    elif model_name == 'random':
        model = RandomReport(train_dataset)

    elif model_name == 'most-similar-image':
        if similar_run_name:
            compiled_model = load_compiled_model_classification(
                similar_run_name, debug=debug, device=device)
            cnn = compiled_model.model.to(device)
            compiled_model.optimizer = None # Not needed
        else:
            cnn = create_cnn(**similar_cnn_kwargs).to(device)

        model = MostSimilarImage(cnn, vocab)
        model.eval()

        LOGGER.info('Fitting model...')
        model.fit(train_dataloader, device=device)

    else:
        raise Exception(f'Model not recognized: {model_name}')


    dataloaders = [
        train_dataloader,
        val_dataloader,
        test_dataloader
    ]

    # Evaluate
    evaluate_model_and_save(
        run_name,
        model,
        dataloaders,
        hierarchical=is_hierarchical,
        free_values=free_values,
        debug=debug,
        device=device,
        )

    LOGGER.info('Evaluated %s', run_name)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, default=None,
                        choices=_AVAILABLE_DUMMY_MODELS, help='Dummy model to use')
    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='Batch size to use')
    parser.add_argument('-k', '--k-first', type=int, default=100,
                        help='Top k selected for common-words and common-sentences')
    parser.add_argument('--cnn-run-name', type=str, default=None,
                        help='MostSimilarImage: cnn run name to use as feature extractor')
    parser.add_argument('--cnn-name', type=str, default=None,
                        help='MostSimilarImage: CNN name to create')
    parser.add_argument('--imagenet', action='store_true',
                        help='MostSimilarImage: args to CNN')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')
    parsers.add_args_free_values(parser)

    args = parser.parse_args()

    args.similar_run_name = args.cnn_run_name
    args.similar_cnn_kwargs = {
        'model_name': args.cnn_name,
        'labels': [],
        'imagenet': args.imagenet,
        'freeze': True,
    }

    if args.model_name == 'most-similar-image':
        if args.similar_run_name is None and args.cnn_name is None:
            parser.error('most-similar-image: needs --cnn-run-name or --cnn-name')

    parsers.build_args_free_values_(args, parser)

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_dummy_model(
        ARGS.model_name,
        batch_size=ARGS.batch_size,
        k_first=ARGS.k_first,
        similar_run_name=ARGS.similar_run_name,
        similar_cnn_kwargs=ARGS.similar_cnn_kwargs,
        debug=not ARGS.no_debug,
        free_values=ARGS.free_values,
        device=DEVICE,
        )
