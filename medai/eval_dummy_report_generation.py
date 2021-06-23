import argparse
import logging
import torch

from medai.datasets import prepare_data_report_generation, AVAILABLE_REPORT_DATASETS
from medai.datasets.common import LATEST_REPORTS_VERSION
from medai.models.checkpoint import load_compiled_model
from medai.models.classification import create_cnn
from medai.models.report_generation.dummy.constant import (
    ConstantReport,
    AVAILABLE_CONSTANT_VERSIONS,
)
from medai.models.report_generation.dummy.common_words import MostCommonWords
from medai.models.report_generation.dummy.common_sentences import MostCommonSentences
from medai.models.report_generation.dummy.random import RandomReport
from medai.models.report_generation.dummy.most_similar_image import MostSimilarImage
from medai.eval_report_generation import evaluate_model_and_save
from medai.utils import (
    get_timestamp,
    config_logging,
    timeit_main,
    parsers,
    print_hw_options,
    RunId,
)


LOGGER = logging.getLogger('medai.rg.eval.dummy')


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
                         dataset_name='iu-x-ray',
                         reports_version=LATEST_REPORTS_VERSION,
                         batch_size=20,
                         dataset_types=['train', 'val', 'test'],
                         constant_version='iu',
                         k_first=100,
                         similar_cnn_id=None,
                         similar_cnn_kwargs={},
                         image_size=256,
                         norm_by_sample=False,
                         free_values=[False, True],
                         frontal_only=True,
                         medical_correctness=True,
                         debug=True,
                         device='cuda',
                         max_samples=None,
                         quiet=False,
                         ):
    # Run name
    run_name = f'{get_timestamp()}_{dataset_name}_dummy-{model_name}'
    if 'common-' in model_name:
        run_name += f'-{str(k_first)}'
    if model_name == 'constant':
        if constant_version != 'iu':
            run_name += f'-{constant_version}'
    if model_name == 'most-similar-image':
        if similar_cnn_id:
            run_name += f'_{similar_cnn_id.short_clean_name}'
        else:
            cnn_name = similar_cnn_kwargs.get('model_name', None)
            run_name += f'_{cnn_name}'
        if image_size != 256:
            run_name += f'_size{image_size}'
    if frontal_only:
        run_name += '_front'
    if reports_version != LATEST_REPORTS_VERSION:
        run_name += f'_reports-{reports_version}'

    run_id = RunId(run_name, debug, 'rg')

    LOGGER.info('Evaluating %s', run_id)

    # Decide hierarchical
    hierarchical = _is_hierarchical(model_name)

    # Load datasets
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'hierarchical': hierarchical,
        'max_samples': max_samples,
        'norm_by_sample': norm_by_sample,
        'image_size': (image_size, image_size),
        'batch_size': batch_size,
        'frontal_only': frontal_only,
        'reports_version': reports_version,
        'do_not_load_image': model_name != 'most-similar-image',
    }
    train_dataloader = prepare_data_report_generation(
        dataset_type='train',
        **dataset_kwargs,
    )
    train_dataset = train_dataloader.dataset
    vocab = train_dataloader.dataset.get_vocab()
    dataset_kwargs['vocab'] = vocab

    val_dataloader = prepare_data_report_generation(
        dataset_type='val',
        **dataset_kwargs,
    )
    test_dataloader = prepare_data_report_generation(
        dataset_type='test',
        **dataset_kwargs,
    )

    # Choose model
    if model_name == 'constant':
        model = ConstantReport(vocab, version=constant_version)

    elif model_name == 'common-words':
        model = MostCommonWords(train_dataset, k_first)

    elif model_name == 'common-sentences':
        model = MostCommonSentences(train_dataset, k_first)

    elif model_name == 'random':
        model = RandomReport(train_dataset)

    elif model_name == 'most-similar-image':
        if similar_cnn_id:
            compiled_model = load_compiled_model(similar_cnn_id, device=device)
            cnn = compiled_model.model.to(device)
            compiled_model.optimizer = None # Not needed
        else:
            cnn = create_cnn(**similar_cnn_kwargs).to(device)

        model = MostSimilarImage(cnn, vocab)
        model.eval()

        LOGGER.info('Precalculating feature vectors...')
        model.fit(train_dataloader, device=device)

    else:
        raise Exception(f'Model not recognized: {model_name}')


    # FIXME: this will not allow loading other split than train/val/test
    dataloaders = [
        d
        for d in (
            train_dataloader,
            val_dataloader,
            test_dataloader,
        )
        if d.dataset.dataset_type in dataset_types
    ]

    # Evaluate
    evaluate_model_and_save(
        run_id,
        model,
        dataloaders,
        hierarchical=hierarchical,
        free_values=free_values,
        medical_correctness=medical_correctness,
        device=device,
        check_unclean=False,
        quiet=quiet,
        model_name=model_name,
    )

    LOGGER.info('Evaluated %s', run_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_name', type=str, default=None,
                        choices=_AVAILABLE_DUMMY_MODELS, help='Dummy model to use')
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('-d', '--dataset', type=str, default='iu-x-ray',
                        help='Batch size', choices=AVAILABLE_REPORT_DATASETS)
    parser.add_argument('-bs', '--batch_size', type=int, default=20,
                        help='Batch size to use')
    parser.add_argument('-k', '--k-first', type=int, default=100,
                        help='Top k selected for common-words and common-sentences')
    parser.add_argument('--constant-version', type=str, default='iu',
                        choices=AVAILABLE_CONSTANT_VERSIONS,
                        help='Constant: version to use')
    parser.add_argument('--cnn-run-name', type=str, default=None,
                        help='MostSimilarImage: cnn run name to use as feature extractor')
    parser.add_argument('--cnn-run-task', type=str, default='cls',
                        choices=('cls', 'cls-seg', 'cls-spatial'),
                        help='MostSimilarImage: cnn run task to load the feature extractor from')
    parser.add_argument('--cnn-name', type=str, default=None,
                        help='MostSimilarImage: CNN name to create')
    parser.add_argument('--imagenet', action='store_true',
                        help='MostSimilarImage: args to CNN')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--norm-by-sample', action='store_true',
                        help='Normalize each sample (instead of by dataset stats)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('--no-med', action='store_true',
                        help='If present, do not use medical-correctness metrics')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics')
    parser.add_argument('--reports-version', type=str, default=LATEST_REPORTS_VERSION,
                        help='Reports version to use')
    parsers.add_args_free_values(parser)

    args = parser.parse_args()


    if args.model_name == 'most-similar-image':
        if args.cnn_run_name is None and args.cnn_name is None:
            parser.error('most-similar-image: needs --cnn-run-name or --cnn-name')

    if args.cnn_run_name:
        args.similar_cnn_id = RunId(args.cnn_run_name, False, args.cnn_run_task)
        args.similar_cnn_kwargs = {}
    else:
        args.similar_cnn_id = None
        args.similar_cnn_kwargs = {
            'model_name': args.cnn_name,
            'labels': [],
            'imagenet': args.imagenet,
            'freeze': True,
        }

    parsers.build_args_free_values_(args, parser)

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_dummy_model(
        ARGS.model_name,
        dataset_name=ARGS.dataset,
        dataset_types=ARGS.eval_in,
        reports_version=ARGS.reports_version,
        constant_version=ARGS.constant_version,
        batch_size=ARGS.batch_size,
        norm_by_sample=ARGS.norm_by_sample,
        k_first=ARGS.k_first,
        similar_cnn_id=ARGS.similar_cnn_id,
        similar_cnn_kwargs=ARGS.similar_cnn_kwargs,
        debug=not ARGS.no_debug,
        free_values=ARGS.free_values,
        medical_correctness=not ARGS.no_med,
        device=DEVICE,
        max_samples=ARGS.max_samples,
        quiet=ARGS.quiet,
    )
