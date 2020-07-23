import time
import argparse
import os
from pprint import pprint

import torch

from mrg.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
)
from mrg.models.checkpoint import load_compiled_model_classification
from mrg.train_classification import evaluate_and_save
from mrg.utils import get_timestamp, duration_to_str


def run_evaluation(run_name,
                   dataset_name,
                   eval_in=['train', 'val', 'test'],
                   max_samples=None,
                   batch_size=10,
                   n_epochs=1,
                   frontal_only=False,
                   labels=None,
                   image_size=512,
                   debug=True,
                   multiple_gpu=False,
                   device='cuda',
                   ):
    """Evaluate a model."""
    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        device=device,
                                                        multiple_gpu=multiple_gpu)

    # Metadata (contains all configuration)
    metadata = compiled_model.metadata

    # Load data
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'labels': labels,
        'max_samples': max_samples,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
        'frontal_only': frontal_only,
    }

    dataloaders = [
        prepare_data_classification(dataset_type=dataset_type, **dataset_kwargs)
        for dataset_type in eval_in
    ]

    # Load stuff from metadata
    hparams = metadata.get('hparams')
    loss_name = hparams.get('loss_name', 'cross-entropy')
    loss_kwargs = hparams.get('loss_kwargs', {})

    # Evaluate
    suffix = f'{dataset_name}_size{image_size}'
    if frontal_only:
        suffix += '_frontal'

    metrics = evaluate_and_save(run_name,
                                compiled_model.model,
                                dataloaders,
                                loss_name,
                                loss_kwargs=loss_kwargs,
                                suffix=suffix,
                                debug=debug,
                                device=device)
    
    pprint(metrics)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run-name to load')
    parser.add_argument('-d', '--dataset', type=str, default=None, required=True,
                        choices=AVAILABLE_CLASSIFICATION_DATASETS,
                        help='Choose dataset to evaluate on')
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--image-size', type=int, default=512,
                        help='Image size in pixels')
    parser.add_argument('--labels', type=str, nargs='*', default=None,
                        help='Subset of labels')
    parser.add_argument('--frontal-only', action='store_true',
                        help='Use only frontal images')
    parser.add_argument('--multiple-gpu', action='store_true',
                        help='Use multiple gpus')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU only')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    _CUDA_AVAIL = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    print('Using device: ', device, _CUDA_AVAIL)

    start_time = time.time()

    run_evaluation(args.run_name,
                   args.dataset,
                   eval_in=args.eval_in,
                   max_samples=args.max_samples,
                   batch_size=args.batch_size,
                   frontal_only=args.frontal_only,
                   n_epochs=args.epochs,
                   labels=args.labels,
                   image_size=args.image_size,
                   debug=not args.no_debug,
                   multiple_gpu=args.multiple_gpu,
                   device=device,
                   )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
