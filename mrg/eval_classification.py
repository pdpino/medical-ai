import time
import argparse
import os

import torch

from mrg.datasets import (
    prepare_data_classification,
    AVAILABLE_CLASSIFICATION_DATASETS,
)
from mrg.models.checkpoint import (
    load_metadata,
    load_compiled_model_classification,
)
from mrg.train_classification import evaluate_and_save
from mrg.utils import get_timestamp, duration_to_str


def run_evaluation(run_name,
                   dataset_name,
                   max_samples=None,
                   batch_size=10,
                   n_epochs=1,
                   labels=None,
                   image_size=512,
                   debug=True,
                   multiple_gpu=False,
                   device='cuda',
                   ):
    """Evaluate a model."""
    # Load metadata (contains all configuration)
    metadata = load_metadata(run_name, classification=True, debug=debug)

    # Load data
    dataset_kwargs = {
        'dataset_name': dataset_name,
        'labels': labels,
        'max_samples': max_samples,
        'batch_size': batch_size,
        'image_size': (image_size, image_size),
    }

    train_dataloader = prepare_data_classification(dataset_type='train', **dataset_kwargs)
    val_dataloader = prepare_data_classification(dataset_type='val', **dataset_kwargs)
    test_dataloader = prepare_data_classification(dataset_type='test', **dataset_kwargs)

    # Load stuff from metadata
    hparams = metadata.get('hparams')
    loss_name = hparams.get('loss_name', 'cross-entropy')
    loss_kwargs = hparams.get('loss_kwargs', {})

    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        device=device,
                                                        multiple_gpu=multiple_gpu)

    # Evaluate
    evaluate_and_save(run_name,
                      compiled_model.model,
                      train_dataloader,
                      val_dataloader,
                      test_dataloader,
                      loss_name,
                      loss_kwargs=loss_kwargs,
                      suffix=dataset_name,
                      debug=debug,
                      device=device)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Run-name to load')
    parser.add_argument('-d', '--dataset', type=str, default=None, required=True,
                        choices=AVAILABLE_CLASSIFICATION_DATASETS,
                        help='Choose dataset to evaluate on')
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
                   max_samples=args.max_samples,
                   batch_size=args.batch_size,
                   n_epochs=args.epochs,
                   labels=args.labels,
                   image_size=args.image_size,
                   debug=not args.no_debug,
                   multiple_gpu=args.multiple_gpu,
                   device=device,
                   )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
