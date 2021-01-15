import argparse
import logging
from pprint import pprint
import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_classification
from medai.metrics import save_results
from medai.metrics.segmentation import attach_metrics_image_saliency
from medai.models.checkpoint import load_compiled_model_classification
from medai.training.classification.grad_cam import create_grad_cam, get_step_fn
from medai.utils import (
    config_logging,
    print_hw_options,
    parsers,
    timeit_main,
)

config_logging()
LOGGER = logging.getLogger('cl-eval-saliency')
LOGGER.setLevel(logging.INFO)


@timeit_main(LOGGER)
def run_evaluation(run_name,
                   debug=True,
                   device='cuda',
                   max_samples=None,
                   batch_size=10,
                   thresh=0.5,
                   image_size=None,
                   quiet=False,
                   multiple_gpu=False,
                   ):
    # Load model
    compiled_model = load_compiled_model_classification(run_name,
                                                        debug=debug,
                                                        device=device,
                                                        multiple_gpu=False,
                                                        )

    # Load data
    dataset_type = 'test-bbox'
    kwargs = {
        'dataset_name': 'cxr14',
        'dataset_type': dataset_type,
        'labels': compiled_model.metadata.get('dataset_kwargs', {}).get('labels', None),
        'max_samples': max_samples,
        'batch_size': batch_size,
        'masks': True,
    }
    if image_size is not None:
        kwargs['image_size'] = (image_size, image_size)

    dataloader = prepare_data_classification(**kwargs)
    labels = dataloader.dataset.labels

    # Prepare GradCAM
    grad_cam = create_grad_cam(compiled_model, device, multiple_gpu)

    # Create engine
    engine = Engine(get_step_fn(grad_cam, labels, thresh=thresh, device=device))
    keys = [
        ('bbox', 'bboxes_map', 'bboxes_valid'),
        ('masks', 'masks', None),
    ]
    attach_metrics_image_saliency(engine, labels, keys, multilabel=True, device=device)

    # Run!
    engine.run(dataloader, 1)

    metrics = engine.state.metrics

    if not quiet:
        pprint(metrics)

    metrics = { dataset_type: metrics }
    save_results(metrics, run_name, task='cls', debug=debug, suffix='grad-cam')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='Select run name to evaluate')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch-size', type=int, default=20,
                        help='Batch size')
    parser.add_argument('--thresh', type=float, default=0.5,
                        help='Threshold for Grad-CAM activations')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--quiet', action='store_true',
                        help='If present, do not print metrics to stdout')

    images_group = parser.add_argument_group('Images params')
    images_group.add_argument('--image-size', type=int, default=512,
                              help='Image size in pixels')

    parsers.add_args_hw(parser, num_workers=2)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    if ARGS.num_threads > 0:
        torch.set_num_threads(ARGS.num_threads)

    DEVICE = torch.device('cuda' if not ARGS.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(DEVICE, ARGS)

    if ARGS.multiple_gpu:
        LOGGER.warning('--multiple-gpu option is not working with Grad-CAM')

    run_evaluation(ARGS.run_name,
                   debug=not ARGS.no_debug,
                   device=DEVICE,
                   max_samples=ARGS.max_samples,
                   batch_size=ARGS.batch_size,
                   thresh=ARGS.thresh,
                   image_size=ARGS.image_size,
                   quiet=ARGS.quiet,
                   multiple_gpu=False, # ARGS.multiple_gpu, # FIXME: not working
                   )

    # --multiple-gpu error:
    # RuntimeError: All input tensors must be on the same device. Received cuda:0 and cuda:1
