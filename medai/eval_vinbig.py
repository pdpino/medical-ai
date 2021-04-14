import argparse
import logging
from pprint import pprint
from functools import partial

import torch
from ignite.engine import Engine

from medai.datasets import prepare_data_classification, UP_TO_DATE_MASKS_VERSION
from medai.metrics import save_results
from medai.metrics.detection import attach_mAP_coco
from medai.metrics.detection.coco_writer import attach_vinbig_writer
from medai.models.checkpoint import (
    load_compiled_model_detection_seg,
)
from medai.training.detection.h2bb import get_h2bb_method, AVAILABLE_H2BB_METHODS
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    RunId,
)


LOGGER = logging.getLogger('medai.det.eval')

def simple_step_fn(unused_engine, data_batch, model=None, device='cuda'):
    # Move inputs to GPU
    images = data_batch.image.to(device)
    # shape: batch_size, channels=3, height, width

    gt_labels = data_batch.labels.to(device)
    # shape(multilabel=True): batch_size, n_labels

    gt_masks = data_batch.masks.to(device)
    # shape: batch_size, n_diseases, height, width

    # Forward
    with torch.no_grad():
        pred_labels, pred_masks = model(images)
        # pred_labels shape: batch_size, n_labels
        # pred_masks shape: batch_size, n_labels, height, width

    pred_labels = torch.sigmoid(pred_labels)
    pred_masks = torch.sigmoid(pred_masks)

    return {
        'pred_labels': pred_labels,
        'gt_labels': gt_labels,
        'activations': pred_masks,
        'gt_activations': gt_masks,
        'image_fnames': data_batch.image_fname,
        'original_sizes': data_batch.original_size,
    }


def h2bb_middleware(original_step_fn, h2bb_names, h2bb_methods):
    def wrapped_step_fn(engine, batch):
        output = original_step_fn(engine, batch)

        batch_heatmaps = output['activations']
        batch_preds = output['pred_labels']
        original_sizes = output['original_sizes']

        for h2bb_name, h2bb_method in zip(h2bb_names, h2bb_methods):
            coco_predictions = h2bb_method(batch_preds, batch_heatmaps, original_sizes)

            output[f'coco_predictions_{h2bb_name}'] = coco_predictions

        return output
    return wrapped_step_fn



def get_method2_forward_fn(cl_model, seg_model):
    def forward_fn(images):
        output_tuple = cl_model(images)
        pred_labels = output_tuple[0]
        # shape: batch_size, n_labels

        pred_labels = torch.sigmoid(pred_labels)

        pred_masks = seg_model(images)
        # shape: batch_size, n_labels, height, width

        pred_masks = torch.sigmoid(pred_masks)

        return pred_labels, pred_masks

    return forward_fn


def get_model(args, device='cuda'):
    # TODO: rename to: cls-seg, ensemble-simple, ensemble-multiple

    if args.method == 'method1':
        run_id = RunId(args.run_name, args.debug, 'det').resolve()
        compiled_model = load_compiled_model_detection_seg(
            run_id,
            device=device,
            multiple_gpu=False,
        )
        compiled_model.model.eval()
        dataset_kwargs = compiled_model.metadata['dataset_kwargs']

        return compiled_model.model, dataset_kwargs, run_id

    if args.method == 'method2':
        raise NotImplementedError()
        # # Load CL model
        # compiled_model = load_compiled_model_classification(
        #     args.cl_run_name,
        #     debug=args.debug,
        #     task='cls',
        #     device=device,
        #     multiple_gpu=False,
        # )
        # dataset_kwargs = compiled_model.metadata['dataset_kwargs']
        # cl_model = compiled_model.model
        # cl_model.eval()

        # compiled_model = load_compiled_model_segmentation(
        #     args.seg_run_name,
        #     debug=args.debug,
        #     device=device,
        #     multiple_gpu=False,
        # )
        # seg_model = compiled_model.model
        # seg_model.eval()

        # forward_fn = get_method2_forward_fn(cl_model, seg_model)

        # run_details = (args.run_name, args.debug, 'cls')

        # return forward_fn, dataset_kwargs, run_details

    if args.method == 'method3':
        raise NotImplementedError()

    raise Exception(f'Method not valid: {args.method}')


@timeit_main(LOGGER)
def evaluate_run(args, device='cuda'):
    """Evaluate a model."""
    model_forward_fn, dataset_kwargs, run_id = get_model(args, device)

    # TODO: implement override
    # if not override and are_results_saved(run_name, task=task, debug=debug):
    #     LOGGER.warning('Skipping run, already calculated')
    #     return {}

    dataset_kwargs.update({
        'dataset_name': 'vinbig',
        'masks': True,
        'masks_version': UP_TO_DATE_MASKS_VERSION,
        # 'fallback_organs': True,
        'labels': None,
        'max_samples': args.max_samples,
    })

    if args.batch_size is not None:
        dataset_kwargs['batch_size'] = args.batch_size

    dataloaders = [
        prepare_data_classification(dataset_type=dataset_type, **dataset_kwargs)
        for dataset_type in args.eval_in
    ]

    metrics = {}

    # Choose h2bb method
    h2bb_methods = []
    h2bb_names = []
    for method_name in AVAILABLE_H2BB_METHODS:
        for cls_thresh in args.cls_threshs:
            for heat_thresh in args.heat_threshs:
                h2bb_names.append(f'{method_name}-{cls_thresh}-{heat_thresh}')
                h2bb_methods.append(get_h2bb_method(method_name, {
                    'cls_thresh': cls_thresh,
                    'heat_thresh': heat_thresh,
                    'norm_heat': True,
                }))

    step_fn = partial(simple_step_fn, model=model_forward_fn, device=device)
    step_fn = h2bb_middleware(step_fn, h2bb_names, h2bb_methods)

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        dataset_type = dataloader.dataset.dataset_type
        LOGGER.info('Evaluating model in %s...', dataset_type)

        engine = Engine(step_fn)

        for name in h2bb_names:
            if dataset_type != 'test':
                attach_mAP_coco(engine, dataloader, run_id,
                                suffix=name, device=device)

            attach_vinbig_writer(
                engine, dataloader, run_id, suffix=name,
            )

        engine.run(dataloader, args.epochs)

        metrics[dataset_type] = engine.state.metrics

    save_results(metrics, run_id)

    if not args.quiet:
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
    # parser.add_argument('--task', type=str, default='det', choices=['cls', 'det'],
    #                     help='Task to load the model from')
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

    h2bb_group = parser.add_argument_group('H2BB params')
    h2bb_group.add_argument('--cls-threshs', nargs='+', type=float, default=[0.3],
                            help='CLS thresholds to apply')
    h2bb_group.add_argument('--heat-threshs', nargs='+', type=float, default=[0.8],
                            help='Heatmap thresholds to apply')

    args = parser.parse_args()

    # For now:
    args.method = 'method1'

    args.debug = not args.no_debug

    args.h2bb_method_kwargs = {
        'cls_threshs': args.cls_threshs,
        'heat_threshs': args.heat_threshs,
    }

    return args


if __name__ == '__main__':
    ARGS = parse_args()

    config_logging()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(ARGS, DEVICE)
