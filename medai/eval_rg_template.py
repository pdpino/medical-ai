import argparse
import logging
import numbers
import torch
from torch.utils.data.dataset import Subset
from ignite.engine import Engine

from medai.datasets import prepare_data_report_generation
from medai.datasets.common import LATEST_REPORTS_VERSION
from medai.metrics import save_results, are_results_saved
from medai.metrics.classification.optimize_threshold import load_optimal_threshold
from medai.metrics.report_generation import (
    attach_metrics_report_generation,
    print_rg_metrics,
)
from medai.metrics.report_generation.labeler_correctness import attach_medical_correctness
from medai.metrics.report_generation.writer import (
    attach_report_writer,
    delete_previous_outputs,
)
from medai.models.report_generation.templates import (
    create_rg_template_model,
    AVAILABLE_TEMPLATE_SETS,
)
from medai.models.checkpoint import load_compiled_model, save_metadata
from medai.training.report_generation.flat import clean_gt_reports
from medai.utils import (
    print_hw_options,
    parsers,
    config_logging,
    timeit_main,
    RunId,
    get_timestamp,
)

LOGGER = logging.getLogger('medai.rg.eval')

def _get_step_fn_classify_plus_templates(cl_model, rg_templates_model,
                                         threshold=0.5, device='cuda'):
    """Return a step_fn that uses a CNN and templates to generate a report.

    Args:
        threshold -- number or tensor of shape (n_diseases,)
    """
    def step_fn(unused_engine, data_batch):
        # Move inputs to GPU
        images = data_batch.images.to(device)
        # shape: batch_size, channels=3, height, width

        # labels = data_batch.labels.to(device)
        # labels = labels.float()
        # shape(multilabel=True): batch_size, n_labels

        # Forward
        with torch.no_grad():
            output_tuple = cl_model(images)
        outputs = torch.sigmoid(output_tuple[0])
        # shape: batch_size, n_labels

        labels = (outputs >= threshold).type(torch.uint8)
        # shape: batch_size, n_labels

        # Compute reports with templates
        flat_reports_gen = rg_templates_model(labels)
        # list of lists with reports (word indices)

        reports = data_batch.reports.long()
        # shape: batch_size, max_sentence_len

        flat_reports_gt = clean_gt_reports(reports)
        # list of lists with indices

        return {
            'flat_clean_reports_gen': flat_reports_gen,
            'flat_clean_reports_gt': flat_reports_gt,
        }

    return step_fn


def _evaluate_model_in_dataloader(
        run_id,
        cl_model,
        rg_templates_model,
        dataloader,
        threshold=0.5,
        n_epochs=1,
        medical_correctness=True,
        device='cuda'):
    """Evaluate a report-generation model on a dataloader."""
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        dataset = dataset.dataset # HACK

    vocab = dataset.get_vocab()

    engine = Engine(_get_step_fn_classify_plus_templates(
        cl_model,
        rg_templates_model,
        threshold=threshold,
        device=device,
    ))
    attach_metrics_report_generation(engine, free=True, device=device)
    attach_report_writer(engine, run_id,
                         vocab, assert_n_samples=len(dataset), free=True)

    if medical_correctness:
        attach_medical_correctness(engine, None, vocab, device=device)

    LOGGER.info('Evaluating model in %s', dataset.dataset_type)

    engine.run(dataloader, n_epochs)

    return engine.state.metrics


def evaluate_model_and_save(
        run_id,
        cl_model,
        rg_templates_model,
        dataloaders,
        device='cuda',
        medical_correctness=True,
        threshold=0.5,
        n_epochs=1,
        ):
    """Evaluates a model in ."""
    delete_previous_outputs(run_id, free=True)

    metrics = {}

    for dataloader in dataloaders:
        if dataloader is None:
            continue
        dataset_type = dataloader.dataset.dataset_type
        metrics[dataset_type] = _evaluate_model_in_dataloader(
            run_id,
            cl_model,
            rg_templates_model,
            dataloader,
            threshold=threshold,
            device=device,
            medical_correctness=medical_correctness,
            n_epochs=n_epochs,
        )

    save_results(metrics, run_id, suffix='free')

    return metrics


def _get_diseases_from_cl_metadata(metadata):
    """Return the list of labels used for a model in the metadata."""
    model_kwargs = metadata['model_kwargs']

    for key in ('cl_labels', 'labels'):
        labels = model_kwargs.get(key)
        if labels is not None:
            return labels

    raise Exception(f'Labels not found in metadata: {metadata}')


def _get_threshold(cl_run_id, mode, diseases, device='cuda'):
    if isinstance(mode, numbers.Number):
        # hardcoded threshold
        return mode

    if not isinstance(mode, str):
        raise Exception(f'Threshold type not recognized: {type(mode)} - {mode}')

    thresh_dict = load_optimal_threshold(cl_run_id, mode)

    if not set(diseases).issubset(thresh_dict):
        raise Exception(
            f'Threshold not found for all diseases: diseases={diseases}, thresh={thresh_dict}',
        )

    return torch.FloatTensor([
        thresh_dict[disease]
        for disease in diseases
    ]).to(device)


_BEST_CHEXPERT_ORDER = [
    'Cardiomegaly',
    'Enlarged Cardiomediastinum',
    'Consolidation',
    'Lung Opacity',
    'Atelectasis',
    'Support Devices',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Edema',
    'Lung Lesion',
    'Fracture',
]

def _get_disease_order(order, dataset_name, diseases):
    if not order or order.lower() == 'none':
        return None

    if dataset_name in ('iu-x-ray', 'mini-mimic', 'mimic-cxr'):
        ordered_diseases = _BEST_CHEXPERT_ORDER
        if set(diseases) != set(ordered_diseases):
            raise Exception('Using different set of diseases: ', diseases)
    else:
        raise Exception(f'Order not implemented for dataset {dataset_name}')

    if order == 'best':
        return ordered_diseases
    elif order == 'worst':
        ordered_diseases.reverse()
        return ordered_diseases
    else:
        raise Exception(f'Order not recognized: {order}')


@timeit_main(LOGGER)
def evaluate_run(cl_run_id,
                 template_set,
                 n_epochs=1,
                 debug=True,
                 device='cuda',
                 multiple_gpu=False,
                 batch_size=None,
                 dataset_types=('train','val','test'),
                 medical_correctness=True,
                 max_samples=None,
                 override=False,
                 quiet=False,
                 order='best',
                 reports_version=LATEST_REPORTS_VERSION,
                 ):
    """Evaluates a saved run."""
    # Load CL model
    compiled_model = load_compiled_model(cl_run_id, device=device, multiple_gpu=multiple_gpu)
    compiled_model.model.eval()

    # Extract useful dataset kwargs
    cl_dataset_kwargs = compiled_model.metadata['dataset_kwargs']
    dataset_name = cl_dataset_kwargs['dataset_name']
    norm_by_sample = cl_dataset_kwargs['norm_by_sample']
    frontal_only = cl_dataset_kwargs['frontal_only']
    image_format = cl_dataset_kwargs['image_format']
    image_size = cl_dataset_kwargs['image_size']

    # Extract other useful kwargs
    diseases = _get_diseases_from_cl_metadata(compiled_model.metadata)
    cnn_name = compiled_model.metadata['model_kwargs']['model_name']

    # Define order
    ordered_diseases = _get_disease_order(order, dataset_name, diseases)

    # Create new run_id
    run_name = f'{get_timestamp()}_{dataset_name}'
    run_name += f'_tpl-{template_set}'
    if order:
        run_name += f'-ord{order}'
    run_name += f'_cnn-{cl_run_id.short_clean_name}_{cnn_name}'
    if reports_version != LATEST_REPORTS_VERSION:
        run_name += f'_reports-{reports_version}'
    run_id = RunId(run_name, debug, 'rg')

    LOGGER.info('Evaluating RG-template run %s', run_id)

    # Check if overriding
    if not override and are_results_saved(run_id, suffix='free'):
        LOGGER.info('Already calculated, skipping')
        return

    dataset_kwargs = {
        'dataset_name': dataset_name,
        'norm_by_sample': norm_by_sample,
        'frontal_only': frontal_only,
        'image_format': image_format,
        'image_size': image_size,
        'hierarchical': False,
        'batch_size': batch_size or cl_dataset_kwargs['batch_size'],
        'max_samples': max_samples,
        'masks': False,
        'sentence_embeddings': False,
        'reports_version': reports_version,
    }

    # Create dataloaders
    dataloaders = [
        prepare_data_report_generation(
            dataset_type=dataset_type,
            **dataset_kwargs,
        )
        for dataset_type in dataset_types
    ]

    # Decide threshold
    threshold = _get_threshold(cl_run_id, 'pr', diseases, device=device)

    # Create RG templates model
    rg_templates_kwargs = {
        'name': template_set,
        'diseases': diseases,
        'vocab': dataloaders[0].dataset.get_vocab(),
        'order': ordered_diseases,
    }
    rg_template_model = create_rg_template_model(**rg_templates_kwargs)

    # Save model metadata
    metadata = {
        'rg_templates_kwargs': rg_templates_kwargs,
        'reports_version': reports_version,
    }
    save_metadata(metadata, run_id)

    metrics = evaluate_model_and_save(
        run_id,
        compiled_model.model,
        rg_template_model,
        dataloaders,
        device=device,
        threshold=threshold,
        medical_correctness=medical_correctness,
        n_epochs=n_epochs,
    )

    if not quiet:
        print_rg_metrics(metrics, ignore=diseases)

    LOGGER.info('Finished run: %s', run_id)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--run-name', type=str, default=None, required=True,
                        help='CL run-name to load')
    parser.add_argument('--task', type=str, default='cls', choices=('cls', 'cls-seg'),
                        help='CL run task')
    parser.add_argument('--templates', type=str, default='chex-v1',
                        help='Template set to use', choices=AVAILABLE_TEMPLATE_SETS)
    parser.add_argument('--eval-in', nargs='*', default=['train', 'val', 'test'],
                        help='Eval in datasets')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max samples to load (debugging)')
    parser.add_argument('-bs', '--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='Number of epochs')
    parser.add_argument('--no-debug', action='store_true',
                        help='If is a non-debugging run')
    parser.add_argument('--override', action='store_true',
                        help='Whether to override previous results')
    parser.add_argument('--no-med', action='store_true',
                        help='If present, do not use medical-correctness metrics')
    parser.add_argument('--quiet', action='store_true',
                        help='Do not print metrics to stdout')
    parser.add_argument('--order', type=str, default='best',
                        help='Default order to use')
    parser.add_argument('--reports-version', type=str, default=LATEST_REPORTS_VERSION,
                        help='Specify an reports-version')

    parsers.add_args_hw(parser, num_workers=4)

    args = parser.parse_args()

    args.cl_run_id = RunId(args.run_name, False, args.task)

    return args


if __name__ == '__main__':
    config_logging()

    ARGS = parse_args()

    DEVICE = torch.device('cuda' if torch.cuda.is_available() and not ARGS.cpu else 'cpu')

    print_hw_options(DEVICE, ARGS)

    evaluate_run(cl_run_id=ARGS.cl_run_id,
                 template_set=ARGS.templates,
                 dataset_types=ARGS.eval_in,
                 debug=not ARGS.no_debug,
                 multiple_gpu=ARGS.multiple_gpu,
                 device=DEVICE,
                 medical_correctness=not ARGS.no_med,
                 n_epochs=ARGS.epochs,
                 max_samples=ARGS.max_samples,
                 override=ARGS.override,
                 batch_size=ARGS.batch_size,
                 quiet=ARGS.quiet,
                 order=ARGS.order,
                 reports_version=ARGS.reports_version,
                 )
