import argparse
import time
from pprint import pprint
import torch
from torch import nn
from ignite.engine import Engine, Events
import captum
from captum.attr import LayerGradCam
from torch.nn.functional import interpolate

from medai.datasets import prepare_data_classification
from medai.metrics import save_results
from medai.metrics.segmentation import attach_metrics_segmentation
from medai.models.checkpoint import load_compiled_model_classification
from medai.utils import tensor_to_range01, duration_to_str, print_hw_options


class ModelWrapper(nn.Module):
    """Wraps a model to pass it through captum.LayerGradCam."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)[0]
        output = torch.sigmoid(output)
        return output


def _calculate_image_scale(image_size, device='cuda'):
    ORIGINAL_SIZE = 1024
    height, width = image_size
    if height == width:
        return ORIGINAL_SIZE // height

    scale_height = ORIGINAL_SIZE / height
    scale_width = ORIGINAL_SIZE / width

    return torch.tensor((scale_height, scale_width, scale_height, scale_width)).to(device)


def _get_last_layer(compiled_model):
    model_name = compiled_model.metadata['model_kwargs']['model_name']

    model = compiled_model.model
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model = model.module

    if model_name == 'mobilenet':
        # return model.base_cnn.features[-1][0] # Last conv
        return model.base_cnn.features[-1][-1] # Actual last
    if model_name == 'densenet-121':
        # return model.base_cnn.features.denseblock4.denselayer16.conv2 # Last conv
        return model.base_cnn.features.norm5 # Actual last
    if model_name == 'resnet-50':
        # return model.base_cnn.layer4[-1].conv3 # Last conv
        return model.base_cnn.layer4[-1].relu # Actual last

    raise Exception(f'Last layer not hardcoded for: {model_name}')


def threshold_attributions(attributions, thresh=0.5):
    ones = torch.ones(attributions.size()).to(attributions.device)
    zeros = torch.zeros(attributions.size()).to(attributions.device)
    attributions = torch.where(attributions >= thresh, ones, zeros)
    # shape: batch_size, h, w
    
    return attributions


def bbox_coordinates_to_map(bboxes, valid, image_size):
    """Transform bbox (x,y,w,h) pairs to binary maps.
    
    Args:
      bboxes - tensor of shape (batch_size, n_diseases, 4)
      valid - tensor of shape (batch_size, n_diseases)
    Returns:
      tensor of shape (batch_size, n_diseases, height, width)
    """
    batch_size, n_labels = valid.size()
    bboxes_map = torch.zeros(batch_size, n_labels, *image_size).to(bboxes.device)

    for i_batch in range(batch_size):
        for j_label in range(n_labels):
            if valid[i_batch, j_label].item() == 0:
                continue

            min_x, min_y, width, height = bboxes[i_batch, j_label].tolist()
            max_x = min_x + width
            max_y = min_y + height

            bboxes_map[i_batch, j_label, min_y:max_y, min_x:max_x] = 1
            
    return bboxes_map


def calculate_attributions(grad_cam, images, label_index, image_size):
    attributions = grad_cam.attribute(images, label_index).detach()
    # shape: batch_size, 1, layer_h, layer_w
    
    attributions = interpolate(attributions, image_size)
    # shape: batch_size, 1, h, w

    attributions = tensor_to_range01(attributions)
    # shape: batch_size, 1, h, w
    
    attributions = attributions.squeeze(1)
    # shape: batch_size, h, w
    
    return attributions
    

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
    kwargs = {
        'dataset_name': 'cxr14',
        'dataset_type': 'test-bbox',
        'max_samples': max_samples,
        'batch_size': batch_size,
    }
    if image_size is not None:
        kwargs['image_size'] = (image_size, image_size)

    dataloader = prepare_data_classification(**kwargs)
    image_size = dataloader.dataset.image_size
    labels = dataloader.dataset.labels
    
    # Image scaling
    scale = _calculate_image_scale(image_size, device=device)

    # Prepare GradCAM
    wrapped_model = ModelWrapper(compiled_model.model)
    if multiple_gpu:
        wrapped_model = nn.DataParallel(wrapped_model)

    layer = _get_last_layer(compiled_model)
    grad_cam = LayerGradCam(wrapped_model, layer)

    # Create engine
    def step_fn(engine, batch):
        ## Calculate bounding boxes
        images = batch.image.to(device)
        # shape: batch_size, 3, h, w

        bboxes_valid = batch.bboxes_valid.to(device) # shape: batch_size, n_labels
        bboxes = (batch.bboxes.to(device) / scale).long() # shape: batch_size, n_labels, 4
        bboxes_map = bbox_coordinates_to_map(bboxes, bboxes_valid, image_size)
        # shape: batch_size, n_labels, height, width
        
        ## Calculate attributions
        attributions = []
        for index, _ in enumerate(labels):
            attrs = calculate_attributions(grad_cam, images, index, image_size)
            attributions.append(attrs)
        attributions = torch.stack(attributions, dim=1)
        # shape: batch_size, n_labels, h, w
        
        attributions = threshold_attributions(attributions)
        # shape: batch_size, n_labels, h, w

        return (attributions, bboxes_map, bboxes_valid)

    engine = Engine(step_fn)
    attach_metrics_segmentation(engine, len(labels))

    # Run!
    engine.run(dataloader, 1)

    # Prettify metrics
    metrics = {}
    for key, value in engine.state.metrics.items():
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu()
            for v, label in zip(arr, labels):
                metrics[f'{key}-{label}'] = v.item()
        else:
            metrics[key] = value

    if not quiet:
        pprint(metrics)
    save_results(metrics, run_name, classification=True, debug=debug, suffix='grad-cam')


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

    hw_group = parser.add_argument_group('Hardware params')
    hw_group.add_argument('--multiple-gpu', action='store_true',
                          help='Use multiple gpus')
    hw_group.add_argument('--cpu', action='store_true',
                          help='Use CPU only')
    hw_group.add_argument('--num-workers', type=int, default=2,
                          help='Number of workers for dataloader')
    hw_group.add_argument('--num-threads', type=int, default=1,
                          help='Number of threads for pytorch')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)

    device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

    print_hw_options(device, args)

    start_time = time.time()

    run_evaluation(args.run_name,
                   debug=not args.no_debug,
                   device=device,
                   max_samples=args.max_samples,
                   batch_size=args.batch_size,
                   thresh=args.thresh,
                   image_size=args.image_size,
                   quiet=args.quiet,
                   multiple_gpu=False, # args.multiple_gpu # FIXME: not working
                   )

    total_time = time.time() - start_time
    print(f'Total time: {duration_to_str(total_time)}')
    print('=' * 80)
