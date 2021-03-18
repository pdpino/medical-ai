"""Provide checkpoint save/load functionality."""
import os
import re
import json
import logging

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver

from medai.models.classification import create_cnn
from medai.models.report_generation import create_decoder
from medai.models.segmentation import create_fcn
from medai.models.report_generation.cnn_to_seq import CNN2Seq
from medai.models.checkpoint.compiled_model import CompiledModel
from medai.utils.files import get_checkpoint_folder


LOGGER = logging.getLogger(__name__)

_CHECKPOINT_EPOCH_REGEX = re.compile(r'_\d+')

def _get_checkpoint_fname_epoch(fname):
    epoch = _CHECKPOINT_EPOCH_REGEX.search(fname)
    if not epoch:
        # Not a checkpoint file
        return -1

    epoch = epoch.group(0).strip('_')
    return int(epoch)


def _get_latest_filepath(folder):
    files = [
        (_get_checkpoint_fname_epoch(fname), fname)
        for fname in os.listdir(folder)
    ]

    if len(files) == 0:
        raise Exception('Model filepath empty:', folder)

    latest_epoch, latest_fname = max(files)

    if latest_epoch == -1:
        raise Exception('Model filepath not found: ', files)

    return os.path.join(folder, latest_fname)


def _load_meta(folder, run_name):
    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'run_name' not in data:
        data['run_name'] = run_name

    return data


def load_metadata(run_name, task, debug=True):
    """Loads metadata for a run."""
    folder = get_checkpoint_folder(run_name,
                                   task=task,
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    return _load_meta(folder, run_name)


def save_metadata(data, run_name, task, debug=True):
    """Saves run metadata to file."""
    folder = get_checkpoint_folder(run_name,
                                   task=task,
                                   debug=debug,
                                   save_mode=True)

    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'w') as f:
        json.dump(data, f)

    return


def load_compiled_model_classification(run_name,
                                       debug=True,
                                       device='cuda',
                                       multiple_gpu=False,
                                       task='cls',
                                       ):
    """Load a compiled classification model."""
    assert task in ('cls', 'det'), 'Task not available'

    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_name,
                                   task=task,
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder, run_name)

    # Create empty model and optimizer
    model = create_cnn(allow_deprecated=True, **metadata['model_kwargs']).to(device)
    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), **metadata['opt_kwargs'])

    # Create LR scheduler
    lr_scheduler_kwargs = metadata.get('lr_scheduler_kwargs', None)
    if lr_scheduler_kwargs is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    else:
        lr_scheduler = None

    # Compile model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Filepath for the latest checkpoint
    filepath = _get_latest_filepath(folder)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


def load_compiled_model_segmentation(run_name,
                                     debug=True,
                                     device='cuda',
                                     multiple_gpu=False,
                                     ):
    """Load a compiled segmentation model.

    NOTE: function copied from classification models
    """
    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_name,
                                   task='seg',
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder, run_name)

    # Create empty model and optimizer
    model = create_fcn(**metadata['model_kwargs']).to(device)
    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), **metadata['opt_kwargs'])

    # Create LR Scheduler
    lr_scheduler_kwargs = metadata.get('lr_scheduler_kwargs', None)
    if lr_scheduler_kwargs is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    else:
        lr_scheduler = None

    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Filepath for the latest checkpoint
    filepath = _get_latest_filepath(folder)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


def load_compiled_model_report_generation(run_name,
                                          debug=True,
                                          device='cuda',
                                          multiple_gpu=False,
                                          ):
    """Loads a report-generation CNN2Seq model."""
    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_name,
                                   task='rg',
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder, run_name)

    # Create CNN
    cnn_kwargs = metadata.get('cnn_kwargs', None)
    assert cnn_kwargs, 'CNN kwargs are not present in metadata'
    cnn = create_cnn(**cnn_kwargs)

    # Create Decoder
    decoder = create_decoder(**metadata['decoder_kwargs'])

    # Create CNN2Seq model and optimizer
    model = CNN2Seq(cnn, decoder).to(device)
    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    # Create optimizer
    opt_kwargs = metadata['opt_kwargs']
    optimizer = optim.Adam(model.parameters(), **opt_kwargs)

    # Create LR scheduler
    lr_scheduler_kwargs = metadata.get('lr_scheduler_kwargs', None)
    if lr_scheduler_kwargs is not None:
        lr_scheduler = ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    else:
        lr_scheduler = None

    # Compiled model
    compiled_model = CompiledModel(model, optimizer, lr_scheduler, metadata)

    # Filepath for the latest checkpoint
    filepath = _get_latest_filepath(folder)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


def attach_checkpoint_saver(run_name,
                            compiled_model,
                            trainer,
                            validator,
                            task,
                            metric=None,
                            debug=True,
                            dryrun=False,
                            ):
    """Attach a Checkpoint handler to a validator to persist to disk a CompiledModel."""
    if dryrun:
        return

    def score_fn(unused_engine):
        value = validator.state.metrics.get(metric, -1)
        if value == -1:
            LOGGER.warning(
                'Checkpoint-saver received %s=-1, will keep oldest checkpoint',
                metric,
            )
        if metric == 'loss':
            value = -value
        return value

    if metric is not None:
        early_kwargs = {
            'score_function': score_fn,
            'score_name': metric,
            'greater_or_equal': True,
        }
        LOGGER.info('Saving checkpoint by best "%s" value', metric)
    else:
        LOGGER.warning('Model checkpoint is not saved by best metric value (not provided)')
        early_kwargs = {}

    initial_epoch = compiled_model.get_current_epoch()

    folderpath = get_checkpoint_folder(run_name,
                                       task=task,
                                       debug=debug,
                                       save_mode=True,
                                       )
    checkpoint = Checkpoint(
        compiled_model.to_save_checkpoint(),
        DiskSaver(folderpath, require_empty=False, atomic=False),
        global_step_transform=lambda _a, _b: trainer.state.epoch + initial_epoch,
        **early_kwargs,
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    return
