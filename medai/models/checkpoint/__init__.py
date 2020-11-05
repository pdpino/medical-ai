"""Provide checkpoint save/load functionality."""
import os
import re
import json

import torch
from torch import optim
from torch import nn
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver

from medai.models.classification import create_cnn
from medai.models.report_generation import create_decoder
from medai.models.report_generation.cnn_to_seq import CNN2Seq
from medai.models.checkpoint.compiled_model import CompiledModel
from medai.utils.files import get_checkpoint_folder


_CHECKPOINT_EPOCH_REGEX = re.compile(r'\d+')

def _get_checkpoint_fname_epoch(fname):
    epoch = _CHECKPOINT_EPOCH_REGEX.search(fname)
    if not epoch:
        # Not a checkpoint file
        return -1

    return int(epoch.group(0))


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


def _load_meta(folder):
    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def load_metadata(run_name, task, debug=True):
    """Public wrapper to call _load_meta()."""
    folder = get_checkpoint_folder(run_name,
                                   task=task,
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    return _load_meta(folder)


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
                                       ):
    """Load a compiled classification model."""
    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_name,
                                   task='cls',
                                   debug=debug,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder)

    # Create empty model and optimizer
    model = create_cnn(**metadata['model_kwargs']).to(device)
    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), **metadata['opt_kwargs'])

    compiled_model = CompiledModel(model, optimizer, metadata)

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
    metadata = _load_meta(folder)
    hparams = metadata.get('hparams', {})

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

    # Compiled model
    compiled_model = CompiledModel(model, optimizer, metadata)

    # Filepath for the latest checkpoint
    filepath = _get_latest_filepath(folder)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


def attach_checkpoint_saver(run_name,
                            compiled_model,
                            engine,
                            task,
                            debug=True,
                            epoch_freq=1,
                            dryrun=False,
                            ):
    """Attach a Checkpoint handler to an engine to persist to disk a CompiledModel."""
    if dryrun:
        return

    initial_epoch = compiled_model.get_current_epoch()

    folderpath = get_checkpoint_folder(run_name,
                                       task=task,
                                       debug=debug,
                                       save_mode=True,
                                       )
    checkpoint = Checkpoint(
        compiled_model.to_save_checkpoint(),
        DiskSaver(folderpath, require_empty=False, atomic=False),
        global_step_transform=lambda eng, _: eng.state.epoch + initial_epoch,
    )
    if epoch_freq is None:
        evt = Events.COMPLETED
    else:
        evt = Events.EPOCH_COMPLETED(every=epoch_freq)

    engine.add_event_handler(evt, checkpoint)

    return