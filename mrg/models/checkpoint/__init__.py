"""Provide checkpoint save/load functionality."""
import os
import re
import json

import torch
from torch import optim
from torch import nn
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver

from mrg.models.classification import init_empty_model
from mrg.models.checkpoint.compiled_model import CompiledModel
from mrg.utils.common import WORKSPACE_DIR


def _get_checkpoint_folder(run_name, classification=True, debug=True, save_mode=False):
    mode_folder = 'classification' if classification else 'report_generation'
    debug_folder = 'debug' if debug else ''

    folder = os.path.join(WORKSPACE_DIR, mode_folder, 'models', debug_folder)
    folder = os.path.join(folder, run_name)

    if save_mode:
        os.makedirs(folder, exist_ok=True)
    else:
        assert os.path.isdir(folder), f'Run folder does not exist: {folder}'

    return folder


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

    latest_epoch, latest_fname = max(files)

    return os.path.join(folder, latest_fname)


def _load_meta(folder):
    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


def load_metadata(run_name, classification=True, debug=True):
    """Public wrapper to call _load_meta()."""
    folder = _get_checkpoint_folder(run_name, classification=True, debug=debug, save_mode=False)

    return _load_meta(folder)    


def save_metadata(data, run_name, classification=True, debug=True):
    """Saves run metadata to file."""
    folder = _get_checkpoint_folder(run_name, classification=classification, debug=debug,
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
    """Load a compiled model.
    
    NOTE: only works for classification models for now, missing: init_empty_model()
    """
    # Folder contains all pertaining files
    folder = _get_checkpoint_folder(run_name, classification=True, debug=debug, save_mode=False)

    # Load metadata
    metadata = _load_meta(folder)

    # Create empty model and optimizer
    model = init_empty_model(**metadata['model_kwargs']).to(device)
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



def attach_checkpoint_saver(run_name,
                            compiled_model,
                            engine,
                            classification=True,
                            debug=True,
                            epoch_freq=1,
                            dryrun=False,
                            ):
    """Attach a Checkpoint handler to an engine to persist to disk a CompiledModel."""
    if dryrun:
        return

    initial_epoch = compiled_model.get_current_epoch()

    folderpath = _get_checkpoint_folder(run_name, classification=classification, debug=debug,
                                        save_mode=True)
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