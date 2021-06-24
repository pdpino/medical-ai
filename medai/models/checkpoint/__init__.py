"""Provide checkpoint save/load functionality."""
import os
import json
import logging
from functools import partial

import torch
from torch import nn
from torch import optim
from ignite.engine import Events
from ignite.handlers import Checkpoint, DiskSaver

from medai.losses.optimizers import create_optimizer
from medai.losses.schedulers import create_lr_sch_handler
from medai.models.classification import create_cnn
from medai.models.report_generation import create_rg_model
from medai.models.segmentation import create_fcn
from medai.models.cls_spatial import create_cls_spatial_model
from medai.models.detection import create_detection_seg_model
from medai.models.cls_seg import create_cls_seg_model
from medai.models.checkpoint.compiled_model import CompiledModel
from medai.models.checkpoint.filenames import get_checkpoint_filepath
from medai.utils.files import get_checkpoint_folder


LOGGER = logging.getLogger(__name__)


def _load_meta(folder, run_id):
    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'run_id' not in data:
        data['run_id'] = run_id.to_dict()

    return data


def load_metadata(run_id):
    """Loads metadata for a run."""
    folder = get_checkpoint_folder(run_id,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    return _load_meta(folder, run_id)


def save_metadata(data, run_id, dryrun=False):
    """Saves run metadata to file."""
    if dryrun:
        LOGGER.warning('Not saving metadata (dryrun)')
        return

    folder = get_checkpoint_folder(run_id,
                                   save_mode=True)

    filepath = os.path.join(folder, 'metadata.json')

    with open(filepath, 'w') as f:
        json.dump(data, f)



def _load_compiled_model_base(run_id,
                              device='cuda',
                              multiple_gpu=False,
                              constructor=None,
                              assert_task=None,
                              mode='best',
                              ):
    """Load a compiled model."""
    assert constructor is not None

    if assert_task is not None:
        assert run_id.task == assert_task

    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_id,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder, run_id)

    # Create empty model and optimizer
    model = constructor(allow_deprecated=True, **metadata['model_kwargs']).to(device)
    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), **metadata['opt_kwargs'])

    # Create LR scheduler
    # TODO: also pass last_epoch?
    # Problem: that info comes from CompiledModel.get_current_epoch() (i.e. its state)
    lr_sch_kwargs = metadata.get('lr_sch_kwargs', None) or {}
    lr_sch_handler = create_lr_sch_handler(optimizer, quiet=True, **lr_sch_kwargs)

    # Compile model
    compiled_model = CompiledModel(run_id, model, optimizer, lr_sch_handler, metadata)

    # Filepath for the latest checkpoint
    filepath = get_checkpoint_filepath(folder, mode=mode)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


load_compiled_model_classification = partial(
    _load_compiled_model_base,
    constructor=create_cnn,
)

load_compiled_model_segmentation = partial(
    _load_compiled_model_base,
    assert_task='seg',
    constructor=create_fcn,
)

load_compiled_model_detection_seg = partial(
    _load_compiled_model_base,
    assert_task='det',
    constructor=create_detection_seg_model,
)

load_compiled_model_cls_seg = partial(
    _load_compiled_model_base,
    assert_task='cls-seg',
    constructor=create_cls_seg_model,
)

load_compiled_model_cls_spatial = partial(
    _load_compiled_model_base,
    assert_task='cls-spatial',
    constructor=create_cls_spatial_model,
)

def load_compiled_model(run_id, **kwargs):
    # TODO: this function should be preferred instead of using a specific one?? refactor?
    _load_fns = {
        'cls': load_compiled_model_classification,
        'cls-seg': load_compiled_model_cls_seg,
        'cls-spatial': load_compiled_model_cls_spatial,
        'det': load_compiled_model_detection_seg,
        'seg': load_compiled_model_segmentation,
        'rg': load_compiled_model_report_generation,
    }

    if run_id.task not in _load_fns:
        raise Exception(f'Task not found in loaders: {run_id.task}')
    load_fn = _load_fns[run_id.task]
    return load_fn(run_id, **kwargs)


def load_compiled_model_report_generation(run_id,
                                          device='cuda',
                                          multiple_gpu=False,
                                          mode='best',
                                          ):
    """Loads a report-generation CNN2Seq model."""
    # TODO: reuse _load_compiled_model_base instead, as the other tasks

    assert run_id.task == 'rg'

    # Folder contains all pertaining files
    folder = get_checkpoint_folder(run_id,
                                   save_mode=False,
                                   assert_exists=True,
                                   )

    # Load metadata
    metadata = _load_meta(folder, run_id)

    # Create model
    model = create_rg_model(**metadata['model_kwargs']).to(device)

    if multiple_gpu:
        # TODO: use DistributedDataParallel instead
        model = nn.DataParallel(model)

    # Create optimizer
    opt_kwargs = metadata['opt_kwargs']
    optimizer = create_optimizer(model, **opt_kwargs)

    # Create LR scheduler
    lr_sch_kwargs = metadata.get('lr_sch_kwargs', None) or {}
    lr_sch_handler = create_lr_sch_handler(optimizer, quiet=True, **lr_sch_kwargs)

    # Compiled model
    compiled_model = CompiledModel(run_id, model, optimizer, lr_sch_handler, metadata)

    # Filepath for the latest checkpoint
    filepath = get_checkpoint_filepath(folder, mode=mode)

    # Actually load data
    checkpoint = torch.load(filepath, map_location=device)
    Checkpoint.load_objects(compiled_model.to_save_checkpoint(), checkpoint)

    return compiled_model


def attach_checkpoint_saver(run_id,
                            compiled_model,
                            trainer,
                            validator,
                            metric=None,
                            dryrun=False,
                            ):
    """Attach a Checkpoint handler to a validator to persist to disk a CompiledModel.

    By default:
    - if metric(s) is (are) provided, only will save best models (not saving last model)
    - if no metrics provided, will save last model
    """
    if dryrun:
        LOGGER.warning('Checkpoint dry run: not saving model to disk')
        return

    def _get_score_kwargs(metric):
        """Given a metric, return extra-kwargs to save a checkpoint with the best metric value."""
        if metric is None:
            return {}

        _SHOULD_IGNORE_WARNING_METRICS = metric.startswith('chex_')

        def score_fn(unused_engine):
            value = validator.state.metrics.get(metric, -1)
            if value == -1:
                if not _SHOULD_IGNORE_WARNING_METRICS:
                    LOGGER.warning(
                        'Checkpoint-saver received %s=-1, will keep with greater_or_equal',
                        metric,
                    )
            if metric == 'loss':
                value = -value
            return value

        return {
            'score_function': score_fn,
            'score_name': metric,
            'greater_or_equal': True,
        }

    if not isinstance(metric, (list, tuple)):
        options = [metric]
    else:
        options = metric

    if len(options) == 0:
        options.append(None)

    if options == [None,]:
        LOGGER.warning(
            'Model checkpoint is not saved by best metric-value (only LAST saved)',
        )
    else:
        LOGGER.info(
            'Saving checkpoints by best/last: %s',
            [o if o is not None else 'LAST' for o in options]
        )


    initial_epoch = compiled_model.get_current_epoch()
    folderpath = get_checkpoint_folder(run_id, save_mode=True)

    for option in options:
        early_kwargs = _get_score_kwargs(option)

        checkpoint = Checkpoint(
            compiled_model.to_save_checkpoint(),
            DiskSaver(folderpath, require_empty=False, atomic=False),
            global_step_transform=lambda _a, _b: trainer.state.epoch + initial_epoch,
            **early_kwargs,
        )

        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint)

    return
