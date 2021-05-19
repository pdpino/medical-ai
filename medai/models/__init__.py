import os
import json
import logging

from medai.utils.files import get_checkpoint_folder

LOGGER = logging.getLogger(__name__)

def save_training_stats(run_id,
                        batch_size,
                        epochs,
                        secs_per_epoch,
                        hw_options,
                        initial_epoch=0,
                        dryrun=False,
                        ):
    if dryrun:
        return
    training_stats = {
        'secs_per_epoch': secs_per_epoch,
        'batch_size': batch_size,
        'initial_epoch': initial_epoch,
        'n_epochs': epochs,
        'hw_options': hw_options,
    }
    folder = get_checkpoint_folder(run_id, save_mode=True)
    final_epoch = initial_epoch + epochs
    fpath = os.path.join(folder, f'training-stats-{initial_epoch}-{final_epoch}.json')

    with open(fpath, 'w') as f:
        json.dump(training_stats, f, indent=2)

    LOGGER.debug('Saved training stats to %s', fpath)
