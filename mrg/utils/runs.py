import os
import json

from mrg.utils.common import WORKSPACE_DIR

RUN_STATE_DIR = os.path.join(WORKSPACE_DIR, 'train_state')

INITIAL_TRAIN_STATE = {
    'current_epoch': 0,
}

def _get_run_state_filepath(run_name, classification=True, debug=True):
    mode_folder = 'classification' if classification else 'report_generation'
    debug_folder = 'debug' if debug else ''
    folder = os.path.join(RUN_STATE_DIR, mode_folder, debug_folder)
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f'{run_name}.json')

    return filepath

class RunState:
    def __init__(self, run_name, classification=True, debug=True):
        self.run_name = run_name
        self.filepath = _get_run_state_filepath(run_name, classification=classification, debug=debug)

        if os.path.isfile(self.filepath):
            with open(self.filepath, 'r') as f:
                self.state = json.load(f)
        else:
            self.state = dict(INITIAL_TRAIN_STATE)

    def save_state(self, current_epoch):
        self.state['current_epoch'] = current_epoch

        with open(self.filepath, 'w') as f:
            json.dump(self.state, f)

    def current_epoch(self):
        return self.state['current_epoch']

    