import os
import re

from mrg.utils.common import WORKSPACE_DIR

_EPOCH_REGEX = re.compile(r'\d+')

def get_checkpoint_folder(run_name, classification=True, debug=True):
    mode_folder = 'classification' if classification else 'report_generation'
    debug_folder = 'debug' if debug else ''

    folder = os.path.join(WORKSPACE_DIR, mode_folder, 'models', debug_folder)
    os.makedirs(folder, exist_ok=True)

    folder = os.path.join(folder, run_name)
    return folder

def get_latest_filepath(run_name, classification=True, debug=True):
    find_epoch = lambda fname: int(_EPOCH_REGEX.search(fname).group(0))

    folder = get_checkpoint_folder(run_name, classification=True, debug=True)
    files = [(find_epoch(fname), fname) for fname in os.listdir(folder)]

    latest_epoch, latest_fname = max(files)

    return os.path.join(folder, latest_fname)


class MetadataDict:
    """Helper class to store a dict of metadata with the state_dict interface."""
    def __init__(self):
        self.data = {}
        
    def upsert(self, key, value):
        self.data[key] = value

    def get(self, key, default_value=0):
        return self.data.get(key, default_value)
        
    def state_dict(self):
        return self.data
    
    def load_state_dict(self, new_data):
        self.data = dict(new_data)


class CompiledModel:
    """Stores a model and optimizer together."""
    def __init__(self, model, optimizer, epoch=0):
        self.model = model
        self.optimizer = optimizer
        
        # Init metadata
        self.metadata = MetadataDict()
        self.save_current_epoch(epoch)
        
    def save_current_epoch(self, epoch):
        self.metadata.upsert('current_epoch', epoch)
        
    def get_current_epoch(self):
        return self.metadata.get('current_epoch')
        
    def state(self):
        return self.model, self.optimizer
        
    def to_save_checkpoint(self):
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'metadata': self.metadata,
        }

        
        