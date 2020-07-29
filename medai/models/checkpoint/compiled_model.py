class SmartDict:
    """Helper class to store a dict with the state_dict interface."""
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
    """Handles a model and optimizer together."""
    def __init__(self, model, optimizer, metadata=None, epoch=0):
        self.model = model
        self.optimizer = optimizer
        self.metadata = metadata # NOTE: is not persisted here, use save_metadata()

        self.state = SmartDict()
        self.save_current_epoch(epoch)
        
    def save_current_epoch(self, epoch):
        self.state.upsert('current_epoch', epoch)
        
    def get_current_epoch(self):
        return self.state.get('current_epoch')
        
    def get_model_optimizer(self):
        return self.model, self.optimizer
        
    def to_save_checkpoint(self):
        return {
            'model': self.model,
            'optimizer': self.optimizer,
            'state': self.state,
        }
