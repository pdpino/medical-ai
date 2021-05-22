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
    def __init__(self, run_id, model, optimizer, lr_sch_handler=None, metadata=None, epoch=0):
        self.run_id = run_id
        self.model = model
        self.optimizer = optimizer
        self.lr_sch_handler = lr_sch_handler
        self.metadata = metadata # NOTE: is not persisted here, use save_metadata()

        self.state = SmartDict()
        self.save_current_epoch(epoch)

    def save_current_epoch(self, epoch):
        self.state.upsert('current_epoch', epoch)

    def get_current_epoch(self):
        return self.state.get('current_epoch')

    def get_elements(self):
        return self.model, self.optimizer, self.lr_sch_handler

    def to_save_checkpoint(self):
        d = {
            'model': self.model,
            'optimizer': self.optimizer,
            'state': self.state,
        }
        if self.lr_sch_handler is not None and self.lr_sch_handler.scheduler is not None:
            d['lr_scheduler'] = self.lr_sch_handler.scheduler
        return d
