from ignite.engine import Events
from ignite.metrics import MetricUsage

class EveryNEpochs(MetricUsage):
    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

        # pylint: disable=not-callable
        super().__init__(
            started=Events.EPOCH_STARTED(self._every_n_epochs),
            completed=Events.EPOCH_COMPLETED(self._every_n_epochs),
            iteration_completed=Events.ITERATION_COMPLETED(self._every_n_epochs)
        )

    def _every_n_epochs(self, engine, unused_event):
        return engine.state.epoch % self.n_epochs == 0
