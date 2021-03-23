import operator
from ignite.engine import Events
from ignite.metrics import MetricUsage, MetricsLambda

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


def attach_metric_for_labels(engine, labels, metric, metric_name, average=True):
    """Attaches a metric to an engine for multiple labels.

    Expects a metric that returns an array/tensor of values, one for each label.
    Args:
        engine -- to attach the metric to.
        labels -- array/list of labels/diseases
        metric -- Metric object, which returns an array of values of size n_diseases.
        metric_name -- metrics will be named as: "{metric_name}-{label}"
        average -- if True, it will also attach the macro average of the metrics.
    """
    for index, label in enumerate(labels):
        metric_for_label_i = MetricsLambda(operator.itemgetter(index), metric)
        metric_for_label_i.attach(engine, f'{metric_name}-{label}')

    if average:
        metric_average = MetricsLambda(lambda x: x.mean().item(), metric)
        metric_average.attach(engine, metric_name)