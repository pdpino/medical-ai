import numpy as np

_singleton_loaded_models = {}

# HACK: only import when using it!
DYN_IMPORTS = {}

class HuggingFaceMetric:
    metric_name = "none"
    metric_args = tuple()
    n_metrics = 1

    def __init__(self):
        if self.metric_name not in _singleton_loaded_models:
            self.preload()

        self._hg_metric = _singleton_loaded_models[self.metric_name]

    def update(self, generated, gt):
        self._hg_metric.add(prediction=generated, reference=gt)

    def __iadd__(self, tup):
        assert isinstance(tup, tuple) and len(tup) == 2
        gen, gt_list = tup
        assert len(gt_list) == 1
        self.update(gen, gt_list[0])
        return self

    def compute_score(self):
        scores = np.array(self._hg_metric.compute()['scores'])

        return np.mean(scores), scores

    @classmethod
    def preload(cls):
        if 'evaluate' not in DYN_IMPORTS or DYN_IMPORTS['evaluate'] is None:
            # pylint:disable=import-outside-toplevel
            import evaluate
            DYN_IMPORTS['evaluate'] = evaluate

        evaluate = DYN_IMPORTS.get('evaluate')
        if evaluate is None:
            raise ModuleNotFoundError('evaluate module has not been loaded!')

        if cls.metric_name not in _singleton_loaded_models:
            _singleton_loaded_models[cls.metric_name] = evaluate.load(
                cls.metric_name, *cls.metric_args, module_type="metric",
            )

class BLEURT(HuggingFaceMetric):
    metric_name = 'bleurt'
    metric_args = ("BLEURT-20",)

class BertScore(HuggingFaceMetric):
    metric_name = "bertscore"
    metric_args = ("microsoft/deberta-xlarge-mnli",)
    metric_names = ['prec', 'recall', 'f1']
    n_metrics = 3

    def update(self, generated, gt):
        # bertscore.add() has a bug!
        self._hg_metric.add_batch(predictions=[generated], references=[gt])

    def compute_score(self):
        keys = ['precision', 'recall', 'f1']
        results = self._hg_metric.compute(model_type="microsoft/deberta-xlarge-mnli")
        scores = np.array([results[k] for k in keys]) # n_metrics, n_samples

        return np.mean(scores, axis=1), scores
