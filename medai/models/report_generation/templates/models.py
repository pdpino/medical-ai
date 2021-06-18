import numbers

from medai.utils.nlp import END_OF_SENTENCE_IDX, ReportReader

class BaseTemplateRGModel:
    """Base class for template-rg models

    FIXME: class-inheritance may not be ideal here.
    """
    def __init__(self, diseases=[], vocab={}, templates={}, order=None, prefix=None):
        self._report_reader = ReportReader(vocab)
        self._vocab = vocab

        self.diseases = list(diseases)

        self.prefix = self.str_to_idxs(prefix, force_dot=False) if prefix is not None else None

        # Filter given-diseases
        self.templates = dict()

        for disease in diseases:
            if disease not in templates:
                raise Exception(f'Disease {disease} not found in templates')

            if not set([0, 1]).issubset(set(templates[disease].keys())):
                raise Exception(f'Values missing for {disease}: {templates[disease].keys()}')

            self.templates[disease] = dict()
            for value, template in templates[disease].items():
                self.check_template_validity(template)

                if len(template) == 0:
                    continue

                self.templates[disease][value] = self.str_to_idxs(template)

        if order is None:
            self.disease_order = list(range(len(self.templates)))
        else:
            if set(order) != set(self.diseases):
                raise Exception(f'Order contains invalid diseases: {order} vs {self.diseases}')
            self.disease_order = [self.diseases.index(d) for d in order]

    def check_template_validity(self, template):
        assert isinstance(template, str), f'Template must be str, got: {type(template)}'

        for token in template.split():
            if token not in self._vocab:
                raise Exception(f'Template token not in vocab: {token}')


    def str_to_idxs(self, template, force_dot=True):
        template_as_idxs = self._report_reader.text_to_idx(template)

        if force_dot:
            if len(template_as_idxs) == 0 or template_as_idxs[-1] != END_OF_SENTENCE_IDX:
                template_as_idxs.append(END_OF_SENTENCE_IDX)

        return template_as_idxs


class StaticTemplateRGModel(BaseTemplateRGModel):
    """Transforms classification scores into fixed templates."""
    def __call__(self, labels):
        # labels shape: batch_size, n_diseases (binary)

        reports = []
        labels = labels.tolist()

        for sample_predictions in labels:
            # shape: n_diseases

            report = []

            if self.prefix is not None:
                report.extend(self.prefix)

            for disease_index in self.disease_order:
                pred_value = sample_predictions[disease_index]
                disease_name = self.diseases[disease_index]
                sentence = self.templates[disease_name].get(pred_value, [])

                report.extend(sentence)

            reports.append(report)

        return reports


class GroupedTemplateRGModel(BaseTemplateRGModel):
    def __init__(self, *args, groups=[], **kwargs):
        super().__init__(*args, **kwargs)

        self.groups = []
        for group_diseases, target, template in groups:
            self.check_template_validity(template)

            if isinstance(group_diseases, str):
                group_diseases = (group_diseases,)

            if isinstance(target, numbers.Number):
                targets = [target] * len(group_diseases)
            elif isinstance(target, (list, tuple)):
                targets = target
            else:
                raise Exception(f'Internal error: target type not recognized {type(target)}')

            assert len(targets) == len(group_diseases)

            self.groups.append((group_diseases, targets, self.str_to_idxs(template)))


    def __call__(self, labels):
        reports = []

        labels = labels.tolist()

        for sample_predictions in labels:
            report = []

            if self.prefix is not None:
                report.extend(self.prefix)

            # First, fill the report with the groups
            preds_by_disease = dict(zip(self.diseases, sample_predictions))
            covered_diseases = set()

            for group_diseases, targets, sentence in self.groups:
                if all(preds_by_disease[d] == t for d, t in zip(group_diseases, targets)):
                    report.extend(sentence)
                    for d in group_diseases:
                        covered_diseases.add(d)

            # Last, fill the report with the diseases that were not covered in the groups
            for disease_index in self.disease_order:
                pred_value = sample_predictions[disease_index]
                disease_name = self.diseases[disease_index]
                if disease_name in covered_diseases:
                    continue

                sentence = self.templates[disease_name].get(pred_value, [])
                report.extend(sentence)

            reports.append(report)

        return reports
