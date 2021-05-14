import logging
import torch

from medai.datasets.common import CHEXPERT_DISEASES
from medai.models.report_generation.templates import chex_v1
from medai.utils.nlp import END_OF_SENTENCE_IDX, ReportReader

_TEMPLATE_SETS = {
    'chex-v1': chex_v1.TEMPLATES_CHEXPERT_v1,
}

AVAILABLE_TEMPLATE_SETS = list(_TEMPLATE_SETS)

LOGGER = logging.getLogger(__name__)


class TemplateRGModel:
    """Transforms classification scores into fixed templates."""
    def __init__(self, given_templates, diseases, vocab):
        report_reader = ReportReader(vocab)

        self.diseases = list(diseases)

        # Filter given-diseases
        self.templates = dict()

        for disease in diseases:
            if disease not in given_templates:
                raise Exception(f'Disease {disease} not found in templates')

            if not set([0, 1]).issubset(set(given_templates[disease].keys())):
                raise Exception(f'Values missing for {disease}: {given_templates[disease].keys()}')

            self.templates[disease] = dict()
            for value, template in given_templates[disease].items():
                for token in template.split():
                    if token not in vocab:
                        raise Exception(f'Template token not in vocab: {token}')

                if len(template) == 0:
                    continue

                template_as_idxs = report_reader.text_to_idx(template)

                if len(template_as_idxs) == 0 or template_as_idxs[-1] != END_OF_SENTENCE_IDX:
                    template_as_idxs.append(END_OF_SENTENCE_IDX)

                self.templates[disease][value] = template_as_idxs


    def __call__(self, labels, threshold=0.5):
        # labels shape: batch_size, n_diseases

        labels = (labels >= threshold).type(torch.uint8)
        # same shape

        reports = []
        for sample_predictions in labels:
            # shape: n_diseases

            report = []
            for disease_name, value in zip(self.diseases, sample_predictions):
                value = value.item()
                report.extend(self.templates[disease_name].get(value, []))

            reports.append(report)

        return reports


def create_rg_template_model(name, diseases, vocab):
    if name not in _TEMPLATE_SETS:
        raise Exception(f'Template set not found: {name}')

    LOGGER.info('Creating RG-template from %s', name)
    templates = _TEMPLATE_SETS[name]

    model = TemplateRGModel(templates, diseases, vocab)
    return model
