import logging

from medai.datasets.common import CHEXPERT_DISEASES
from medai.models.report_generation.templates import chex_v1, chex_v2, chex_v3, chex_group
from medai.models.report_generation.templates.models import (
    StaticTemplateRGModel,
    GroupedTemplateRGModel,
)
from medai.utils import partialclass


_TEMPLATE_SETS = {
    'chex-v1': partialclass(StaticTemplateRGModel, templates=chex_v1.TEMPLATES_CHEXPERT_v1),
    'chex-v1-noisy': partialclass(
        StaticTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1_noisy,
    ),
    'chex-v2': partialclass(StaticTemplateRGModel, templates=chex_v2.TEMPLATES_CHEXPERT_v2),
    'chex-v3': partialclass(StaticTemplateRGModel, templates=chex_v3.TEMPLATES_CHEXPERT_v3),
    'chex-v1-grouped': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_v1,
    ),
    'chex-v2-grouped': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_v2,
    ),
    'm-chex-grouped-v1': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v1,
    ),
    'm-chex-grouped-v2': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v2,
    ),
}

AVAILABLE_TEMPLATE_SETS = list(_TEMPLATE_SETS)

LOGGER = logging.getLogger(__name__)


def create_rg_template_model(name, diseases, vocab, **kwargs):
    if name not in _TEMPLATE_SETS:
        raise Exception(f'Template set not found: {name}')

    LOGGER.info('Creating RG-template: name=%s', name)
    ModelClass = _TEMPLATE_SETS[name]

    return ModelClass(diseases=diseases, vocab=vocab, **kwargs)
