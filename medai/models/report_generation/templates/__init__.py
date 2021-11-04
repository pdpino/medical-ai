import logging

from medai.datasets.common import CHEXPERT_DISEASES
from medai.models.report_generation.templates import (
    chex_v1, chex_v2, chex_v3, chex_v4, chex_v5, chex_v6,
    chex_group,
)
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
    'm-chex-grouped-v3': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v3,
    ),
    'm-chex-grouped-v3-prefixed': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v3_woPREFIX,
        prefix=chex_group.PREFIX_mimic,
    ),
    'm-chex-grouped-v4': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v4,
    ),
    'm-chex-grouped-v5': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v5,
        prefix=chex_group.PREFIX_mimic_v2,
    ),
    'm-chex-grouped-v6': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1,
        groups=chex_group.GROUPS_mimic_v6,
    ),
    'chex-v1-gaming-rm-neg': partialclass(
        GroupedTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1_gaming_rm_neg,
        groups=chex_group.GROUPS_no_active_disease,
    ),
    'chex-v1-gaming-dup': partialclass(
        StaticTemplateRGModel,
        templates=chex_v1.TEMPLATES_CHEXPERT_v1_gaming_dup,
    ),
    'chex-v4-syn': partialclass(
        StaticTemplateRGModel,
        templates=chex_v4.TEMPLATES_CHEXPERT_v4_syn,
    ),
    'chex-v4-noisy': partialclass(
        StaticTemplateRGModel,
        templates=chex_v4.TEMPLATES_CHEXPERT_v4_noisy,
    ),
    'chex-v4-dup-noisy': partialclass(
        StaticTemplateRGModel,
        templates=chex_v4.TEMPLATES_CHEXPERT_v4_gaming_dup_noisy,
    ),
    'chex-v4-dup-noisy-inv': partialclass(
        StaticTemplateRGModel,
        templates=chex_v4.TEMPLATES_CHEXPERT_v4_gaming_dup_noisy_inv,
    ),
    'chex-v5-clean': partialclass(
        StaticTemplateRGModel,
        templates=chex_v5.TEMPLATES_CHEXPERT_v5_clean,
    ),
    'chex-v5-verbose': partialclass(
        StaticTemplateRGModel,
        templates=chex_v5.TEMPLATES_CHEXPERT_v5_verbose,
    ),
    'chex-v5-2-verbose': partialclass(
        StaticTemplateRGModel,
        templates=chex_v5.TEMPLATES_CHEXPERT_v5_2_verbose,
    ),
    'chex-v6-minimal': partialclass(
        StaticTemplateRGModel,
        templates=chex_v6.TEMPLATES_CHEXPERT_v6_minimal,
    ),
    'chex-v6-verbose': partialclass(
        StaticTemplateRGModel,
        templates=chex_v6.TEMPLATES_CHEXPERT_v6_verbose,
    ),
    'chex-v6-verbose-amb': partialclass(
        StaticTemplateRGModel,
        templates=chex_v6.TEMPLATES_CHEXPERT_v6_verbose_amb,
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
