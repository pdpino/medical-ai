from medai.training.report_generation.coatt import get_step_fn_coatt
from medai.training.report_generation.flat import get_step_fn_flat
from medai.training.report_generation.hierarchical import get_step_fn_hierarchical

def get_step_fn_getter(model_name):
    if model_name == 'coatt':
        return get_step_fn_coatt
    if model_name.startswith('h-'):
        return get_step_fn_hierarchical
    return get_step_fn_flat
