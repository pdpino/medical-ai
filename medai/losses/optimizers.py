import logging
from torch import optim


LOGGER = logging.getLogger(__name__)

def _filter_params_names(model, include):
    assert isinstance(include, str)
    included = []
    for name, _ in model.named_parameters():
        if include in name:
            included.append(name)

    return included

def _iter_params_exclude(model, exclude):
    assert isinstance(exclude, list)

    for name, param in model.named_parameters():
        if name in exclude:
            LOGGER.debug('Excluding %s', name)
            continue
        yield param

def _iter_params_include(model, include):
    assert isinstance(include, list), f'include must be list, got {type(include)}'
    for name, param in model.named_parameters():
        if name in include:
            yield param


def create_optimizer(model, custom_lr=None, **kwargs):
    _log_info = dict(kwargs)

    if custom_lr is None:
        params = model.parameters()
    else:
        names_with_custom_lr = []
        params = []

        for param_name, lr_value in custom_lr.items():
            if lr_value is None:
                continue

            included_params = _filter_params_names(model, param_name)
            if len(included_params) == 0:
                LOGGER.warning('Could not find params that matched %s', param_name)
                continue

            params.append(
                { 'params': _iter_params_include(model, included_params), 'lr': lr_value }
            )
            names_with_custom_lr.extend(included_params)

            for p in included_params:
                _log_info[f'lr_{p}'] = lr_value

        if len(params) == 0:
            # No custom-lr provided
            params = model.parameters()
        else:
            # Append the rest of the params
            params.append(
                { 'params': _iter_params_exclude(model, names_with_custom_lr) }
            )

    # Log info
    info_str = ' '.join(f"{k}={v}" for k, v in _log_info.items())
    LOGGER.info('Creating optimizer: %s', info_str)

    # Create optimizer
    optimizer = optim.Adam(params, **kwargs)

    return optimizer
