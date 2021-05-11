import os
import re
import logging

LOGGER = logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):
    """Custom logging Formatter.

    * Adds color to warning and error messages
    * If is a medai-logger, ignores the logger-name in the message written

    HACK: This formatter instance has two attributes that are also formatter instances.
    Could not find a way to apply conditional formatting with just one formatter instance.

    See here for an explanation on color codes: https://stackoverflow.com/a/33206814/9951939
    """

    separator_re = re.compile(r'\A[-_=]+\Z')
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[45;37m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    datefmt = '%m-%d %H:%M'
    global_formatter = logging.Formatter(
        '[%(name)s] %(levelname)s(%(asctime)s) %(message)s', # '(%(filename)s:%(lineno)d)',
        datefmt=datefmt,
    )
    medai_formatter = logging.Formatter(
        '(%(asctime)s) %(message)s',
        datefmt=datefmt,
    )

    COLORS = {
        # logging.DEBUG: gray,
        # logging.INFO: gray,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        # Choose format
        if record.name.startswith('medai'):
            chosen_formatter = self.medai_formatter
        else:
            chosen_formatter = self.global_formatter
        s = chosen_formatter.format(record)

        # Choose color
        level = record.levelno
        if level == logging.INFO and self.separator_re.search(record.message):
            color = self.purple
        else:
            color = self.COLORS.get(level)

        if color is not None:
            s = color + s + self.reset

        return s


def config_logging(basic_level=logging.WARNING):
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(basic_level)

    # Console handler with color
    ch = logging.StreamHandler()
    ch.setFormatter(CustomFormatter())
    root_logger.addHandler(ch)

    # Own loggers level
    medai_logger = logging.getLogger('medai')
    medai_logger.setLevel(logging.INFO)


def print_hw_options(device, args):
    """Prints hardware options (device selected) and args provided."""
    def _safe_get_attr(obj, attr, default_value=None):
        if not hasattr(obj, attr):
            return default_value
        return getattr(obj, attr)

    _CUDA_VISIBLE = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    d = {
        'device': device,
        'visible': _CUDA_VISIBLE,
        'multiple': _safe_get_attr(args, 'multiple_gpu'),
        'num_workers': _safe_get_attr(args, 'num_workers'),
        'num_threads': _safe_get_attr(args, 'num_threads'),
    }
    info_str = ' '.join(f'{k}={v}' for k, v in d.items())
    LOGGER.info('Using: %s', info_str)
