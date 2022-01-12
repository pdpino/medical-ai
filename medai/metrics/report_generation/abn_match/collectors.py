from medai.metrics.report_generation.abn_match.matchers import (
    WordCollectorPattern,
)

LUNG_LOCATIONS = WordCollectorPattern(
    'lung',
    'left', 'right',
    'lobe',
    'basal', r'\bbase',
    'basilar', # 'bibasilar',
    'bilateral', 'lateral',
    'lower', 'upper', 'midlung', 'middle', 'central',
    'biapical', 'apex', 'apical',
    'hilar|hilum',
    'costophrenic', 'retrocardiac',
    'lingula', # 'lingular', # lingula*
    'anterior', 'posterior',
)
