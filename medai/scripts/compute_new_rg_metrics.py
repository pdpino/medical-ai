import argparse
import os
import pandas as pd

from medai.utils import RunId
from medai.utils.files import get_results_folder
from medai.metrics.report_generation.nlp import huggingface
from medai.metrics.report_generation import build_suffix
from medai.metrics import save_results

RUNS = [
    # MIMIC benchmark:
    # '0702_145200', # constant-v2
    # '1112_125550', # top-words-100
    # '1112_131626', # top-sentences-100
    # '0702_150811', # random
    # '1210_212245', # 1-NN
    # '1102_115221', # tpl-abnormal-only
    # '1102_190559', # tpl-single
    # '1129_212630', # tpl-grouped
    ('1202_161321', 'lighter-chex_f1', 0), # show tell
    ('1201_150847', 'lighter-chex_f1', 0), # show attend tell
    # '0703_144847', # CoAtt

    # MIMIC stress tests:
    ## 1102_190559, 1102_115221, 0702_160242, 1104_134722, 1102_205924
    # '0702_160242', # constant-v1
    # '1104_134722', # constant-short
    # '1102_205924', # constant-long

    # IU stress tests:
    # '0612_160823', # constant-v1
    # '1102_213308', # short
    # '1104_134628', # long
    # '0623_142422', # tpl-single
    # '1026_112451', # tpl-abnormal-only

    # IU benchmark
    # '0612_160823', # constant
    # '1103_133310', # top-words-100
    # '1103_133405', # top-sentences-100
    # '0612_160842', # random
    # '1210_212248', # 1-NN
    # '1118_210509', # tpl-single
    # '1130_114158', # tpl-abnormal-only
    # '1118_210821', # tpl-grouped
    ('1119_183609', 'lighter-chex_f1', 0), # show tell
    ('1123_001440', 'lighter-chex_f1', 0), # show attend tell
    # '0623_202003', # CoAtt
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--metric', type=str, default=None, required=True,
                        help='Metric', choices=['bleurt', 'bert'])

    args = parser.parse_args()

    return args

def main(scorer, max_samples=None):
    for run_name in RUNS:
        if isinstance(run_name, tuple):
            run_name, best, beam_size = run_name
        else:
            best = None
            beam_size = None

        run_id = RunId(run_name, False, 'rg')

        suffix = build_suffix(True, best, beam_size)
        reports_path = os.path.join(get_results_folder(run_id), f'outputs-{suffix}.csv')

        if not os.path.isfile(reports_path):
            print('File not found: ', reports_path)
            continue
        reports_df = pd.read_csv(reports_path)
        reports_df = reports_df.loc[reports_df['dataset_type'] == 'test']

        if max_samples is not None:
            reports_df = reports_df.head(max_samples)

        gts = list(reports_df['ground_truth'])
        gens = list(reports_df['generated'])

        # pylint: disable=protected-access
        scorer._hg_metric.add_batch(predictions=gens, references=gts)

        print(f'Computing for {run_id} (n_samples={len(reports_df)})')
        scores, _ = scorer.compute_score()
        # shape: (n_metrics,)

        if scores.ndim > 0:
            results = {
                f'{scorer.metric_name}-{key}': value
                for key, value in zip(scorer.metric_names, scores)
            }
        else:
            results = {
                scorer.metric_name: scores.item(),
            }

        save_results({ "test": results }, run_id, suffix=scorer.metric_name)


if __name__ == '__main__':
    ARGS = parse_args()

    if ARGS.metric == 'bleurt':
        SCORER = huggingface.BLEURT()
    elif ARGS.metric == 'bert':
        SCORER = huggingface.BertScore()
    else:
        raise NameError(f'Metric not found: {ARGS.metric}')

    main(SCORER)
