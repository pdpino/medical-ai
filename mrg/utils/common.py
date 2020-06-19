from datetime import datetime
import time
import os
# import json
# from pprint import pprint

PAD_TOKEN = 'PAD'
PAD_IDX = 0
END_TOKEN = 'END'
END_IDX = 1
START_TOKEN = 'START'
START_IDX = 2
UNKNOWN_TOKEN = 'UNK'
UNKNOWN_IDX = 3

WORKSPACE_DIR = os.environ['MRG_WORKSPACE_DIR']

def get_timestamp(short=True):
    now = datetime.fromtimestamp(time.time())

    if short:
        return now.strftime('%m%d_%H%M%S')
    return now.strftime('%Y-%m-%d_%H-%M-%S')

def duration_to_str(all_seconds):
    all_seconds = int(all_seconds)
    seconds = all_seconds % 60
    minutes = all_seconds // 60
    hours = minutes // 60
    
    minutes = minutes % 60

    return '{}h {}m {}s'.format(hours, minutes, int(seconds))

def labels_to_str(values, labels, thresh=0.5, sort_values=False, show_value=False):
    positive = [(val, label) for val, label in zip(values, labels) if val > thresh]
    if sort_values:
        positive = sorted(positive, reverse=True)

    if len(positive) > 0:
        format_str = '{1} ({0:.3f})' if show_value else '{1}'
        return ', '.join(format_str.format(val, label) for val, label in positive)
    else:
        return 'No Findings'


def compute_vocab(reports_iterator):
    word_to_idx = {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNKNOWN_TOKEN: UNKNOWN_IDX,
    }

    for report in reports_iterator:
        for token in report:
            if token not in word_to_idx:
                word_to_idx[token] = len(word_to_idx)

    return word_to_idx

# def write_to_txt(arr, fname, sep='\n'):
#     """Writes a list of strings to a file"""
#     with open(fname, 'w') as f:
#         for line in arr:
#             f.write(line + sep)


# def save_hparams(run_name, hparams_dict, experiment_mode="debug", base_dir=BASE_DIR):
#     hparams_dir = os.path.join(base_dir, "hparams", experiment_mode)
#     os.makedirs(hparams_dir, exist_ok=True)
#     fname = f"{hparams_dir}/{run_name}.json"

#     with open(fname, "w") as f:
#         json.dump(hparams_dict, f)
#     print("Saved hparams to: ", fname)


# def load_hparams(run_name, experiment_mode="debug", base_dir=BASE_DIR):
#     hparams_dir = os.path.join(base_dir, "hparams", experiment_mode)
#     fname = f"{hparams_dir}/{run_name}.json"

#     with open(fname, "r") as f:
#         hparams_dict = json.load(f)
#     pprint(hparams_dict)
#     return hparams_dict