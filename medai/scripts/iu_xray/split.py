import random
import json
import os

DATASET_DIR = os.environ['DATASET_DIR_IU_XRAY']
REPORTS_DIR = os.path.join(DATASET_DIR, 'reports')


def save_list(items, name):
    filepath = os.path.join(REPORTS_DIR, f'{name}.txt')
    with open(filepath, 'w') as f:
        for item in items:
            f.write(f'{item}\n')

    print(f'List saved to: {filepath}')


def main(val_split=0.1, test_split=0.1):
    reports_fname = os.path.join(REPORTS_DIR, 'reports.clean.json')
    with open(reports_fname, 'r') as f:
        reports_names = list(json.load(f))

    n_reports = len(reports_names)
    n_val = int(val_split * n_reports)
    n_test = int(test_split * n_reports)
    val_test_reports = random.sample(reports_names, (n_val + n_test))

    val_reports = val_test_reports[:n_val]
    test_reports = val_test_reports[n_val:]

    train_reports = [name for name in reports_names if name not in val_test_reports]

    save_list(train_reports, 'train')
    save_list(val_reports, 'val')
    save_list(test_reports, 'test')



if __name__ == '__main__':
    main()
