import os
import json

def _get_clean_reports_fpath(reports_dir, version):
    return os.path.join(reports_dir, f'reports.clean.{version}.json')


def assert_reports_not_exist(reports_dir, version, override=False):
    fpath = _get_clean_reports_fpath(reports_dir, version)
    if os.path.isfile(fpath):
        msg = f'Reports version {version} already exist, override={override}'
        if override:
            print(msg)
        else:
            raise Exception(msg)

def save_clean_reports(reports_dir, reports_dict, version):
    fpath = _get_clean_reports_fpath(reports_dir, version)

    with open(fpath, 'w') as f:
        json.dump(reports_dict, f)

    print('Saved reports to: ', fpath)
