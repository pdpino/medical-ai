import os
import re

from medai.utils.common import WORKSPACE_DIR

def _get_task_folder(task):
    folder_by_task = {
        'cls': 'classification',
        'rg': 'report_generation',
        'seg': 'segmentation',
    }
    task = task.lower()
    if task not in folder_by_task:
        raise Exception(f'Task does not exist: {task}')

    return folder_by_task[task]


_TIMESTAMP_ONLY_REGEX = re.compile(r'^\d{4}_\d{6}$')

def _build_dir_getter(folder):
    def _get_run_folder(run_name, task='cls', debug=True,
                        workspace_dir=WORKSPACE_DIR,
                        save_mode=False,
                        assert_exists=False,
                        ):
        """Returns the run folder.

        Args:
            run_name -- str with the full run-name, or the timestamp
                (i.e. DATE_HOUR format).
        """
        debug_folder = 'debug' if debug else ''
        task_folder = _get_task_folder(task)

        parent_folder = os.path.join(
            workspace_dir,
            task_folder,
            folder,
            debug_folder,
        )

        if _TIMESTAMP_ONLY_REGEX.match(run_name):
            # Search the run with timestamp
            if not os.path.isdir(parent_folder):
                raise FileNotFoundError(f'No folder to search run: {parent_folder}')
            matching_runs = [
                saved_run
                for saved_run in os.listdir(parent_folder)
                if saved_run.startswith(run_name)
            ]
            assert len(matching_runs) == 1, f'Matching runs != 1: {matching_runs}'
            run_name = matching_runs[0]

        run_folder = os.path.join(parent_folder, run_name)

        if save_mode:
            os.makedirs(run_folder, exist_ok=True)
        elif assert_exists:
            assert os.path.isdir(run_folder), f'Run folder does not exist: {run_folder}'

        return run_folder

    return _get_run_folder


get_tb_log_folder = _build_dir_getter('runs')
get_tb_large_log_folder = _build_dir_getter('runs-large')
get_results_folder = _build_dir_getter('results')
get_checkpoint_folder = _build_dir_getter('models')
