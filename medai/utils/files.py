import os
import re
from collections import namedtuple

from medai.utils.common import WORKSPACE_DIR

def _get_task_folder(task):
    folder_by_task = {
        'cls': 'classification',
        'rg': 'report_generation',
        'seg': 'segmentation',
        'det': 'detection',
        'cls-seg': 'cls_seg',
    }
    task = task.lower()
    if task not in folder_by_task:
        raise Exception(f'Task does not exist: {task}')

    return folder_by_task[task]


def _get_parent_folder(folder, debug, task, workspace_dir=WORKSPACE_DIR):
    debug_folder = 'debug' if debug else ''
    task_folder = _get_task_folder(task)

    parent_folder = os.path.join(
        workspace_dir,
        task_folder,
        folder,
        debug_folder,
    )

    return parent_folder



_TIMESTAMP_ONLY_REGEX = re.compile(r'^\d{4}_\d{6}$')


def _build_dir_getter(folder):
    def _get_run_folder(run_id,
                        workspace_dir=WORKSPACE_DIR,
                        save_mode=False,
                        assert_exists=False,
                        ):
        """Returns the run folder.

        Args:
            run_id -- Resolved RunId.
            save_mode -- If True, ensures the folder is created
            assert_exists -- If True raise an Exception if the folder does not exist.
                (Notice if save_mode is True, the folder will be created, and no
                exception will be raised).
        """
        parent_folder = _get_parent_folder(
            folder, run_id.debug, run_id.task, workspace_dir=workspace_dir,
        )

        run_folder = os.path.join(parent_folder, run_id.name)

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


def _resolve_run_name(run_id, search_in='runs', workspace_dir=WORKSPACE_DIR):
    """Given a run_id with a timestamp-only run_name, expands the run_name."""
    assert search_in in ('runs', 'runs-large', 'models', 'results')

    if _TIMESTAMP_ONLY_REGEX.match(run_id.name):
        parent_folder = _get_parent_folder(
            search_in, run_id.debug, run_id.task, workspace_dir=workspace_dir,
        )

        # Search the run with timestamp
        if not os.path.isdir(parent_folder):
            raise FileNotFoundError(f'No folder to search run: {parent_folder}')
        matching_runs = [
            saved_run
            for saved_run in os.listdir(parent_folder)
            if saved_run.startswith(run_id.name)
        ]
        assert len(matching_runs) == 1, f'Matching runs != 1: {matching_runs}'

        return matching_runs[0]

    return None


class RunId(namedtuple('RunId', ['name', 'debug', 'task'])):
    """Class to hold run identification."""

    def __str__(self):
        return f'{self.name} (task={self.task}, debug={self.debug})'

    def resolve(self):
        """Resolve the run-name."""
        resolved_name = _resolve_run_name(self)
        if resolved_name is not None:
            return self._replace(name=resolved_name)

        return self

RunId.__new__.__defaults__ = (None, True, None)
