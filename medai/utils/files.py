import os
import re
from medai.utils.common import WORKSPACE_DIR

def _get_task_folder(task):
    folder_by_task = {
        'cls': 'classification',
        'rg': 'report_generation',
        'seg': 'segmentation',
        'det': 'detection',
        'cls-seg': 'cls_seg',
        'cls-spatial': 'cls_spatial',
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

        run_folder = os.path.join(parent_folder, run_id.full_name)

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


def _resolve_run_name(run_id, search_in=('runs', 'results', 'models'), workspace_dir=WORKSPACE_DIR):
    """Given a run_id with a timestamp-only run_name, expands the run_name."""

    if _TIMESTAMP_ONLY_REGEX.match(run_id.name):
        matching_runs = []

        for search_folder in search_in:
            assert search_folder in ('runs', 'runs-large', 'models', 'results')

            parent_folder = _get_parent_folder(
                search_folder, run_id.debug, run_id.task, workspace_dir=workspace_dir,
            )

            # Search the run with timestamp
            if not os.path.isdir(parent_folder):
                continue

            matching_runs = [
                saved_run
                for saved_run in os.listdir(parent_folder)
                if saved_run.startswith(run_id.name)
            ]

            if len(matching_runs) > 0:
                break

        assert len(matching_runs) < 2, f'Found more than one run: {matching_runs}'
        assert len(matching_runs) > 0, f'Could not find any run in {search_in}'

        return matching_runs[0]

    return None



class RunId:
    """Class to hold run identification."""
    _experiment_regex = re.compile(r'[\w\-]+__([A-Za-z\-]+)\Z')

    def __init__(self, name='', debug=True, task=None, experiment=''):
        self.name = name
        self.debug = debug
        self.task = task
        self.experiment = experiment or ''

        self.experiment = self.experiment.replace('_', '-')

        self.resolve()

    def __str__(self):
        return f'{self.full_name}\n\t(task={self.task}, exp={self.experiment}, debug={self.debug})'

    def __repr__(self):
        return self.__str__().replace('\n\t', ' ')

    def get_dataset_name(self):
        name_without_timestamp = self.name[12:] # dataset-name is at the front
        lodash_index = name_without_timestamp.index('_')
        dataset_name = name_without_timestamp[:lodash_index]

        if self.task == 'rg':
            # Hardcoded for older runs that do not display dataset-name
            if 'mimic' not in dataset_name:
                dataset_name = 'iu-x-ray'
        return dataset_name

    @property
    def full_name(self):
        if self.experiment:
            suffix = f'__{self.experiment}'
            if not self.name.endswith(suffix):
                return self.name + suffix
        return self.name

    @property
    def short_name(self):
        # timestamp is 11 characters long
        return self.name[:11]

    @property
    def short_clean_name(self):
        return self.short_name.replace('_', '-')

    def resolve(self):
        """Resolve the run-name."""
        resolved_name = _resolve_run_name(self)
        if resolved_name is not None:
            self.name = resolved_name

            exp_found = self._experiment_regex.search(resolved_name)
            if exp_found:
                self.experiment = exp_found.group(1)

        return self

    def to_dict(self):
        return {
            'name': self.name,
            'debug': self.debug,
            'task': self.task,
            'experiment': self.experiment,
        }
