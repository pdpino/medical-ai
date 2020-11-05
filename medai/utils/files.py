import os

from medai.utils.common import WORKSPACE_DIR

def _get_task_folder(task):
    folder_by_task = {
        'cls': 'classification',
        'rg': 'report_generation',
        'seg': 'segmentation',
    }
    if task not in folder_by_task:
        raise Exception(f'Task does not exist: {task}')

    return folder_by_task[task]


def _build_dir_getter(folder):
    def _get_run_folder(run_name, task='cls', debug=True,
                        workspace_dir=WORKSPACE_DIR,
                        save_mode=False,
                        assert_exists=False,
                        ):
        debug_folder = 'debug' if debug else ''
        task_folder = _get_task_folder(task)

        run_folder = os.path.join(
            workspace_dir,
            task_folder,
            folder,
            debug_folder,
            run_name,
        )

        if save_mode:
            os.makedirs(run_folder, exist_ok=True)
        elif assert_exists:
            assert os.path.isdir(run_folder), f'Run folder does not exist: {run_folder}'

        return run_folder

    return _get_run_folder


get_tb_log_folder = _build_dir_getter('runs')
get_results_folder = _build_dir_getter('results')
get_checkpoint_folder = _build_dir_getter('models')