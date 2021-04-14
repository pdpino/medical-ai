import os

from medai.utils.files import get_results_folder


def get_outputs_fpath(run_id, dataset_type, prefix='temp', suffix=None):
    folder = get_results_folder(run_id, save_mode=True)

    filename = '-'.join(
        s
        for s in [prefix, f'outputs-{dataset_type}', suffix]
        if s
    )
    # filename is in the form: '{prefix}-outputs-{dtype}-suffix.csv'

    path = os.path.join(folder, f'{filename}.csv')

    return path


class CocoResultsWriter:
    """Writes BB predictions to a csv in the challenge specification."""
    def __init__(self, fpath):
        self.filepath = fpath
        self.file_pointer = None

    def _write_header(self):
        header = 'image_id,PredictionString\n'
        self.file_pointer.write(header)

    def open(self):
        self.file_pointer = open(self.filepath, 'a')

        self._write_header()

    def reset(self):
        self.close()

        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

        self.open()

    def _format_prediction(self, pred):
        return ' '.join(str(v) for v in pred)

    def write(self, output):
        """Writes a line of predictions.

        Args:
            output -- iterator of tuples (image_id, predictions), where
                image_id is a string, and predictions is a list, each prediction is
                a tuple with values: (class_id, score, x_min, y_min, x_max, y_max).
        """
        if self.file_pointer is None:
            return

        for image_id, predictions in output:
            preds_str = ' '.join(
                self._format_prediction(pred)
                for pred in predictions
            )

            line = f'{image_id},{preds_str}\n'
            self.file_pointer.write(line)

    def close(self):
        if self.file_pointer:
            self.file_pointer.close()
