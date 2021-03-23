import os


class CocoResultsWriter:
    """Writes BB predictions to a csv in the challenge specification."""
    def __init__(self, temp_fpath):
        self.filepath = temp_fpath
        self.file_pointer = None

    def _write_header(self):
        header = 'image_id,PredictionString\n'
        self.file_pointer.write(header)

    def reset(self):
        self.close()

        if os.path.isfile(self.filepath):
            os.remove(self.filepath)

        self.file_pointer = open(self.filepath, 'a')

        self._write_header()

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