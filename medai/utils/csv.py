import os

class CSVWriter:
    def __init__(self, filepath, columns=None, assert_folder=True):
        if assert_folder:
            folder = os.path.dirname(filepath)
            os.makedirs(folder, exist_ok=True)

        self.csv_file = None
        self.filepath = filepath
        self.columns = columns

    def open(self):
        if self.csv_file:
            self.csv_file.close()

        already_exists = os.path.isfile(self.filepath)

        self.csv_file = open(self.filepath, 'a')

        if not already_exists and self.columns:
            self.write(*self.columns, quote=False)

    def write(self, *values, quote=False):
        if self.csv_file is None:
            return

        s = '"{}"' if quote else '{}'
        line = ','.join(s.format(v) for v in values) + '\n'
        self.csv_file.write(line)

    def close(self):
        if self.csv_file is None:
            return

        self.csv_file.close()

    def __enter__(self):
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
