from timeit import default_timer as timer

class Timer:
    def __init__(self):
        self.total = 0
        self._start = 0

    def reset(self):
        self.total = 0

    def start(self):
        self._start = timer()

    def stop(self):
        self.total += timer() - self._start

    def __enter__(self):
        self.start()

    def __exit__(self, _1, _2, _3):
        self.stop()