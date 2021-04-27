import logging
import os
from filelock import FileLock, Timeout

class SyncLock:
    """Implements a sync-lock.

    Default behavior: if it cannot acquire the lock, returns False instead of raising an error.
    """
    def __init__(self, folder, name, raise_error=False, verbose=False):
        self.name = name

        fpath = os.path.join(folder, f'{name}.lock')
        self.lock = FileLock(fpath)

        self.raise_error = raise_error

        self.logger = logging.getLogger(f'{__name__}.{name}')
        self.logger.setLevel(
            logging.INFO if verbose else logging.WARNING
        )

    def acquire(self, timeout=10):
        try:
            self.lock.acquire(timeout=timeout)
            self.logger.info('Acquired lock for %s', self.name)
            return True
        except Timeout:
            if self.raise_error:
                raise
            return False

    def release(self):
        self.lock.release()
        self.logger.info('Released lock for %s', self.name)

    def __enter__(self):
        self.acquire()

    def __exit__(self, _1, _2, _3):
        self.release()


def with_lock(folder, lockname_key, **other):
    """Wraps a function using a SyncLock.

    The folder where the lock is stored is fixed, but the name is
    obtained from the function call parameters.

    Args:
        folder -- folder to save the lock
        lockname_key -- argument key to obtain the lockname
        **other -- other arguments passed to the SyncLock constructor
    """

    def wrapper(fn):
        def wrapped(*args, **kwargs):
            lockname = kwargs.get(lockname_key, fn.__name__)

            with SyncLock(folder, lockname, **other):
                return fn(*args, **kwargs)

        return wrapped

    return wrapper
