import logging
import os
from filelock import FileLock, Timeout

LOGGER = logging.getLogger(__name__)

class SyncLock:
    """Implements a sync-lock.

    If it cannot acquire the lock, returns False instead of raising an error.
    """
    def __init__(self, folder, name):
        self.name = name

        fpath = os.path.join(folder, f'{name}.lock')
        self.lock = FileLock(fpath)

    def acquire(self, timeout=10):
        try:
            self.lock.acquire(timeout=timeout)
            LOGGER.info('Acquired lock for %s', self.name)
            return True
        except Timeout:
            return False

    def release(self):
        self.lock.release()
        LOGGER.info('Released lock for %s', self.name)
