"""Persistent heartbeat for long scientific stages without changing their inputs."""
from contextlib import contextmanager
import logging
import threading
import time

LOG = logging.getLogger(__name__)


class _LastOperation(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.message = 'initializing'
        self.updated = time.monotonic()

    def emit(self, record):
        if record.name != __name__:
            self.message = record.getMessage()
            self.updated = time.monotonic()


@contextmanager
def progress_stage(label, *, interval=30):
    """Log an elapsed-time heartbeat only when the stage has otherwise been quiet."""
    stop = threading.Event()
    last = _LastOperation()
    root = logging.getLogger()
    original_handlers = set(root.handlers)
    root.addHandler(last)
    started = time.monotonic()

    def heartbeat():
        while not stop.wait(interval):
            if time.monotonic()-last.updated >= interval:
                LOG.info('%s still running; elapsed=%.1fs; current operation: %s',
                         label, time.monotonic()-started, last.message)

    thread = threading.Thread(target=heartbeat, name='scientific-progress', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)
        root.removeHandler(last)
        last.close()
        # Completed run logs must not accumulate messages from later stages.
        for handler in list(root.handlers):
            if isinstance(handler, logging.FileHandler) and handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
