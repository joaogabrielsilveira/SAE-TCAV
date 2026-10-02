"""Whole-system memory recovery; never combine outputs across fitted devices."""
import logging
import os
from pathlib import Path
import json

from artifact_storage import atomic_write_json

LOG = logging.getLogger(__name__)


def configure_allocator():
    # Must be called before CUDA initialization, including forecasting imports.
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')


def run_with_cpu_fallback(operation, workspace, requested_device, *, allow_cpu_fallback=True, retry_cuda=False):
    """Retry the entire callback on CPU after CUDA OOM, using an isolated cache.

    The device decision survives interruption. Non-memory failures propagate.
    All GPU traceback references are released before starting the CPU callback.
    """
    import torch
    from temporal_performance_windows import _is_cuda_out_of_memory, _release_cuda_memory
    workspace = Path(workspace)
    marker = workspace/'device_recovery.json'
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if requested_device == 'auto' else requested_device
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('selected_device') == 'cpu' and not retry_cuda:
            device = 'cpu'
            LOG.info('Resuming CPU recovery recorded in %s', marker)
    while True:
        attempt = workspace / f'attempt_{device}' / workspace.parent.name / workspace.name
        attempt.mkdir(parents=True, exist_ok=True)
        LOG.info('Fitted-system attempt: device=%s workspace=%s', device, attempt)
        failure = None
        try:
            result = operation(device, attempt)
        except RuntimeError as error:
            if device != 'cuda' or not _is_cuda_out_of_memory(error):
                raise
            if not allow_cpu_fallback:
                LOG.error('GPU memory exhausted under the requested GPU-only policy; preserving checkpoints and stopping')
                raise
            failure = str(error)
            LOG.warning('GPU memory exhausted; discarding this attempt and recreating the whole system on CPU. %s', failure)
            error.__traceback__ = None
        if failure is None:
            atomic_write_json(marker, {'selected_device': device, 'complete': True})
            return result, device
        atomic_write_json(marker, {'selected_device': 'cpu', 'complete': False,
                                   'reason': 'cuda_out_of_memory', 'message': failure})
        _release_cuda_memory()
        device = 'cpu'
