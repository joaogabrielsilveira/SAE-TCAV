"""Bounded GPU execution and checksum-validated, atomic batch checkpoints."""
import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np

from artifact_storage import atomic_write_json, file_sha256

LOG = logging.getLogger(__name__)
GPU_POLICY = {'version': 1, 'internal_batch': 1, 'peak_memory_factor': 64,
              'prediction_batch': 16, 'embedding_batch': 16, 'gradient_batch': 16}


def add_gpu_arguments(parser):
    parser.add_argument('--gpu-settings', type=Path, help='Validated benchmark JSON with recommended_policy')
    for name in ('prediction_batch', 'embedding_batch', 'gradient_batch', 'peak_memory_factor', 'internal_batch'):
        parser.add_argument('--' + name.replace('_', '-'), type=int, default=GPU_POLICY[name])


def apply_gpu_arguments(args):
    if args.gpu_settings is not None:
        benchmark = json.loads(args.gpu_settings.read_text())
        policy = benchmark.get('recommended_policy')
        if not benchmark.get('complete') or not policy:
            raise ValueError('GPU benchmark has no accepted policy')
        for name, value in policy.items():
            if name != 'version':
                setattr(args, name, value)
    for name in ('prediction_batch', 'embedding_batch', 'gradient_batch', 'peak_memory_factor', 'internal_batch'):
        value = getattr(args, name)
        if value < 1:
            raise ValueError(f'{name} must be positive')
        GPU_POLICY[name] = value


def enforce_peak_chunks(layer, args):
    # TabPFN resets this to 8 at every predict call; apply the stronger bound
    # immediately before each layer without changing its arithmetic or weights.
    layer.save_peak_mem_factor = getattr(layer, '_temporal_peak_memory_factor', GPU_POLICY['peak_memory_factor'])


def configure_gpu_model(model):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU execution requested but CUDA is unavailable')
    if not str(getattr(model, 'device_', '')).startswith('cuda'):
        raise RuntimeError('GPU memory policy requires a CUDA-fitted model')
    model.batch_size_inference = GPU_POLICY['internal_batch']
    model.save_peak_memory = 'True'
    # Preserve the estimator's existing mixed-precision and ensemble settings.
    layers = model.model_processed_.transformer_encoder.layers
    for layer in layers:
        layer._temporal_peak_memory_factor = GPU_POLICY['peak_memory_factor']
        if not hasattr(layer, 'save_peak_mem_factor'):
            raise RuntimeError('This transformer does not support memory chunking')
        if not any(h is enforce_peak_chunks for h in layer._forward_pre_hooks.values()):
            layer.register_forward_pre_hook(enforce_peak_chunks)
    model._temporal_gpu_memory_safe = True
    LOG.info('GPU policy: %s fp16=%s ensemble=%s; all training records retained',
             GPU_POLICY, model.fp16_inference, model.N_ensemble_configurations)


def array_identity(*values):
    h = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class BatchStore:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        manifest = self.root/'identity.json'
        if manifest.exists() and json.loads(manifest.read_text()) != {'identity': identity}:
            raise ValueError(f'Batch checkpoint identity mismatch: {self.root}')
        atomic_write_json(manifest, {'identity': identity})

    def load(self, start):
        descriptor = self.root/f'{start:09d}.json'
        if not descriptor.exists():
            return None
        d = json.loads(descriptor.read_text())
        path = self.root/f'{start:09d}.npy'
        if d['identity'] != self.identity or d['start'] != start or file_sha256(path) != d['sha256']:
            raise ValueError(f'Invalid batch checkpoint: {path}')
        values = np.load(path, allow_pickle=False)
        if len(values) != d['end']-start or not np.isfinite(values).all():
            raise ValueError(f'Invalid batch values: {path}')
        return values

    def save(self, start, values):
        values = np.asarray(values)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Cannot checkpoint empty or non-finite model output')
        path = self.root/f'{start:09d}.npy'
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as handle:
            np.save(handle, values, allow_pickle=False)
        os.replace(tmp, path)
        atomic_write_json(self.root/f'{start:09d}.json', {
            'identity': self.identity, 'start': start, 'end': start+len(values),
            'sha256': file_sha256(path)})


def model_batch_store(model, stage, X, domains):
    root = getattr(model, '_temporal_batch_root', None)
    if root is None:
        return None
    identity = array_identity(X, domains)
    return BatchStore(Path(root)/stage/identity, model._temporal_fit_identity + ':' + identity)


def log_batch(stage, start, end, total, started, reused=False):
    import time
    import torch
    elapsed = time.monotonic()-started
    LOG.info('%s %s: rows %d-%d/%d (%.1f%%), elapsed %.1fs, GPU allocated %.0f MiB reserved %.0f MiB',
             stage, 'reused' if reused else 'saved', start, end, total, 100*end/max(1,total), elapsed,
             torch.cuda.memory_allocated()/2**20, torch.cuda.memory_reserved()/2**20)
