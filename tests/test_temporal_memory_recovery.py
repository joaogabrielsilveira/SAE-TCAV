import json
from pathlib import Path

import pytest
import torch

from temporal_memory_recovery import configure_allocator, run_with_cpu_fallback


def test_allocator_preserves_explicit_user_configuration(monkeypatch):
    monkeypatch.delenv('PYTORCH_CUDA_ALLOC_CONF', raising=False)
    configure_allocator()
    import os
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'max_split_size_mb:128'
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF','max_split_size_mb:64')
    configure_allocator()
    assert os.environ['PYTORCH_CUDA_ALLOC_CONF'] == 'max_split_size_mb:64'


@pytest.mark.parametrize('failing_stage', ['predictions','embeddings','gradients'])
def test_whole_system_cpu_retry_and_resume(tmp_path, monkeypatch, failing_stage):
    import temporal_performance_windows as windows
    calls, cleaned = [], []
    monkeypatch.setattr(windows,'_release_cuda_memory',lambda: cleaned.append(True))
    def operation(device, path):
        stages = []
        calls.append((device,path,stages))
        for stage in ('fit','predictions','embeddings','gradients'):
            stages.append(stage)
            if device == 'cuda' and stage == failing_stage:
                (path/'partial.json').write_text('GPU partial')
                raise torch.cuda.OutOfMemoryError('CUDA out of memory')
        assert not (path/'partial.json').exists()
        return {stage:device for stage in stages}
    root=tmp_path/'reference_2015'/'split_42'
    result,device=run_with_cpu_fallback(operation,root,'cuda')
    assert device == 'cpu' and set(result.values()) == {'cpu'}
    assert calls[0][1] != calls[1][1]
    assert calls[1][2] == ['fit','predictions','embeddings','gradients']
    assert cleaned == [True]
    assert json.loads((root/'device_recovery.json').read_text())['selected_device'] == 'cpu'
    calls.clear()
    run_with_cpu_fallback(operation,root,'cuda')
    assert [c[0] for c in calls] == ['cpu']


def test_non_memory_error_never_silently_retries(tmp_path):
    calls=[]
    def operation(device,path):
        calls.append(device)
        raise RuntimeError('invalid domain mapping')
    with pytest.raises(RuntimeError,match='invalid domain'):
        run_with_cpu_fallback(operation,tmp_path,'cuda')
    assert calls == ['cuda']


def test_cpu_failure_is_reported_without_loop(tmp_path):
    calls=[]
    def operation(device,path):
        calls.append(device)
        raise RuntimeError('CPU failed')
    with pytest.raises(RuntimeError,match='CPU failed'):
        run_with_cpu_fallback(operation,tmp_path,'cpu')
    assert calls == ['cpu']


@pytest.mark.parametrize('change', [None, 'membership', 'source', 'oom'])
def test_reviewed_artifact_migration(tmp_path, monkeypatch, change):
    import temporal_window_concepts as module
    from temporal_concept_forecasting import digest, write_table, checked_manifest, table
    from artifact_storage import atomic_write_json
    sources = {'temporal_window_concepts.py': 'old', 'scientific.py': 'unchanged'}
    monkeypatch.setattr(module, '_REVIEWED_LEGACY_SOURCES', digest(sources))
    inputs = {'sources': sources, 'device': 'cuda', 'protocol': module.PROTOCOL}
    old_identity = digest(inputs)[:20]
    job = {'ref': 2015, 'seed': 42, 'train': [1, 2]}
    old_job = digest({'run': old_identity, **job})[:24]
    root = tmp_path / ('window_concepts_' + old_identity)
    source = root/'fits'/old_job/'reference_2015'/'split_42'
    source.mkdir(parents=True)
    artifacts = {'performance': write_table(source, 'performance', [{'score': 0.4}])}
    if change != 'oom':
        atomic_write_json(source/'completed.json', {'complete': True, 'identity': old_job, 'artifacts': artifacts})
    atomic_write_json(root/'incomplete_manifest.json', {'schema_version': module.PROTOCOL,
        'source_code': sources, 'failures': [{'reference_year': 2015, 'patient_split_seed': 42,
                                            'window': 'all_history', 'message': 'CUDA out of memory'}]})
    current = {**inputs, 'sources': {**sources, 'temporal_window_concepts.py': 'new'}}
    if change == 'source':
        current['sources']['scientific.py'] = 'different'
    if change == 'membership':
        job = {**job, 'train': [1, 3]}
    target = tmp_path/'new'/'reference_2015'/'split_42'
    reused = module.reuse_reviewed_completed_job(tmp_path, current, job, target, 'new_job', 'all_history')
    assert reused == (change is None)
    if reused:
        manifest = checked_manifest(target/'completed.json')
        assert manifest['identity'] == 'new_job'
        assert table(target/'completed.json', manifest, 'performance') == [{'score': 0.4}]
    if change == 'oom':
        assert json.loads((target/'device_recovery.json').read_text())['selected_device'] == 'cpu'
    elif change is not None:
        assert not (target/'device_recovery.json').exists()


def test_gpu_only_retry_ignores_cpu_marker_and_never_falls_back(tmp_path):
    (tmp_path/'device_recovery.json').write_text(json.dumps({'selected_device': 'cpu'}))
    calls = []
    def operation(device, path):
        calls.append(device)
        raise torch.cuda.OutOfMemoryError('CUDA out of memory')
    with pytest.raises(torch.cuda.OutOfMemoryError):
        run_with_cpu_fallback(operation, tmp_path, 'cuda', retry_cuda=True, allow_cpu_fallback=False)
    assert calls == ['cuda']


def test_atomic_batch_resume_rejects_corruption_and_different_inputs(tmp_path):
    import numpy as np
    from temporal_gpu_execution import BatchStore, array_identity
    values = np.array([[0.2, 0.8], [0.7, 0.3]])
    identity = array_identity(values)
    store = BatchStore(tmp_path, identity)
    store.save(0, values)
    assert store.load(2) is None
    np.testing.assert_array_equal(BatchStore(tmp_path, identity).load(0), values)
    with pytest.raises(ValueError, match='identity mismatch'):
        BatchStore(tmp_path, array_identity(values[::-1]))
    (tmp_path/'000000000.npy').write_bytes(b'partial write')
    with pytest.raises(ValueError, match='Invalid batch checkpoint'):
        store.load(0)


def test_gpu_layer_chunk_hook_preserves_outputs_and_serializes(tmp_path):
    import pickle
    from temporal_gpu_execution import enforce_peak_chunks
    layer = torch.nn.Linear(3, 2)
    layer.save_peak_mem_factor = 8
    inputs = torch.ones((4, 3))
    expected = layer(inputs)
    layer.register_forward_pre_hook(enforce_peak_chunks)
    restored = pickle.loads(pickle.dumps(layer))
    actual = restored(inputs)
    assert restored.save_peak_mem_factor == 64
    torch.testing.assert_close(actual, expected)
