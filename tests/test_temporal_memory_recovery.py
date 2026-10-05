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


# Source-hash bundle and inputs recorded by the prior window run whose completed fits are reused.
REVIEWED_PRIOR_INPUTS = {
    "configs": {
        "comparison_config_path": "780d2bc5a00a3d7caad0e46e557be676b188cfdbf0caa5b727622b872449bcdf",
        "semantic_config_path": "607e06110677160f932f1a6d8455ba5e89055b00350b42e4c933bced2a5f529b"},
    "device": "cuda",
    "environment": {"cuda_version": "12.1", "cudnn_version": 8902, "numpy": "1.26.4",
                    "resolved_device": "cuda", "scipy": "1.15.3", "sklearn": "1.5.2", "torch": "2.1.2+cu121"},
    "gpu_memory_policy": {"embedding_batch": 16, "gradient_batch": 16, "internal_batch": 1,
                          "peak_memory_factor": 64, "prediction_batch": 16, "version": 1},
    "import_handoff_sha256": "21ac0a165325a7384abc50b3a925602b83b08099e40659668c7aa3ad40e3a9e6",
    "model_checkpoint": {"path_name": "tabpfn_dist_model_1.cpkt",
                         "sha256": "e653d21bd9b2713cf857696f98aabb2a59a4009212d1a63744feccbca94462db"},
    "parent": "db383bc72050076b0640060475a399c5b4101d285cbf17732c06fb55484ad445",
    "protocol": "expanded_patient_disjoint_concepts_v1",
    "sources": {
        "comparison_runner.py": "a5b55e8f51db4c75b5a8f393f07a83187f1a4f398f1713c8636daa553ecd1cd2",
        "tabpfn_model.py": "64633e937ad7a1e77953aed33f5c3fff399a3ee7d5eb52211b8b53a2ad0d640f",
        "tcav.py": "84cfc2e65508da522e4ad7a8ebb03fa96de45862c7ff87ded43e8cb738b91cf1",
        "temporal_cav.py": "ebac477aaf5e0c8ce50713dac3209b41e1efb39a498fb8f69ac6d331f8e4c140",
        "temporal_concept_forecasting.py": "8063b87e64f78058d8bae6ed5f1c0bbdc1e1c4ad364015b82c59ffa2a791227f",
        "temporal_cri.py": "c31fe79873ef4df95d11ea03746610241085c3d56eeb851f55e78f5cb811dbec",
        "temporal_gpu_execution.py": "16201fa9b78c8956dd37e4cad8bee43a0bcb1d14d40c01624872fd3aa8802866",
        "temporal_handoff.py": "2a3a5840fbfd02bf018e68f001b5e7fc58d337030425dbb35a2193f4baccacc3",
        "temporal_memory_recovery.py": "a86203ce2973a766270a8243aa9a6f7bd4c4e9a181d6898962eccec583ebb9b8",
        "temporal_performance_windows.py": "e54ed7915630a38c837edea2d7c7a1b3e18813fba6b012c9d17374de9b87e4c9",
        "temporal_production.py": "f0ba50525f3ba413dbeabd01b79afc58dae8f5edeae6f52af9c750aa266fb713",
        "temporal_rules.py": "4e8b96c453c0d50171f5b5e187f8961c12389a72fe0d1147379e0f4592068d24",
        "temporal_unified_analysis.py": "4eb1b8778732477d19990b7eed8e87a04c6caf6f017eb5cf2444094ff3772edd",
        "temporal_unified_enrichment.py": "0f6f60294e97aee05cf5267153773667d2b1fc142c04449f193e67c97d391eac",
        "temporal_window_concepts.py": "99f4488cdc0b053c5ce268b3049ae6341ac949f3c6412119775130598811dde5"},
    "windows": "866eec322c5575f49c2647bb7700de5cfd4f92d9982af338ac1e7b6f42fe2f9b",
}
REVIEWED_PRIOR_IDENTITY = "5b1d98fd385f1b76c2df"


def _current_inputs(**changes):
    inputs = {k: v for k, v in REVIEWED_PRIOR_INPUTS.items() if k != "import_handoff_sha256"}
    inputs["sources"] = {name: "edited-" + name for name in inputs["sources"]}
    return {**inputs, **changes}


def _job(**changes):
    import numpy as np
    return {"ref": 2008, "seed": 42, "years": [2007, 2008], "train": np.array([3, 4, 5, 9]),
            "roles": {"sae_discovery": np.array([4]), "rule_discovery": np.array([5])},
            "evaluation": np.array([3, 4, 5, 9, 12]), "domains": {2007: 0, 2008: 1}, **changes}


def _snapshot(root):
    from artifact_storage import file_sha256
    return {str(p.relative_to(root)): (file_sha256(p), p.stat().st_mtime_ns)
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def _prior_run(tmp_path, inputs, job, rows=None):
    """Synthetic completed-fit tree in the layout written by a prior window run."""
    from temporal_concept_forecasting import digest, write_table
    from artifact_storage import atomic_write_json
    identity = digest(inputs)[:20]
    fit_identity = digest({k: v for k, v in inputs.items() if k != "selection"})[:20]
    old_job = digest({"run": fit_identity, **job})[:24]
    root = tmp_path / "prior" / f"window_concepts_{identity}"
    source = root / "fits" / old_job / f"reference_{job['ref']}" / f"split_{job['seed']}"
    source.mkdir(parents=True)
    atomic_write_json(root / "run_identity.json", {"identity": identity, "inputs": inputs})
    rows = rows or [{"test_year": 2008, "score": 0.25}, {"test_year": 2009, "score": 0.5}]
    artifacts = {name: write_table(source, name, rows) for name in ("performance", "record_probabilities")}
    atomic_write_json(source / "completed.json", {"complete": True, "identity": old_job, "artifacts": artifacts,
                                                  "actual_device": "cuda", "effective_years": job["years"]})
    return tmp_path / "prior", source


def test_recorded_prior_inputs_reproduce_the_prior_run_identity():
    from temporal_concept_forecasting import digest
    assert digest(REVIEWED_PRIOR_INPUTS)[:20] == REVIEWED_PRIOR_IDENTITY
    assert len(REVIEWED_PRIOR_INPUTS["sources"]) == 15


def test_reviewed_source_bundle_candidate_is_reused_without_touching_the_source(tmp_path):
    import temporal_window_concepts as module
    from temporal_concept_forecasting import checked_manifest, table
    source_root, source = _prior_run(tmp_path, REVIEWED_PRIOR_INPUTS, _job())
    before = _snapshot(source_root)
    target = tmp_path / "new" / "reference_2008" / "split_42"
    assert module.reuse_reviewed_completed_from_root(
        source_root, _current_inputs(), _job(), target, "new_job") is True
    manifest = checked_manifest(target / "completed.json")
    assert manifest["identity"] == "new_job"
    assert manifest["reuse_basis"] == "reviewed_source_bundle_exact_scientific_inputs_and_membership"
    assert manifest["reused_manifest_sha256"] == before[str((source / "completed.json").relative_to(source_root))][0]
    assert table(target / "completed.json", manifest, "performance") == [
        {"test_year": 2008, "score": 0.25}, {"test_year": 2009, "score": 0.5}]
    assert _snapshot(source_root) == before


def test_candidate_with_the_current_sources_needs_no_review(tmp_path):
    import temporal_window_concepts as module
    current = _current_inputs()
    source_root, _ = _prior_run(tmp_path, {**current, "import_handoff_sha256": "other"}, _job())
    target = tmp_path / "new" / "reference_2008" / "split_42"
    assert module.reuse_reviewed_completed_from_root(source_root, current, _job(), target, "new_job") is True


@pytest.mark.parametrize("change", ["source", "config", "model", "environment", "gpu_policy", "no_gpu_policy",
                                    "device", "membership"])
def test_incompatible_candidates_are_rejected_and_logged(tmp_path, caplog, change):
    import copy
    import temporal_window_concepts as module
    prior, current, job = copy.deepcopy(REVIEWED_PRIOR_INPUTS), _current_inputs(), _job()
    if change == "source":
        prior["sources"]["tcav.py"] = "0" + prior["sources"]["tcav.py"][1:]
    elif change == "config":
        current["configs"]["semantic_config_path"] = "changed"
    elif change == "model":
        current["model_checkpoint"] = {**current["model_checkpoint"], "sha256": "changed"}
    elif change == "environment":
        current["environment"] = {**current["environment"], "torch": "9.9"}
    elif change == "gpu_policy":
        current["gpu_memory_policy"] = {**current["gpu_memory_policy"], "prediction_batch": 8}
    elif change == "no_gpu_policy":
        del current["gpu_memory_policy"]
    elif change == "device":
        current["device"] = "cpu"
    source_root, _ = _prior_run(tmp_path, prior, job)
    if change == "membership":
        job = _job(train=job["train"][:-1])
    before = _snapshot(source_root)
    target = tmp_path / "new" / "reference_2008" / "split_42"
    with caplog.at_level("INFO"):
        assert module.reuse_reviewed_completed_from_root(source_root, current, job, target, "new_job") is False
    assert not (target / "completed.json").exists()
    assert _snapshot(source_root) == before
    if change != "membership":
        assert "Rejected prior run" in caplog.text


@pytest.mark.parametrize("damage", ["checksum", "missing", "identity"])
def test_matching_candidate_with_damaged_artifact_fails_closed(tmp_path, damage):
    import temporal_window_concepts as module
    from artifact_storage import atomic_write_json
    source_root, source = _prior_run(tmp_path, REVIEWED_PRIOR_INPUTS, _job())
    if damage == "checksum":
        (source / "performance.jsonl.gz").write_bytes(b"corrupted")
    elif damage == "missing":
        (source / "record_probabilities.jsonl.gz").unlink()
    else:
        completed = json.loads((source / "completed.json").read_text())
        atomic_write_json(source / "completed.json", {**completed, "identity": "someone_else"})
    before = _snapshot(source_root)
    target = tmp_path / "new" / "reference_2008" / "split_42"
    with pytest.raises(module.ReuseIntegrityError):
        module.reuse_reviewed_completed_from_root(source_root, _current_inputs(), _job(), target, "new_job")
    assert not (target / "completed.json").exists()
    assert _snapshot(source_root) == before
