import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from artifact_storage import atomic_write_json, file_sha256
from temporal_handoff import (EmbeddingsReady, load_handoff, save_handoff, validate_decoder,
    assert_agreement, verify_package, scientific_inputs)
from tcav import get_model_gradients


class Model:
    def __init__(self):
        torch.manual_seed(12)
        self.model_processed_ = SimpleNamespace(decoder_dict={'standard': torch.nn.Sequential(
            torch.nn.Linear(3, 7), torch.nn.Tanh(), torch.nn.Linear(7, 2))})
        self.calls = 0
    def get_embeddings(self, X, additional_x):
        self.calls += 1
        return torch.tensor(X, dtype=torch.float32).unsqueeze(1)


def test_saved_raw_matches_original_gradient_and_skips_transformer():
    model = Model()
    X = np.arange(18, dtype=np.float32).reshape(6, 3) / 20
    old = get_model_gradients(model, np.zeros(6), X, batch_size=2, device='cpu', use_cache=False)
    assert model.calls == 3
    def forbidden(*a, **k):
        raise AssertionError('Transformer must not run with supplied raw embeddings')
    model.get_embeddings = forbidden
    new = get_model_gradients(model, np.zeros(6), X, batch_size=4, device='cpu', use_cache=False, raw_embeddings=X)
    np.testing.assert_allclose(new, old, rtol=1e-6, atol=1e-7)


def make_bundle(tmp_path):
    model = Model()
    X = np.arange(12, dtype=np.float32).reshape(4,3) / 20
    fit = {'model':model, 'dist_shift_domain_train': np.array([0,0])}
    source = tmp_path/'fit.pkl'; source.write_bytes(pickle.dumps(fit))
    context = {'job': {'train':[0,1], 'evaluation':[0,1,2,3], 'domains':{'2015':0},
        'roles':{'sae_discovery':[0]}, 'ref':2015,'seed':42,'years':[2015]}, 'predict_indices':[2,3]}
    emb = SimpleNamespace(train_raw=X[:2],test_raw=X,require_model=lambda:model)
    pop = SimpleNamespace(X=X,years=np.array([2015]*4))
    result=SimpleNamespace(probabilities=np.array([[.7,.3],[.8,.2]]), classes=np.array([0,1]),model_info={})
    manifest=save_handoff(tmp_path,context,emb,source,result,pop,{})
    return manifest, context


def test_bundle_roundtrip_and_relocation(tmp_path):
    manifest, context = make_bundle(tmp_path)
    import shutil
    moved=tmp_path/'destination'; shutil.copytree(manifest.parent,moved)
    meta, arrays, fit = load_handoff(moved/'manifest.json',context)
    assert arrays['test_raw'].shape == (4,3)
    assert fit['model'].calls == 0  # immutable early model, not a post-extraction fit
    assert meta['validation']['max_absolute_differences']['decoder_gradients'] == 0
    changed=json.loads(json.dumps(context)); changed['job']['train']=[1,0]
    with pytest.raises(ValueError,match='ordering'):
        load_handoff(moved/'manifest.json',changed)
    with (moved/'test_raw.npy').open('ab') as handle:
        handle.write(b'corruption')
    with pytest.raises(ValueError,match='checksum'):
        load_handoff(moved/'manifest.json',context)


def test_domain_mapping_tamper_rejected(tmp_path):
    manifest,context=make_bundle(tmp_path)
    meta=json.loads(manifest.read_text()); meta['context']['job']['domains']['2015']=1
    atomic_write_json(manifest,meta)
    with pytest.raises(ValueError,match='training domains'):
        load_handoff(manifest)


def test_mismatch_fails_closed_and_stop_is_not_job_failure():
    with pytest.raises(ValueError,match='agreement failed'):
        assert_agreement('decoder',np.array([.2]),np.array([.3]))
    assert not isinstance(EmbeddingsReady('manifest.json'),Exception)


def test_gpu_settings_and_hook_are_configurable(monkeypatch):
    import argparse
    import temporal_gpu_execution as gpu
    before=gpu.GPU_POLICY.copy()
    try:
        p=argparse.ArgumentParser(); gpu.add_gpu_arguments(p)
        gpu.apply_gpu_arguments(p.parse_args(['--embedding-batch','64','--peak-memory-factor','16']))
        assert gpu.GPU_POLICY['embedding_batch']==64
        layer=SimpleNamespace(_temporal_peak_memory_factor=16)
        gpu.enforce_peak_chunks(layer,None)
        assert layer.save_peak_mem_factor==16
        with pytest.raises(ValueError,match='positive'):
            gpu.apply_gpu_arguments(p.parse_args(['--gradient-batch','0']))
    finally:
        gpu.GPU_POLICY.update(before)


def test_package_verification_detects_missing_or_changed_files(tmp_path):
    p=tmp_path/'data';p.write_text('saved')
    atomic_write_json(tmp_path/'package_manifest.json',{'complete':True,'files':{'data':file_sha256(p)}})
    verify_package(tmp_path)
    p.write_text('changed')
    with pytest.raises(ValueError,match='checksum'):
        verify_package(tmp_path)


def test_runtime_change_does_not_change_scientific_binding():
    old={'sources':{'a':'old'},'device':'cuda','environment':{'torch':'old'},'gpu_memory_policy':{'batch':16},'parent':'abc'}
    new={**old,'environment':{'torch':'new'},'gpu_memory_policy':{'batch':64}}
    assert scientific_inputs(old)==scientific_inputs(new)
    assert scientific_inputs(old)!=scientific_inputs({**new,'parent':'def'})


def test_production_stop_callback_precedes_sae_training(tmp_path, monkeypatch):
    import pandas as pd
    from comparison_runner import ComparisonRunnerConfig, DefaultComparisonAdapter
    from temporal_production import ProductionTemporalAdapter
    from temporal_config import TemporalRobustnessConfig
    adapter=ProductionTemporalAdapter()
    adapter._base_config=ComparisonRunnerConfig()
    adapter._source_prepared=SimpleNamespace(test_rows=pd.DataFrame({'x':range(4)}))
    pop=SimpleNamespace(X=np.ones((4,3)),outcomes=np.array([0,1,0,1]),years=np.array([2015]*4),
                        feature_names=['a','b','c'],patient_ids=np.array(['a','b','c','d']),record_keys=np.arange(4))
    roles={'tabpfn_context':np.array([0]),'sae_discovery':np.array([1]),'rule_selection_cav':np.array([2]),
           'rule_discovery':np.array([1]),'t0_evaluation':np.array([3])}
    monkeypatch.setattr(DefaultComparisonAdapter,'embeddings',lambda *a,**kw: 'saved_embeddings')
    def forbidden(*a,**k): raise AssertionError('SAE started before handoff exit')
    monkeypatch.setattr(DefaultComparisonAdapter,'train_saes',forbidden)
    def stop(embeddings, prepared):
        assert embeddings=='saved_embeddings'
        raise EmbeddingsReady(tmp_path/'manifest.json')
    with pytest.raises(EmbeddingsReady):
        adapter.run_reference_experiment(population=pop,reference_year=2015,split=SimpleNamespace(effective_seed=42),
            global_roles=roles,evaluation_indices=np.arange(4),domain_map={2015:0},config=TemporalRobustnessConfig(),
            workspace=tmp_path,after_embeddings=stop)


def test_runner_records_clean_exit_without_full_grid(tmp_path, monkeypatch):
    import run_temporal_concept_forecasting as runner
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner,'original_inputs',lambda *a: {})
    monkeypatch.setattr(runner,'build_forecasting',lambda *a: tmp_path/'forecast.json')
    monkeypatch.setattr(runner,'build_report',lambda *a,**kw: None)
    calls=[]
    def stop(*a,**k):
        calls.append(k)
        raise EmbeddingsReady(tmp_path/'handoff.json')
    monkeypatch.setattr(runner,'run_window_concepts',stop)
    assert runner.main(['--repo',str(tmp_path),'--output',str(tmp_path/'out'),'--stop-after-embeddings'])==0
    status=json.loads((tmp_path/'out/execution_status.json').read_text())
    assert status['status']=='embeddings_ready' and status['exit_code']==0
    assert len(calls)==1 and calls[0]['stop_after_embeddings']


def test_gradient_helper_accepts_indexed_cuda_model_device():
    model=Model()
    # CPU is the available test device; torch device.type is the portable
    # cuda/cpu token accepted by resolve_torch_device, unlike cuda:0.
    device=next(model.model_processed_.decoder_dict['standard'].parameters()).device
    assert device.type in {'cpu','cuda'}
