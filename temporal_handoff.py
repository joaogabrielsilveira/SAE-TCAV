"""Explicit, checksum-validated handoff of one fitted temporal system.

Pickles are loaded only after manifest validation. Use bundles from a trusted
source: checksums detect corruption, not malicious Python pickle payloads.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from artifact_storage import atomic_write_json, file_sha256, jsonable
from temporal_concept_forecasting import digest

LOG = logging.getLogger(__name__)
SCHEMA = 'temporal_embedding_handoff_v1'
DERIVED_SCHEMA = 'temporal_embedding_handoff_v2'
RTOL, ATOL = 1e-3, 1e-4


class EmbeddingsReady(BaseException):
    """Successful stopping boundary; must bypass per-job failure handlers."""
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        super().__init__(f'Embeddings ready: {manifest}')


def scientific_inputs(inputs):
    return {k: v for k, v in inputs.items() if k not in {'sources', 'device', 'environment', 'gpu_memory_policy', 'import_handoff_sha256'}}


def handoff_context(inputs, job, population, predict):
    from temporal_gpu_execution import array_identity
    return jsonable({'scientific_inputs': scientific_inputs(inputs), 'job': job,
        'predict_indices': predict, 'population': {
            n: array_identity(np.asarray(getattr(population, n)).astype(str) if n in ('patient_ids', 'record_keys')
                              else getattr(population, n))
            for n in ('X', 'outcomes', 'years', 'patient_ids', 'record_keys')},
        'feature_names': list(population.feature_names)})


class ImportedRaw(tuple):
    """An unchanged raw pair carrying validated source provenance."""
    def __new__(cls, train, retained, provenance):
        value = super().__new__(cls, (train, retained))
        value.training_provenance = json.loads(json.dumps(provenance))
        return value


def _handoff_training_provenance(meta):
    from semantic_artifacts import array_fingerprint
    legacy = {"origin": "independent_queries_v1", "mapping": None}
    provenance = meta.get("training_provenance")
    if meta["schema"] == SCHEMA:
        if provenance is not None and provenance != legacy:
            raise ValueError("Legacy schema cannot declare a derived training origin")
        return legacy
    if not isinstance(provenance, dict) or meta.get("training_provenance_sha256") != digest(provenance):
        raise ValueError("Missing or mismatched training provenance hash")
    origin, mapping = provenance.get("origin"), provenance.get("mapping")
    if origin in ("independent_queries_v1", "imported_preserved_v1"):
        if mapping is not None:
            raise ValueError("Independent/imported origin must not declare a derived mapping")
        return provenance
    if origin != "retained_rows_v1" or not isinstance(mapping, dict):
        raise ValueError("Unsupported training provenance origin or mapping")
    job = meta["context"]["job"]
    train, retained = np.asarray(job["train"]), np.asarray(job["evaluation"])
    for indices in (train, retained):
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) == 0
                or len(np.unique(indices)) != len(indices) or np.any(indices < 0)):
            raise ValueError("Derived provenance requires unique integer record identities")
    local = {int(index): position for position, index in enumerate(retained)}
    if not set(train.tolist()).issubset(local):
        raise ValueError("Derived mapping requires complete training identity containment")
    expected = np.asarray([local[int(index)] for index in train], dtype=np.int64)
    actual = np.asarray(mapping.get("indices"))
    if (actual.ndim != 1 or actual.dtype.kind not in "iu"
            or not np.array_equal(actual, expected)
            or mapping.get("sha256") != array_fingerprint(expected)):
        raise ValueError("Derived mapping or hash disagrees with exact record identities")
    return provenance


def checked_files(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    data = json.loads(manifest_path.read_text())
    if data.get('schema') not in (SCHEMA, DERIVED_SCHEMA) or data.get('complete') is not True:
        raise ValueError('Incomplete or unsupported handoff')
    data["training_provenance"] = _handoff_training_provenance(data)
    root = manifest_path.parent
    for name, desc in data['files'].items():
        path = (root / desc['path']).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != desc['sha256']:
            raise ValueError(f'Handoff checksum/path mismatch: {name}')
    return data


def load_handoff(path, expected_context=None):
    path = Path(path)
    meta = checked_files(path)
    if expected_context is not None and digest(meta['context']) != digest(expected_context):
        raise ValueError('Handoff mismatch: scientific inputs, patient ordering, training membership, prediction rows, roles or domains')
    root = path.parent
    arrays = {}
    for name in ('train_raw', 'test_raw', 'probabilities', 'classes', 'probe_outputs', 'probe_gradients'):
        desc = meta['files'][name]
        a = np.load(root/desc['path'], allow_pickle=False)
        if list(a.shape) != desc['shape'] or a.dtype.str != desc['dtype'] or not np.isfinite(a).all():
            raise ValueError(f'Invalid handoff array: {name}')
        arrays[name] = a
    job = meta['context']['job']
    if (len(arrays['train_raw']) != len(job['train']) or len(arrays['test_raw']) != len(job['evaluation'])
            or len(arrays['probabilities']) != len(meta['context']['predict_indices'])):
        raise ValueError('Handoff arrays do not match membership counts')
    if arrays['probabilities'].shape != (len(meta['context']['predict_indices']), len(arrays['classes'])):
        raise ValueError('Probability classes or dimensions differ')
    if arrays['train_raw'].ndim != 2 or arrays['test_raw'].shape[1:] != arrays['train_raw'].shape[1:]:
        raise ValueError('Invalid embedding dimensions')
    provenance = meta["training_provenance"]
    if provenance["origin"] == "retained_rows_v1":
        indices = np.asarray(provenance["mapping"]["indices"], dtype=np.int64)
        if not np.array_equal(arrays["train_raw"], arrays["test_raw"][indices]):
            raise ValueError("Derived training arrays disagree with mapped retained embeddings")
    arrays["raw_pair"] = ImportedRaw(arrays["train_raw"], arrays["test_raw"], provenance)
    with (root/meta['files']['fitted_state']['path']).open('rb') as handle:
        fit = pickle.load(handle)
    expected_domains = np.array([job['domains'][str(y)] for y in meta['training_years']])
    if not np.array_equal(fit['dist_shift_domain_train'], expected_domains):
        raise ValueError('Fitted training domains differ from manifest')
    model = fit['model']
    # Runtime checkpoint paths belong to the destination, never to the source.
    for attr in ('_temporal_batch_root', '_temporal_fit_identity'):
        if hasattr(model, attr):
            delattr(model, attr)
    return meta, arrays, fit


def reuse_completed_from_transfer(output, inputs, job, workspace, identity):
    from temporal_concept_forecasting import checked_manifest, table, write_table
    for state in sorted(Path(output).glob('window_concepts_*/run_identity.json')):
        old = json.loads(state.read_text())
        # Fits are identified without the aggregate selection that scopes the prior run.
        fit_inputs = {k: v for k, v in old['inputs'].items() if k != 'selection'}
        if digest(old['inputs'])[:20] != old['identity'] or digest(scientific_inputs(fit_inputs)) != digest(scientific_inputs(inputs)):
            continue
        old_job = digest({'run': digest(fit_inputs)[:20], **job})[:24]
        complete = state.parent/'fits'/old_job/workspace.parent.name/workspace.name/'completed.json'
        if not complete.exists():
            continue
        meta = checked_manifest(complete)
        if meta['identity'] != old_job:
            raise ValueError('Transferred completed system identity mismatch')
        artifacts = {n:write_table(workspace,n,table(complete,meta,n)) for n in meta['artifacts']}
        atomic_write_json(workspace/'completed.json', {**meta,'identity':identity,'artifacts':artifacts,
            'reuse_basis':'explicit_transfer_matching_scientific_inputs_and_exact_membership',
            'source_manifest_sha256':file_sha256(complete),'source_runtime':old['inputs']})
        LOG.info('HANDOFF: reused completed concepts and results for corresponding system %s',old_job)
        return True
    return False


def decoder_values(model, raw):
    import torch
    decoder = model.model_processed_.decoder_dict['standard']
    device = next(decoder.parameters()).device
    with torch.no_grad():
        return decoder(torch.as_tensor(np.array(raw, copy=True), device=device, dtype=torch.float32)).detach().cpu().numpy()


def decoder_gradients(model, raw):
    from tcav import get_model_gradients
    device = next(model.model_processed_.decoder_dict['standard'].parameters()).device.type
    return get_model_gradients(model, np.zeros(len(raw), dtype=int), np.zeros((len(raw), 1)),
        batch_size=len(raw), device=device, use_cache=False, raw_embeddings=raw)


def assert_agreement(label, actual, expected):
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=RTOL, atol=ATOL):
        delta = float(np.max(np.abs(actual-expected))) if actual.shape == expected.shape else None
        raise ValueError(f'{label} agreement failed: max_abs={delta}, rtol={RTOL}, atol={ATOL}')
    return float(np.max(np.abs(actual-expected)))


def validate_decoder(model, X, domains, raw, *, reference=None, sample_size=16):
    """One transformer sample, then exactly the existing decoder derivative."""
    import torch
    count = min(sample_size, len(X))
    if count < 1:
        raise ValueError('Cannot validate empty embeddings')
    with torch.no_grad():
        fresh = model.get_embeddings(np.asarray(X[:count], dtype=np.float32), additional_x={
            'dist_shift_domain': torch.tensor(domains[:count], dtype=torch.long, device='cpu').reshape(-1, 1, 1)})
        if fresh.ndim == 3 and fresh.shape[0] == 1:
            fresh = fresh[0]
        elif fresh.ndim == 3 and fresh.shape[1] == 1:
            fresh = fresh.squeeze(1)
        fresh = fresh.detach().cpu().numpy()
    outputs, gradients = decoder_values(model, raw[:count]), decoder_gradients(model, raw[:count])
    differences = {
        'raw_embeddings': assert_agreement('raw embeddings', fresh, raw[:count]),
        'decoder_outputs': assert_agreement('decoder outputs', decoder_values(model, fresh), outputs),
        'decoder_gradients': assert_agreement('decoder gradients', decoder_gradients(model, fresh), gradients)}
    if reference is not None:
        differences['source_outputs'] = assert_agreement('source outputs', outputs, reference['probe_outputs'])
        differences['source_gradients'] = assert_agreement('source gradients', gradients, reference['probe_gradients'])
    LOG.info('HANDOFF validation passed: sample=%d rtol=%g atol=%g max_absolute_differences=%s', count, RTOL, ATOL, differences)
    return outputs, gradients, {'sample_rows': list(range(count)), 'rtol': RTOL, 'atol': ATOL, 'max_absolute_differences': differences}


def save_handoff(workspace, context, embeddings, fit_path, result, population, provenance):
    from temporal_gpu_execution import GPU_POLICY
    training = getattr(embeddings, "training_provenance", None)
    header = {"schema": SCHEMA if training is None else DERIVED_SCHEMA, "context": context}
    if training is not None:
        header.update(training_provenance=jsonable(training), training_provenance_sha256=digest(training))
    validated = _handoff_training_provenance(header)
    if validated["origin"] == "retained_rows_v1":
        indices = np.asarray(validated["mapping"]["indices"], dtype=np.int64)
        if not np.array_equal(embeddings.train_raw, embeddings.test_raw[indices]):
            raise ValueError("Derived training arrays disagree with mapped retained embeddings")
    root = Path(workspace)/'handoff'
    root.mkdir(parents=True, exist_ok=True)
    evaluation = np.asarray(context['job']['evaluation'])
    domains = np.array([context['job']['domains'][str(y)] for y in population.years[evaluation]])
    LOG.info('Saving completed raw embeddings and verifying decoder reuse before handoff')
    # Persist expensive outputs before even the small numerical verification.
    values = {'train_raw': embeddings.train_raw, 'test_raw': embeddings.test_raw,
              'probabilities': result.probabilities, 'classes': result.classes}
    files = {}
    def save_array(name, value):
        a = np.asarray(value); path = root/(name+'.npy')
        with path.with_suffix('.tmp').open('wb') as handle:
            np.save(handle, a, allow_pickle=False)
        os.replace(path.with_suffix('.tmp'), path)
        files[name] = {'path': path.name, 'sha256': file_sha256(path), 'shape': list(a.shape), 'dtype': a.dtype.str}
    for name, value in values.items():
        save_array(name, value)
    shutil.copy2(fit_path, root/'fitted_state.pkl')
    files['fitted_state'] = {'path': 'fitted_state.pkl', 'sha256': file_sha256(root/'fitted_state.pkl')}
    outputs, gradients, validation = validate_decoder(embeddings.require_model(), population.X[evaluation], domains,
                                                       embeddings.test_raw, sample_size=GPU_POLICY['embedding_batch'])
    save_array('probe_outputs', outputs); save_array('probe_gradients', gradients)
    manifest = root/'manifest.json'
    atomic_write_json(manifest, {**header, 'complete': True, 'context': context, 'files': files,
        'training_years': population.years[np.asarray(context['job']['train'])],
        'model_info': result.model_info, 'source_gpu_policy': dict(GPU_POLICY),
        'provenance': provenance, 'validation': validation})
    LOG.info('HANDOFF ready: %s; SAE training has not started', manifest)
    return manifest


def validate_import(meta, arrays, fit, population, workspace):
    from temporal_gpu_execution import GPU_POLICY, configure_gpu_model
    import torch
    requested = dict(GPU_POLICY)
    # Match the source sample and arithmetic policy before testing destination tuning.
    try:
        GPU_POLICY.update(meta['source_gpu_policy'])
        configure_gpu_model(fit['model'])
        evaluation = np.asarray(meta['context']['job']['evaluation'])
        domains = np.array([meta['context']['job']['domains'][str(y)] for y in population.years[evaluation]])
        _, _, report = validate_decoder(fit['model'], population.X[evaluation], domains, arrays['test_raw'],
            reference=arrays, sample_size=len(arrays['probe_outputs']))
    finally:
        GPU_POLICY.update(requested)
        configure_gpu_model(fit['model'])
    atomic_write_json(Path(workspace)/'handoff_validation.json', {**report, 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'destination_policy': requested})


def export_package(repo, handoff, destination):
    """Copy an independent repository snapshot, including retained upstream results."""
    repo, destination = Path(repo).resolve(), Path(destination).resolve()
    checked_files(handoff)
    if destination.exists():
        raise FileExistsError(f'Export destination already exists: {destination}')
    destination.mkdir(parents=True)
    target = destination/'repo'; target.mkdir()
    LOG.info('Exporting portable repository to %s (manifest-linked research artifacts only)', target)
    names = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'], cwd=repo).decode().split('\0')
    for name in names:
        source = repo/name
        if not name or not source.is_file() or source.resolve().is_relative_to(destination) or source.suffix in {'.pyc', '.log'}:
            continue
        if name.split('/')[0] in {'stats', 'handoff', 'handoffs', '.venv', '.git', '.env'}:
            continue
        out = target/name; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source,out)
    def copy_verified_artifact(manifest_path, manifest, name, source_root, target_root):
        from artifact_storage import validate_descriptor
        descriptor = manifest['aggregate_artifacts'][name] if 'aggregate_artifacts' in manifest else manifest['artifacts'][name]
        validate_descriptor(Path(manifest_path).parent, descriptor)
        source = Path(manifest_path).parent/descriptor['path']
        relative = source.relative_to(source_root)
        destination_path = target_root/relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination_path)

    # Keep the complete upstream provenance chain and only manifest-linked data.
    # The robustness tree contains 1.2M disposable cache files that the resume
    # path never reads.
    from temporal_concept_forecasting import PARENT, WINDOWS, ENRICHMENT, CRI, checked_manifest
    parent_root = repo/'stats/temporal_robustness'/PARENT
    target_parent = target/'stats/temporal_robustness'/PARENT
    target_parent.mkdir(parents=True, exist_ok=True)
    pp = parent_root/'parent_manifest.json'
    parent = checked_manifest(pp)
    shutil.copy2(pp, target_parent/'parent_manifest.json')
    for name in parent['aggregate_artifacts']:
        copy_verified_artifact(pp,parent,name,parent_root,target_parent)
    shutil.copy2(parent_root/'summary.json',target_parent/'summary.json')
    for success in parent['successful_experiments']:
        old=Path(success['manifest'])
        relative=Path(old.parent.parent.name)/old.parent.name/old.name
        source=parent_root/relative
        if file_sha256(source)!=success['manifest_fingerprint']:
            raise ValueError(f'Parent split changed during export: {source}')
        split=checked_manifest(source,('reference_roles',))
        destination_split=target_parent/relative
        destination_split.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination_split)
        copy_verified_artifact(source,split,'reference_roles',parent_root,target_parent)

    # Select the one prepared population whose fingerprint matches the parent.
    import pickle
    from temporal_robustness import _fingerprint_population
    from temporal_production import ProductionTemporalAdapter
    selected_population=None
    for source in sorted((repo/'stats/temporal_robustness/_population_cache/prepared').glob('*/prepared.pkl')):
        with source.open('rb') as handle:
            prepared=pickle.load(handle)
        population=ProductionTemporalAdapter()._population_from_prepared(prepared)
        if _fingerprint_population(population)==parent['population_fingerprints']:
            relative=source.relative_to(repo/'stats/temporal_robustness')
            destination_path=target/'stats/temporal_robustness'/relative
            destination_path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination_path)
            selected_population=str(relative)
            break
    if selected_population is None:
        raise ValueError('No retained prepared population matches parent during export')

    # Forecast reconstruction needs only these four checked window artifacts.
    wp=repo/'stats/temporal_performance_windows'/WINDOWS/'manifest.json'
    wm=checked_manifest(wp,('record_probabilities','thresholds','yearly_metrics','legacy_parent_metric_parity'))
    target_windows=target/'stats/temporal_performance_windows'/WINDOWS
    target_windows.mkdir(parents=True,exist_ok=True)
    shutil.copy2(wp,target_windows/'manifest.json')
    for name in ('record_probabilities','thresholds','yearly_metrics','legacy_parent_metric_parity'):
        copy_verified_artifact(wp,wm,name,wp.parent,target_windows)

    # Original concept forecasts require the retained enrichment and CRI tables.
    for folder,manifest_name,required in (
        (repo/'stats/temporal_robustness'/PARENT/'derived'/ENRICHMENT,'manifest.json',('headline_factor_metrics','tcav_significance')),
        (repo/'stats/temporal_robustness'/PARENT/'derived'/CRI,'manifest.json',('cri_family_universe',))):
        source_manifest=folder/manifest_name
        checked=checked_manifest(source_manifest,required)
        relative=folder.relative_to(repo/'stats/temporal_robustness')
        target_folder=target/'stats/temporal_robustness'/relative
        target_folder.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_manifest,target_folder/manifest_name)
        for name in required:
            copy_verified_artifact(source_manifest,checked,name,folder,target_folder)

    # Completed forecast outputs and the current resumable concept pilot.
    for name in ('stats/temporal_concept_forecasting',):
        if (repo/name).exists():
            LOG.info('Exporting %s',name)
            shutil.copytree(repo/name,target/name,dirs_exist_ok=True,ignore=shutil.ignore_patterns('*.tmp','*.log','__pycache__'))
    for run_name in ('window_concepts_4d3ee7449803eaeb1e68',):
        source=repo/'stats/temporal_window_concepts'/run_name
        if source.exists():
            LOG.info('Exporting resumable concept results %s',run_name)
            shutil.copytree(source,target/'stats/temporal_window_concepts'/run_name,dirs_exist_ok=True,
                ignore=shutil.ignore_patterns('*.tmp','*.log','__pycache__'))
    shutil.copytree(Path(handoff).parent, target/'handoff', dirs_exist_ok=True)
    import tabpfn
    shutil.copytree(Path(tabpfn.__file__).parent, destination/'vendor/tabpfn', ignore=shutil.ignore_patterns('__pycache__'))
    packages = sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions() if d.metadata.get('Name'))
    import torch
    cuda_index = '' if torch.version.cuda is None else '--extra-index-url https://download.pytorch.org/whl/cu' + torch.version.cuda.replace('.', '') + '\n'
    (destination/'requirements-lock.txt').write_text(cuda_index + '\n'.join(
        f'{n}=={v}' for n,v in packages if n.lower() not in {'tabpfn', 'drift-resilient-tabpfn'})+'\n')
    installed = importlib.metadata.distribution('drift-resilient-tabpfn')
    shutil.copytree(installed._path, destination/'vendor'/installed._path.name)

    atomic_write_json(destination/'environment.json', {'python':sys.version,'platform':platform.platform(),'packages':packages})
    (destination/'README.md').write_text('''# Temporal experiment handoff

Contains patient-level research data; transfer to your intended research machine.
Use Python 3.11, create a new venv, and install requirements-lock.txt. The source
Torch/CUDA wheels are pinned; install a compatible CUDA wheel if the new driver
requires it. The bundled TabPFN source and checkpoint take precedence on PYTHONPATH.

From this directory:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
export PYTHONPATH="$PWD/vendor:$PWD/repo"
cd repo
../.venv/bin/python temporal_handoff.py verify --package ..
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../.venv/bin/python temporal_handoff.py benchmark --handoff handoff/manifest.json --output gpu-benchmark.json
```

Review gpu-benchmark.json. It contains a recommended policy only if a candidate
passes numerical agreement and stays below 80% of GPU memory. Then launch:

```bash
mkdir -p logs
tmux new-session -s temporal-resume 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 ../.venv/bin/python -u run_temporal_concept_forecasting.py --device cuda --gpu-memory-safe --gpu-settings gpu-benchmark.json --import-handoff handoff/manifest.json >> logs/temporal-resume.log 2>&1'
```

Import verifies membership, ordering, domains, checksums and decoder agreement
(rtol=0.001, atol=0.0001) before SAE training. It does one small transformer
validation sample; complete embedding extraction and prediction are skipped for
the imported system. Other fitted systems still require their own extraction.
''')
    files = {}
    for index, path in enumerate(sorted(destination.rglob('*'))):
        if path.is_file():
            files[str(path.relative_to(destination))] = file_sha256(path)
            if index % 500 == 0:
                LOG.info('Checksumming transfer package: %d entries', index)
    atomic_write_json(destination/'package_manifest.json', {'complete': True, 'retained_population': selected_population, 'files': files})
    LOG.info('Portable package complete: %s', destination)
    return destination


def verify_package(root):
    root = Path(root).resolve()
    manifest = json.loads((root/'package_manifest.json').read_text())
    if manifest.get('complete') is not True:
        raise ValueError('Incomplete package')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError(f'Package checksum/path mismatch: {name}')
    LOG.info('All %d package checksums verified', len(manifest['files']))


def benchmark(handoff, output, batches, factors, headroom=0.20):
    import torch
    from temporal_gpu_execution import GPU_POLICY, configure_gpu_model
    from temporal_production import ProductionTemporalAdapter
    from temporal_config import TemporalRobustnessConfig
    from temporal_concept_forecasting import PARENT
    meta, arrays, fit = load_handoff(handoff)
    repo = Path(__file__).resolve().parent
    pp = repo/'stats/temporal_robustness'/PARENT/'parent_manifest.json'
    parent = json.loads(pp.read_text()); cfg = parent['config'].copy()
    for name in ('comparison_config_path','semantic_config_path','dataset_path'):
        cfg[name] = str(repo/Path(cfg[name]).name)
    cfg['device'] = 'cuda'
    loader = ProductionTemporalAdapter()
    population = loader.load_retained_population(TemporalRobustnessConfig.from_dict(cfg), pp.parent.parent, parent['population_fingerprints'])
    if any(n < 1 for n in batches + factors):
        raise ValueError('Benchmark batch sizes and memory factors must be positive')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    validate_import(meta, arrays, fit, population, Path(output).resolve().parent)
    indices = np.asarray(meta['context']['job']['evaluation'])
    domains = np.array([meta['context']['job']['domains'][str(y)] for y in population.years[indices]])
    model = fit['model']; results = []
    for factor in factors:
        for batch in batches:
            if batch > len(indices):
                continue
            GPU_POLICY.update(peak_memory_factor=factor, embedding_batch=batch, prediction_batch=batch)
            configure_gpu_model(model)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            free, total = torch.cuda.mem_get_info()
            external = total-free-torch.cuda.memory_reserved()
            start = time.monotonic(); row = {'batch':batch, 'peak_memory_factor':factor}
            try:
                with torch.no_grad():
                    domain = torch.tensor(domains[:batch], dtype=torch.long).reshape(-1,1,1)
                    x = np.asarray(population.X[indices[:batch]], dtype=np.float32)
                    emb = model.get_embeddings(x, additional_x={'dist_shift_domain':domain})
                    emb = emb.detach().cpu().numpy().reshape(batch,-1)
                    assert_agreement('benchmark embeddings', emb, arrays['test_raw'][:batch])
                    prediction_rows = np.asarray(meta['context']['predict_indices'])[:batch]
                    prediction_domains = torch.tensor([meta['context']['job']['domains'][str(y)] for y in
                        population.years[prediction_rows]], dtype=torch.long).reshape(-1,1,1)
                    pred = model.predict_proba(np.asarray(population.X[prediction_rows], dtype=np.float32),
                        additional_x={'dist_shift_domain':prediction_domains})
                    assert_agreement('benchmark predictions', np.asarray(pred), arrays['probabilities'][:len(prediction_rows)])
                    gradients = decoder_gradients(model, emb)
                    if not np.isfinite(gradients).all():
                        raise ValueError('Non-finite benchmark gradients')
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_reserved()
                row.update(seconds=time.monotonic()-start, peak_reserved_bytes=peak, external_bytes=external,
                           accepted=(peak+external <= total*(1-headroom)))
                row['records_per_second'] = batch/row['seconds']
            except (torch.cuda.OutOfMemoryError, ValueError) as error:
                row.update(accepted=False, error=str(error))
                error.__traceback__ = None
            results.append(row)
            LOG.info('GPU benchmark: %s', row)
            atomic_write_json(output, {'complete':False,'results':results})
            import gc
            gc.collect(); torch.cuda.empty_cache()
            if not row['accepted']:
                break  # Increasing this batch further cannot improve headroom.
    accepted = [r for r in results if r['accepted']]
    best = max(accepted, key=lambda r:r['records_per_second']) if accepted else None
    policy = None if best is None else {**GPU_POLICY, 'prediction_batch':best['batch'], 'embedding_batch':best['batch'],
        'gradient_batch':best['batch'], 'peak_memory_factor':best['peak_memory_factor']}
    atomic_write_json(output, {'complete':True,'gpu':torch.cuda.get_device_name(),'headroom':headroom,
        'results':results,'recommended_policy':policy})
    if best is None:
        raise RuntimeError('No numerically valid GPU setting with requested headroom; see benchmark report')
    LOG.info('Recommended measured GPU policy: %s', policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p=sub.add_parser('export'); p.add_argument('--repo', type=Path, default=Path.cwd()); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify'); p.add_argument('--package',type=Path,required=True)
    p=sub.add_parser('benchmark'); p.add_argument('--handoff',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batches',type=int,nargs='+',default=[16,32,64]); p.add_argument('--factors',type=int,nargs='+',default=[64,32,16,8])
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    if args.command=='export': export_package(args.repo,args.handoff,args.output)
    elif args.command=='verify': verify_package(args.package)
    else: benchmark(args.handoff,args.output,args.batches,args.factors)


if __name__=='__main__':
    main()
