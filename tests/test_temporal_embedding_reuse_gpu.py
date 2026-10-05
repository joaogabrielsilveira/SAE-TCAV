"""Bounded real-CUDA smoke: skipping the temporal training-query pass preserves retained results.

The temporal injected-fit path derives training embeddings from the retained rows instead of
running a separate training query. This test replays both arms on a real, immutable, completed
fit and compares them on CUDA:

* arm A: the current derived path (``train_to_retained_indices`` mapping, no training query);
* arm B: the legacy path (the full training-query extraction first, then the retained extraction).

Both arms call ``DefaultComparisonAdapter.embeddings`` with a freshly unpickled model, the
production TF32 setting and the GPU memory-safe policy. Only a bounded, deterministic sample of
retained batches is extracted. It always contains the first batch, the final (partial) retained
batch, the batches holding the last training-query rows and batches holding discovery rows.

Required (exact, bitwise) results: retained raw embeddings, decoder outputs and decoder
gradients agree between arms; derived ``train_raw`` equals ``retained_raw[mapping]``; arm A
issues no training query while arm B does; the discovery-only scaler is identical; the fixture
is byte-for-byte unchanged. Legacy-versus-derived training rows and the archived arrays are
reported separately and never used to excuse a mismatch. The scaler checks run on CPU
(scikit-learn) and are labelled as such.

Fixture: ``TEMPORAL_REUSE_SMOKE_MANIFEST`` is the path of a completed fit's ``completed.json``
(``.../window_concepts_<run>/fits/<job>/reference_<year>/split_<seed>/completed.json``). The
fitted state is read from its ``attempt_cuda/reference_<year>/split_<seed>`` directory. The job
membership is rebuilt exactly as the window runner builds it and must reproduce ``<job>``.
Nothing is ever written below the fixture or its data repository.

Environment variables:
  TEMPORAL_REUSE_SMOKE_MANIFEST        required; the test skips when unset or CUDA is unavailable
  TEMPORAL_REUSE_SMOKE_OUT             JSON summary path (``*.json``) or directory; default tmp_path
  TEMPORAL_REUSE_SMOKE_WINDOW          logical window of the fit (default reference_only_common)
  TEMPORAL_REUSE_SMOKE_BATCHES         random extra retained batches (default 8)
  TEMPORAL_REUSE_SMOKE_TRAIN_TAIL_ROWS final training-query rows whose batches are covered (default 16)
  TEMPORAL_REUSE_SMOKE_MIN_FIT_ROWS    discovery rows the sample must cover (default 8)
  TEMPORAL_REUSE_SMOKE_SEED            sampling seed (default 11)
  TEMPORAL_REUSE_SMOKE_MAX_BUSY_MIB    refuse to run when the GPU already holds more (default 1024)
  TEMPORAL_REUSE_SMOKE_PREFLIGHT       set to 1 to run only the read-only CPU fixture preflight

Run on an idle GPU (no sharing), from a clean checkout::

    env TEMPORAL_REUSE_SMOKE_MANIFEST=<completed.json> TEMPORAL_REUSE_SMOKE_OUT=<out>/summary.json \\
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        ~/SAE-TCAV/.venv/bin/python -B -m pytest -q -rs -p no:cacheprovider \\
        tests/test_temporal_embedding_reuse_gpu.py
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import datetime
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

ENV_MANIFEST = "TEMPORAL_REUSE_SMOKE_MANIFEST"
ENV_OUT = "TEMPORAL_REUSE_SMOKE_OUT"
ENV_WINDOW = "TEMPORAL_REUSE_SMOKE_WINDOW"
ENV_BATCHES = "TEMPORAL_REUSE_SMOKE_BATCHES"
ENV_TRAIN_TAIL = "TEMPORAL_REUSE_SMOKE_TRAIN_TAIL_ROWS"
ENV_MIN_FIT = "TEMPORAL_REUSE_SMOKE_MIN_FIT_ROWS"
ENV_SEED = "TEMPORAL_REUSE_SMOKE_SEED"
ENV_MAX_BUSY = "TEMPORAL_REUSE_SMOKE_MAX_BUSY_MIB"
ENV_PREFLIGHT = "TEMPORAL_REUSE_SMOKE_PREFLIGHT"
SUMMARY_NAME = "temporal_embedding_reuse_smoke.json"
SCHEMA = "temporal_embedding_reuse_smoke_v1"
REQUIRED_DEVICE = "cuda"
TRAIN_DESCRIPTION = "Extracting train embeddings"
TEST_DESCRIPTION = "Extracting test embeddings"


# --------------------------------------------------------------------------- pure helpers


def smoke_manifest_path(environ=None, cuda_available=None):
    """Return the fixture manifest path, or skip before reading anything heavy."""
    environ = os.environ if environ is None else environ
    raw = str(environ.get(ENV_MANIFEST, "")).strip()
    if not raw:
        pytest.skip(f"{ENV_MANIFEST} is not set; no completed fit to replay")
    if cuda_available is None:
        import torch
        cuda_available = torch.cuda.is_available()
    if not cuda_available:
        pytest.skip("CUDA unavailable: the skipped-training-pass smoke needs a CUDA device, "
                    "there is no CPU fallback")
    return Path(raw).expanduser()


def fixture_layout(manifest):
    """Locate the immutable fit, its fitted state and its data repository from ``completed.json``."""
    path = Path(manifest).expanduser().resolve()
    if path.name != "completed.json" or len(path.parents) < 8:
        raise ValueError(f"manifest must be a completed.json inside a window-concepts run: {path}")
    split_dir, reference_dir, job_dir, fits_dir, run_root = path.parents[:5]
    reference = re.fullmatch(r"reference_(\d{4})", reference_dir.name)
    split = re.fullmatch(r"split_(\d+)", split_dir.name)
    if (not reference or not split or fits_dir.name != "fits"
            or not run_root.name.startswith("window_concepts_")
            or run_root.parent.name != "temporal_window_concepts" or run_root.parent.parent.name != "stats"):
        raise ValueError(f"unexpected completed-fit location: {path}")
    return SimpleNamespace(
        manifest=path, split_dir=split_dir, run_root=run_root, data_repo=run_root.parents[2],
        attempt_dir=split_dir / "attempt_cuda" / reference_dir.name / split_dir.name,
        reference_year=int(reference.group(1)), patient_split_seed=int(split.group(1)),
        job_identity=job_dir.name, run_identity=run_root.name.removeprefix("window_concepts_"))


def resolve_summary_path(environ, default_dir, forbidden_roots=()):
    raw = str(environ.get(ENV_OUT, "")).strip()
    if raw:
        target = Path(raw).expanduser()
        target = target if target.suffix == ".json" else target / SUMMARY_NAME
    else:
        target = Path(default_dir) / SUMMARY_NAME
    target = target.resolve()
    for root in forbidden_roots:
        root = Path(root).resolve()
        if target == root or root in target.parents:
            raise ValueError(f"refusing to write the summary inside the protected root {root}")
    return target


def tree_listing_fingerprint(root):
    """Fingerprint (relative path, kind, size, mtime) of everything below root; reads no content."""
    root = Path(root)
    digest = hashlib.sha256()
    files = size = 0
    for path in sorted(root.rglob("*")):
        status = path.lstat()
        is_file = path.is_file() and not path.is_symlink()
        digest.update(f"{path.relative_to(root).as_posix()}\0{'f' if is_file else 'd'}\0"
                      f"{status.st_size if is_file else 0}\0{status.st_mtime_ns}\n".encode())
        files += is_file
        size += status.st_size if is_file else 0
    return {"method": "listing", "files": int(files), "bytes": int(size), "sha256": digest.hexdigest()}


def select_retained_blocks(n_rows, batch_size, train_positions, discovery_positions, *,
                           extra_blocks, train_tail_rows, min_fit_rows, seed):
    """Choose production-aligned retained batches: first, last, training tail, discovery, random."""
    if n_rows < 1 or batch_size < 1 or min(extra_blocks, train_tail_rows, min_fit_rows) < 0:
        raise ValueError("invalid sample bounds")
    train_positions = np.asarray(train_positions, dtype=np.int64)
    discovery_positions = np.asarray(discovery_positions, dtype=np.int64)
    for name, values in (("training", train_positions), ("discovery", discovery_positions)):
        if values.size and (values.min() < 0 or values.max() >= n_rows):
            raise ValueError(f"{name} positions are outside the retained population")
    if discovery_positions.size == 0:
        raise ValueError("the sample needs discovery rows to fit the discovery-only scaler")
    n_blocks = -(-n_rows // batch_size)
    chosen = {0, n_blocks - 1}
    if train_tail_rows and train_positions.size:
        chosen |= {int(position) // batch_size for position in train_positions[-train_tail_rows:]}
    per_block = Counter(int(position) // batch_size for position in discovery_positions)
    needed = min(min_fit_rows, int(discovery_positions.size))
    covered = sum(count for block, count in per_block.items() if block in chosen)
    for block, count in sorted(per_block.items(), key=lambda item: (-item[1], item[0])):
        if covered >= needed:
            break
        if block not in chosen:
            chosen.add(block)
            covered += count
    pool = np.array(sorted(set(range(n_blocks)) - chosen), dtype=np.int64)
    if extra_blocks and pool.size:
        picked = np.random.default_rng(seed).choice(pool, size=min(extra_blocks, pool.size), replace=False)
        chosen |= {int(block) for block in picked}
    return sorted(chosen)


def block_rows(blocks, n_rows, batch_size):
    """Retained positions of the chosen batches, ascending, so each keeps its production alignment."""
    blocks = [int(block) for block in blocks]
    if blocks != sorted(set(blocks)) or not blocks or blocks[0] < 0 or blocks[-1] * batch_size >= n_rows:
        raise ValueError("blocks must be unique, ascending and inside the retained population")
    return np.concatenate([np.arange(block * batch_size, min((block + 1) * batch_size, n_rows))
                           for block in blocks]).astype(np.int64)


def exact_report(label, actual, expected, *, max_listed=20):
    """Bitwise comparison; signed zeros, NaN payloads and one-ulp differences all count."""
    a, e = np.ascontiguousarray(actual), np.ascontiguousarray(expected)
    report = {"label": label, "shape": list(a.shape), "expected_shape": list(e.shape),
              "dtype": str(a.dtype), "expected_dtype": str(e.dtype)}
    if a.shape != e.shape or a.dtype != e.dtype or a.size == 0:
        return {**report, "equal": False, "rows": int(len(a)) if a.ndim else 0, "n_differing": None,
                "differing_rows": [], "reason": "shape, dtype or emptiness differs"}
    rows = len(a)
    differing = np.flatnonzero((a.view(np.uint8).reshape(rows, -1) != e.view(np.uint8).reshape(rows, -1)).any(axis=1))
    report.update(equal=bool(differing.size == 0), rows=int(rows), n_differing=int(differing.size),
                  differing_rows=[int(row) for row in differing[:max_listed]],
                  sha256_actual=hashlib.sha256(a.tobytes()).hexdigest(),
                  sha256_expected=hashlib.sha256(e.tobytes()).hexdigest())
    if a.dtype.kind == "f":
        delta = np.abs(a.astype(np.float64) - e.astype(np.float64))
        report["max_abs_difference"] = float(np.max(delta)) if np.isfinite(delta).all() else None
    return report


def make_extraction_recorder(real, calls, synchronize=lambda: None):
    """Wrap the extraction entry point; a plain function, since callers fingerprint its source."""
    def recording_extract(**kwargs):
        synchronize()
        started = time.perf_counter()
        result = real(**kwargs)
        synchronize()
        cfg = kwargs["cfg"]
        calls.append({"description": str(cfg.progress_desc), "rows": int(len(kwargs["X"])),
                      "batch_size": int(cfg.batch_size), "strict_domains": bool(cfg.strict_domains),
                      "seconds": time.perf_counter() - started})
        return result
    return recording_extract


# --------------------------------------------------------------------------- fixture and job


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _git(*args):
    try:
        done = subprocess.run(["git", *args], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                              text=True, timeout=60, check=True)
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_job(layout, window, scratch, device):
    """Rebuild the fit's job exactly as the window runner builds it; reads only, all CPU."""
    from artifact_storage import file_sha256
    import temporal_window_concepts as runner  # sets the allocator policy before CUDA initializes
    from temporal_concept_forecasting import PARENT, WINDOWS, checked_manifest, digest
    from comparison_runner import _tabpfn_checkpoint_fingerprint
    from semantic_artifacts import array_fingerprint
    from temporal_config import TemporalRobustnessConfig
    from temporal_gpu_execution import GPU_POLICY
    from temporal_metric_synthesis import _numerical_environment
    from temporal_performance_windows import (
        WindowExperimentConfig, build_training_indices, effective_window_years,
        model_domain_mapping, post_death_exclusion_mask)
    from temporal_production import ProductionTemporalAdapter, _train_to_retained_mapping

    completed = checked_manifest(layout.manifest)
    _require(completed.get("identity") == layout.job_identity, "completed.json identity differs from its directory")
    _require(completed.get("actual_device") == REQUIRED_DEVICE, "the fixture fit did not run on CUDA")
    state = json.loads((layout.run_root / "run_identity.json").read_text())
    inputs = state["inputs"]
    _require(digest(inputs)[:20] == state["identity"] == layout.run_identity, "run identity is inconsistent")
    fit_identity = digest({k: v for k, v in inputs.items() if k != "selection"})[:20]

    repo = layout.data_repo
    parent_path = repo / "stats/temporal_robustness" / PARENT / "parent_manifest.json"
    windows_path = repo / "stats/temporal_performance_windows" / WINDOWS / "manifest.json"
    parent = checked_manifest(parent_path)
    windows = checked_manifest(windows_path, ("record_probabilities", "thresholds", "role_exposure_audit"))
    _require(windows["dependency_sha256"]["parent_manifest"] == file_sha256(parent_path),
             "window and parent manifests disagree")
    _require((inputs["parent"], inputs["windows"]) == (file_sha256(parent_path), file_sha256(windows_path)),
             "the fixture run used different parent or window manifests")
    cfg = dict(parent["config"])
    for name in ("comparison_config_path", "semantic_config_path", "dataset_path"):
        candidate = Path(cfg[name])
        cfg[name] = str(candidate if candidate.is_file() else repo / candidate.name)
    for name in ("comparison_config_path", "semantic_config_path"):
        actual = file_sha256(cfg[name])
        _require(actual == parent["dependent_config_fingerprints"][name]["sha256"] == inputs["configs"][name],
                 f"scientific configuration differs from the fixture run: {name}")
    cfg.update(device=device, artifact_dir=str(scratch), show_progress=False)
    config = TemporalRobustnessConfig.from_dict(cfg)
    loader = ProductionTemporalAdapter()
    population = loader.load_retained_population(config, parent_path.parent.parent, parent["population_fingerprints"])
    population.validate()
    window_config = WindowExperimentConfig.from_dict({
        **windows["config"], "parent_manifest": str(parent_path),
        "comparison_config_path": cfg["comparison_config_path"], "device": device,
        "artifact_dir": str(scratch), "show_progress": False})
    _require(_tabpfn_checkpoint_fingerprint(loader._base_config.tabpfn.model_name) == inputs["model_checkpoint"],
             "the model checkpoint differs from the fixture run")
    _require(_numerical_environment(device) == inputs["environment"],
             "the numerical environment differs from the fixture run; archived comparisons would be void")
    _require(dict(GPU_POLICY) == inputs["gpu_memory_policy"], "the GPU memory policy differs from the fixture run")

    ref, seed = layout.reference_year, layout.patient_split_seed
    keep, _ = post_death_exclusion_mask(population.patient_ids, population.years, population.outcomes)
    reference_roles = runner._reference_roles(parent_path, parent, ref, seed)
    years = effective_window_years(window, ref)
    roles, _ = runner.historical_roles(population, ref, seed, reference_roles, years, keep)
    train = build_training_indices(population=population, reference_year=ref, logical_window=window,
                                   global_roles=reference_roles, common_keep=keep)
    evaluation = np.flatnonzero((population.years >= min(years)) & keep)
    domains = model_domain_mapping(population.years[train], population.years[evaluation])
    job_inputs = {"ref": ref, "seed": seed, "years": years, "train": train, "roles": roles,
                  "evaluation": evaluation, "domains": domains}
    _require(digest({"run": fit_identity, **job_inputs})[:24] == layout.job_identity,
             "the rebuilt membership does not reproduce the fixture job identity; wrong window?")
    _require(completed["effective_years"] == list(years), "effective years differ from the fixture")
    _require(all(np.all(np.diff(a) > 0) for a in (train, evaluation)), "membership must be ascending and unique")

    fit_path = layout.attempt_dir / "fitted_state.pkl"
    fit_sha = file_sha256(fit_path)
    _require(json.loads(fit_path.with_suffix(".pkl.json").read_text())
             == {"identity": layout.job_identity, "sha256": fit_sha}, "fitted-state sidecar does not match")
    train_to_retained = _train_to_retained_mapping(population, train, evaluation, domains)
    _require(np.array_equal(evaluation[train_to_retained], train), "full-job mapping does not identify the training rows")
    return SimpleNamespace(
        layout=layout, window=window, population=population, loader=loader, config=config,
        window_config=window_config, reference_year=ref, seed=seed, years=years, train=train,
        roles=roles, evaluation=evaluation, domains=domains, job_inputs=job_inputs,
        job_identity=layout.job_identity, fit_path=fit_path, fit_sha=fit_sha, fixture_inputs=inputs,
        full_mapping=train_to_retained, context_sha256=digest(job_inputs),
        population_sha256=array_fingerprint(population.X))


def resolve_device():
    from runtime_acceleration import resolve_torch_device
    return resolve_torch_device(REQUIRED_DEVICE)


def fixture_state(job):
    """Everything that must be identical before and after the replay."""
    from artifact_storage import file_sha256
    from semantic_artifacts import array_fingerprint
    from temporal_concept_forecasting import digest
    layout = job.layout
    names = [layout.manifest, layout.run_root / "run_identity.json", job.fit_path,
             job.fit_path.with_suffix(".pkl.json"), layout.attempt_dir / "embeddings.npz",
             layout.attempt_dir / "embedding_scaler.pkl", layout.attempt_dir / "embedding_scaler_provenance.json"]
    return {"files": {path.name: file_sha256(path) for path in names if path.exists()},
            "tree": tree_listing_fingerprint(layout.split_dir),
            "context_sha256": digest(job.job_inputs),
            "population_features_sha256": array_fingerprint(job.population.X)}


def build_sample(job, *, batch_size, extra_blocks, train_tail_rows, min_fit_rows, seed):
    from temporal_production import _train_to_retained_mapping
    evaluation, train = job.evaluation, job.train
    train_positions = np.searchsorted(evaluation, train)
    discovery = np.asarray(job.roles["sae_discovery"])
    discovery_positions = np.searchsorted(evaluation, discovery)
    _require(np.array_equal(evaluation[train_positions], train)
             and np.array_equal(evaluation[discovery_positions], discovery),
             "training and discovery rows must be retained rows")
    blocks = select_retained_blocks(len(evaluation), batch_size, train_positions, discovery_positions,
                                    extra_blocks=extra_blocks, train_tail_rows=train_tail_rows,
                                    min_fit_rows=min_fit_rows, seed=seed)
    local = block_rows(blocks, len(evaluation), batch_size)
    position = {int(p): i for i, p in enumerate(local)}
    train_select = np.array([i for i, p in enumerate(train_positions) if int(p) in position], dtype=np.int64)
    retained = evaluation[local]
    train_sample = train[train_select]
    mapping = _train_to_retained_mapping(job.population, train_sample, retained, job.domains)
    _require(np.array_equal(retained[mapping], train_sample), "sample mapping does not identify the training rows")
    fit_local = np.array([position[int(p)] for p in discovery_positions if int(p) in position], dtype=np.int64)
    n_blocks = -(-len(evaluation) // batch_size)
    return SimpleNamespace(
        blocks=blocks, local=local, retained=retained, train_select=train_select, train=train_sample,
        mapping=mapping, fit_local=fit_local, batch_size=batch_size, n_blocks=n_blocks,
        tail_rows=len(evaluation) - (n_blocks - 1) * batch_size,
        train_tail_start=(len(train) // batch_size) * batch_size)


def prepared_rows(job, train_global, retained_global):
    from comparison_runner import _PreparedData
    population, source = job.population, job.loader._source_prepared
    return _PreparedData(
        train_rows=source.test_rows.iloc[train_global].reset_index(drop=True),
        test_rows=source.test_rows.iloc[retained_global].reset_index(drop=True),
        feature_names=population.feature_names,
        X_train=population.X[train_global], y_train=population.outcomes[train_global],
        years_train=population.years[train_global], X_test=population.X[retained_global],
        y_test=population.outcomes[retained_global], years_test=population.years[retained_global],
        patient_ids=population.patient_ids[retained_global], record_keys=population.record_keys[retained_global],
        domain_reference_year=int(job.reference_year))


def archived_coverage(job, sample, check):
    """CPU-only, read-only checks against the fixture's archived arrays and scaler."""
    from comparison_runner import _scale_embeddings_from_semantic_fit
    from semantic_artifacts import array_fingerprint
    layout = job.layout
    with np.load(layout.attempt_dir / "embeddings.npz", allow_pickle=False) as values:
        train_raw, test_raw = np.asarray(values["train_raw"]), np.asarray(values["test_raw"])
    check("archived_array_shapes", train_raw.shape[0] == len(job.train) and test_raw.shape[0] == len(job.evaluation),
          {"train_raw": list(train_raw.shape), "test_raw": list(test_raw.shape)})
    fit_full = np.searchsorted(job.evaluation, np.asarray(job.roles["sae_discovery"]))
    provenance = json.loads((layout.attempt_dir / "embedding_scaler_provenance.json").read_text())
    check("archived_scaler_fit_rows_match_provenance",
          array_fingerprint(fit_full) == provenance["fit_indices_fingerprint"]
          and len(fit_full) == provenance["fit_row_count"], {"fit_rows": int(len(fit_full))})
    _, test_scaled, scaler = _scale_embeddings_from_semantic_fit(train_raw, test_raw, fit_full)
    check("archived_scaler_reproduced_on_cpu",
          array_fingerprint(np.asarray(scaler.mean_)) == provenance["mean_fingerprint"]
          and array_fingerprint(np.asarray(scaler.scale_)) == provenance["scale_fingerprint"])
    with (layout.attempt_dir / "embedding_scaler.pkl").open("rb") as handle:
        stored = pickle.load(handle)
    check("archived_scaler_object_matches_recomputation",
          np.array_equal(stored.mean_, scaler.mean_) and np.array_equal(stored.scale_, scaler.scale_))
    derived = test_raw[job.full_mapping]
    _, derived_test_scaled, derived_scaler = _scale_embeddings_from_semantic_fit(derived, test_raw, fit_full)
    check("scaler_is_discovery_only_whatever_the_training_origin (cpu, archived arrays)",
          np.array_equal(derived_scaler.mean_, scaler.mean_) and np.array_equal(derived_scaler.scale_, scaler.scale_)
          and np.array_equal(derived_test_scaled, test_scaled))
    legacy = exact_report("archived legacy train_raw vs archived retained_raw[mapping]", derived, train_raw,
                          max_listed=len(job.train))
    partial = set(range(sample.train_tail_start, len(job.train)))
    check("archived_legacy_vs_derived_training_rows", True, {
        **{k: legacy[k] for k in ("n_differing", "rows", "differing_rows")},
        "confined_to_final_partial_training_batch": set(legacy["differing_rows"]) <= partial,
        "final_partial_training_batch_rows": sorted(partial)}, required=False)
    return SimpleNamespace(train_raw=train_raw, test_raw=test_raw, fit_full=fit_full)


@contextmanager
def production_precision(window_config):
    """The precision settings the production fit/prediction stage applies before extraction."""
    import torch
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
             torch.get_float32_matmul_precision())
    torch.backends.cuda.matmul.allow_tf32 = bool(window_config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(window_config.allow_tf32)
    torch.set_float32_matmul_precision("high" if window_config.allow_tf32 else "highest")
    try:
        yield {"allow_tf32": bool(window_config.allow_tf32),
               "float32_matmul_precision": torch.get_float32_matmul_precision()}
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[:2]
        torch.set_float32_matmul_precision(saved[2])


def load_fit(job):
    """A fresh, memory-safe-configured model from the immutable pickle; never checkpoints to disk."""
    from temporal_gpu_execution import configure_gpu_model
    from temporal_window_concepts import load_binary_checkpoint
    fit = load_binary_checkpoint(job.fit_path, job.job_identity)
    _require(fit is not None, "fitted state is missing")
    model = fit["model"]
    stale = [name for name in ("_temporal_batch_root", "_temporal_fit_identity") if hasattr(model, name)]
    for name in stale:
        delattr(model, name)
    configure_gpu_model(model)
    _require(getattr(model, "_temporal_batch_root", None) is None, "model must not checkpoint into the fixture")
    expected = np.array([job.domains[int(y)] for y in job.population.years[job.train]])
    _require(np.array_equal(fit["dist_shift_domain_train"], expected), "fitted training domains differ")
    return fit, stale


def runner_config(job):
    from dataclasses import replace
    from temporal_gpu_execution import GPU_POLICY
    from temporal_splits import ReferenceSplit
    base = job.loader._base_config
    split = ReferenceSplit(job.seed, job.seed, 0, {}, {})
    return replace(
        base, seed=split.effective_seed, use_cache=False, show_progress=False,
        accelerator=replace(base.accelerator, device=resolve_device()),
        tabpfn=replace(base.tabpfn, run_walkforward=False, batch_size=GPU_POLICY["embedding_batch"]))


def decoder_outputs(model, raw):
    from temporal_gpu_execution import GPU_POLICY
    from temporal_handoff import decoder_gradients, decoder_values
    step = GPU_POLICY["gradient_batch"]
    values = np.concatenate([decoder_values(model, raw[i:i + step]) for i in range(0, len(raw), step)])
    gradients = np.concatenate([decoder_gradients(model, raw[i:i + step]) for i in range(0, len(raw), step)])
    return values, gradients


def run_arm(name, job, sample, scratch, *, derived, controls):
    """One inference arm through the real adapter on a freshly loaded model."""
    import torch
    import tabpfn_model
    from comparison_runner import DefaultComparisonAdapter
    from tabpfn_model import EmbeddingExtractConfig, flatten_embeddings
    from temporal_gpu_execution import GPU_POLICY
    fit, stale = load_fit(job)
    model = fit["model"]
    train_global = sample.train if derived else job.train
    prepared = prepared_rows(job, train_global, sample.retained)
    workspace = scratch / name
    workspace.mkdir()
    calls = []
    recorder = make_extraction_recorder(tabpfn_model.extract_embeddings_robust, calls, torch.cuda.synchronize)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with mock.patch.object(tabpfn_model, "extract_embeddings_robust", recorder):
        result = DefaultComparisonAdapter(None).embeddings(
            prepared, {"idx_semantic_fit": sample.fit_local}, runner_config(job), workspace, force=False,
            fitted_state=fit, explicit_domain_map=job.domains, fitted_identity=job.fit_sha,
            train_to_retained_indices=sample.mapping if derived else None)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    peak = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())}
    started = time.perf_counter()
    outputs, gradients = decoder_outputs(model, result.test_raw)
    torch.cuda.synchronize()
    decoder_seconds = time.perf_counter() - started
    handoff = None
    if derived:
        from temporal_handoff import validate_decoder
        first = np.arange(min(GPU_POLICY["embedding_batch"], len(sample.retained)))
        domains = np.array([job.domains[int(y)] for y in job.population.years[sample.retained[first]]])
        try:
            _, _, handoff = validate_decoder(model, job.population.X[sample.retained[first]], domains,
                                             result.test_raw, sample_size=len(first))
        except ValueError as error:
            handoff = {"error": str(error)}
    repeats = []
    if controls:
        extraction = EmbeddingExtractConfig()
        extraction.strict_domains, extraction.batch_size = True, GPU_POLICY["embedding_batch"]
        extraction.use_cache, extraction.show_progress = False, False
        for block in sorted({sample.blocks[0], sample.blocks[-1]}):
            rows = np.flatnonzero(np.isin(sample.local // sample.batch_size, [block]))
            again = flatten_embeddings(tabpfn_model.extract_embeddings_robust(
                model=model, X=prepared.X_test[rows], years=prepared.years_test[rows],
                year_to_domain_map=job.domains, cfg=extraction, device=fit["model_add_x_device"],
                is_train=True, ctx_idx=None, example_add_shape=fit["example_add_shape"]))
            repeats.append(exact_report(f"repeat of retained batch {block}", again, result.test_raw[rows]))
    arm = SimpleNamespace(
        name=name, test_raw=np.asarray(result.test_raw), train_raw=np.asarray(result.train_raw),
        test_scaled=np.asarray(result.test_scaled), train_scaled=np.asarray(result.train_scaled),
        scaler=result.scaler, outputs=outputs, gradients=gradients, calls=calls, handoff=handoff,
        provenance=result.training_provenance, repeats=repeats, stale_attributes=stale,
        metrics={"extraction_seconds": seconds, "decoder_seconds": decoder_seconds, **peak,
                 "training_origin": result.training_provenance["origin"], "extraction_calls": calls})
    del result, model, fit, prepared
    gc.collect()
    torch.cuda.empty_cache()
    return arm


def source_comparison(job):
    from artifact_storage import file_sha256
    root = Path(__file__).resolve().parents[1]
    return {name: {"fixture": recorded, "current": file_sha256(root / name) if (root / name).exists() else None}
            for name, recorded in sorted(job.fixture_inputs["sources"].items())}


def new_checker(checks):
    def check(name, passed, detail=None, required=True):
        checks.append({"name": name, "passed": bool(passed), "required": bool(required), "detail": detail})
        return bool(passed)
    return check


def sample_parameters(environ):
    return {"extra_blocks": int(environ.get(ENV_BATCHES, 8)), "train_tail_rows": int(environ.get(ENV_TRAIN_TAIL, 16)),
            "min_fit_rows": int(environ.get(ENV_MIN_FIT, 8)), "seed": int(environ.get(ENV_SEED, 11))}


def describe_sample(job, sample, train_tail_rows):
    return {"batch_size": sample.batch_size, "retained_rows_total": int(len(job.evaluation)),
            "training_rows_total": int(len(job.train)), "retained_batches_total": int(sample.n_blocks),
            "sampled_batch_indices": [int(b) for b in sample.blocks],
            "sampled_batch_row_ranges": [[int(b * sample.batch_size), int(min((b + 1) * sample.batch_size, len(job.evaluation)))]
                                         for b in sample.blocks],
            "sampled_retained_rows": int(len(sample.retained)),
            "includes_first_batch": sample.blocks[0] == 0,
            "includes_retained_tail_batch": sample.blocks[-1] == sample.n_blocks - 1,
            "retained_tail_batch_rows": int(sample.tail_rows),
            "sampled_training_rows": int(len(sample.train)),
            "sampled_discovery_rows": int(len(sample.fit_local)),
            "final_partial_training_batch_rows": int(len(job.train) - sample.train_tail_start),
            "requested_final_training_rows": int(train_tail_rows),
            "covered_final_training_rows": int(np.isin(job.train[-train_tail_rows:], sample.train).sum()) if train_tail_rows else 0}


def run_preflight(manifest, tmp_path, environ):
    layout = fixture_layout(manifest)
    scratch = tmp_path / "preflight"
    scratch.mkdir()
    checks = []
    check = new_checker(checks)
    window = environ.get(ENV_WINDOW, "reference_only_common")
    out = resolve_summary_path(environ, tmp_path, [layout.data_repo])
    job = load_job(layout, window, scratch, REQUIRED_DEVICE)
    before = fixture_state(job)
    parameters = sample_parameters(environ)
    from temporal_gpu_execution import GPU_POLICY
    sample = build_sample(job, batch_size=GPU_POLICY["embedding_batch"], **parameters)
    archived_coverage(job, sample, check)
    check("fixture_unchanged", fixture_state(job) == before)
    from artifact_storage import atomic_write_json
    atomic_write_json(out, {"schema": SCHEMA, "mode": "cpu_preflight", "sample": describe_sample(job, sample, parameters["train_tail_rows"]),
                            "checks": checks, "job_identity": job.job_identity}, compact=False)
    failed = [c for c in checks if c["required"] and not c["passed"]]
    assert not failed, f"preflight failed: {[c['name'] for c in failed]}"


# --------------------------------------------------------------------------- the CUDA smoke


def _execute(layout, scratch, environ, summary, check, holder):
    import torch
    from runtime_acceleration import accelerator_manifest
    from temporal_gpu_execution import GPU_POLICY
    from temporal_memory_recovery import configure_allocator
    timings = summary["timings_seconds"]
    configure_allocator()  # the allocator policy must be set before CUDA initializes
    device = resolve_device()
    summary["accelerator"] = accelerator_manifest(device)
    free, total = torch.cuda.mem_get_info()
    busy_mib = (total - free) / 2**20
    limit = float(environ.get(ENV_MAX_BUSY, 1024))
    summary["gpu_in_use_before_mib"] = busy_mib
    _require(busy_mib <= limit, f"GPU already holds {busy_mib:.0f} MiB (> {limit:.0f}); sharing is not approved")

    started = time.perf_counter()
    window = environ.get(ENV_WINDOW, "reference_only_common")
    job = load_job(layout, window, scratch, device)
    holder["job"], holder["before"] = job, fixture_state(job)
    timings["load_and_rebuild_job_cpu"] = time.perf_counter() - started
    summary["fixture"].update(window=window, fitted_state_sha256=job.fit_sha, context_sha256=job.context_sha256,
                              rebuilt_job_identity_matches=True, training_rows=int(len(job.train)),
                              retained_rows=int(len(job.evaluation)))
    summary["source_files"] = source_comparison(job)
    parameters = sample_parameters(environ)
    sample = build_sample(job, batch_size=GPU_POLICY["embedding_batch"], **parameters)
    summary["parameters"].update(parameters)
    summary["sample"] = describe_sample(job, sample, parameters["train_tail_rows"])
    check("sample_includes_first_and_retained_tail_batches",
          summary["sample"]["includes_first_batch"] and summary["sample"]["includes_retained_tail_batch"],
          {"tail_batch_rows": int(sample.tail_rows)})
    started = time.perf_counter()
    archived = archived_coverage(job, sample, check)
    timings["archived_cpu_checks"] = time.perf_counter() - started

    with production_precision(job.window_config) as precision:
        summary["precision"] = precision
        started = time.perf_counter()
        a = run_arm("arm_a_derived", job, sample, scratch, derived=True, controls=True)
        timings["arm_a_derived"] = time.perf_counter() - started
        started = time.perf_counter()
        b = run_arm("arm_b_legacy", job, sample, scratch, derived=False, controls=False)
        timings["arm_b_legacy"] = time.perf_counter() - started
    summary["arms"] = {"A_derived": a.metrics, "B_legacy": b.metrics}

    n, expected_batch = len(sample.retained), GPU_POLICY["embedding_batch"]
    check("arm_a_issues_no_training_query",
          [(c["description"], c["rows"]) for c in a.calls] == [(TEST_DESCRIPTION, n)],
          {"calls": [(c["description"], c["rows"]) for c in a.calls]})
    check("arm_b_runs_training_query_before_retained_query",
          [(c["description"], c["rows"]) for c in b.calls] == [(TRAIN_DESCRIPTION, len(job.train)), (TEST_DESCRIPTION, n)],
          {"calls": [(c["description"], c["rows"]) for c in b.calls]})
    check("extraction_policy_matches_production",
          all(c["batch_size"] == expected_batch and c["strict_domains"] for c in a.calls + b.calls),
          {"batch_size": expected_batch})
    check("training_provenance_recorded",
          a.provenance["origin"] == "retained_rows_v1" and a.provenance["mapping"]["indices"] == sample.mapping.tolist()
          and b.provenance["origin"] == "independent_queries_v1",
          {"arm_a": a.provenance["origin"], "arm_b": b.provenance["origin"]})
    check("outputs_are_finite", all(np.isfinite(x).all() for x in (
        a.test_raw, b.test_raw, a.train_raw, b.train_raw, a.outputs, b.outputs, a.gradients, b.gradients)))
    check("handoff_decoder_contract_on_first_retained_batch", a.handoff is not None and "error" not in a.handoff,
          a.handoff)

    exact = {
        "retained_raw_embeddings": exact_report("retained raw, arm A vs arm B", a.test_raw, b.test_raw),
        "derived_train_raw_equals_mapped_retained_raw": exact_report(
            "arm A train_raw vs arm A retained_raw[mapping]", a.train_raw, a.test_raw[sample.mapping]),
        "decoder_outputs": exact_report("decoder outputs, arm A vs arm B", a.outputs, b.outputs),
        "decoder_gradients": exact_report("decoder gradients, arm A vs arm B", a.gradients, b.gradients),
        "scaler_mean (cpu)": exact_report("scaler mean_, arm A vs arm B", a.scaler.mean_, b.scaler.mean_),
        "scaler_scale (cpu)": exact_report("scaler scale_, arm A vs arm B", a.scaler.scale_, b.scaler.scale_),
        "test_scaled (cpu)": exact_report("test_scaled, arm A vs arm B", a.test_scaled, b.test_scaled)}
    for name, report in exact.items():
        check(f"exact_{name}", report["equal"], report)
    blocks = []
    for block in sample.blocks:
        rows = np.flatnonzero(sample.local // sample.batch_size == block)
        blocks.append({"batch_index": int(block), "retained_start": int(sample.local[rows[0]]), "rows": int(len(rows)),
                       "equal_arm_a_vs_arm_b": exact_report("block", a.test_raw[rows], b.test_raw[rows])["equal"],
                       "equal_arm_a_vs_archived": exact_report("block", a.test_raw[rows], archived.test_raw[sample.local[rows]])["equal"]})
    summary["per_batch"] = blocks
    check("exact_every_sampled_batch_arm_a_vs_arm_b", all(item["equal_arm_a_vs_arm_b"] for item in blocks),
          [item["batch_index"] for item in blocks if not item["equal_arm_a_vs_arm_b"]])

    tail = sample.train_tail_start
    fresh = exact_report("fresh legacy train rows vs derived", b.train_raw[sample.train_select], a.train_raw,
                         max_listed=len(sample.train))
    differing_in_job = [int(sample.train_select[row]) for row in fresh["differing_rows"]]
    info = {
        "fresh_legacy_vs_derived_training_rows": {
            **{k: fresh[k] for k in ("rows", "n_differing")}, "differing_training_positions": differing_in_job,
            "confined_to_final_partial_training_batch": all(row >= tail for row in differing_in_job)},
        "fresh_legacy_train_raw_vs_archived_legacy_train_raw": exact_report(
            "arm B train_raw vs archived train_raw", b.train_raw, archived.train_raw, max_listed=len(job.train)),
        "arm_a_retained_vs_archived_retained": exact_report(
            "arm A retained vs archived retained", a.test_raw, archived.test_raw[sample.local]),
        "train_scaled_arm_a_vs_arm_b (cpu)": exact_report(
            "train_scaled on shared rows", a.train_scaled, b.train_scaled[sample.train_select]),
        "repeat_control_arm_a": a.repeats,
        "note": "informational; legacy-versus-derived differences are a documented caveat, not a tolerance."}
    summary["informational"] = info
    check("fresh_legacy_vs_derived_training_rows", True, info["fresh_legacy_vs_derived_training_rows"], required=False)
    check("repeat_control_retained_batches_bitwise_stable", all(r["equal"] for r in a.repeats),
          [r["label"] for r in a.repeats if not r["equal"]], required=False)
    check("arm_a_retained_equals_archived_retained", info["arm_a_retained_vs_archived_retained"]["equal"],
          {"n_differing": info["arm_a_retained_vs_archived_retained"]["n_differing"]}, required=False)


def run_smoke(manifest, tmp_path, environ):
    layout = fixture_layout(manifest)
    out = resolve_summary_path(environ, tmp_path, [layout.data_repo])
    from artifact_storage import atomic_write_json
    checks, holder, started = [], {}, time.perf_counter()
    check = new_checker(checks)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    summary = {
        "schema": SCHEMA, "mode": "cuda_smoke", "status": "running", "started_at": _utc_now(),
        "fixture": {"manifest": str(layout.manifest), "run_identity": layout.run_identity,
                    "job_identity": layout.job_identity, "reference_year": layout.reference_year,
                    "patient_split_seed": layout.patient_split_seed, "data_repo": str(layout.data_repo)},
        "parameters": {}, "timings_seconds": {}, "checks": checks,
        "code": {"head": _git("rev-parse", "HEAD"), "status_porcelain": _git("status", "--porcelain")},
        "coverage_labels": {
            "cuda_sampled": "retained raw embeddings, decoder outputs and gradients on sampled retained batches",
            "cpu_scikit_learn": "discovery-only scaler fit equality (bounded fresh arrays and full archived arrays)",
            "archived_read_only": "fixture arrays; informational comparisons never excuse a mismatch",
            "not_covered": "full-population retained extraction, SAE training, rule/CAV selection, downstream metrics"}}
    write = lambda: atomic_write_json(out, summary, compact=False)
    write()
    error = None
    try:
        _execute(layout, scratch, environ, summary, check, holder)
    except BaseException as exc:  # recorded in the summary, then re-raised
        error = exc
        summary["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    if "job" in holder:
        try:
            after = fixture_state(holder["job"])
            summary["fixture_state"] = {"before": holder["before"], "after": after}
            check("fixture_unchanged_before_and_after (files, tree listing, context, features)",
                  after == holder["before"])
        except Exception as exc:  # pragma: no cover - evidence capture must not hide the primary failure
            check("fixture_unchanged_before_and_after", False, {"error": repr(exc)})
    failed = [c["name"] for c in checks if c["required"] and not c["passed"]]
    summary.update(status="error" if error else ("failed" if failed else "passed"), finished_at=_utc_now(),
                   total_seconds=time.perf_counter() - started, failed_checks=failed)
    write()
    if error is not None:
        raise error
    assert not failed, f"failed checks {failed}; summary: {out}"


# --------------------------------------------------------------------------- CPU contracts


def test_gate_skips_when_manifest_variable_is_unset():
    with pytest.raises(pytest.skip.Exception, match=ENV_MANIFEST):
        smoke_manifest_path({}, cuda_available=True)


def test_gate_skips_without_cuda_before_touching_the_fixture(tmp_path):
    missing = tmp_path / "absent" / "completed.json"
    with pytest.raises(pytest.skip.Exception, match="CUDA unavailable"):
        smoke_manifest_path({ENV_MANIFEST: str(missing)}, cuda_available=False)
    assert not missing.parent.exists()


def test_gate_returns_the_manifest_path_when_cuda_is_present(tmp_path):
    path = smoke_manifest_path({ENV_MANIFEST: str(tmp_path / "completed.json")}, cuda_available=True)
    assert path == tmp_path / "completed.json"


def _fixture_tree(root, window="window_concepts_abc", job="b5c22217fb69bfb1246189dd",
                  reference="reference_2007", split="split_42"):
    split_dir = root / "stats" / "temporal_window_concepts" / window / "fits" / job / reference / split
    split_dir.mkdir(parents=True)
    (split_dir / "completed.json").write_text("{}")
    return split_dir


def test_fixture_layout_parses_the_completed_manifest_path(tmp_path):
    split_dir = _fixture_tree(tmp_path)
    layout = fixture_layout(split_dir / "completed.json")
    assert layout.reference_year == 2007 and layout.patient_split_seed == 42
    assert layout.job_identity == "b5c22217fb69bfb1246189dd"
    assert layout.run_identity == "abc"
    assert layout.data_repo == tmp_path.resolve()
    assert layout.split_dir == split_dir.resolve()
    assert layout.attempt_dir == split_dir.resolve() / "attempt_cuda" / "reference_2007" / "split_42"


@pytest.mark.parametrize("mutation", ["name", "window", "reference", "split"])
def test_fixture_layout_rejects_unexpected_locations(tmp_path, mutation):
    options = {"window": "other_run", "reference": "ref_2007", "split": "seed_42"}
    split_dir = _fixture_tree(tmp_path, **({} if mutation == "name" else {mutation: options[mutation]}))
    target = split_dir / ("fit.json" if mutation == "name" else "completed.json")
    if mutation == "name":
        (split_dir / "completed.json").rename(target)
    with pytest.raises(ValueError):
        fixture_layout(target)


def test_summary_path_defaults_to_the_scratch_directory(tmp_path):
    assert resolve_summary_path({}, tmp_path) == (tmp_path / SUMMARY_NAME).resolve()


def test_summary_path_accepts_a_file_or_a_directory(tmp_path):
    named = tmp_path / "evidence" / "smoke.json"
    assert resolve_summary_path({ENV_OUT: str(named)}, tmp_path / "x") == named.resolve()
    assert resolve_summary_path({ENV_OUT: str(tmp_path / "evidence")}, tmp_path / "x") == (
        tmp_path / "evidence" / SUMMARY_NAME).resolve()


def test_summary_path_refuses_to_write_inside_protected_roots(tmp_path):
    protected = tmp_path / "repo"
    protected.mkdir()
    for target in (protected / "stats" / "out.json", protected / "out.json", protected):
        with pytest.raises(ValueError, match="protected"):
            resolve_summary_path({ENV_OUT: str(target)}, tmp_path, forbidden_roots=[protected])


def test_tree_listing_fingerprint_detects_any_change(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.bin").write_bytes(b"abc")
    first = tree_listing_fingerprint(tmp_path)
    assert first == tree_listing_fingerprint(tmp_path)
    assert first["files"] == 1 and first["bytes"] == 3 and first["method"] == "listing"
    (tmp_path / "a" / "y.bin").write_bytes(b"")
    added = tree_listing_fingerprint(tmp_path)
    assert added["sha256"] != first["sha256"]
    (tmp_path / "a" / "x.bin").write_bytes(b"abcd")
    assert tree_listing_fingerprint(tmp_path)["sha256"] != added["sha256"]


def _selection(**overrides):
    options = dict(extra_blocks=2, train_tail_rows=4, min_fit_rows=3, seed=5)
    options.update(overrides)
    return select_retained_blocks(
        100, 16, np.arange(10, 60), np.array([12, 13, 40, 41, 42]), **options)


def test_block_selection_always_covers_first_tail_train_tail_and_discovery():
    blocks = _selection()
    assert blocks == sorted(set(blocks))
    assert {0, 6, 3, 2} <= set(blocks)  # first, final partial, last training rows, discovery rows
    assert len(blocks) == 6
    assert blocks == _selection()


def test_block_selection_is_bounded_and_seeded():
    assert _selection(extra_blocks=0) == [0, 2, 3, 6]
    assert len(_selection(extra_blocks=99)) == 7
    assert _selection(seed=1, extra_blocks=1) == _selection(seed=1, extra_blocks=1)
    assert len({tuple(_selection(seed=seed, extra_blocks=1)) for seed in range(12)}) > 1


def test_block_selection_rejects_unusable_input():
    with pytest.raises(ValueError, match="discovery"):
        select_retained_blocks(100, 16, np.arange(10), np.array([], dtype=int), extra_blocks=1,
                               train_tail_rows=1, min_fit_rows=1, seed=0)
    with pytest.raises(ValueError, match="outside"):
        select_retained_blocks(100, 16, np.arange(10), np.array([100]), extra_blocks=1,
                               train_tail_rows=1, min_fit_rows=1, seed=0)


def test_block_rows_keep_production_batch_alignment_and_the_partial_tail():
    rows = block_rows([0, 3, 6], 100, 16)
    assert rows.tolist() == list(range(0, 16)) + list(range(48, 64)) + list(range(96, 100))
    assert len(rows) % 16 == 4 and rows[-1] == 99
    with pytest.raises(ValueError):
        block_rows([0, 7], 100, 16)
    with pytest.raises(ValueError):
        block_rows([3, 0], 100, 16)


def test_exact_report_is_bitwise():
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert exact_report("same", values, values.copy())["equal"] is True
    nudged = values.copy()
    nudged[2, 1] = np.nextafter(nudged[2, 1], np.float32(100))
    report = exact_report("ulp", nudged, values)
    assert report["equal"] is False and report["differing_rows"] == [2] and report["n_differing"] == 1
    assert report["max_abs_difference"] > 0
    zero = np.zeros((2, 2), dtype=np.float32)
    assert exact_report("signed zero", -zero, zero)["equal"] is False
    assert exact_report("dtype", values.astype(np.float64), values)["equal"] is False
    assert exact_report("shape", values[:2], values)["equal"] is False
    assert exact_report("empty", values[:0], values[:0])["equal"] is False
    json.dumps(report, allow_nan=False)


def test_extraction_recorder_logs_calls_and_stays_inspectable():
    calls = []

    def real(**kwargs):
        return np.zeros((len(kwargs["X"]), 2))

    recorder = make_extraction_recorder(real, calls)
    cfg = SimpleNamespace(progress_desc=TEST_DESCRIPTION, batch_size=16, strict_domains=True)
    recorder(model=None, X=np.zeros((5, 3)), years=np.zeros(5), cfg=cfg)
    assert [(c["description"], c["rows"], c["batch_size"], c["strict_domains"]) for c in calls] == [
        (TEST_DESCRIPTION, 5, 16, True)]
    assert calls[0]["seconds"] >= 0
    assert inspect.getsource(recorder)  # the adapter fingerprints extraction callables by source
    with pytest.raises(TypeError):
        recorder(None, np.zeros((1, 1)))


# --------------------------------------------------------------------------- fixture-backed tests


@pytest.fixture
def smoke_manifest():
    return smoke_manifest_path()


def test_cuda_skipped_training_pass_matches_the_legacy_path(smoke_manifest, tmp_path):
    run_smoke(smoke_manifest, tmp_path, os.environ)


def test_fixture_preflight_on_cpu_is_read_only(tmp_path):
    if os.environ.get(ENV_PREFLIGHT) != "1":
        pytest.skip(f"{ENV_PREFLIGHT}=1 is not set; the CPU fixture preflight is opt-in")
    if not os.environ.get(ENV_MANIFEST, "").strip():
        pytest.skip(f"{ENV_MANIFEST} is not set; no fixture to inspect")
    run_preflight(Path(os.environ[ENV_MANIFEST]), tmp_path, os.environ)
