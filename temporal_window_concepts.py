"""Resumable, serial reconstruction of corresponding window models and concepts.

Run --pilot first to exercise reference-only and the largest all-history context.
The full command reuses pilot model/concept checkpoints and expands to 81 jobs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
import logging
import os
from pathlib import Path
import pickle
import time

from temporal_memory_recovery import configure_allocator, run_with_cpu_fallback
configure_allocator()

import numpy as np

from artifact_storage import atomic_write_json, file_sha256
from temporal_concept_forecasting import (PARENT, WINDOWS, checked_manifest,
    configure_logging, digest, table, write_table)

LOG = logging.getLogger(__name__)
STRATEGIES = ("reference_only_common", "last_3", "all_history")
CONCEPT_ROLES = ("sae_discovery", "rule_discovery", "rule_selection_cav")
PROTOCOL = "expanded_patient_disjoint_concepts_v1"


def select_window_jobs(reference_years, windows, patient_split_seeds):
    """Validate and canonicalize supported immutable-parent logical systems."""
    def integers(values, allowed, name):
        values = list(values)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or int(value) not in allowed for value in values
        ):
            raise ValueError(f"{name} must be a nonempty list within {sorted(allowed)}")
        return sorted({int(value) for value in values})
    years = integers(reference_years, set(range(2007, 2016)), "reference years")
    seeds = integers(patient_split_seeds, {42, 43, 44}, "patient split seeds")
    requested = list(windows)
    if not requested or not set(requested).issubset(STRATEGIES):
        raise ValueError(f"windows must be a nonempty selection from {STRATEGIES}")
    ordered = [window for window in STRATEGIES if window in requested]
    return [(year, seed, window) for year in years for seed in seeds for window in ordered]


def canonical_window_jobs(jobs):
    """Validate an explicit, not necessarily cartesian, selection of logical systems."""
    if jobs is None:
        return None
    if isinstance(jobs, (str, bytes)) or not hasattr(jobs, "__iter__"):
        raise ValueError("jobs must be a nonempty collection of (reference_year, seed, window) tuples")
    requested = list(jobs)
    if not requested:
        raise ValueError("jobs must be a nonempty collection of (reference_year, seed, window) tuples")
    selected = set()
    for job in requested:
        if not isinstance(job, (tuple, list)) or len(job) != 3 or not isinstance(job[2], str):
            raise ValueError(f"job must be (reference_year, seed, window) with a named window, got {job!r}")
        year, seed, window = job
        selected.update(select_window_jobs([year], [window], [seed]))
    return sorted(selected, key=lambda job: (job[0], job[1], STRATEGIES.index(job[2])))


# Reviewed pre-recovery implementation: only memory execution policy changes.
# No other historical source version is eligible for this migration.
_REVIEWED_LEGACY_SOURCES = "e858bb196bf7a6c9739c9493bf91d040d152b2fc46ff90d1a29edd8cd4eb7ec4"


def reuse_reviewed_completed_job(output, identity_inputs, job_inputs, workspace, job_identity, logical_window=None, gpu_memory_safe=False):
    for old_root in sorted(Path(output).glob("window_concepts_*")):
        old_manifest = old_root / "incomplete_manifest.json"
        if not old_manifest.exists():
            continue
        old = json.loads(old_manifest.read_text())
        if old.get("schema_version") != PROTOCOL or digest(old.get("source_code", {})) != _REVIEWED_LEGACY_SOURCES:
            continue
        old_sources = old["source_code"]
        runtime_changes = {"temporal_window_concepts.py", "temporal_performance_windows.py", "temporal_production.py", "tabpfn_model.py", "tcav.py"}
        if any(identity_inputs["sources"].get(n) != v for n, v in old_sources.items() if n not in runtime_changes):
            continue
        legacy_inputs = {k: v for k, v in identity_inputs.items() if k != "gpu_memory_policy"}
        old_identity = digest({**legacy_inputs, "sources": old_sources})[:20]
        if old_root.name != f"window_concepts_{old_identity}":
            continue  # Config, model checkpoint, device or numerical environment changed.
        old_job = digest({"run": old_identity, **job_inputs})[:24]
        source = old_root / "fits" / old_job / workspace.parent.name / workspace.name / "completed.json"
        if not source.exists():
            known_oom = [f for f in old.get("failures", []) if
                         f.get("reference_year") == job_inputs["ref"] and
                         f.get("patient_split_seed") == job_inputs["seed"] and
                         f.get("window") == logical_window and
                         "cuda" in f.get("message", "").lower() and
                         "out of memory" in f.get("message", "").lower()]
            recovery = workspace / "device_recovery.json"
            if source.parent.exists() and known_oom and not recovery.exists() and not gpu_memory_safe:
                atomic_write_json(recovery, {"selected_device": "cpu", "complete": False,
                    "reason": "verified_prior_cuda_oom", "source_manifest": str(old_manifest.resolve()),
                    "source_manifest_sha256": file_sha256(old_manifest)})
                LOG.info("Reusing verified OOM diagnosis: this identical system will start on CPU")
            continue
        completed = checked_manifest(source)
        if completed.get("identity") != old_job:
            raise ValueError("Legacy completed-system identity mismatch")
        artifacts = {name: write_table(workspace, name, table(source, completed, name)) for name in completed["artifacts"]}
        atomic_write_json(workspace / "completed.json", {
            **completed, "identity": job_identity, "artifacts": artifacts,
            "reused_from": str(source.resolve()), "reused_manifest_sha256": file_sha256(source),
            "reuse_basis": "reviewed_memory_only_change_and_exact_membership_identity"})
        LOG.info("Reused validated completed system without refitting or extracting: %s", source)
        return True
    return False


# Prior-run source bundle whose completed fits may be reused. A completed result
# is independent of the retained-row embedding path added since: only fresh jobs
# take it, so results recorded under exactly this bundle remain scientifically
# equivalent. Every other historical source version stays ineligible.
_REVIEWED_COMPLETED_SOURCES = {
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
    "temporal_window_concepts.py": "99f4488cdc0b053c5ce268b3049ae6341ac949f3c6412119775130598811dde5",
}


class ReuseIntegrityError(ValueError):
    """A prior completed fit matches this job but fails validation; never recompute silently."""


def reuse_reviewed_completed_from_root(source_root, identity_inputs, job_inputs, workspace, job_identity):
    """Copy a matching completed fit out of a read-only prior root; nothing is written there."""
    def bound(inputs):
        return digest({k: v for k, v in inputs.items() if k not in ("sources", "import_handoff_sha256", "selection")})
    device = identity_inputs.get("environment", {}).get("resolved_device", identity_inputs["device"])
    for state in sorted(Path(source_root).glob("window_concepts_*/run_identity.json")):
        old = json.loads(state.read_text())
        inputs = old["inputs"]
        if digest(inputs)[:20] != old["identity"] or state.parent.name != f"window_concepts_{old['identity']}":
            continue
        if bound(inputs) != bound(identity_inputs):
            LOG.info("Rejected prior run %s: configuration, model, environment, device or GPU policy differ", old["identity"])
            continue
        if inputs["sources"] not in (identity_inputs["sources"], _REVIEWED_COMPLETED_SOURCES):
            LOG.info("Rejected prior run %s: scientific sources are not reviewed", old["identity"])
            continue
        # The per-fit identity excludes the aggregate selection; job identity binds exact membership.
        fit = digest({k: v for k, v in inputs.items() if k != "selection"})[:20]
        old_job = digest({"run": fit, **job_inputs})[:24]
        source = state.parent / "fits" / old_job / workspace.parent.name / workspace.name / "completed.json"
        if not source.exists():
            continue
        try:
            completed = checked_manifest(source)
            if completed.get("identity") != old_job:
                raise ValueError("completed-system identity mismatch")
            if completed.get("actual_device") != device:
                LOG.info("Rejected prior fit %s: it ran on %s, not %s", old_job, completed.get("actual_device"), device)
                continue
            rows = {name: table(source, completed, name) for name in completed["artifacts"]}
            artifacts = {name: write_table(workspace, name, values) for name, values in rows.items()}
            if any(table(workspace / "completed.json", {"artifacts": artifacts}, name) != values for name, values in rows.items()):
                raise ValueError("copied artifacts differ from the source")
        except (ValueError, KeyError, OSError) as error:
            raise ReuseIntegrityError(f"Matching prior completed fit {source} failed validation: {error}") from error
        atomic_write_json(workspace / "completed.json", {
            **completed, "identity": job_identity, "artifacts": artifacts,
            "reused_from": str(source.resolve()), "reused_manifest_sha256": file_sha256(source),
            "source_run_identity": old["identity"],
            "reuse_basis": "reviewed_source_bundle_exact_scientific_inputs_and_membership"})
        LOG.info("Reused validated completed system without refitting or extracting: %s", source)
        return True
    return False


def historical_roles(population, reference_year, seed, reference_roles, years, keep):
    """Assign once using all pre-reference patients, independently of window/outcome.

    Hash ordering followed by round robin gives equally sized historical-only
    groups. Restricting that assignment to shorter windows preserves nesting.
    """
    patients = np.asarray(population.patient_ids).astype(str)
    assignments = {}
    for role, indices in reference_roles.items():
        for patient in set(patients[indices]):
            if patient in assignments and assignments[patient] != role:
                raise ValueError("Reference roles are not patient-disjoint")
            assignments[patient] = role
    historical = set(patients[(population.years >= 2007) & (population.years < reference_year)]) - assignments.keys()
    ordered = sorted(historical, key=lambda p: (digest([reference_year, seed, p]), p))
    for i, patient in enumerate(ordered):
        assignments[patient] = CONCEPT_ROLES[i % 3]
    output = {role: [] for role in reference_roles}
    audit = []
    for index in np.flatnonzero(np.isin(population.years, years) & keep):
        patient = patients[index]
        role = assignments[patient]
        if role in CONCEPT_ROLES:
            output[role].append(int(index))
        audit.append({"row_index": int(index), "patient_id": patient, "year": int(population.years[index]),
                      "assigned_role": role, "concept_eligible": role in CONCEPT_ROLES,
                      "historical_only_patient": patient in historical})
    for role in ("tabpfn_context", "t0_evaluation"):
        output[role] = [int(i) for i in reference_roles[role] if keep[i]]
    output = {r: np.asarray(v, dtype=int) for r, v in output.items()}
    groups = [set(patients[output[r]]) for r in CONCEPT_ROLES]
    excluded = set(patients[reference_roles["t0_evaluation"]])
    if any(a & b for i, a in enumerate(groups) for b in groups[i+1:]) or any(g & excluded for g in groups):
        raise AssertionError("Historical concept roles overlap each other or reference evaluation")
    if any(np.any(population.years[output[r]] > reference_year) for r in CONCEPT_ROLES):
        raise AssertionError("Future data entered concept fitting")
    return output, audit


def cohort_masks(population, ref, evaluation, roles, training, validation):
    """One cohort definition shared by all measurements and probability scoring."""
    patients, years = population.patient_ids[evaluation].astype(str), population.years[evaluation]
    exposed_indices = np.unique(np.concatenate([training, validation] + [roles[r] for r in CONCEPT_ROLES]))
    exposed = set(population.patient_ids[exposed_indices].astype(str))
    unseen = np.array([p not in exposed for p in patients])
    result = {}
    for year in sorted(set(years)):
        if year < ref:
            continue
        mask = np.isin(evaluation, roles["t0_evaluation"]) if year == ref else years == year
        result[(int(year), "all_comer")] = mask
        # Historical model training can expose some t0 patients. Do not alias d0.
        result[(int(year), "pipeline_unseen")] = mask & unseen
    return result


def load_binary_checkpoint(path, identity):
    descriptor_path = path.with_suffix(path.suffix + ".json")
    if not descriptor_path.exists():
        return None
    d = json.loads(descriptor_path.read_text())
    if d["identity"] != identity or not path.exists() or d["sha256"] != file_sha256(path):
        raise ValueError(f"Invalid fitted-state checkpoint {path}")
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_binary_checkpoint(path, value, identity):
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    atomic_write_json(path.with_suffix(path.suffix + ".json"), {"identity": identity, "sha256": file_sha256(path)})


def _reference_roles(parent_path, parent, ref, seed):
    source = next(r for r in parent["successful_experiments"] if Path(r["manifest"]).parent.name == f"split_{seed}"
                  and Path(r["manifest"]).parent.parent.name == f"reference_{ref}")
    path = parent_path.parent / f"reference_{ref}" / f"split_{seed}" / "manifest.json"
    if file_sha256(path) != source["manifest_fingerprint"]:
        raise ValueError("Reference split manifest checksum mismatch")
    sm = checked_manifest(path, ("reference_roles",))
    roles = defaultdict(list)
    for row in table(path, sm, "reference_roles"):
        roles[row["role"]].append(row["row_index"])
    return {r: np.array(sorted(v), dtype=int) for r, v in roles.items()}


def _reproduction_audit(wp, wm, ref, seed, window, records):
    old = {}
    with gzip.open(wp.parent / wm["artifacts"]["record_probabilities"]["path"], "rt") as handle:
        for line in handle:
            r = json.loads(line)
            if r["reference_year"] == ref and r["patient_split_seed"] == seed and r["window"] == window:
                old[r["row_index"]] = r
    if set(old) != {r["row_index"] for r in records}:
        raise ValueError("Recreated model evaluation membership differs from window protocol")
    deltas = [abs(r["death_probability"]-old[r["row_index"]]["death_probability"]) for r in records]
    return {"reference_year": ref, "patient_split_seed": seed, "window": window,
            "records": len(records), "max_probability_difference": max(deltas, default=0.),
            "exact_probability_reproduction": all(d == 0 for d in deltas),
            "recreated_threshold": records[0]["frozen_threshold"] if records else None,
            "old_threshold": next(iter(old.values()))["frozen_threshold"] if old else None,
            "outcome_source": "this_recreated_fit"}


def run_window_concepts(repo, output, *, pilot=False, device="cuda", jobs=None, gpu_memory_safe=False, stop_after_embeddings=False, import_handoff=None, resume_extraction=None, reuse_completed_from=None):
    explicit_jobs = canonical_window_jobs(jobs)
    from temporal_handoff import (EmbeddingsReady, handoff_context, load_handoff, save_handoff,
        validate_import, checked_files, scientific_inputs)
    from comparison_runner import _tabpfn_checkpoint_fingerprint
    from temporal_config import TemporalRobustnessConfig
    from temporal_production import ProductionTemporalAdapter
    from temporal_performance_windows import (WindowExperimentConfig, ProductionWindowAdapter,
        build_training_indices, effective_window_years, model_domain_mapping,
        post_death_exclusion_mask, death_probabilities, select_frozen_threshold, metric_bundle)
    from temporal_splits import ReferenceSplit
    from temporal_unified_analysis import UnifiedAnalysisConfig, summarize_tcav_repetitions
    from temporal_unified_enrichment import activation_magnitude_rows, tcav_repetition_rows, _tcav_headline_views
    from temporal_cri import build_family_universe, CRIAnalysisConfig

    repo, output = Path(repo).resolve(), Path(output)
    configure_logging(output)
    pp = repo / "stats/temporal_robustness" / PARENT / "parent_manifest.json"
    wp = repo / "stats/temporal_performance_windows" / WINDOWS / "manifest.json"
    parent = checked_manifest(pp)
    wm = checked_manifest(wp, ("record_probabilities", "thresholds", "role_exposure_audit"))
    if wm["dependency_sha256"]["parent_manifest"] != file_sha256(pp):
        raise ValueError("Window/parent identity mismatch")
    cfg = dict(parent["config"])
    for name in ("comparison_config_path", "semantic_config_path", "dataset_path"):
        p = Path(cfg[name])
        cfg[name] = str(p if p.is_file() else repo / p.name)
    for name in ("comparison_config_path", "semantic_config_path"):
        if file_sha256(cfg[name]) != parent["dependent_config_fingerprints"][name]["sha256"]:
            raise ValueError(f"Parent scientific configuration changed: {name}")
    cfg.update(device=device, artifact_dir=str(output), show_progress=False)
    config = TemporalRobustnessConfig.from_dict(cfg)
    loader = ProductionTemporalAdapter()
    population = loader.load_retained_population(config, pp.parent.parent, parent["population_fingerprints"])
    population.validate()
    wc = dict(wm["config"])
    wc.update(parent_manifest=str(pp), comparison_config_path=cfg["comparison_config_path"], device=device,
              artifact_dir=str(output), show_progress=False)
    window_config = WindowExperimentConfig.from_dict(wc)
    checkpoint = _tabpfn_checkpoint_fingerprint(loader._base_config.tabpfn.model_name)
    if checkpoint.get("sha256") != wm["dependency_sha256"]["tabpfn_checkpoint"]["sha256"]:
        raise ValueError("Model checkpoint differs from existing window experiment")
    from temporal_metric_synthesis import _numerical_environment
    source_names = ("temporal_window_concepts.py", "temporal_concept_forecasting.py", "temporal_production.py",
                    "temporal_performance_windows.py", "temporal_unified_enrichment.py", "comparison_runner.py",
                    "temporal_rules.py", "temporal_cav.py", "temporal_cri.py", "temporal_unified_analysis.py", "tabpfn_model.py", "tcav.py", "temporal_memory_recovery.py", "temporal_gpu_execution.py", "temporal_handoff.py")
    source_hashes = {n: file_sha256(repo / n) for n in source_names}
    identity_inputs = {"parent": file_sha256(pp), "windows": file_sha256(wp), "protocol": PROTOCOL,
                       "sources": source_hashes, "device": device, "model_checkpoint": checkpoint,
                       "environment": _numerical_environment(device),
                       "configs": {n: file_sha256(cfg[n]) for n in ("comparison_config_path", "semantic_config_path")}}
    if gpu_memory_safe:
        from temporal_gpu_execution import GPU_POLICY
        if device != 'cuda':
            raise ValueError('GPU memory policy requires --device cuda')
        identity_inputs['gpu_memory_policy'] = GPU_POLICY
        window_config = replace(window_config, batch_size=GPU_POLICY['prediction_batch'])
    if import_handoff is not None:
        identity_inputs['import_handoff_sha256'] = file_sha256(import_handoff)
    all_jobs = [(ref, seed, w) for ref in range(2007, 2016) for seed in (42, 43, 44) for w in STRATEGIES]
    selected_jobs = explicit_jobs or ([(2015, 42, "reference_only_common"), (2015, 42, "all_history")] if pilot else all_jobs)
    selection = [list(job) for job in selected_jobs]
    # Per-fit identities exclude the scope so equivalent fits (for example the 2007
    # reference-only and last-3 systems) stay aliased; only the aggregate binds it.
    fit_identity = digest(identity_inputs)[:20]
    aggregate_inputs = identity_inputs if explicit_jobs is None else {**identity_inputs, 'selection': selection}
    identity = digest(aggregate_inputs)[:20]
    if resume_extraction is not None:
        saved = json.loads(Path(resume_extraction).read_text())
        if saved['identity'] != digest(saved['inputs'])[:20]:
            raise ValueError('Invalid extraction run identity')
        if saved['inputs'].get('selection', selection) != selection:
            raise ValueError('Partial extraction requires the identical job selection')
        before = {k:v for k,v in saved['inputs'].items() if k not in ('sources', 'selection')}
        after = {k:v for k,v in identity_inputs.items() if k != 'sources'}
        if digest(before) != digest(after):
            raise ValueError('Partial extraction requires identical scientific inputs, environment and GPU policy')
        identity = saved['identity']
        fit_identity = digest({k:v for k,v in saved['inputs'].items() if k != 'selection'})[:20]
        LOG.info('Explicit extraction resume: %s; source changes recorded separately', identity)
    imported_meta = checked_files(import_handoff) if import_handoff else None
    root = output / f"window_concepts_{identity}"
    configure_logging(root)
    atomic_write_json(root/'resume_code_provenance.json', {'current_inputs':identity_inputs,
        'resume_extraction': str(resume_extraction) if resume_extraction else None,
        'import_handoff_sha256': file_sha256(import_handoff) if import_handoff else None})
    if not (root/'run_identity.json').exists():
        atomic_write_json(root/'run_identity.json', {'identity':identity,'inputs':aggregate_inputs})
    run_manifest = root / ("pilot_manifest.json" if pilot else "manifest.json")
    if run_manifest.exists():
        checked_manifest(run_manifest)
        LOG.info("Verified complete window extraction: %s", run_manifest)
        return run_manifest
    keep, post_death_audit = post_death_exclusion_mask(population.patient_ids, population.years, population.outcomes)
    LOG.info("Window extraction: %d jobs, %d post-death exclusions, serial model execution", len(selected_jobs), len(post_death_audit))
    aggregated = {w: defaultdict(list) for w in STRATEGIES}
    aliases, failures = [], []
    started = time.monotonic()
    for number, (ref, seed, window) in enumerate(selected_jobs, 1):
        job_start = time.monotonic()
        stage = "membership"
        try:
            reference_roles = _reference_roles(pp, parent, ref, seed)
            years = effective_window_years(window, ref)
            roles, membership = historical_roles(population, ref, seed, reference_roles, years, keep)
            train = build_training_indices(population=population, reference_year=ref, logical_window=window,
                                           global_roles=reference_roles, common_keep=keep)
            validation = reference_roles["rule_selection_cav"]
            validation = validation[keep[validation]]
            evaluation = np.flatnonzero((population.years >= min(years)) & keep)
            masks = cohort_masks(population, ref, evaluation, roles, train, validation)
            eval_records = np.unique(np.concatenate([evaluation[m] for (y, c), m in masks.items() if c == "all_comer"]))
            predict = np.unique(np.concatenate([validation, eval_records]))
            domains = model_domain_mapping(population.years[train], population.years[evaluation])
            job_inputs = {"ref": ref, "seed": seed, "years": years,
                          "train": train, "roles": roles, "evaluation": evaluation, "domains": domains}
            context = handoff_context(identity_inputs, job_inputs, population, predict)
            job_identity = digest({"run": fit_identity, **job_inputs})[:24]
            workspace = root / "fits" / job_identity / f"reference_{ref}" / f"split_{seed}"
            workspace.mkdir(parents=True, exist_ok=True)
            complete = workspace / "completed.json"
            LOG.info("Job %d/%d ref=%d split=%d window=%s years=%s training=%d concepts=%s", number, len(selected_jobs),
                     ref, seed, window, years, len(train), {r: len(roles[r]) for r in CONCEPT_ROLES})
            if not complete.exists() and imported_meta is not None:
                # Explicit reuse of completed corresponding fits from the same transfer.
                from temporal_handoff import reuse_completed_from_transfer
                reuse_completed_from_transfer(output, identity_inputs, job_inputs, workspace, job_identity)
            if not complete.exists() and reuse_completed_from is not None:
                reuse_reviewed_completed_from_root(reuse_completed_from, identity_inputs, job_inputs, workspace, job_identity)
            if not complete.exists():
                reuse_reviewed_completed_job(output, identity_inputs, job_inputs, workspace, job_identity, window, gpu_memory_safe)
            if complete.exists():
                result_manifest = checked_manifest(complete)
                data = {n: table(complete, result_manifest, n) for n in result_manifest["artifacts"]}
                LOG.info("Reusing validated completed fit/concepts: %s", job_identity)
            else:
                base_config, base_window_config = config, window_config
                def compute_system(attempt_device, workspace):
                    nonlocal stage
                    config = replace(base_config, device=attempt_device)
                    window_config = replace(base_window_config, device=attempt_device)
                    loader._configure(config)
                    stage = "model_and_probabilities"
                    adapter = ProductionWindowAdapter()
                    adapter._model_name = loader._base_config.tabpfn.model_name
                    fit_path = workspace / "fitted_state.pkl"
                    fitted = load_binary_checkpoint(fit_path, job_identity)
                    reused_fit = fitted is not None
                    if reused_fit:
                        LOG.info("Resuming retained fitted state %s", job_identity)
                    imported_arrays = None
                    imported_job = imported_meta['context']['job'] if imported_meta else None
                    importing = imported_job is not None and (imported_job['ref'], imported_job['seed'], imported_job['years']) == (ref, seed, list(years))
                    if importing:
                        import shutil
                        from temporal_performance_windows import ProbabilityResult
                        meta, imported_arrays, fitted = load_handoff(import_handoff, context)
                        source_fit = Path(import_handoff).parent/meta['files']['fitted_state']['path']
                        if fit_path.exists() and file_sha256(fit_path) != file_sha256(source_fit):
                            raise ValueError('Destination contains a different fitted state')
                        if not fit_path.exists():
                            shutil.copy2(source_fit, fit_path)
                        atomic_write_json(fit_path.with_suffix('.pkl.json'), {'identity':job_identity,'sha256':file_sha256(fit_path)})
                        import torch
                        torch.backends.cuda.matmul.allow_tf32 = window_config.allow_tf32
                        torch.backends.cudnn.allow_tf32 = window_config.allow_tf32
                        torch.set_float32_matmul_precision('high' if window_config.allow_tf32 else 'highest')
                        validate_import(meta, imported_arrays, fitted, population, workspace)
                        fitted['model']._temporal_batch_root = str(workspace/'batches')
                        fitted['model']._temporal_fit_identity = file_sha256(fit_path)
                        result = ProbabilityResult(probabilities=imported_arrays['probabilities'],
                            classes=imported_arrays['classes'], model_info=meta['model_info'])
                        LOG.info('HANDOFF: fitted model and predictions imported; fitting and prediction SKIPPED')
                    else:
                        result = adapter.fit_predict(population=population, train_indices=train, predict_indices=predict,
                            model_domain_ids=np.array([domains[int(y)] for y in population.years[train]]),
                            prediction_domain_ids=np.array([domains[int(y)] for y in population.years[predict]]),
                            seed=seed, config=window_config, retain_fitted_state=True, fitted_state=fitted,
                            gpu_memory_safe=gpu_memory_safe, checkpoint_workspace=workspace,
                            checkpoint_identity=job_identity)
                        fitted = adapter.fitted_state
                    requested_domains = np.array([domains[int(y)] for y in population.years[train]])
                    if not np.array_equal(fitted["dist_shift_domain_train"], requested_domains):
                        raise ValueError("Fitted model training domains differ from the explicit domain mapping")
                    # Bind all subsequent stages to this exact retained state.
                    fitted_identity = file_sha256(fit_path)
                    probabilities = death_probabilities(result)
                    lookup = {int(i): float(p) for i, p in zip(predict, probabilities)}
                    threshold = select_frozen_threshold(population.outcomes[validation], [lookup[int(i)] for i in validation])
                    cutoff = threshold["threshold"]
                    records = [{"reference_year": ref, "patient_split_seed": seed, "test_year": int(population.years[i]),
                                "row_index": int(i), "record_key": str(population.record_keys[i]), "patient_id": str(population.patient_ids[i]),
                                "outcome": int(population.outcomes[i]), "death_probability": lookup[int(i)],
                                "frozen_threshold": cutoff, "fitted_identity": fitted_identity,
                                "model_domain_id": domains[int(population.years[i])]} for i in eval_records]
                    perf, audits = [], []
                    for (year, cohort), mask in masks.items():
                        indices = evaluation[mask]
                        support = digest(sorted((int(i), str(population.record_keys[i]), str(population.patient_ids[i]), int(population.outcomes[i])) for i in indices))
                        meta = {"reference_year": ref, "patient_split_seed": seed, "test_year": year, "temporal_distance": year-ref,
                                "cohort_view": cohort, "cohort_definition": PROTOCOL, "concept_protocol": PROTOCOL,
                                "support_fingerprint": support, "fitted_identity": fitted_identity, "frozen_threshold": cutoff}
                        metrics = metric_bundle(population.outcomes[indices], [lookup[int(i)] for i in indices], cutoff,
                                                minimum_deaths=10, minimum_survivors=30) if len(indices) else {"valid": False, "failure_reason": "empty_cohort", "record_count": 0}
                        perf.append({**meta, **metrics})
                        audits.append({**meta, "record_count": len(indices), "d0_alias_verified": False})
                    reproduction = _reproduction_audit(wp, wm, ref, seed, window, records)
                    LOG.info("Model ready: exact old-probability reproduction=%s max_difference=%.6g cutoff=%.6g",
                             reproduction["exact_probability_reproduction"], reproduction["max_probability_difference"], cutoff)
                    stage = "concept_learning_and_gradients"
                    LOG.info("Starting embeddings, SAE fitting, rule selection and gradients for %s", job_identity)
                    # Production roles have expanded concept rows; training is passed separately.
                    def after_embeddings(embeddings, prepared):
                        if not stop_after_embeddings:
                            return
                        manifest = save_handoff(workspace, context, embeddings, fit_path, result, population,
                            {'source_run':fit_identity,'source_job':job_identity,'inputs':identity_inputs,
                             'resumed_from': str(resume_extraction) if resume_extraction else None})
                        raise EmbeddingsReady(manifest)
                    output_data = loader.run_reference_experiment(population=population, reference_year=ref,
                        split=ReferenceSplit(seed, seed, 0, {}, {}), global_roles=roles,
                        evaluation_indices=evaluation, domain_map=domains, config=config, workspace=workspace,
                        fitted_state=fitted, fitted_identity=fitted_identity, training_indices=train,
                        imported_raw=None if imported_arrays is None else imported_arrays['raw_pair'],
                        after_embeddings=after_embeddings, performance_override=perf)
                    expected_domains = np.array([domains[int(y)] for y in population.years[evaluation]])
                    for name, actual in output_data["stage_domains"].items():
                        if not np.array_equal(actual, expected_domains):
                            raise AssertionError(f"{name} used different model domains")
                    retained_tables = {n: write_table(workspace, n, output_data[n]) for n in ("rules", "matching_recurrence", "cavs", "factor_families")}
                    atomic_write_json(workspace / "manifest.json", {"complete": True, "artifacts": retained_tables,
                        "reference_year": ref, "patient_split_seed": seed, "fitted_identity": fitted_identity})
                    stage = "concept_measurements"
                    LOG.info("Measuring concept trajectories and TCAV repetitions for %s", job_identity)
                    unified = UnifiedAnalysisConfig()
                    options = dict(evaluation_indices=evaluation, roles_override=roles, masks_override=masks)
                    factors = activation_magnitude_rows(workspace, population, config.to_dict(), unified, **options)
                    repetitions = tcav_repetition_rows(workspace, population, config.to_dict(), unified, **options)
                    tcav = _tcav_headline_views(summarize_tcav_repetitions(repetitions, unified), output_data["matching_recurrence"], unified)
                    support_lookup = {(r["test_year"], r["cohort_view"]): r for r in audits}
                    for measured in factors + tcav:
                        support_row = support_lookup[(measured["test_year"], measured["cohort_view"])]
                        measured.update(fitted_identity=fitted_identity, support_fingerprint=support_row["support_fingerprint"],
                                        cohort_definition=PROTOCOL, concept_protocol=PROTOCOL)
                        if measured.get("future_denominator", support_row["record_count"]) != support_row["record_count"]:
                            raise ValueError("Concept and probability support differ")
                    universe = build_family_universe(factors, unified, CRIAnalysisConfig())
                    data = {"performance": perf, "factors": factors, "tcav": tcav, "universe": universe,
                            "membership_audit": audits, "historical_roles": membership, "record_probabilities": records,
                            "reproduction_audit": [reproduction], "tcav_repetitions": repetitions,
                            "threshold": [{**threshold, "reference_year": ref, "patient_split_seed": seed,
                                           "validation_indices": validation.tolist(), "fitted_identity": fitted_identity}],
                            "domain_audit": [{"year": y, "model_domain_id": d, "reported_distance": y-ref,
                                              "fitted_identity": fitted_identity, "stages": ["probabilities", "embeddings", "gradients"]} for y, d in domains.items()]}
                    return {name: [{**row, "actual_device": attempt_device} for row in values] for name, values in data.items()}
                data, actual_device = run_with_cpu_fallback(compute_system, workspace, device, allow_cpu_fallback=not gpu_memory_safe, retry_cuda=gpu_memory_safe)
                artifacts = {n: write_table(workspace, n, values) for n, values in data.items()}
                atomic_write_json(complete, {"complete": True, "identity": job_identity, "artifacts": artifacts,
                                           "actual_device": actual_device, "effective_years": years})
                from temporal_performance_windows import _release_cuda_memory
                _release_cuda_memory()
            for name, values in data.items():
                aggregated[window][name].extend({**v, "system": window, "concept_protocol": PROTOCOL} for v in values)
            aliases.append({"reference_year": ref, "patient_split_seed": seed, "window": window,
                            "fit_identity": job_identity, "completed_manifest": str(complete.relative_to(root)),
                            "completed_manifest_sha256": file_sha256(complete), "effective_years": years})
            LOG.info("Job %d/%d complete in %.1fs; elapsed %.1fs; ETA %.1fs", number, len(selected_jobs),
                     time.monotonic()-job_start, time.monotonic()-started,
                     (time.monotonic()-started)/number*(len(selected_jobs)-number))
        except Exception as error:
            if import_handoff or resume_extraction or isinstance(error, ReuseIntegrityError):
                raise  # A failed handoff or damaged reuse candidate must not silently proceed into other expensive jobs.
            failure = {"reference_year": ref, "patient_split_seed": seed, "window": window,
                       "stage": stage, "error": type(error).__name__, "message": str(error)}
            failures.append(failure)
            atomic_write_json(root / "failures.json", failures)
            LOG.exception("Job failed at %s; completed checkpoints retained", stage)
            # Continue independent jobs; never publish an incomplete scientific manifest.
    artifacts = {f"{w}_{name}": write_table(root, f"{w}_{name}", values)
                 for w, tables in aggregated.items() for name, values in tables.items()}
    artifacts["aliases"] = write_table(root, "aliases", aliases)
    artifacts["post_death_exclusions"] = write_table(root, "post_death_exclusions", post_death_audit)
    manifest = {"schema_version": PROTOCOL, "complete": not failures, "pilot": pilot, "selection": selection, "systems": sorted({w for _, _, w in selected_jobs}),
                "sources": {"parent": file_sha256(pp), "windows": file_sha256(wp)}, "source_code": source_hashes,
                "artifacts": artifacts, "failures": failures, "logical_jobs": len(selected_jobs),
                "distinct_fits": len({r["fit_identity"] for r in aliases}), "persistent_log": "progress.log"}
    atomic_write_json(run_manifest if not failures else root / "incomplete_manifest.json", manifest)
    if failures:
        raise RuntimeError(f"{len(failures)} window extraction jobs failed; see {root / 'failures.json'}")
    LOG.info("Window extraction complete: %s", run_manifest)
    return run_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("stats/temporal_window_concepts"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--gpu-memory-safe", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    run_window_concepts(args.repo, args.output, pilot=args.pilot, device=args.device, gpu_memory_safe=args.gpu_memory_safe)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
