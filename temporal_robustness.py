"""Parent orchestration for frozen, reference-specific temporal experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from semantic_artifacts import array_fingerprint, stable_hash
from temporal_cohorts import relative_domain_map
from temporal_config import TemporalRobustnessConfig
from temporal_splits import ReferenceSplit, generate_valid_reference_splits
from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl_gzip,
    describe_json,
    descriptor_for_file,
    file_sha256,
    iter_artifact_rows,
    jsonable,
    read_artifact,
    validate_descriptor,
)


TEMPORAL_ESTIMAND = (
    "Each matrix row measures degradation of a TabPFN system re-contextualized "
    "using that row's reference-year data, not degradation of the original pre-2007 context."
)


@dataclass(frozen=True)
class TemporalPopulation:
    X: np.ndarray
    outcomes: np.ndarray
    years: np.ndarray
    patient_ids: np.ndarray
    feature_names: tuple[str, ...]
    first_eligible_year: Mapping[str, int]
    record_keys: np.ndarray | None = None
    feature_selection_max_year: int | None = None

    def validate(self) -> None:
        row_count = len(self.X)
        if np.asarray(self.X).ndim != 2:
            raise ValueError("temporal features must be two-dimensional")
        for name in ("outcomes", "years", "patient_ids"):
            values = np.asarray(getattr(self, name))
            if values.ndim != 1 or len(values) != row_count:
                raise ValueError(f"temporal {name} do not align with features")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature vocabulary does not align with feature matrix")
        if np.any(~np.isin(self.outcomes, (0, 1))):
            raise ValueError("temporal outcomes must be binary")
        if self.record_keys is not None and len(self.record_keys) != row_count:
            raise ValueError("record keys do not align with temporal population")
        patients = np.asarray(self.patient_ids).astype(str)
        years = np.asarray(self.years, dtype=int)
        first = {str(patient): int(year) for patient, year in self.first_eligible_year.items()}
        for patient in np.unique(patients):
            observed = int(np.min(years[patients == patient]))
            if first.get(patient) != observed:
                raise ValueError(
                    f"first eligible year provenance is invalid for patient {patient!r}"
                )


class TemporalExperimentAdapter(Protocol):
    """Expensive stages. Implementations must fit only from supplied role indices."""

    def load_population(self, config: TemporalRobustnessConfig) -> TemporalPopulation: ...

    def run_reference_experiment(
        self,
        *,
        population: TemporalPopulation,
        reference_year: int,
        split: ReferenceSplit,
        global_roles: Mapping[str, np.ndarray],
        evaluation_indices: np.ndarray,
        domain_map: Mapping[int, int],
        config: TemporalRobustnessConfig,
        workspace: Path,
    ) -> Mapping[str, Any]: ...


def _fingerprint_population(population: TemporalPopulation) -> dict[str, Any]:
    return {
        "features": array_fingerprint(np.asarray(population.X)),
        "outcomes": array_fingerprint(np.asarray(population.outcomes)),
        "years": array_fingerprint(np.asarray(population.years)),
        "patients": array_fingerprint(np.asarray(population.patient_ids).astype(str)),
        "records": None if population.record_keys is None else array_fingerprint(np.asarray(population.record_keys).astype(str)),
        "feature_vocabulary": stable_hash(population.feature_names),
    }


def _scientific_config(config: TemporalRobustnessConfig) -> dict[str, Any]:
    value = config.to_dict()
    for name in ("artifact_dir", "use_cache", "show_progress", "force"):
        value.pop(name, None)
    return value


def _dependent_config_fingerprints(
    config: TemporalRobustnessConfig,
) -> dict[str, dict[str, Any]]:
    result = {}
    for name in ("comparison_config_path", "semantic_config_path"):
        path = Path(getattr(config, name))
        result[name] = {
            "path": str(path),
            "sha256": file_sha256(path) if path.is_file() else None,
        }
    return result


def _temporal_source_fingerprints(adapter: TemporalExperimentAdapter) -> dict[str, str]:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "temporal_analysis.py",
        "temporal_cav.py",
        "temporal_cohorts.py",
        "temporal_config.py",
        "temporal_matching.py",
        "temporal_metrics.py",
        "temporal_rules.py",
        "temporal_splits.py",
    ):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    try:
        adapter_source = inspect.getsource(type(adapter))
    except (OSError, TypeError):
        adapter_source = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
    return {
        "scientific_modules": digest.hexdigest(),
        "production_adapter": hashlib.sha256(adapter_source.encode()).hexdigest(),
    }


def _split_identity(
    *,
    population: TemporalPopulation,
    reference_year: int,
    split: ReferenceSplit,
    role_rows: Sequence[Mapping[str, Any]],
    config: TemporalRobustnessConfig,
    adapter: TemporalExperimentAdapter,
) -> dict[str, Any]:
    return {
        "scientific_config_fingerprint": stable_hash(_scientific_config(config)),
        "legacy_config_fingerprint": stable_hash(config.to_dict()),
        "population_fingerprints": _fingerprint_population(population),
        "reference_year": int(reference_year),
        "patient_split_seed": int(split.effective_seed),
        "requested_patient_split_seed": int(split.requested_seed),
        "role_fingerprint": stable_hash(role_rows),
        "dependent_config_fingerprints": _dependent_config_fingerprints(config),
        "source_fingerprints": _temporal_source_fingerprints(adapter),
    }


def _load_completed_split(root: Path, identity: Mapping[str, Any]) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
            or manifest.get("complete") is not True
            or manifest.get("scientific_identity") != identity
        ):
            return None
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return None
        for descriptor in artifacts.values():
            if not isinstance(descriptor, Mapping):
                return None
            validate_descriptor(root, descriptor)
        return manifest
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _write_value_artifact(root: Path, name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        path = root / f"{name}.jsonl.gz"
        descriptor = atomic_write_jsonl_gzip(path, value)
        descriptor["path"] = path.name
        return descriptor
    path = atomic_write_json(root / f"{name}.json", value)
    return describe_json(path, relative_to=root)


def run_reference_experiment(
    *,
    population: TemporalPopulation,
    reference_year: int,
    split: ReferenceSplit,
    config: TemporalRobustnessConfig,
    adapter: TemporalExperimentAdapter,
    workspace: str | Path,
) -> dict[str, Any]:
    """Run one independently namespaced, frozen reference/split experiment."""

    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    reference_indices = np.flatnonzero(population.years == reference_year)
    evaluation_indices = np.flatnonzero(population.years >= reference_year)
    global_roles = {
        role: reference_indices[np.asarray(local_indices, dtype=int)]
        for role, local_indices in split.roles.items()
    }
    domain_map = relative_domain_map(reference_year, population.years[evaluation_indices])
    role_rows = []
    for role, indices in global_roles.items():
        for index in indices:
            role_rows.append(
                {
                    "reference_year": reference_year,
                    "patient_split_seed": split.effective_seed,
                    "role": role,
                    "row_index": int(index),
                    "patient_id": str(population.patient_ids[index]),
                    "outcome": int(population.outcomes[index]),
                }
            )
    identity = _split_identity(
        population=population,
        reference_year=reference_year,
        split=split,
        role_rows=role_rows,
        config=config,
        adapter=adapter,
    )
    if config.use_cache and not config.force:
        completed = _load_completed_split(root, identity)
        if completed is not None:
            return completed

    artifact_files: dict[str, dict[str, Any]] = {
        "reference_roles": _write_value_artifact(root, "reference_roles", role_rows),
        "temporal_domain_map": _write_value_artifact(
            root,
            "temporal_domain_map",
            {str(year): domain for year, domain in domain_map.items()},
        ),
    }

    payload = dict(
        adapter.run_reference_experiment(
            population=population,
            reference_year=reference_year,
            split=split,
            global_roles=global_roles,
            evaluation_indices=evaluation_indices,
            domain_map=domain_map,
            config=config,
            workspace=root,
        )
    )
    stage_domains = payload.pop("stage_domains", None)
    expected_domains = population.years[evaluation_indices] - reference_year
    if stage_domains is not None:
        for stage, values in stage_domains.items():
            actual = np.asarray(values, dtype=int)
            if actual.shape != expected_domains.shape or not np.array_equal(actual, expected_domains):
                raise ValueError(f"{stage} domains differ from reference-relative record domains")

    for name, value in sorted(payload.items()):
        if name.startswith("_"):
            continue
        artifact_files[name] = _write_value_artifact(root, name, value)

    manifest = {
        "schema_version": config.schema_version,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "cache_schema_version": 2,
        "complete": True,
        "reference_year": reference_year,
        "patient_split_seed": split.effective_seed,
        "requested_patient_split_seed": split.requested_seed,
        "canonical_sae_seed": config.canonical_sae_seed,
        "sae_seeds": list(config.sae_seeds),
        "temporal_domain_map": {str(year): domain for year, domain in domain_map.items()},
        "estimand": TEMPORAL_ESTIMAND,
        "support": split.support,
        "support_thresholds": config.support.__dict__,
        "population_fingerprints": _fingerprint_population(population),
        "role_fingerprint": stable_hash(role_rows),
        "config_fingerprint": stable_hash(config.to_dict()),
        "scientific_config_fingerprint": stable_hash(_scientific_config(config)),
        "dependent_config_fingerprints": identity["dependent_config_fingerprints"],
        "source_fingerprints": identity["source_fingerprints"],
        "scientific_identity": identity,
        "cache_namespace": str(root / "cache"),
        "artifacts": artifact_files,
    }
    atomic_write_json(root / "manifest.json", manifest)
    return manifest


def run_temporal_robustness(
    config: TemporalRobustnessConfig,
    *,
    adapter: TemporalExperimentAdapter | None = None,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run all reference years and valid patient splits; index every outcome."""

    if adapter is None:
        from temporal_production import ProductionTemporalAdapter

        adapter = ProductionTemporalAdapter()
    population = adapter.load_population(config)
    population.validate()
    feature_year = population.feature_selection_max_year
    configured_feature_year = config.feature_selection_max_year
    if feature_year is None:
        feature_year = configured_feature_year
    if feature_year is None:
        raise ValueError("feature_selection_max_year provenance is required")
    if int(feature_year) > min(config.reference_years):
        raise ValueError("future-derived feature vocabulary rejected")

    population_fingerprints = _fingerprint_population(population)
    dependent_fingerprints = _dependent_config_fingerprints(config)
    source_fingerprints = _temporal_source_fingerprints(adapter)
    root_hash = stable_hash(
        _scientific_config(config),
        population_fingerprints,
        dependent_fingerprints,
        source_fingerprints,
    )[:20]
    root = Path(config.artifact_dir) / root_hash
    # The pre-v2 namespace included transport toggles and no dependency/source
    # fingerprints. Reuse it only when the exact legacy identity already exists.
    legacy_hash = stable_hash(config.to_dict(), population_fingerprints)[:20]
    legacy_root = Path(config.artifact_dir) / legacy_hash
    using_legacy_root = False
    if not root.exists() and legacy_root.is_dir():
        root_hash = legacy_hash
        root = legacy_root
        using_legacy_root = True
    root.mkdir(parents=True, exist_ok=True)
    recovery_frontier = None
    if using_legacy_root and not config.force:
        parent_path = root / "parent_manifest.json"
        if parent_path.is_file():
            try:
                existing_parent = json.loads(parent_path.read_text(encoding="utf-8"))
                if existing_parent.get("complete") is True:
                    recorded_frontier = existing_parent.get(
                        "recovery_frontier_reference_year"
                    )
                    if recorded_frontier is not None:
                        recovery_frontier = int(recorded_frontier)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        if recovery_frontier is None:
            completed_years = []
            for manifest_path in root.glob("reference_*/split_*/manifest.json"):
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest.get("complete") is True:
                        completed_years.append(int(manifest["reference_year"]))
                except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            if completed_years:
                recovery_frontier = max(completed_years)
    successful = []
    skipped = []
    failed = []
    split_attempts = {}
    for reference_year in config.reference_years:
        if recovery_frontier is not None and reference_year > recovery_frontier:
            skipped.append(
                {
                    "reference_year": reference_year,
                    "reason": "outside_legacy_recovery_scope",
                    "recovery_frontier_reference_year": recovery_frontier,
                }
            )
            continue
        reference_indices = np.flatnonzero(population.years == reference_year)
        if not len(reference_indices):
            skipped.append({"reference_year": reference_year, "reason": "no_reference_records"})
            continue
        valid, attempts = generate_valid_reference_splits(
            population.patient_ids[reference_indices],
            population.outcomes[reference_indices],
            config.patient_split_seeds,
            config.support,
            maximum_attempts=config.maximum_split_attempts,
        )
        split_attempts[str(reference_year)] = attempts
        if len(valid) < len(config.patient_split_seeds):
            skipped.append(
                {
                    "reference_year": reference_year,
                    "reason": "unable_to_obtain_required_valid_splits",
                    "valid_split_count": len(valid),
                    "required_split_count": len(config.patient_split_seeds),
                    "attempt_count": len(attempts),
                }
            )
            continue
        for split in valid:
            experiment_root = root / f"reference_{reference_year}" / f"split_{split.effective_seed}"
            try:
                manifest = run_reference_experiment(
                    population=population,
                    reference_year=reference_year,
                    split=split,
                    config=config,
                    adapter=adapter,
                    workspace=experiment_root,
                )
                successful.append(
                    {
                        "reference_year": reference_year,
                        "patient_split_seed": split.effective_seed,
                        "manifest": str(experiment_root / "manifest.json"),
                        "manifest_fingerprint": _sha256(experiment_root / "manifest.json"),
                    }
                )
            except Exception as error:
                failed.append(
                    {
                        "reference_year": reference_year,
                        "patient_split_seed": split.effective_seed,
                        "reason": type(error).__name__,
                        "message": str(error),
                    }
                )
                if fail_fast:
                    raise
    from temporal_reporting import build_parent_reports

    aggregate_artifacts = _aggregate_artifacts(root, successful)
    reports = build_parent_reports(root, successful, config)
    for name, rows in reports.items():
        path = root / "aggregate" / f"{name}.jsonl.gz"
        descriptor = atomic_write_jsonl_gzip(path, rows)
        descriptor["path"] = str(path.relative_to(root))
        aggregate_artifacts[name] = descriptor
    parent = {
        "schema_version": config.schema_version,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "runner_hash": root_hash,
        "artifact_dir": str(root),
        "estimand": TEMPORAL_ESTIMAND,
        "config": config.to_dict(),
        "feature_selection_max_year": int(feature_year),
        "population_fingerprints": population_fingerprints,
        "scientific_config_fingerprint": stable_hash(_scientific_config(config)),
        "dependent_config_fingerprints": dependent_fingerprints,
        "source_fingerprints": source_fingerprints,
        "recovery_frontier_reference_year": recovery_frontier,
        "successful_experiments": successful,
        "skipped_references": skipped,
        "failed_experiments": failed,
        "split_attempts": split_attempts,
        "aggregate_artifacts": aggregate_artifacts,
    }
    atomic_write_json(root / "summary.json", {
        "runner_hash": root_hash,
        "artifact_dir": str(root),
        "successful_count": len(successful),
        "skipped_reference_count": len(skipped),
        "failed_count": len(failed),
    })
    # This parent completion marker is deliberately the final publication.
    atomic_write_json(root / "parent_manifest.json", parent)
    return parent


def _aggregate_artifacts(root: Path, successful) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[tuple[Path, str | Mapping[str, Any]]]] = {}
    for experiment in successful:
        manifest_path = Path(experiment["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, descriptor in manifest.get("artifacts", {}).items():
            try:
                first = next(iter_artifact_rows(manifest_path.parent, descriptor))
            except (StopIteration, ValueError, OSError, TypeError):
                continue
            if not isinstance(first, Mapping):
                continue
            grouped.setdefault(name, []).append((manifest_path.parent, descriptor))
    output = root / "aggregate"
    files: dict[str, dict[str, Any]] = {}
    for name, sources in sorted(grouped.items()):
        def rows():
            for directory, descriptor in sources:
                yield from iter_artifact_rows(directory, descriptor)

        path = output / f"{name}.jsonl.gz"
        artifact = atomic_write_jsonl_gzip(path, rows())
        artifact["path"] = str(path.relative_to(root))
        files[name] = artifact
    return files


def _jsonable(value):
    return jsonable(value)


def _write_json(path: Path, value) -> None:
    atomic_write_json(path, value, compact=False)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    key: (
                        json.dumps(_jsonable(row.get(key)), sort_keys=True, separators=(",", ":"))
                        if isinstance(row.get(key), (Mapping, list, tuple))
                        else _jsonable(row.get(key))
                    )
                    for key in fieldnames
                })


def _sha256(path: Path) -> str:
    return file_sha256(path)
