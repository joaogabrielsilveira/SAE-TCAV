"""Parent orchestration for frozen, reference-specific temporal experiments."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from semantic_artifacts import array_fingerprint, stable_hash
from temporal_cohorts import relative_domain_map
from temporal_config import TemporalRobustnessConfig
from temporal_splits import ReferenceSplit, generate_valid_reference_splits


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
    _write_json(root / "reference_roles.json", role_rows)
    _write_csv(root / "reference_roles.csv", role_rows)
    _write_json(root / "temporal_domain_map.json", {str(year): domain for year, domain in domain_map.items()})

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

    artifact_files = {}
    for name, value in sorted(payload.items()):
        if name.startswith("_"):
            continue
        path = root / f"{name}.json"
        _write_json(path, value)
        artifact_files[name] = path.name
        if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
            csv_path = root / f"{name}.csv"
            _write_csv(csv_path, value)

    manifest = {
        "schema_version": config.schema_version,
        "cache_schema_version": 1,
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
        "cache_namespace": str(root / "cache"),
        "artifacts": artifact_files,
    }
    _write_json(root / "manifest.json", manifest)
    return manifest


def run_temporal_robustness(
    config: TemporalRobustnessConfig,
    *,
    adapter: TemporalExperimentAdapter,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run all reference years and valid patient splits; index every outcome."""

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

    root_hash = stable_hash(config.to_dict(), _fingerprint_population(population))[:20]
    root = Path(config.artifact_dir) / root_hash
    root.mkdir(parents=True, exist_ok=True)
    successful = []
    skipped = []
    failed = []
    split_attempts = {}
    for reference_year in config.reference_years:
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
    parent = {
        "schema_version": config.schema_version,
        "runner_hash": root_hash,
        "artifact_dir": str(root),
        "estimand": TEMPORAL_ESTIMAND,
        "config": config.to_dict(),
        "feature_selection_max_year": int(feature_year),
        "population_fingerprints": _fingerprint_population(population),
        "successful_experiments": successful,
        "skipped_references": skipped,
        "failed_experiments": failed,
        "split_attempts": split_attempts,
    }
    _write_json(root / "parent_manifest.json", parent)
    _write_json(root / "summary.json", {
        "runner_hash": root_hash,
        "artifact_dir": str(root),
        "successful_count": len(successful),
        "skipped_reference_count": len(skipped),
        "failed_count": len(failed),
    })
    return parent


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _jsonable(row.get(key)) for key in fieldnames})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
