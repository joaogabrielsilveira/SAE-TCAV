"""Natural-prevalence historical-window TabPFN performance experiment.

This module is intentionally independent from temporal robustness and metric
synthesis outputs.  Its public helpers are pure where possible so protocol
invariants can be tested without loading TabPFN.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import gc
import gzip
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import pickle
import sys
import time
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_json_gzip,
    atomic_write_jsonl_gzip,
    descriptor_for_file,
    file_sha256,
    read_artifact,
    validate_descriptor,
)
from semantic_artifacts import array_fingerprint, stable_hash
from temporal_robustness import TemporalPopulation, _fingerprint_population
from temporal_splits import generate_valid_reference_splits


WINDOWS = (
    "legacy_reference_only",
    "reference_only_common",
    "last_2",
    "last_3",
    "last_5",
    "all_history",
)
COHORTS = (
    "all_comer",
    "pipeline_unseen",
    "returning_model_seen",
    "threshold_only_seen",
)
COMMON_FITTING_ROLES = ("tabpfn_context", "sae_discovery", "rule_discovery")
METRIC_NAMES = (
    "death_f1_at_0_5",
    "death_f1_at_frozen_threshold",
    "death_average_precision",
    "auroc",
    "brier_score",
    "calibration_intercept",
    "calibration_slope",
    "observed_death_prevalence",
    "death_precision_at_0_5",
    "death_recall_at_0_5",
    "predicted_positive_rate_at_0_5",
    "death_precision_at_frozen_threshold",
    "death_recall_at_frozen_threshold",
    "predicted_positive_rate_at_frozen_threshold",
    "macro_f1_at_0_5",
    "death_f1_oracle",
)


@dataclass(frozen=True)
class WindowExperimentConfig:
    schema_version: str = "1.0"
    dataset_path: str = "tidy_event_data.feather"
    artifact_dir: str = "stats/temporal_performance_windows"
    comparison_config_path: str = "comparison_runner.example.json"
    parent_manifest: str | None = None
    device: str = "auto"
    reference_years: tuple[int, ...] = tuple(range(2007, 2016))
    patient_split_seeds: tuple[int, ...] = (42, 43, 44)
    windows: tuple[str, ...] = WINDOWS
    first_history_year: int = 2007
    final_evaluation_year: int = 2015
    maximum_split_attempts: int = 100
    minimum_class_count: int = 1
    minimum_evaluation_deaths: int = 10
    minimum_evaluation_survivors: int = 30
    bootstrap_replicates: int = 1000
    bootstrap_seed: int = 42
    # Empirically safe starting point for the largest natural-prevalence window
    # on the project's 8GB GPU. The exact legacy protocol separately retains
    # its parent batch size for reproducibility.
    batch_size: int = 256
    cpu_workers: int = 4
    max_pending_bootstrap_jobs: int = 8
    cuda_allocator_config: str | None = "max_split_size_mb:128"
    allow_tf32: bool = True
    pin_memory: bool = True
    use_cache: bool = True
    show_progress: bool = True
    force: bool = False

    def __post_init__(self) -> None:
        for name in ("reference_years", "patient_split_seeds", "windows"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "reference_years", tuple(int(x) for x in self.reference_years))
        object.__setattr__(self, "patient_split_seeds", tuple(int(x) for x in self.patient_split_seeds))
        unknown = set(self.windows) - set(WINDOWS)
        if unknown:
            raise ValueError(f"unknown windows: {sorted(unknown)}")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.cuda_allocator_config is not None and not self.cuda_allocator_config.strip():
            raise ValueError("cuda_allocator_config must be non-empty or null")
        if self.first_history_year > min(self.reference_years):
            raise ValueError("first_history_year exceeds earliest reference year")
        if max(self.reference_years) > self.final_evaluation_year:
            raise ValueError("reference year exceeds final_evaluation_year")
        if self.maximum_split_attempts < len(self.patient_split_seeds):
            raise ValueError("maximum_split_attempts is too small")
        if min(
            self.minimum_class_count,
            self.minimum_evaluation_deaths,
            self.minimum_evaluation_survivors,
            self.bootstrap_replicates,
            self.batch_size,
            self.cpu_workers,
            self.max_pending_bootstrap_jobs,
        ) < 1:
            raise ValueError("counts, bootstrap_replicates, and batch_size must be positive")

    def development_profile(self) -> "WindowExperimentConfig":
        return replace(
            self,
            reference_years=self.reference_years[:1],
            patient_split_seeds=self.patient_split_seeds[:1],
            windows=("legacy_reference_only", "reference_only_common", "last_2"),
            bootstrap_replicates=min(self.bootstrap_replicates, 100),
        )

    def largest_window_pilot(self) -> "WindowExperimentConfig":
        return replace(
            self,
            reference_years=(max(self.reference_years),),
            patient_split_seeds=self.patient_split_seeds[:1],
            windows=("all_history",),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WindowExperimentConfig":
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown window config fields: {sorted(unknown)}")
        converted = dict(value)
        for name in ("reference_years", "patient_split_seeds", "windows"):
            if name in converted:
                converted[name] = tuple(converted[name])
        return cls(**converted)

    @classmethod
    def from_json(cls, path: str | Path) -> "WindowExperimentConfig":
        source = Path(path).resolve()
        value = json.loads(source.read_text(encoding="utf-8"))
        config = cls.from_dict(value)
        updates = {}
        for name in ("dataset_path", "artifact_dir", "comparison_config_path", "parent_manifest"):
            raw = getattr(config, name)
            if raw is not None and not Path(raw).is_absolute():
                updates[name] = str((source.parent / raw).resolve())
        return replace(config, **updates)


@dataclass(frozen=True)
class ProbabilityResult:
    probabilities: np.ndarray
    classes: np.ndarray
    model_info: Mapping[str, Any] = field(default_factory=dict)


class WindowModelAdapter(Protocol):
    def load_population(self, config: WindowExperimentConfig) -> TemporalPopulation: ...

    def fit_predict(
        self,
        *,
        population: TemporalPopulation,
        train_indices: np.ndarray,
        predict_indices: np.ndarray,
        model_domain_ids: np.ndarray,
        prediction_domain_ids: np.ndarray,
        seed: int,
        config: WindowExperimentConfig,
        require_parent_fit: bool = False,
    ) -> ProbabilityResult: ...


def _is_cuda_out_of_memory(error: BaseException) -> bool:
    return "out of memory" in str(error).lower() and "cuda" in str(error).lower()


def _cuda_oom_backoff_batch_size(
    error: BaseException,
    current_batch_size: int,
    *,
    cuda_requested: bool,
    exact_parent_protocol: bool,
) -> int | None:
    if (
        not cuda_requested
        or exact_parent_protocol
        or current_batch_size <= 1
        or not _is_cuda_out_of_memory(error)
    ):
        return None
    return max(1, int(current_batch_size) // 2)


def _release_cuda_memory() -> None:
    """Release unreachable tensors and cached CUDA allocator blocks."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _configure_cuda_allocator(config: WindowExperimentConfig) -> None:
    if config.device != "cpu" and config.cuda_allocator_config is not None:
        # This runs before the production adapter imports torch/TabPFN.  The
        # setting is also retained in the manifest's numerical environment.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config.cuda_allocator_config


class ProductionWindowAdapter:
    """Thin adapter around existing prepared population and TabPFN fit."""

    def __init__(self) -> None:
        self._prepared = None
        self._model_name = "tabpfn_dist_model_1"
        self._parent_root: Path | None = None
        self._legacy_batch_size = 512
        self._initial_torch_precision: tuple[bool, bool, str] | None = None

    def load_population(self, config: WindowExperimentConfig) -> TemporalPopulation:
        from comparison_cache import ComparisonCache
        from comparison_runner import ComparisonRunnerConfig, DefaultComparisonAdapter

        if config.parent_manifest:
            from temporal_config import TemporalRobustnessConfig
            from temporal_production import ProductionTemporalAdapter

            parent_path = Path(config.parent_manifest)
            parent = json.loads(parent_path.read_text(encoding="utf-8"))
            if parent.get("complete") is not True:
                raise ValueError("parent temporal manifest is incomplete")
            parent_config = TemporalRobustnessConfig.from_dict(parent["config"])
            retained_root = Path(parent["artifact_dir"]).parent
            loader = ProductionTemporalAdapter()
            population = loader.load_retained_population(
                parent_config,
                retained_root,
                parent["population_fingerprints"],
            )
            self._prepared = loader._source_prepared
            self._model_name = loader._base_config.tabpfn.model_name
            self._legacy_batch_size = loader._base_config.tabpfn.batch_size
            self._parent_root = parent_path.parent
            return population

        base = ComparisonRunnerConfig.from_json(config.comparison_config_path)
        base = replace(
            base,
            dataset_path=config.dataset_path,
            artifact_dir=config.artifact_dir,
            use_cache=config.use_cache,
            show_progress=config.show_progress,
            accelerator=replace(base.accelerator, device=config.device),
        )
        self._model_name = base.tabpfn.model_name
        cache = ComparisonCache(Path(config.artifact_dir) / "_population_cache", enabled=config.use_cache)
        workspace = Path(config.artifact_dir) / "_population"
        workspace.mkdir(parents=True, exist_ok=True)
        self._prepared = DefaultComparisonAdapter(cache).prepare(base, workspace, force=config.force)
        patients = self._prepared.patient_ids.astype(str)
        years = self._prepared.years_test.astype(int)
        return TemporalPopulation(
            X=np.asarray(self._prepared.X_test),
            outcomes=np.asarray(self._prepared.y_test),
            years=years,
            patient_ids=patients,
            feature_names=tuple(self._prepared.feature_names),
            first_eligible_year={p: int(np.min(years[patients == p])) for p in np.unique(patients)},
            record_keys=np.asarray(self._prepared.record_keys).astype(str),
            feature_selection_max_year=int(np.max(self._prepared.years_train)),
        )

    def fit_predict(
        self,
        *,
        population: TemporalPopulation,
        train_indices: np.ndarray,
        predict_indices: np.ndarray,
        model_domain_ids: np.ndarray,
        prediction_domain_ids: np.ndarray,
        seed: int,
        config: WindowExperimentConfig,
        require_parent_fit: bool = False,
    ) -> ProbabilityResult:
        import torch
        from tabpfn_model import TabPFNEvalConfig, fit_dr_tabpfn, make_dist_tensor

        cuda_requested = torch.cuda.is_available() and config.device in {"auto", "cuda"}
        if cuda_requested:
            # TabPFN performs CUDA work internally even when its additional_x
            # tensor is retained on CPU.  Release allocator state left by the
            # preceding independently fitted model before measuring this job.
            _release_cuda_memory()
            if self._initial_torch_precision is None:
                self._initial_torch_precision = (
                    bool(torch.backends.cuda.matmul.allow_tf32),
                    bool(torch.backends.cudnn.allow_tf32),
                    torch.get_float32_matmul_precision(),
                )
            if require_parent_fit:
                matmul_tf32, cudnn_tf32, matmul_precision = self._initial_torch_precision
            else:
                matmul_tf32 = cudnn_tf32 = config.allow_tf32
                matmul_precision = "high" if config.allow_tf32 else "highest"
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = cudnn_tf32
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(matmul_precision)
            torch.cuda.reset_peak_memory_stats()

        fit = self._load_parent_fit(population, train_indices, seed) if require_parent_fit else None
        if require_parent_fit and fit is None:
            raise RuntimeError("validated parent TabPFN fit is unavailable for legacy arm")
        if fit is None:
            evaluation = TabPFNEvalConfig()
            evaluation.rng_seed = int(seed)
            evaluation.tabpfn_model_name = self._model_name
            evaluation.device = config.device
            evaluation.batch_size_predict = config.batch_size
            evaluation.show_progress = config.show_progress
            # fit_dr_tabpfn derives contiguous IDs from these values. Passing model
            # domain IDs, not calendar distance, preserves that contract.
            fit = fit_dr_tabpfn(
                population.X[train_indices],
                population.outcomes[train_indices],
                model_domain_ids,
                evaluation,
            )
            fit_origin = "new_fit"
        else:
            fit_origin = "validated_parent_cache"
        model = fit["model"]
        model_device = torch.device(fit["model_add_x_device"])
        chunks = []
        prediction_values = np.ascontiguousarray(
            population.X[predict_indices], dtype=np.float32
        )
        pinned_prediction_values = None
        if model_device.type == "cuda" and config.pin_memory:
            pinned_prediction_values = torch.from_numpy(prediction_values).pin_memory()
        requested_batch_size = self._legacy_batch_size if require_parent_fit else config.batch_size
        active_batch_size = min(requested_batch_size, max(1, len(predict_indices)))
        smallest_batch_size = active_batch_size
        batch_backoff_count = 0
        inference_started = time.perf_counter()
        start = 0
        while start < len(predict_indices):
            end = min(start + active_batch_size, len(predict_indices))
            values = prediction_values[start:end]
            domain = host_values = device_values = output = None
            try:
                domain = make_dist_tensor(
                    prediction_domain_ids[start:end],
                    model_device,
                    fit["example_add_shape"],
                )
                if model_device.type == "cpu":
                    output = model.predict_proba(values, additional_x={"dist_shift_domain": domain})
                else:
                    host_values = (
                        pinned_prediction_values[start:end]
                        if pinned_prediction_values is not None
                        else torch.from_numpy(values)
                    )
                    device_values = host_values.to(
                        model_device,
                        non_blocking=config.pin_memory,
                    )
                    with torch.inference_mode():
                        output = model.predict_proba(
                            device_values,
                            additional_x={"dist_shift_domain": domain},
                        )
            except RuntimeError as error:
                # model_add_x_device can be CPU while TabPFN itself computes on
                # CUDA, so it must not be used to decide whether backoff applies.
                reduced_batch_size = _cuda_oom_backoff_batch_size(
                    error,
                    active_batch_size,
                    cuda_requested=cuda_requested,
                    exact_parent_protocol=require_parent_fit,
                )
                if reduced_batch_size is None:
                    raise
                active_batch_size = reduced_batch_size
                smallest_batch_size = min(smallest_batch_size, active_batch_size)
                batch_backoff_count += 1
                domain = host_values = device_values = output = None
                _release_cuda_memory()
                continue
            if isinstance(output, torch.Tensor):
                output = output.detach().cpu().numpy()
            # Preserve TabPFN's native probability precision.  The parent
            # implementation applies argmax before any dtype conversion; a
            # forced float32 cast can collapse a marginal class preference to
            # an exact 0.5 tie and change the hard label.
            chunks.append(np.asarray(output))
            start = end
        if cuda_requested:
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - inference_started
        probabilities = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=float)
        classes = np.asarray(getattr(model, "classes_", (0, 1)))
        peak_allocated = None
        peak_reserved = None
        if cuda_requested:
            peak_allocated = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            peak_reserved = round(torch.cuda.max_memory_reserved() / 2**20, 1)
        result = ProbabilityResult(
            probabilities=probabilities,
            classes=classes,
            model_info={
                "model_source": fit.get("model_source"),
                "fit_origin": fit_origin,
                "fit_time_seconds": fit.get("fit_time_sec"),
                "model_resolution_error": fit.get("model_resolution_error"),
                "device": "cuda" if cuda_requested else str(model_device),
                "additional_x_device": str(model_device),
                "requested_batch_size": requested_batch_size,
                "effective_batch_size": smallest_batch_size,
                "batch_backoff_count": batch_backoff_count,
                "inference_time_seconds": inference_time,
                "cuda_peak_allocated_mb": peak_allocated,
                "cuda_peak_reserved_mb": peak_reserved,
                "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if cuda_requested else None,
                "pin_memory": config.pin_memory if cuda_requested else None,
            },
        )
        # Drop the fit and model before returning the CPU-only result.  The next
        # job also performs a defensive collection before its fit.
        fit = model = pinned_prediction_values = None
        if cuda_requested:
            _release_cuda_memory()
        return result

    def _load_parent_fit(
        self,
        population: TemporalPopulation,
        train_indices: np.ndarray,
        seed: int,
    ) -> Mapping[str, Any] | None:
        if self._parent_root is None:
            return None
        years = np.asarray(population.years[train_indices], dtype=int)
        if len(np.unique(years)) != 1:
            return None
        reference_year = int(years[0])
        stage_root = (
            self._parent_root
            / f"reference_{reference_year}"
            / f"split_{int(seed)}"
            / "cache"
            / "tabpfn_fit"
        )
        expected = {
            "model_name": self._model_name,
            "seed": int(seed),
            "X_train": array_fingerprint(np.asarray(population.X[train_indices])),
            "y_train": array_fingerprint(np.asarray(population.outcomes[train_indices])),
            "years_train": array_fingerprint(years),
        }
        matches = []
        for manifest_path in stage_root.glob("*/manifest.json.gz"):
            try:
                with gzip.open(manifest_path, "rt", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                dependencies = manifest.get("dependencies", {})
                if manifest.get("complete") is not True or any(
                    dependencies.get(name) != value for name, value in expected.items()
                ):
                    continue
                payload = manifest_path.parent / "fitted_model.pkl"
                descriptor = manifest.get("payloads", {}).get("fitted_model.pkl", {})
                if not payload.is_file() or file_sha256(payload) != descriptor.get("sha256"):
                    continue
                matches.append(payload)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        if len(matches) != 1:
            return None
        with matches[0].open("rb") as handle:
            value = pickle.load(handle)
        return value if isinstance(value, Mapping) and "model" in value else None


def post_death_exclusion_mask(
    patient_ids: Sequence[object], years: Sequence[int], outcomes: Sequence[int]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Keep all rows through first observed death; audit later rows."""

    patients = np.asarray(patient_ids).astype(str)
    year_values = np.asarray(years, dtype=int)
    truth = np.asarray(outcomes, dtype=int)
    if not (patients.ndim == year_values.ndim == truth.ndim == 1) or not (
        len(patients) == len(year_values) == len(truth)
    ):
        raise ValueError("post-death inputs must be aligned vectors")
    first_death = {
        patient: int(np.min(year_values[(patients == patient) & (truth == 1)]))
        for patient in np.unique(patients)
        if np.any((patients == patient) & (truth == 1))
    }
    keep = np.ones(len(patients), dtype=bool)
    audit = []
    for index, (patient, year) in enumerate(zip(patients, year_values)):
        death_year = first_death.get(patient)
        if death_year is not None and int(year) > death_year:
            keep[index] = False
            audit.append(
                {
                    "row_index": int(index),
                    "patient_id": patient,
                    "year": int(year),
                    "first_death_year": death_year,
                    "reason": "after_first_observed_death",
                }
            )
    return keep, audit


def effective_window_years(
    logical_window: str, reference_year: int, first_history_year: int = 2007
) -> tuple[int, ...]:
    if logical_window == "legacy_reference_only" or logical_window == "reference_only_common":
        return (int(reference_year),)
    if logical_window == "all_history":
        start = first_history_year
    elif logical_window.startswith("last_"):
        width = int(logical_window.removeprefix("last_"))
        start = max(first_history_year, int(reference_year) - width + 1)
    else:
        raise ValueError(f"unknown window {logical_window!r}")
    return tuple(range(int(start), int(reference_year) + 1))


def build_training_indices(
    *,
    population: TemporalPopulation,
    reference_year: int,
    logical_window: str,
    global_roles: Mapping[str, np.ndarray],
    common_keep: np.ndarray,
    first_history_year: int = 2007,
) -> np.ndarray:
    years = np.asarray(population.years, dtype=int)
    if logical_window == "legacy_reference_only":
        indices = np.asarray(global_roles["tabpfn_context"], dtype=int)
    else:
        selected_years = effective_window_years(logical_window, reference_year, first_history_year)
        prior = np.flatnonzero(np.isin(years, selected_years) & (years < reference_year))
        current = np.concatenate(
            [np.asarray(global_roles[name], dtype=int) for name in COMMON_FITTING_ROLES]
        )
        indices = np.unique(np.concatenate((prior, current)))
        indices = indices[np.asarray(common_keep, dtype=bool)[indices]]
    if np.any(years[indices] > reference_year):
        raise AssertionError("training contains future rows")
    forbidden = np.concatenate(
        [
            np.asarray(global_roles["rule_selection_cav"], dtype=int),
            np.asarray(global_roles["t0_evaluation"], dtype=int),
        ]
    )
    if np.intersect1d(indices, forbidden).size:
        raise AssertionError("year-R threshold/evaluation row entered training")
    return np.asarray(indices, dtype=int)


def model_domain_mapping(training_years: Sequence[int], prediction_years: Sequence[int]) -> dict[int, int]:
    """Contiguous model IDs, deliberately unrelated to reported distance."""

    years = sorted(set(map(int, training_years)) | set(map(int, prediction_years)))
    return {year: index for index, year in enumerate(years)}


def build_prediction_indices(
    population: TemporalPopulation,
    reference_year: int,
    validation_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    *,
    exact_parent_protocol: bool,
) -> np.ndarray:
    """Build query rows, preserving the parent's full ordering for legacy fits."""

    if exact_parent_protocol:
        return np.flatnonzero(population.years >= int(reference_year))
    return np.unique(np.concatenate((
        np.asarray(validation_indices, dtype=int),
        np.asarray(evaluation_indices, dtype=int),
    )))


def death_probabilities(result: ProbabilityResult) -> np.ndarray:
    matrix = np.asarray(result.probabilities, dtype=float)
    classes = np.asarray(result.classes)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError("probability columns do not align with classes")
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError("model classes must contain death class 1 exactly once")
    values = matrix[:, matches[0]]
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("death probabilities must be finite and lie in [0, 1]")
    return values


def argmax_binary_labels(result: ProbabilityResult) -> np.ndarray:
    """Exact existing hard-label behavior, including 0.5 ties."""

    matrix = np.asarray(result.probabilities, dtype=float)
    classes = np.asarray(result.classes)
    if matrix.ndim != 2 or matrix.shape[1] != len(classes):
        raise ValueError("probability columns do not align with classes")
    labels = classes[np.argmax(matrix, axis=1)]
    if np.any(~np.isin(labels, (0, 1))):
        raise ValueError("model classes must be binary labels 0 and 1")
    return np.asarray(labels, dtype=int)


def select_frozen_threshold(y_true: Sequence[int], death_probability: Sequence[float]) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(death_probability, dtype=float)
    if truth.ndim != 1 or probability.shape != truth.shape or len(truth) == 0:
        raise ValueError("threshold inputs must be non-empty aligned vectors")
    order = np.argsort(-probability, kind="mergesort")
    scores_descending = probability[order]
    truth_descending = truth[order]
    positives = int(np.count_nonzero(truth == 1))
    true_positive = false_positive = 0
    rows = []
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores_descending[end] == scores_descending[start]:
            end += 1
        group_positive = int(np.count_nonzero(truth_descending[start:end] == 1))
        true_positive += group_positive
        false_positive += end - start - group_positive
        false_negative = positives - true_positive
        precision = true_positive / (true_positive + false_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = 2 * true_positive / denominator if denominator else 0.0
        rows.append((float(f1), float(precision), float(scores_descending[start])))
        start = end
    best_f1, best_precision, threshold = max(rows, key=lambda row: (row[0], row[1], row[2]))
    return {
        "threshold": threshold,
        "death_f1": best_f1,
        "death_precision": best_precision,
        "candidate_count": int(len(rows)),
        "tie_breaking": "death_f1_then_precision_then_higher_threshold",
    }


def _calibration(y_true: np.ndarray, probability: np.ndarray) -> tuple[float | None, float | None, str | None]:
    if len(np.unique(y_true)) < 2:
        return None, None, "single_outcome_class"
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    if np.ptp(logit) == 0:
        return None, None, "constant_probability"
    design = np.column_stack((np.ones(len(logit)), logit[:, 0]))
    coefficient = np.zeros(2, dtype=float)
    try:
        for _ in range(100):
            linear = np.clip(design @ coefficient, -30, 30)
            fitted = 1 / (1 + np.exp(-linear))
            weight = np.clip(fitted * (1 - fitted), 1e-9, None)
            information = design.T @ (weight[:, None] * design)
            step = np.linalg.solve(information, design.T @ (y_true - fitted))
            coefficient += step
            if np.max(np.abs(step)) < 1e-9:
                break
        if np.any(~np.isfinite(coefficient)):
            raise FloatingPointError
    except (np.linalg.LinAlgError, FloatingPointError):
        return None, None, "calibration_fit_failed:numerical_degeneracy"
    return float(coefficient[0]), float(coefficient[1]), None


def _binary_scores(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(predicted, dtype=bool)
    tp = int(np.count_nonzero((truth == 1) & pred))
    fp = int(np.count_nonzero((truth == 0) & pred))
    fn = int(np.count_nonzero((truth == 1) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _macro_f1(y_true: np.ndarray, predicted: np.ndarray) -> float:
    positive = _binary_scores(y_true, predicted)["f1"]
    negative = _binary_scores(1 - y_true, ~np.asarray(predicted, dtype=bool))["f1"]
    return float((positive + negative) / 2)


def _average_precision(y_true: np.ndarray, probability: np.ndarray) -> float:
    positives = int(np.sum(y_true == 1))
    if positives == 0:
        return 0.0
    order = np.argsort(-probability, kind="mergesort")
    truth = y_true[order]
    scores = probability[order]
    cumulative_true = np.cumsum(truth == 1)
    cumulative_false = np.cumsum(truth == 0)
    endpoints = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    precision = cumulative_true[endpoints] / (cumulative_true[endpoints] + cumulative_false[endpoints])
    recall = cumulative_true[endpoints] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _auroc(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    positive_count = int(np.count_nonzero(y_true == 1))
    negative_count = int(np.count_nonzero(y_true == 0))
    if not positive_count or not negative_count:
        return None
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(len(probability), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_probability[end] == sorted_probability[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive_rank_sum = float(np.sum(ranks[y_true == 1]))
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def metric_bundle(
    y_true: Sequence[int],
    death_probability: Sequence[float],
    frozen_threshold: float,
    *,
    hard_labels_at_half: Sequence[int] | None = None,
    minimum_deaths: int = 1,
    minimum_survivors: int = 1,
) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(death_probability, dtype=float)
    if truth.ndim != 1 or probability.shape != truth.shape or len(truth) == 0:
        raise ValueError("metric inputs must be non-empty aligned vectors")
    if minimum_deaths < 1 or minimum_survivors < 1:
        raise ValueError("minimum outcome support must be positive")
    at_half = probability > 0.5 if hard_labels_at_half is None else np.asarray(hard_labels_at_half, dtype=int)
    if at_half.shape != truth.shape or np.any(~np.isin(at_half, (0, 1))):
        raise ValueError("hard labels at 0.5 must be aligned binary labels")
    frozen = probability >= float(frozen_threshold)
    death_count = int(np.count_nonzero(truth == 1))
    survivor_count = int(np.count_nonzero(truth == 0))
    valid = death_count >= minimum_deaths and survivor_count >= minimum_survivors
    base = {
        "record_count": int(len(truth)),
        "death_count": death_count,
        "survivor_count": survivor_count,
        "observed_death_prevalence": float(np.mean(truth)),
        "valid": bool(valid),
        "failure_reason": None if valid else "insufficient_class_support",
    }
    if not valid:
        return {
            **{
                name: None
                for name in METRIC_NAMES
                if name != "observed_death_prevalence"
            },
            **base,
            "calibration_failure_reason": "insufficient_class_support",
            "oracle_threshold": None,
            "oracle_minus_frozen_f1": None,
            "oracle_minus_0_5_f1": None,
            "frozen_minus_0_5_f1": None,
        }
    oracle = select_frozen_threshold(truth, probability)
    half_scores = _binary_scores(truth, at_half)
    frozen_scores = _binary_scores(truth, frozen)
    intercept, slope, calibration_reason = _calibration(truth, probability)
    values = {
        **base,
        "death_f1_at_0_5": half_scores["f1"],
        "death_f1_at_frozen_threshold": frozen_scores["f1"],
        "death_average_precision": _average_precision(truth, probability),
        "auroc": _auroc(truth, probability),
        "brier_score": float(np.mean((truth - probability) ** 2)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_failure_reason": calibration_reason,
        "death_precision_at_0_5": half_scores["precision"],
        "death_recall_at_0_5": half_scores["recall"],
        "predicted_positive_rate_at_0_5": float(np.mean(at_half)),
        "death_precision_at_frozen_threshold": frozen_scores["precision"],
        "death_recall_at_frozen_threshold": frozen_scores["recall"],
        "predicted_positive_rate_at_frozen_threshold": float(np.mean(frozen)),
        "macro_f1_at_0_5": _macro_f1(truth, at_half),
        "death_f1_oracle": oracle["death_f1"],
        "oracle_threshold": oracle["threshold"],
        "oracle_minus_frozen_f1": float(oracle["death_f1"] - frozen_scores["f1"]),
        "oracle_minus_0_5_f1": float(oracle["death_f1"] - half_scores["f1"]),
        "frozen_minus_0_5_f1": float(frozen_scores["f1"] - half_scores["f1"]),
    }
    return values


def exposure_cohort_masks(
    patient_ids: Sequence[object], training_patient_ids: Sequence[object], threshold_patient_ids: Sequence[object]
) -> dict[str, np.ndarray]:
    patients = np.asarray(patient_ids).astype(str)
    trained = set(np.asarray(training_patient_ids).astype(str))
    threshold = set(np.asarray(threshold_patient_ids).astype(str))
    model_seen = np.asarray([patient in trained for patient in patients], dtype=bool)
    threshold_seen = np.asarray([patient in threshold for patient in patients], dtype=bool)
    return {
        "all_comer": np.ones(len(patients), dtype=bool),
        "pipeline_unseen": ~(model_seen | threshold_seen),
        "returning_model_seen": model_seen,
        "threshold_only_seen": threshold_seen & ~model_seen,
    }


def _patient_bootstrap(
    truth: np.ndarray,
    probability: np.ndarray,
    hard_labels_at_half: np.ndarray,
    patients: np.ndarray,
    threshold: float,
    *,
    replicates: int,
    seed: int,
    minimum_deaths: int,
    minimum_survivors: int,
) -> dict[str, tuple[float | None, float | None]]:
    patient_values = patients.astype(str)
    unique, inverse = np.unique(patient_values, return_inverse=True)
    if len(unique) < 2:
        return {name: (None, None) for name in METRIC_NAMES}
    row_indices = np.arange(len(patient_values))
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRIC_NAMES}
    for _ in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        multiplicity = np.bincount(selected, minlength=len(unique))
        indices = np.repeat(row_indices, multiplicity[inverse])
        row = metric_bundle(
            truth[indices],
            probability[indices],
            threshold,
            hard_labels_at_half=hard_labels_at_half[indices],
            minimum_deaths=minimum_deaths,
            minimum_survivors=minimum_survivors,
        )
        for name in METRIC_NAMES:
            value = row.get(name)
            if value is not None and np.isfinite(value):
                samples[name].append(float(value))
    return {
        name: (
            tuple(float(x) for x in np.percentile(values, (2.5, 97.5)))
            if values else (None, None)
        )
        for name, values in samples.items()
    }


def _bootstrap_worker_initializer() -> None:
    """Prevent each bootstrap process from starting its own BLAS thread pool."""

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except ImportError:
        pass


def _bootstrap_interval_rows(
    tasks: Sequence[
        tuple[Mapping[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]
    ],
    threshold: float,
    replicates: int,
    minimum_deaths: int,
    minimum_survivors: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base, truth, probabilities, hard_labels, patients, seed in tasks:
        intervals = _patient_bootstrap(
            truth,
            probabilities,
            hard_labels,
            patients,
            threshold,
            replicates=replicates,
            seed=seed,
            minimum_deaths=minimum_deaths,
            minimum_survivors=minimum_survivors,
        )
        for metric, (lower, upper) in intervals.items():
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "lower_95": lower,
                    "upper_95": upper,
                    "resampling_unit": "patient",
                }
            )
    return rows


def _calibration_distance(row: Mapping[str, Any]) -> float | None:
    intercept = row.get("calibration_intercept")
    slope = row.get("calibration_slope")
    if intercept is None or slope is None:
        return None
    if not np.isfinite(intercept) or not np.isfinite(slope):
        return None
    return float(abs(float(intercept)) + abs(float(slope) - 1.0))


def build_diagnostic_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Describe within-trajectory changes without unregistered score cutoffs.

    These labels are descriptive. Aggregate conclusions still require the
    reference-year cluster intervals emitted by the runner.
    """

    baselines = {
        (
            int(row["reference_year"]),
            int(row["patient_split_seed"]),
            str(row["window"]),
            str(row["cohort"]),
        ): row
        for row in metrics
        if int(row.get("temporal_distance", -1)) == 0 and row.get("valid") is True
    }
    output = []
    delta_names = (
        "death_f1_at_0_5",
        "death_f1_at_frozen_threshold",
        "death_average_precision",
        "death_f1_oracle",
        "brier_score",
        "observed_death_prevalence",
        "predicted_positive_rate_at_0_5",
        "predicted_positive_rate_at_frozen_threshold",
    )
    for row in metrics:
        base = {
            "reference_year": row.get("reference_year"),
            "patient_split_seed": row.get("patient_split_seed"),
            "window": row.get("window"),
            "test_year": row.get("test_year"),
            "temporal_distance": row.get("temporal_distance"),
            "cohort": row.get("cohort"),
            "evidence_basis": "within_trajectory_change_from_distance_0",
            "inferential_status": "descriptive_pattern_requires_cluster_interval_confirmation",
        }
        if row.get("valid") is not True:
            output.append({**base, "classification": "non_estimable", "failure_reason": row.get("failure_reason")})
            continue
        key = (
            int(row["reference_year"]),
            int(row["patient_split_seed"]),
            str(row["window"]),
            str(row["cohort"]),
        )
        baseline = baselines.get(key)
        if baseline is None:
            output.append({**base, "classification": "no_distance_zero_baseline", "failure_reason": "missing_valid_distance_zero_cell"})
            continue
        deltas = {
            f"delta_{name}": float(row[name]) - float(baseline[name])
            for name in delta_names
            if row.get(name) is not None and baseline.get(name) is not None
        }
        baseline_calibration = _calibration_distance(baseline)
        current_calibration = _calibration_distance(row)
        calibration_worsened = (
            current_calibration is not None
            and baseline_calibration is not None
            and current_calibration > baseline_calibration
        )
        if int(row["temporal_distance"]) == 0:
            no_skill_ap = float(row["death_average_precision"]) <= float(row["observed_death_prevalence"])
            zero_f1 = max(
                float(row["death_f1_at_0_5"]),
                float(row["death_f1_at_frozen_threshold"]),
                float(row["death_f1_oracle"]),
            ) == 0.0
            classification = "inadequate_reference_signal" if no_skill_ap and zero_f1 else "reference_baseline"
        elif deltas.get("delta_death_average_precision", 0.0) < 0 and deltas.get("delta_death_f1_oracle", 0.0) < 0:
            classification = "discrimination_decay_pattern"
        elif (
            deltas.get("delta_death_average_precision", -np.inf) >= 0
            and deltas.get("delta_death_f1_at_frozen_threshold", 0.0) < 0
            and (deltas.get("delta_brier_score", 0.0) > 0 or calibration_worsened)
        ):
            classification = "calibration_prevalence_or_threshold_drift_pattern"
        elif (
            deltas.get("delta_death_f1_at_0_5", 0.0) < 0
            and deltas.get("delta_death_average_precision", -np.inf) >= 0
            and deltas.get("delta_death_f1_oracle", -np.inf) >= 0
        ):
            classification = "default_threshold_or_scaling_failure_pattern"
        else:
            classification = "mixed_or_no_descriptive_pattern"
        output.append(
            {
                **base,
                **deltas,
                "baseline_calibration_distance": baseline_calibration,
                "current_calibration_distance": current_calibration,
                "calibration_distance_worsened": calibration_worsened,
                "classification": classification,
                "failure_reason": None,
            }
        )
    return output


class _ProgressLog:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = root / "logs" / f"run_{stamp}.log"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"window-experiment-{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(self.path, encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def write(self, phase: str, **fields: Any) -> None:
        telemetry = _telemetry()
        text = " ".join(f"{key}={value}" for key, value in {"phase": phase, **fields, **telemetry}.items())
        self.logger.info(text)
        for handler in self.logger.handlers:
            handler.flush()


def _telemetry() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        result["rss_mb"] = round(process.memory_info().rss / 2**20, 1)
        result["system_memory_percent"] = psutil.virtual_memory().percent
    except ImportError:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            result["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            result["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            free, total = torch.cuda.mem_get_info()
            result["gpu_free_mb"] = round(free / 2**20, 1)
            result["gpu_total_mb"] = round(total / 2**20, 1)
    except ImportError:
        pass
    return result


def _checkpoint_path(root: Path, reference: int, split: int, window: str) -> Path:
    return root / "checkpoints" / f"reference_{reference}" / f"split_{split}" / f"{window}.json"


def _load_checkpoint(path: Path, identity: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("identity") != identity:
            return None
        return value
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _save_probability_cache(path: Path, identity: str, result: ProbabilityResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"identity": identity, "result": result}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_probability_cache(path: Path, identity: str) -> ProbabilityResult | None:
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        if value.get("identity") == identity and isinstance(value.get("result"), ProbabilityResult):
            return value["result"]
    except (OSError, ValueError, TypeError, pickle.UnpicklingError):
        return None
    return None


def _write_table(root: Path, name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = root / "artifacts" / f"{name}.jsonl.gz"
    descriptor = atomic_write_jsonl_gzip(path, rows)
    descriptor["path"] = str(path.relative_to(root))
    return descriptor


def _support_thresholds(config: WindowExperimentConfig):
    class Thresholds:
        context_deaths = config.minimum_class_count
        context_survivors = config.minimum_class_count
        t0_deaths = config.minimum_class_count
        t0_survivors = config.minimum_class_count

    return Thresholds()


def _canonical_pointer(config: WindowExperimentConfig, manifest: Path) -> None:
    pointer = Path(config.artifact_dir) / "latest_manifest.json"
    atomic_write_json(
        pointer,
        {
            "manifest": str(manifest.resolve()),
            "sha256": file_sha256(manifest),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _scientific_source_fingerprints() -> dict[str, str | None]:
    root = Path(__file__).resolve().parent
    names = (
        "temporal_performance_windows.py",
        "tabpfn_model.py",
        "temporal_splits.py",
        "temporal_robustness.py",
        "temporal_production.py",
        "comparison_runner.py",
        "artifact_storage.py",
    )
    return {
        name: file_sha256(root / name) if (root / name).is_file() else None
        for name in names
    }


def _numerical_environment(config: WindowExperimentConfig) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "tabpfn": _package_version("tabpfn"),
        "torch": _package_version("torch"),
        "requested_device": config.device,
        "allow_tf32": config.allow_tf32,
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    try:
        import torch

        result.update(
            {
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        )
    except (ImportError, RuntimeError):
        result.update({"cuda_available": False, "cuda_version": None, "cudnn_version": None, "gpu_names": []})
    return result


def _load_parent_performance(config: WindowExperimentConfig) -> list[dict[str, Any]]:
    if not config.parent_manifest:
        raise ValueError(
            "legacy_reference_only requires --parent-manifest so parity can be checked"
        )
    parent_path = Path(config.parent_manifest)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("complete") is not True:
        raise ValueError("parent temporal manifest is incomplete")
    descriptor = parent.get("aggregate_artifacts", {}).get("performance")
    if not isinstance(descriptor, Mapping):
        raise ValueError("parent temporal manifest has no aggregate performance artifact")
    rows = read_artifact(parent_path.parent, descriptor)
    return [dict(row) for row in rows if row.get("cohort_view") == "all_comer"]


def compare_legacy_parent_row(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    y_true: Sequence[int] | None = None,
    death_probability: Sequence[float] | None = None,
    hard_labels_at_half: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare parent aggregates and audit any exact-0.5 boundary ambiguity.

    The parent retained aggregate metrics but not its record-level probability or
    hard-label vectors. Exact aggregate parity remains the normal requirement. A
    mismatch is accepted only when support and prevalence match exactly, the only
    differing fields are the two hard-label-derived F1 metrics, and assigning
    labels to records whose retained-model death probability is *exactly* 0.5 can
    reproduce both parent F1 values. Current labels and metrics are never changed.
    """

    count_fields = (
        ("record_count", "sample_count"),
        ("death_count", "death_count"),
        ("survivor_count", "survivor_count"),
    )
    metric_fields = (
        ("death_f1_at_0_5", "death_f1"),
        ("macro_f1_at_0_5", "macro_f1"),
        ("observed_death_prevalence", "prevalence"),
    )
    checks: dict[str, bool] = {}
    differences: dict[str, float | None] = {}
    for actual_name, expected_name in count_fields:
        checks[f"{actual_name}_matches"] = int(actual[actual_name]) == int(expected[expected_name])
    for actual_name, expected_name in metric_fields:
        left, right = actual.get(actual_name), expected.get(expected_name)
        if left is None or right is None:
            checks[f"{actual_name}_matches"] = left is None and right is None
            differences[f"{actual_name}_difference"] = None
        else:
            difference = float(left) - float(right)
            differences[f"{actual_name}_difference"] = difference
            checks[f"{actual_name}_matches"] = bool(np.isclose(left, right, rtol=0.0, atol=1e-12))
    result = {
        "reference_year": int(actual["reference_year"]),
        "patient_split_seed": int(actual["patient_split_seed"]),
        "test_year": int(actual["test_year"]),
        "temporal_distance": int(actual["temporal_distance"]),
        "cohort": "all_comer",
        **checks,
        **differences,
        "parity": bool(all(checks.values())),
        "hard_label_vector_parity": None,
        "hard_label_vector_parity_unavailable_reason": "parent artifacts retain aggregate metrics, not record-level hard labels",
    }
    result.update(
        {
            "accepted": result["parity"],
            "acceptance_reason": "exact_aggregate_parity" if result["parity"] else None,
            "exact_half_probability_count": None,
            "half_tie_reconciliation_solution_count": 0,
            "half_tie_minimum_label_changes": None,
        }
    )
    if result["parity"] or any(
        values is None for values in (y_true, death_probability, hard_labels_at_half)
    ):
        return result

    support_checks = (
        "record_count_matches",
        "death_count_matches",
        "survivor_count_matches",
        "observed_death_prevalence_matches",
    )
    f1_checks = ("death_f1_at_0_5_matches", "macro_f1_at_0_5_matches")
    if not all(result[name] for name in support_checks):
        return result
    if any(not result[name] for name in checks if name not in f1_checks):
        return result

    truth = np.asarray(y_true, dtype=int)
    probability = np.asarray(death_probability)
    labels = np.asarray(hard_labels_at_half, dtype=int)
    if truth.ndim != 1 or probability.shape != truth.shape or labels.shape != truth.shape:
        raise ValueError("legacy parity reconciliation vectors must be aligned")
    tie_indices = np.flatnonzero(probability == 0.5)
    result["exact_half_probability_count"] = int(len(tie_indices))
    # More ties would indicate a broader reproducibility issue rather than a
    # defensible isolated boundary ambiguity.
    if not len(tie_indices) or len(tie_indices) > 12:
        return result

    expected_death_f1 = expected.get("death_f1")
    expected_macro_f1 = expected.get("macro_f1")
    if expected_death_f1 is None or expected_macro_f1 is None:
        return result
    solutions: list[int] = []
    for assignment in range(1 << len(tie_indices)):
        candidate = labels.copy()
        candidate[tie_indices] = [
            (assignment >> position) & 1 for position in range(len(tie_indices))
        ]
        death_f1 = _binary_scores(truth, candidate)["f1"]
        macro_f1 = _macro_f1(truth, candidate)
        if np.isclose(death_f1, expected_death_f1, rtol=0.0, atol=1e-12) and np.isclose(
            macro_f1, expected_macro_f1, rtol=0.0, atol=1e-12
        ):
            solutions.append(int(np.count_nonzero(candidate != labels)))
    result["half_tie_reconciliation_solution_count"] = len(solutions)
    if solutions:
        result["half_tie_minimum_label_changes"] = min(solutions)
        result["accepted"] = True
        result["acceptance_reason"] = "parent_aggregates_reproduced_by_exact_0_5_tie_assignment"
    return result


def load_window_experiment(path: str | Path) -> dict[str, Any]:
    """Load canonical pointer or manifest, validating every checksum."""

    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if "manifest" in value and "artifacts" not in value:
        manifest_path = Path(value["manifest"])
        if file_sha256(manifest_path) != value["sha256"]:
            raise ValueError("canonical manifest pointer checksum mismatch")
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = manifest_path
    if value.get("complete") is not True:
        raise ValueError("window experiment manifest is incomplete")
    for descriptor in value.get("artifacts", {}).values():
        validate_descriptor(source.parent, descriptor)
    return {
        "manifest_path": source,
        "manifest": value,
        "artifacts": {
            name: read_artifact(source.parent, descriptor)
            for name, descriptor in value.get("artifacts", {}).items()
        },
    }


def _contrast_rows(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["reference_year"], row["patient_split_seed"], row["test_year"], row["cohort"], row["window"]): row
        for row in metrics
    }
    rows = []
    for key, row in sorted(index.items()):
        reference, split, test_year, cohort, window = key
        if window in {"legacy_reference_only", "reference_only_common"}:
            continue
        baseline = index.get((reference, split, test_year, cohort, "reference_only_common"))
        if baseline is None:
            continue
        valid_pair = row.get("valid") is True and baseline.get("valid") is True
        matched = valid_pair and row.get("support_fingerprint") == baseline.get("support_fingerprint")
        for metric in ("death_f1_at_0_5", "death_f1_at_frozen_threshold", "death_average_precision", "brier_score"):
            left, right = row.get(metric), baseline.get(metric)
            rows.append(
                {
                    "reference_year": reference,
                    "patient_split_seed": split,
                    "test_year": test_year,
                    "temporal_distance": test_year - reference,
                    "cohort": cohort,
                    "window": window,
                    "baseline_window": "reference_only_common",
                    "metric": metric,
                    "window_value": left,
                    "baseline_value": right,
                    "difference": None if not matched or left is None or right is None else float(left - right),
                    "matched_support": bool(matched),
                    "valid_pair": bool(valid_pair),
                    "failure_reason": None if matched else (
                        "non_estimable_metric_cell" if not valid_pair else "unmatched_cohort_support"
                    ),
                }
            )
    return rows


def _trajectory_rows(metrics: Sequence[Mapping[str, Any]], config: WindowExperimentConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in metrics:
        if row.get("valid") is not True:
            continue
        key = (row["reference_year"], row["test_year"], row["temporal_distance"], row["cohort"], row["window"])
        grouped.setdefault(key, []).append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        for metric in ("death_f1_at_0_5", "death_f1_at_frozen_threshold", "death_average_precision", "brier_score"):
            observed = [float(row[metric]) for row in values if row.get(metric) is not None]
            rows.append(
                {
                    "reference_year": key[0], "test_year": key[1], "temporal_distance": key[2],
                    "cohort": key[3], "window": key[4], "metric": metric,
                    "split_mean": float(np.mean(observed)) if observed else None,
                    "split_count": len(observed),
                    "aggregation_order": "split_within_reference_year",
                }
            )
    return rows


def _cluster_inference(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_name: str,
    group_names: Sequence[str],
    replicates: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Average splits within reference year, then bootstrap reference years."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if "valid" in row and row.get("valid") is not True:
            continue
        value = row.get(value_name)
        if value is None or not np.isfinite(value):
            continue
        grouped.setdefault(tuple(row.get(name) for name in group_names), []).append(row)
    output = []
    for group_index, (key, values) in enumerate(sorted(grouped.items(), key=lambda item: str(item[0]))):
        by_reference: dict[int, list[float]] = {}
        for row in values:
            by_reference.setdefault(int(row["reference_year"]), []).append(float(row[value_name]))
        reference_means = np.asarray([np.mean(by_reference[year]) for year in sorted(by_reference)])
        lower = upper = None
        if len(reference_means) >= 2:
            rng = np.random.default_rng(seed + group_index)
            draws = [float(np.mean(rng.choice(reference_means, len(reference_means), replace=True))) for _ in range(replicates)]
            lower, upper = (float(x) for x in np.percentile(draws, (2.5, 97.5)))
        output.append(
            {
                **dict(zip(group_names, key)),
                "estimate": float(np.mean(reference_means)),
                "lower_95": lower,
                "upper_95": upper,
                "reference_year_count": int(len(reference_means)),
                "aggregation_order": "split_mean_then_reference_year_cluster_bootstrap",
                "resampling_unit": "reference_year",
            }
        )
    return output


def run_window_experiment(
    config: WindowExperimentConfig,
    *,
    adapter: WindowModelAdapter | None = None,
    fail_fast: bool = False,
    legacy_parent_performance: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    _configure_cuda_allocator(config)
    if adapter is None:
        adapter = ProductionWindowAdapter()
    population = adapter.load_population(config)
    population.validate()
    fingerprints = _fingerprint_population(population)
    scientific = config.to_dict()
    for name in ("artifact_dir", "device", "use_cache", "show_progress", "force"):
        scientific.pop(name, None)
    source_fingerprints = _scientific_source_fingerprints()
    numerical_environment = _numerical_environment(config)
    from comparison_runner import ComparisonRunnerConfig, _tabpfn_checkpoint_fingerprint

    comparison_config = ComparisonRunnerConfig.from_json(config.comparison_config_path)
    dependency_sha256 = {
        "comparison_config": file_sha256(config.comparison_config_path) if Path(config.comparison_config_path).is_file() else None,
        "parent_manifest": file_sha256(config.parent_manifest) if config.parent_manifest and Path(config.parent_manifest).is_file() else None,
        "tabpfn_checkpoint": _tabpfn_checkpoint_fingerprint(
            comparison_config.tabpfn.model_name
        ),
    }
    run_hash = stable_hash(
        scientific,
        fingerprints,
        source_fingerprints,
        numerical_environment,
        dependency_sha256,
    )[:20]
    root = Path(config.artifact_dir) / f"windows_{run_hash}"
    root.mkdir(parents=True, exist_ok=True)
    progress = _ProgressLog(root)
    progress.write("population_ready", records=len(population.X), deaths=int(np.sum(population.outcomes)))

    common_keep, exclusion_rows = post_death_exclusion_mask(
        population.patient_ids, population.years, population.outcomes
    )
    if "legacy_reference_only" in config.windows:
        if legacy_parent_performance is None:
            legacy_parent_performance = _load_parent_performance(config)
        parent_performance_index = {
            (
                int(row["reference_year"]),
                int(row["patient_split_seed"]),
                int(row["test_year"]),
            ): dict(row)
            for row in legacy_parent_performance
            if row.get("cohort_view") == "all_comer"
        }
    else:
        parent_performance_index = {}
    for row in exclusion_rows:
        if population.record_keys is not None:
            row["record_key"] = str(population.record_keys[row["row_index"]])
    role_audit: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    legacy_parity_rows: list[dict[str, Any]] = []
    exclusion_count_rows: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    split_attempts: dict[str, Any] = {}
    jobs_total = len(config.reference_years) * len(config.patient_split_seeds) * len(config.windows)
    job_number = 0
    completed_jobs = 0
    started = time.monotonic()
    bootstrap_pool = ProcessPoolExecutor(
        max_workers=config.cpu_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_bootstrap_worker_initializer,
    )
    pending_jobs: list[dict[str, Any]] = []

    def complete_pending_job(pending: Mapping[str, Any]) -> None:
        nonlocal completed_jobs
        checkpoint_path = pending["checkpoint_path"]
        try:
            job_intervals: list[dict[str, Any]] = []
            for future in pending["bootstrap_futures"]:
                job_intervals.extend(future.result())
            interval_rows.extend(job_intervals)
            job_result = dict(pending["job_result"])
            job_result["intervals"] = job_intervals
            job_result_path = checkpoint_path.with_name(f"{pending['window']}.result.json.gz")
            atomic_write_json_gzip(job_result_path, job_result)
            job_descriptor = descriptor_for_file(job_result_path, relative_to=root)
            elapsed = time.monotonic() - pending["job_start"]
            atomic_write_json(
                checkpoint_path,
                {
                    "identity": pending["identity"],
                    "complete": True,
                    "stage": "metrics_complete",
                    "cache_hit": pending["cache_hit"],
                    "elapsed_seconds": elapsed,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "result_artifact": job_descriptor,
                },
            )
            completed_jobs += 1
            total_elapsed = time.monotonic() - started
            eta = total_elapsed / completed_jobs * max(0, jobs_total - completed_jobs)
            progress.write(
                "job_complete",
                reference_year=pending["reference_year"],
                split=pending["split"],
                window=pending["window"],
                effective_years=pending["effective_years"],
                job=f"{pending['job_number']}/{jobs_total}",
                cache="hit" if pending["cache_hit"] else "miss",
                elapsed_seconds=round(elapsed, 1),
                eta_seconds=round(eta, 1),
            )
        except Exception as error:
            for future in pending["bootstrap_futures"]:
                future.cancel()
            failure = {
                "reference_year": pending["reference_year"],
                "patient_split_seed": pending["split"],
                "window": pending["window"],
                "reason": type(error).__name__,
                "message": str(error),
            }
            failed.append(failure)
            completed_jobs += 1
            atomic_write_json(
                checkpoint_path,
                {
                    "identity": pending["identity"],
                    "complete": False,
                    "stage": "bootstrap_failed",
                    **failure,
                },
            )
            progress.write("job_failed", **failure)
            if fail_fast:
                bootstrap_pool.shutdown(wait=False, cancel_futures=True)
                raise

    for reference_year in config.reference_years:
        reference_indices = np.flatnonzero(population.years == reference_year)
        if not len(reference_indices):
            failed.append({"reference_year": reference_year, "reason": "no_reference_records"})
            continue
        splits, attempts = generate_valid_reference_splits(
            population.patient_ids[reference_indices],
            population.outcomes[reference_indices],
            config.patient_split_seeds,
            _support_thresholds(config),
            maximum_attempts=config.maximum_split_attempts,
        )
        split_attempts[str(reference_year)] = attempts
        if len(splits) != len(config.patient_split_seeds):
            failed.append({"reference_year": reference_year, "reason": "insufficient_valid_splits"})
            continue
        for split in splits:
            global_roles = {
                role: reference_indices[np.asarray(local, dtype=int)] for role, local in split.roles.items()
            }
            for role, indices in global_roles.items():
                for index in indices:
                    role_audit.append(
                        {
                            "reference_year": reference_year,
                            "patient_split_seed": split.effective_seed,
                            "role": role,
                            "row_index": int(index),
                            "record_key": None if population.record_keys is None else str(population.record_keys[index]),
                            "patient_id": str(population.patient_ids[index]),
                            "year": int(population.years[index]),
                            "outcome": int(population.outcomes[index]),
                            "common_protocol_excluded": not bool(common_keep[index]),
                        }
                    )
            for window in config.windows:
                while len(pending_jobs) >= config.max_pending_bootstrap_jobs:
                    complete_pending_job(pending_jobs.pop(0))
                job_number += 1
                job_start = time.monotonic()
                job_bootstrap_futures: list[Future[list[dict[str, Any]]]] = []
                checkpoint_path = _checkpoint_path(root, reference_year, split.effective_seed, window)
                try:
                    train = build_training_indices(
                        population=population,
                        reference_year=reference_year,
                        logical_window=window,
                        global_roles=global_roles,
                        common_keep=common_keep,
                        first_history_year=config.first_history_year,
                    )
                    validation = np.asarray(global_roles["rule_selection_cav"], dtype=int)
                    evaluation = np.unique(np.concatenate((
                        np.asarray(global_roles["t0_evaluation"], dtype=int),
                        np.flatnonzero(
                            (population.years > reference_year)
                            & (population.years <= config.final_evaluation_year)
                        ),
                    )))
                    if window != "legacy_reference_only":
                        validation = validation[common_keep[validation]]
                        evaluation = evaluation[common_keep[evaluation]]
                    raw_train = train if window == "legacy_reference_only" else build_training_indices(
                        population=population,
                        reference_year=reference_year,
                        logical_window=window,
                        global_roles=global_roles,
                        common_keep=np.ones(len(common_keep), dtype=bool),
                        first_history_year=config.first_history_year,
                    )
                    exclusion_count_rows.extend(
                        {
                            "reference_year": reference_year,
                            "patient_split_seed": split.effective_seed,
                            "window": window,
                            "stage": stage,
                            "eligible_before_exclusion": int(before),
                            "retained_after_exclusion": int(after),
                            "post_death_exclusion_count": int(before - after),
                            "protocol": "legacy_unchanged" if window == "legacy_reference_only" else "common_post_death",
                        }
                        for stage, before, after in (
                            ("model_fitting", len(raw_train), len(train)),
                            ("threshold_selection", len(global_roles["rule_selection_cav"]), len(validation)),
                            (
                                "evaluation",
                                int(len(global_roles["t0_evaluation"]) + np.count_nonzero(
                                    (population.years > reference_year) & (population.years <= config.final_evaluation_year)
                                )),
                                len(evaluation),
                            ),
                        )
                    )
                    # Exact legacy parity requires the parent's full query set
                    # and original ordering.  Query membership and 1,024-row
                    # batch boundaries can affect marginal TabPFN outputs.
                    predict = build_prediction_indices(
                        population,
                        reference_year,
                        validation,
                        evaluation,
                        exact_parent_protocol=window == "legacy_reference_only",
                    )
                    effective_years = effective_window_years(window, reference_year, config.first_history_year)
                    protocol = "legacy" if window == "legacy_reference_only" else "common"
                    identity = stable_hash(
                        fingerprints,
                        reference_year,
                        split.effective_seed,
                        protocol,
                        effective_years,
                        array_fingerprint(train),
                        array_fingerprint(predict),
                    )
                    checkpoint = None if config.force else _load_checkpoint(checkpoint_path, identity)
                    if checkpoint and checkpoint.get("complete") is True and isinstance(checkpoint.get("result_artifact"), Mapping):
                        validate_descriptor(root, checkpoint["result_artifact"])
                        saved = read_artifact(root, checkpoint["result_artifact"])
                        threshold_rows.extend(saved["thresholds"])
                        probability_rows.extend(saved["probabilities"])
                        metric_rows.extend(saved["metrics"])
                        interval_rows.extend(saved["intervals"])
                        legacy_parity_rows.extend(saved.get("legacy_parity", []))
                        progress.write(
                            "job_resume", reference_year=reference_year, split=split.effective_seed,
                            window=window, effective_years=effective_years, job=f"{job_number}/{jobs_total}",
                            stage="metrics_complete",
                        )
                        completed_jobs += 1
                        continue
                    starts = {
                        "thresholds": len(threshold_rows), "probabilities": len(probability_rows),
                        "metrics": len(metric_rows),
                        "legacy_parity": len(legacy_parity_rows),
                    }
                    cache_path = root / "fit_cache" / f"{identity}.pkl"
                    result = None if config.force else _load_probability_cache(cache_path, identity)
                    cache_hit = result is not None
                    mapping = model_domain_mapping(population.years[train], population.years[predict])
                    if result is None:
                        progress.write(
                            "fit_start", reference_year=reference_year, split=split.effective_seed,
                            window=window, effective_years=effective_years, job=f"{job_number}/{jobs_total}",
                            records=len(train), deaths=int(np.sum(population.outcomes[train])), cache="miss",
                        )
                        result = adapter.fit_predict(
                            population=population,
                            train_indices=train,
                            predict_indices=predict,
                            model_domain_ids=np.asarray([mapping[int(year)] for year in population.years[train]], dtype=int),
                            prediction_domain_ids=np.asarray([mapping[int(year)] for year in population.years[predict]], dtype=int),
                            seed=split.effective_seed,
                            config=config,
                            require_parent_fit=window == "legacy_reference_only",
                        )
                        _save_probability_cache(cache_path, identity, result)
                    probability = death_probabilities(result)
                    hard = argmax_binary_labels(result)
                    lookup = {int(index): local for local, index in enumerate(predict)}
                    val_local = np.asarray([lookup[int(index)] for index in validation], dtype=int)
                    validation_truth = population.outcomes[validation]
                    if (
                        len(val_local) == 0
                        or np.count_nonzero(validation_truth == 1) < config.minimum_class_count
                        or np.count_nonzero(validation_truth == 0) < config.minimum_class_count
                    ):
                        raise RuntimeError("threshold_validation_insufficient_class_support")
                    threshold = select_frozen_threshold(validation_truth, probability[val_local])
                    progress.write(
                        "threshold_selected", reference_year=reference_year, split=split.effective_seed,
                        window=window, threshold=threshold["threshold"], validation_records=len(validation),
                    )
                    threshold_rows.append(
                        {
                            "reference_year": reference_year,
                            "patient_split_seed": split.effective_seed,
                            "window": window,
                            "logical_window": window,
                            "effective_window_years": list(effective_years),
                            "truncated_early_window": window.startswith("last_") and len(effective_years) < int(window.split("_")[1]),
                            "training_record_count": int(len(train)),
                            "training_death_count": int(np.sum(population.outcomes[train])),
                            "validation_record_count": int(len(validation)),
                            "model_info": dict(result.model_info),
                            **threshold,
                        }
                    )
                    progress.write(
                        "inference_complete", reference_year=reference_year, split=split.effective_seed,
                        window=window, prediction_records=len(predict), **result.model_info,
                    )
                    train_patients = population.patient_ids[train]
                    threshold_patients = population.patient_ids[validation]
                    for test_year in range(reference_year, config.final_evaluation_year + 1):
                        year_indices = evaluation[population.years[evaluation] == test_year]
                        if test_year == reference_year:
                            allowed = set(map(int, global_roles["t0_evaluation"]))
                            year_indices = np.asarray([x for x in year_indices if int(x) in allowed], dtype=int)
                        if not len(year_indices):
                            for cohort in COHORTS:
                                metric_rows.append({
                                    "reference_year": reference_year, "patient_split_seed": split.effective_seed,
                                    "window": window, "test_year": test_year, "temporal_distance": test_year-reference_year,
                                    "cohort": cohort, "valid": False, "failure_reason": "no_evaluation_records",
                                    "record_count": 0, "support_fingerprint": stable_hash([]),
                                })
                            continue
                        local = np.asarray([lookup[int(index)] for index in year_indices], dtype=int)
                        cohorts = exposure_cohort_masks(population.patient_ids[year_indices], train_patients, threshold_patients)
                        bootstrap_tasks = []
                        for cohort, mask in cohorts.items():
                            selected = np.flatnonzero(mask)
                            base = {
                                "reference_year": reference_year,
                                "patient_split_seed": split.effective_seed,
                                "window": window,
                                "test_year": test_year,
                                "temporal_distance": test_year - reference_year,
                                "cohort": cohort,
                                "support_fingerprint": stable_hash(
                                    [] if len(selected) == 0 else [
                                        str(population.record_keys[index]) if population.record_keys is not None else int(index)
                                        for index in year_indices[selected]
                                    ]
                                ),
                            }
                            if len(selected) == 0:
                                metric_rows.append({**base, "valid": False, "failure_reason": "empty_cohort", "record_count": 0})
                                continue
                            cohort_local = local[selected]
                            truth = population.outcomes[year_indices[selected]]
                            values = metric_bundle(
                                truth,
                                probability[cohort_local],
                                threshold["threshold"],
                                hard_labels_at_half=hard[cohort_local],
                                minimum_deaths=config.minimum_evaluation_deaths,
                                minimum_survivors=config.minimum_evaluation_survivors,
                            )
                            row = {**base, **values}
                            metric_rows.append(row)
                            if window == "legacy_reference_only" and cohort == "all_comer":
                                expected = parent_performance_index.get(
                                    (reference_year, split.effective_seed, test_year)
                                )
                                if expected is None:
                                    raise RuntimeError("legacy_parent_performance_row_missing")
                                parity_row = compare_legacy_parent_row(
                                    row,
                                    expected,
                                    y_true=truth,
                                    death_probability=probability[cohort_local],
                                    hard_labels_at_half=hard[cohort_local],
                                )
                                legacy_parity_rows.append(parity_row)
                                if not parity_row["accepted"]:
                                    mismatches = {
                                        key: value
                                        for key, value in parity_row.items()
                                        if key.endswith("_matches") and value is False
                                    }
                                    differences = {
                                        key: value
                                        for key, value in parity_row.items()
                                        if key.endswith("_difference") and value is not None
                                    }
                                    raise RuntimeError(
                                        "legacy_parent_metric_parity_failed:"
                                        f"reference={reference_year}:split={split.effective_seed}:year={test_year}:"
                                        f"mismatches={json.dumps(mismatches, sort_keys=True)}:"
                                        f"differences={json.dumps(differences, sort_keys=True)}"
                                    )
                            if values["valid"] is True:
                                bootstrap_tasks.append((
                                    base,
                                    truth,
                                    probability[cohort_local],
                                    hard[cohort_local],
                                    population.patient_ids[year_indices[selected]],
                                ))
                        if bootstrap_tasks:
                            tasks_with_seeds = [
                                (
                                    base,
                                    truth,
                                    probabilities,
                                    hard_labels,
                                    patients,
                                    config.bootstrap_seed
                                    + reference_year
                                    + split.effective_seed
                                    + test_year
                                    + task_index,
                                )
                                for task_index, (base, truth, probabilities, hard_labels, patients) in enumerate(bootstrap_tasks)
                            ]
                            job_bootstrap_futures.append(
                                bootstrap_pool.submit(
                                    _bootstrap_interval_rows,
                                    tasks_with_seeds,
                                    threshold["threshold"],
                                    config.bootstrap_replicates,
                                    config.minimum_evaluation_deaths,
                                    config.minimum_evaluation_survivors,
                                )
                            )
                        cohort_labels = np.full(len(year_indices), "pipeline_unseen", dtype=object)
                        masks = exposure_cohort_masks(population.patient_ids[year_indices], train_patients, threshold_patients)
                        cohort_labels[masks["returning_model_seen"]] = "returning_model_seen"
                        cohort_labels[masks["threshold_only_seen"]] = "threshold_only_seen"
                        for position, index in enumerate(year_indices):
                            probability_rows.append(
                                {
                                    "reference_year": reference_year, "patient_split_seed": split.effective_seed,
                                    "window": window, "test_year": test_year, "temporal_distance": test_year-reference_year,
                                    "row_index": int(index),
                                    "record_key": None if population.record_keys is None else str(population.record_keys[index]),
                                    "patient_id": str(population.patient_ids[index]),
                                    "outcome": int(population.outcomes[index]),
                                    "death_probability": float(probability[local[position]]),
                                    "hard_label_at_0_5": int(hard[local[position]]),
                                    "frozen_threshold": float(threshold["threshold"]),
                                    "hard_label_at_frozen_threshold": int(probability[local[position]] >= threshold["threshold"]),
                                    "exposure_cohort": str(cohort_labels[position]),
                                    "model_domain_id": int(mapping[test_year]),
                                    "reported_temporal_distance": test_year-reference_year,
                                }
                            )
                    job_result = {
                        "thresholds": threshold_rows[starts["thresholds"]:],
                        "probabilities": probability_rows[starts["probabilities"]:],
                        "metrics": metric_rows[starts["metrics"]:],
                        "intervals": [],
                        "legacy_parity": legacy_parity_rows[starts["legacy_parity"]:],
                    }
                    atomic_write_json(
                        checkpoint_path,
                        {
                            "identity": identity,
                            "complete": False,
                            "stage": "bootstrap_queued",
                            "cache_hit": cache_hit,
                        },
                    )
                    pending_jobs.append(
                        {
                            "reference_year": reference_year,
                            "split": split.effective_seed,
                            "window": window,
                            "effective_years": effective_years,
                            "job_number": job_number,
                            "job_start": job_start,
                            "checkpoint_path": checkpoint_path,
                            "identity": identity,
                            "cache_hit": cache_hit,
                            "job_result": job_result,
                            "bootstrap_futures": job_bootstrap_futures,
                        }
                    )
                    progress.write(
                        "bootstrap_queued", reference_year=reference_year, split=split.effective_seed,
                        window=window, effective_years=effective_years, job=f"{job_number}/{jobs_total}",
                        tasks=len(job_bootstrap_futures), pending_jobs=len(pending_jobs),
                    )
                except Exception as error:
                    for future in job_bootstrap_futures:
                        future.cancel()
                    _release_cuda_memory()
                    failure = {
                        "reference_year": reference_year, "patient_split_seed": split.effective_seed,
                        "window": window, "reason": type(error).__name__, "message": str(error),
                    }
                    failed.append(failure)
                    atomic_write_json(checkpoint_path, {"complete": False, "stage": "failed", **failure})
                    progress.write("job_failed", **failure)
                    completed_jobs += 1
                    if fail_fast:
                        bootstrap_pool.shutdown(wait=False, cancel_futures=True)
                        raise

    while pending_jobs:
        complete_pending_job(pending_jobs.pop(0))
    bootstrap_pool.shutdown(wait=True)

    contrasts = _contrast_rows(metric_rows)
    trajectories = _trajectory_rows(metric_rows, config)
    diagnostic_rows = build_diagnostic_rows(metric_rows)
    trajectory_inference = _cluster_inference(
        metric_rows,
        value_name="death_f1_at_0_5",
        group_names=("window", "cohort", "temporal_distance"),
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed,
    )
    contrast_inference = _cluster_inference(
        [row for row in contrasts if row.get("matched_support")],
        value_name="difference",
        group_names=("window", "cohort", "temporal_distance", "metric"),
        replicates=config.bootstrap_replicates,
        seed=config.bootstrap_seed + 10_000,
    )
    artifacts = {
        "population_exclusions": _write_table(root, "population_exclusions", exclusion_rows),
        "per_window_exclusion_counts": _write_table(root, "per_window_exclusion_counts", exclusion_count_rows),
        "role_exposure_audit": _write_table(root, "role_exposure_audit", role_audit),
        "thresholds": _write_table(root, "thresholds", threshold_rows),
        "record_probabilities": _write_table(root, "record_probabilities", probability_rows),
        "yearly_metrics": _write_table(root, "yearly_metrics", metric_rows),
        "patient_bootstrap_intervals": _write_table(root, "patient_bootstrap_intervals", interval_rows),
        "paired_window_contrasts": _write_table(root, "paired_window_contrasts", contrasts),
        "split_averaged_trajectories": _write_table(root, "split_averaged_trajectories", trajectories),
        "trajectory_cluster_bootstrap": _write_table(root, "trajectory_cluster_bootstrap", trajectory_inference),
        "contrast_cluster_bootstrap": _write_table(root, "contrast_cluster_bootstrap", contrast_inference),
        "diagnostic_classifications": _write_table(root, "diagnostic_classifications", diagnostic_rows),
        "legacy_parent_metric_parity": _write_table(root, "legacy_parent_metric_parity", legacy_parity_rows),
    }
    manifest = {
        "schema_version": config.schema_version,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "complete": not failed,
        "experiment": "natural_prevalence_historical_windows",
        "headline_metric": "death_f1_at_0_5",
        "oracle_metrics_are_diagnostic_only": True,
        "legacy_parity_scope": (
            "all parent-retained all-comer aggregate metrics; exact parity required except a "
            "mathematically verified exact-0.5 hard-label boundary ambiguity; parent record-level "
            "hard labels are unavailable and reconciled rows remain explicitly non-parity"
        ),
        "config": config.to_dict(),
        "population_fingerprints": fingerprints,
        "post_death_exclusion_count": len(exclusion_rows),
        "expected_current_post_death_exclusion_count": 222,
        "post_death_count_matches_planned_audit": len(exclusion_rows) == 222,
        "source_fingerprints": source_fingerprints,
        "numerical_environment": numerical_environment,
        "dependency_sha256": dependency_sha256,
        "split_attempts": split_attempts,
        "failures": failed,
        "persistent_log": str(progress.path.relative_to(root)),
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    if manifest["complete"]:
        _canonical_pointer(config, manifest_path)
    progress.write("run_complete", complete=manifest["complete"], failures=len(failed), manifest=manifest_path)
    return {**manifest, "artifact_dir": str(root), "manifest_path": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="temporal_performance_windows.example.json")
    parser.add_argument("--parent-manifest")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, help="Requested inference batch; CUDA OOM halves it automatically")
    parser.add_argument("--cpu-workers", type=int, help="Spawned patient-bootstrap worker processes")
    parser.add_argument("--max-pending-bootstrap-jobs", type=int, help="Bound queued CPU jobs while GPU advances")
    parser.add_argument("--strict-fp32", action="store_true", help="Disable CUDA TF32 acceleration")
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned nonblocking CUDA input copies")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse valid checkpoints and fit caches")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--largest-window-pilot", action="store_true")
    parser.add_argument("--validate", metavar="MANIFEST", help="Validate and load an existing result")
    args = parser.parse_args(argv)
    if args.validate:
        loaded = load_window_experiment(args.validate)
        print(loaded["manifest_path"])
        return 0
    config = WindowExperimentConfig.from_json(args.config)
    if args.parent_manifest:
        config = replace(config, parent_manifest=str(Path(args.parent_manifest).resolve()))
    if args.device:
        config = replace(config, device=args.device)
    performance_updates = {}
    if args.batch_size is not None:
        performance_updates["batch_size"] = args.batch_size
    if args.cpu_workers is not None:
        performance_updates["cpu_workers"] = args.cpu_workers
    if args.max_pending_bootstrap_jobs is not None:
        performance_updates["max_pending_bootstrap_jobs"] = args.max_pending_bootstrap_jobs
    if args.strict_fp32:
        performance_updates["allow_tf32"] = False
    if args.no_pin_memory:
        performance_updates["pin_memory"] = False
    if performance_updates:
        config = replace(config, **performance_updates)
    if args.development:
        config = config.development_profile()
    if args.largest_window_pilot:
        config = config.largest_window_pilot()
    if args.force:
        config = replace(config, force=True)
    elif args.resume:
        config = replace(config, use_cache=True, force=False)
    result = run_window_experiment(config)
    print(result["manifest_path"])
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
