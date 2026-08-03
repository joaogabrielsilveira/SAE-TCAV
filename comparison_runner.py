"""End-to-end renal SAE comparison orchestration.

The public interface is deliberately small: load one configuration and call
``run_comparison``.  Expensive legacy stages stay behind an adapter seam so the
orchestration can be tested without training TabPFN or SAEs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import inspect
import json
import math
from pathlib import Path
import pickle
import platform
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from semantic_artifacts import array_fingerprint, stable_hash
from runtime_acceleration import StageTelemetry, accelerator_manifest
from comparison_cache import ComparisonCache


@dataclass(frozen=True)
class AcceleratorRunnerConfig:
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError(
                "accelerator.device must be 'auto', 'cpu', or 'cuda'")


@dataclass(frozen=True)
class TabPFNRunnerConfig:
    model_name: str = "tabpfn_dist_model_1"
    batch_size: int = 512
    run_walkforward: bool = True

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("tabpfn.model_name cannot be empty")
        if self.batch_size < 1:
            raise ValueError("tabpfn.batch_size must be positive")
        if not isinstance(self.run_walkforward, bool):
            raise ValueError("tabpfn.run_walkforward must be a boolean")


@dataclass(frozen=True)
class SAERunnerConfig:
    seeds: tuple[int, ...] = (42, 135)
    model_type: str = "ReLU"
    epochs: int = 1000
    alpha: float = 0.10
    scaling_factor: float = 1.50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    k: int = 16
    k_aux: int = 64
    encoding_batch_size: int = 4096

    def __post_init__(self) -> None:
        object.__setattr__(self, "seeds", tuple(int(seed)
                           for seed in self.seeds))
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "sae.seeds must contain at least two unique values")
        if self.model_type not in {"ReLU", "TopK"}:
            raise ValueError("sae.model_type must be 'ReLU' or 'TopK'")
        if self.epochs < 1:
            raise ValueError("sae.epochs must be positive")
        if self.alpha < 0 or self.scaling_factor <= 0:
            raise ValueError("sae alpha/scaling_factor values are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("sae optimizer values are invalid")
        if self.k < 1 or self.k_aux < 1 or self.encoding_batch_size < 1:
            raise ValueError(
                "sae k and encoding batch values must be positive")


@dataclass(frozen=True)
class MatchingRunnerConfig:
    scope: str = "all"
    criterion: str = "cos_sim"
    max_pairs_per_run_pair: int | None = None
    analysis_percentiles: tuple[int, ...] = (70, 80, 90)
    cosine_analysis_threshold: float = 0.60
    overlap_analysis_threshold: float = 0.70
    nearest_neighbor_top_k: int = 3
    alternative_score_deltas: tuple[float, ...] = (0.05, 0.10)

    def __post_init__(self) -> None:
        raw_percentiles = tuple(self.analysis_percentiles)
        percentiles = tuple(int(value) for value in raw_percentiles)
        deltas = tuple(float(value) for value in self.alternative_score_deltas)
        object.__setattr__(self, "analysis_percentiles", percentiles)
        object.__setattr__(self, "alternative_score_deltas", deltas)
        if self.scope not in {"all", "baseline"}:
            raise ValueError("matching.scope must be 'all' or 'baseline'")
        if self.criterion not in {"cos_sim", "overlap"}:
            raise ValueError(
                "matching.criterion must be 'cos_sim' or 'overlap'")
        if (
            self.max_pairs_per_run_pair is not None
            and self.max_pairs_per_run_pair < 1
        ):
            raise ValueError(
                "matching.max_pairs_per_run_pair must be positive")
        if (
            not percentiles
            or any(
                isinstance(value, bool) or float(value) != int(value)
                for value in raw_percentiles
            )
            or percentiles != tuple(sorted(set(percentiles)))
            or any(not 0 <= value <= 100 for value in percentiles)
        ):
            raise ValueError(
                "matching.analysis_percentiles must be ordered, unique, and lie in [0, 100]"
            )
        if not -1 <= self.cosine_analysis_threshold <= 1:
            raise ValueError(
                "matching.cosine_analysis_threshold must lie in [-1, 1]"
            )
        if not 0 <= self.overlap_analysis_threshold <= 1:
            raise ValueError(
                "matching.overlap_analysis_threshold must lie in [0, 1]"
            )
        if self.nearest_neighbor_top_k < 1:
            raise ValueError(
                "matching.nearest_neighbor_top_k must be positive")
        if (
            not deltas
            or deltas != tuple(sorted(set(deltas)))
            or any(not 0 < value <= 2 for value in deltas)
        ):
            raise ValueError(
                "matching.alternative_score_deltas must be ordered, unique, and lie in (0, 2]"
            )


@dataclass(frozen=True)
class FunctionalRunnerConfig:
    enabled: bool = True
    tree_max_depth: int = 15
    minimum_rule_samples: int = 50
    minimum_cav_samples: int = 50
    forced_rule_fallback: bool = True
    significance_runs: int = 15
    gradient_batch_size: int = 512

    def __post_init__(self) -> None:
        for name in ("enabled", "forced_rule_fallback"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"functional.{name} must be a boolean")
        if self.tree_max_depth < 1:
            raise ValueError("functional.tree_max_depth must be positive")
        if self.minimum_rule_samples < 1 or self.minimum_cav_samples < 2:
            raise ValueError("functional sample limits are invalid")
        if self.significance_runs < 0 or self.gradient_batch_size < 1:
            raise ValueError(
                "functional significance runs and gradient batch are invalid"
            )


@dataclass(frozen=True)
class ComparisonRunnerConfig:
    dataset_path: str = "data/renal/tidy_event_data.feather"
    semantic_config_path: str = "semantic_experiment.example.json"
    artifact_dir: str = "stats/comparison"
    use_cache: bool = True
    cache_dir: str | None = None
    cache_verification: Literal["manifest", "checksum"] = "checksum"
    show_progress: bool = True
    seed: int = 42
    accelerator: AcceleratorRunnerConfig = field(
        default_factory=AcceleratorRunnerConfig
    )
    tabpfn: TabPFNRunnerConfig = field(default_factory=TabPFNRunnerConfig)
    sae: SAERunnerConfig = field(default_factory=SAERunnerConfig)
    matching: MatchingRunnerConfig = field(
        default_factory=MatchingRunnerConfig)
    functional: FunctionalRunnerConfig = field(
        default_factory=FunctionalRunnerConfig)

    def __post_init__(self) -> None:
        if not self.dataset_path or not self.semantic_config_path or not self.artifact_dir:
            raise ValueError(
                "dataset, semantic config, and artifact paths are required")
        if not isinstance(self.use_cache, bool) or not isinstance(
            self.show_progress, bool
        ):
            raise ValueError("use_cache and show_progress must be booleans")
        if self.cache_verification not in {"manifest", "checksum"}:
            raise ValueError(
                "cache_verification must be 'manifest' or 'checksum'"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ComparisonRunnerConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("Comparison configuration must be a JSON object")
        known = {
            "dataset_path",
            "semantic_config_path",
            "artifact_dir",
            "use_cache",
            "cache_dir",
            "cache_verification",
            "show_progress",
            "seed",
            "accelerator",
            "tabpfn",
            "sae",
            "matching",
            "functional",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"Unknown comparison config fields: {sorted(unknown)}")
        return cls(
            dataset_path=str(raw.get("dataset_path", cls.dataset_path)),
            semantic_config_path=str(
                raw.get("semantic_config_path", cls.semantic_config_path)
            ),
            artifact_dir=str(raw.get("artifact_dir", cls.artifact_dir)),
            use_cache=raw.get("use_cache", True),
            cache_dir=(
                None if raw.get("cache_dir") is None else str(raw["cache_dir"])
            ),
            cache_verification=str(
                raw.get("cache_verification", "checksum")
            ),
            show_progress=raw.get("show_progress", True),
            seed=int(raw.get("seed", 42)),
            accelerator=_nested_config(
                AcceleratorRunnerConfig,
                raw.get("accelerator", {}),
                "accelerator",
            ),
            tabpfn=_nested_config(
                TabPFNRunnerConfig, raw.get("tabpfn", {}), "tabpfn"
            ),
            sae=_nested_config(SAERunnerConfig, raw.get("sae", {}), "sae"),
            matching=_nested_config(
                MatchingRunnerConfig, raw.get("matching", {}), "matching"
            ),
            functional=_nested_config(
                FunctionalRunnerConfig, raw.get("functional", {}), "functional"
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ComparisonRunnerConfig":
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            config = cls.from_dict(json.load(handle))
        base = config_path.parent
        return replace(
            config,
            dataset_path=str(_resolve_path(base, config.dataset_path)),
            semantic_config_path=str(
                _resolve_path(base, config.semantic_config_path)
            ),
            artifact_dir=str(_resolve_path(base, config.artifact_dir)),
            cache_dir=(
                None
                if config.cache_dir is None
                else str(_resolve_path(base, config.cache_dir))
            ),
        )


def _nested_config(config_type, raw: Any, name: str):
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    fields = set(config_type.__dataclass_fields__)
    unknown = set(raw) - fields
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    values = dict(raw)
    if config_type is SAERunnerConfig and "seeds" in values:
        values["seeds"] = tuple(values["seeds"])
    if config_type is MatchingRunnerConfig:
        for field_name in ("analysis_percentiles", "alternative_score_deltas"):
            if field_name in values:
                values[field_name] = tuple(values[field_name])
    return config_type(**values)


def _resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


@dataclass
class _PreparedData:
    train_rows: Any
    test_rows: Any
    feature_names: tuple[str, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    years_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    years_test: np.ndarray
    patient_ids: np.ndarray
    record_keys: np.ndarray


@dataclass
class _EmbeddingData:
    model: Any
    model_device: Any
    example_add_shape: Any
    year_to_domain: dict[int, int]
    train_raw: np.ndarray
    test_raw: np.ndarray
    train_scaled: np.ndarray
    test_scaled: np.ndarray
    scaler: Any
    fit_time_seconds: float
    walkforward_metrics: list[dict[str, Any]]
    model_identity: str | None = None
    model_provider: Callable[[], Mapping[str, Any]] | None = None

    def require_model(self, *, require_decoder: bool = False) -> Any:
        if self.model is None:
            if self.model_provider is None:
                raise RuntimeError("TabPFN model is unavailable")
            fit = self.model_provider()
            self.model = fit["model"]
            self.model_device = fit["model_add_x_device"]
            self.example_add_shape = fit["example_add_shape"]
            self.fit_time_seconds = float(fit["fit_time_sec"])
        if require_decoder:
            processed = getattr(self.model, "model_processed_", None)
            decoder_dict = getattr(processed, "decoder_dict", None)
            if (
                decoder_dict is None
                or "standard" not in decoder_dict
            ):
                raise ValueError(
                    "TabPFN model lacks standard decoder access required by TCAV"
                )
        return self.model


@dataclass
class _SAEData:
    runs: list[dict[str, Any]]
    activations: dict[int, np.ndarray]


class DefaultComparisonAdapter:
    """Production adapter for expensive legacy model and filesystem stages."""

    def __init__(self, cache: ComparisonCache | None = None) -> None:
        self.cache = cache

    def prepare(
        self,
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> _PreparedData:
        from database import (
            TabPFNPrepConfig,
            get_tabpfn_arrays,
            open_feather,
            prepare_database,
        )

        dataset_path = Path(config.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(
                f"Renal Feather file not found: {dataset_path}")
        preparation_config = TabPFNPrepConfig()
        preparation_config.rng_seed = config.seed

        def compute() -> _PreparedData:
            frame = open_feather(str(dataset_path))
            required = {"patient_id", "date", "event"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Renal Feather missing columns: {sorted(missing)}"
                )
            if not frame["event"].astype(str).eq("DEATH").any():
                raise ValueError(
                    "Renal Feather must contain an exact 'DEATH' event")
            prepared_rows = prepare_database(frame, cfg=preparation_config)
            arrays = get_tabpfn_arrays(prepared_rows)
            train_rows = prepared_rows["train_rows"].reset_index(drop=True)
            test_rows = prepared_rows["test_rows"].reset_index(drop=True)
            features = tuple(str(name)
                             for name in prepared_rows["top_k_events"])
            patient_ids = test_rows["patient_id"].astype(str).to_numpy()
            years_test = np.asarray(arrays["years_test"], dtype=int)
            record_keys = np.asarray(
                [
                    f"patient:{patient}|year:{year}"
                    for patient, year in zip(patient_ids, years_test)
                ],
                dtype=str,
            )
            return _PreparedData(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_names=features,
                X_train=np.asarray(arrays["X_train"], dtype=np.float32),
                y_train=np.asarray(arrays["y_train"], dtype=int),
                years_train=np.asarray(arrays["years_train"], dtype=int),
                X_test=np.asarray(arrays["X_test"], dtype=np.float32),
                y_test=np.asarray(arrays["y_test"], dtype=int),
                years_test=years_test,
                patient_ids=patient_ids,
                record_keys=record_keys,
            )

        if self.cache is None:
            prepared = compute()
        else:
            prep_values = {
                name: getattr(preparation_config, name)
                for name in (
                    "rng_seed",
                    "target_pos_lines",
                    "target_neg_lines",
                    "max_total_rows",
                    "final_top_k",
                    "m_candidates",
                    "forced_train_year_start",
                    "forced_test_year_start",
                )
            }
            result = self.cache.resolve(
                stage="prepared",
                item="renal-dataset",
                dependencies={
                    "dataset_sha256": _file_fingerprint(dataset_path),
                    "preparation": prep_values,
                    "versions": _package_versions(
                        "numpy", "pandas", "lightgbm", "pyarrow"
                    ),
                },
                source_fingerprint=_source_files_fingerprint("database.py"),
                load=lambda directory: _pickle_load(
                    directory / "prepared.pkl"),
                compute=compute,
                store=lambda directory, value: _pickle_dump(
                    directory / "prepared.pkl", value
                ),
                validate=_validate_prepared,
                fingerprint=_prepared_fingerprints,
            )
            prepared = result.value
        _validate_prepared(prepared)
        with (workspace / "prepared.pkl").open("wb") as handle:
            pickle.dump(prepared, handle)
        return prepared

    def embeddings(
        self,
        prepared: _PreparedData,
        splits: Mapping[str, np.ndarray],
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> _EmbeddingData:
        from tabpfn_model import (
            EmbeddingExtractConfig,
            TabPFNEvalConfig,
            batch_get_embeddings,
            ensure_test_feature_columns,
            extract_embeddings_robust,
            fit_dr_tabpfn,
            flatten_embeddings,
            infer_model_additional_x_info,
            make_dist_tensor,
            walkforward_evaluate_tabpfn,
        )

        evaluation = TabPFNEvalConfig()
        evaluation.rng_seed = config.seed
        evaluation.tabpfn_model_name = config.tabpfn.model_name
        evaluation.batch_size_predict = config.tabpfn.batch_size
        evaluation.device = config.accelerator.device
        evaluation.show_progress = config.show_progress
        domain_map = {
            int(year): index
            for index, year in enumerate(
                sorted(
                    set(prepared.years_train.tolist())
                    | set(prepared.years_test.tolist())
                )
            )
        }

        numerical_environment = _numerical_environment_fingerprint(
            config.accelerator.device, "numpy", "torch", "tabpfn"
        )
        model_dependencies = {
            "model_name": config.tabpfn.model_name,
            "checkpoint": _tabpfn_checkpoint_fingerprint(
                config.tabpfn.model_name
            ),
            "seed": config.seed,
            "X_train": array_fingerprint(prepared.X_train),
            "y_train": array_fingerprint(prepared.y_train),
            "years_train": array_fingerprint(prepared.years_train),
        }
        fit_source = _callable_source_fingerprint(
            fit_dr_tabpfn, infer_model_additional_x_info
        )
        extraction_source = _callable_source_fingerprint(
            extract_embeddings_robust,
            flatten_embeddings,
            batch_get_embeddings,
            make_dist_tensor,
        )
        walkforward_source = _callable_source_fingerprint(
            walkforward_evaluate_tabpfn,
            ensure_test_feature_columns,
            make_dist_tensor,
        )
        scaling_source = _callable_source_fingerprint(
            _scale_embeddings_from_semantic_fit
        )
        model_identity = stable_hash(
            model_dependencies,
            fit_source,
            numerical_environment,
        )
        fit_holder: dict[str, Mapping[str, Any]] = {}

        def fit_model() -> Mapping[str, Any]:
            if "fit" in fit_holder:
                return fit_holder["fit"]

            def compute_fit():
                return fit_dr_tabpfn(
                    prepared.X_train,
                    prepared.y_train,
                    prepared.years_train,
                    evaluation,
                )

            if self.cache is None:
                fit = compute_fit()
            else:
                fit_result = self.cache.resolve(
                    stage="tabpfn_fit",
                    item=config.tabpfn.model_name,
                    dependencies=model_dependencies,
                    source_fingerprint=fit_source,
                    environment_fingerprint=numerical_environment,
                    load=lambda directory: _pickle_load(
                        directory / "fitted_model.pkl"
                    ),
                    compute=compute_fit,
                    store=_store_tabpfn_fit,
                    validate=_validate_tabpfn_fit,
                    fingerprint=lambda value: {
                        "model_identity": stable_hash(
                            model_dependencies,
                            type(value["model"]).__qualname__,
                            value.get("model_source"),
                        )
                    },
                    ignore_store_errors=True,
                )
                fit = fit_result.value
            fit_holder["fit"] = fit
            return fit

        def compute_raw() -> tuple[np.ndarray, np.ndarray]:
            fit = fit_model()
            model = fit["model"]
            extraction = EmbeddingExtractConfig()
            extraction.batch_size = config.tabpfn.batch_size
            extraction.use_cache = False
            extraction.show_progress = config.show_progress
            extraction.progress_desc = "Extracting train embeddings"
            train_raw = flatten_embeddings(
                extract_embeddings_robust(
                    model=model,
                    X=prepared.X_train,
                    years=prepared.years_train,
                    year_to_domain_map=domain_map,
                    cfg=extraction,
                    device=fit["model_add_x_device"],
                    is_train=True,
                    ctx_idx=None,
                    example_add_shape=fit["example_add_shape"],
                )
            )
            extraction.progress_desc = "Extracting test embeddings"
            test_raw = flatten_embeddings(
                extract_embeddings_robust(
                    model=model,
                    X=prepared.X_test,
                    years=prepared.years_test,
                    year_to_domain_map=domain_map,
                    cfg=extraction,
                    device=fit["model_add_x_device"],
                    is_train=True,
                    ctx_idx=None,
                    example_add_shape=fit["example_add_shape"],
                )
            )
            return np.asarray(train_raw), np.asarray(test_raw)

        raw_dependencies = {
            "model": model_dependencies,
            "X_train": array_fingerprint(prepared.X_train),
            "years_train": array_fingerprint(prepared.years_train),
            "X_test": array_fingerprint(prepared.X_test),
            "years_test": array_fingerprint(prepared.years_test),
            "domain_map": domain_map,
        }
        if self.cache is None:
            train_raw, test_raw = compute_raw()
        else:
            raw_result = self.cache.resolve(
                stage="embeddings_raw",
                item="train-and-test",
                dependencies=raw_dependencies,
                source_fingerprint=extraction_source,
                environment_fingerprint=numerical_environment,
                load=_load_embedding_pair,
                compute=compute_raw,
                store=_store_embedding_pair,
                validate=lambda value: _validate_embedding_pair(
                    value, prepared),
                fingerprint=lambda value: {
                    "train_raw": array_fingerprint(value[0]),
                    "test_raw": array_fingerprint(value[1]),
                },
            )
            train_raw, test_raw = raw_result.value

        walkforward_metrics: list[dict[str, Any]] = []
        if config.tabpfn.run_walkforward:
            def compute_walkforward() -> list[dict[str, Any]]:
                fit = fit_model()
                walkforward = walkforward_evaluate_tabpfn(
                    drift_model=fit["model"],
                    test_rows=prepared.test_rows,
                    top_k_events=list(prepared.feature_names),
                    train_years=prepared.years_train,
                    model_add_x_device=fit["model_add_x_device"],
                    batch_size_predict=config.tabpfn.batch_size,
                    example_add_shape=fit["example_add_shape"],
                    use_cache=False,
                    show_progress=config.show_progress,
                )
                return [
                    _jsonable(row) for row in walkforward["results_per_year"]
                ]

            if self.cache is None:
                walkforward_metrics = compute_walkforward()
            else:
                walk_result = self.cache.resolve(
                    stage="walkforward",
                    item="test-years",
                    dependencies={
                        "model": model_dependencies,
                        "test_rows": array_fingerprint(
                            prepared.test_rows[
                                list(prepared.feature_names) +
                                ["DEATH", "year"]
                            ].to_numpy()
                        ),
                        "record_keys": array_fingerprint(
                            prepared.record_keys
                        ),
                        "feature_names": prepared.feature_names,
                        "domain_map": domain_map,
                    },
                    source_fingerprint=walkforward_source,
                    environment_fingerprint=numerical_environment,
                    load=lambda directory: _read_json(
                        directory / "metrics.json"
                    ),
                    compute=compute_walkforward,
                    store=lambda directory, value: _write_json(
                        directory / "metrics.json", value
                    ),
                    validate=lambda value: _validate_walkforward(value),
                    fingerprint=lambda value: {
                        "metrics": stable_hash(value)
                    },
                )
                walkforward_metrics = walk_result.value

        fit_indices = np.asarray(splits["idx_semantic_fit"], dtype=int)
        if (
            fit_indices.ndim != 1
            or len(fit_indices) == 0
            or np.any(fit_indices < 0)
            or np.any(fit_indices >= len(test_raw))
            or len(np.unique(fit_indices)) != len(fit_indices)
        ):
            raise ValueError(
                "idx_semantic_fit must contain unique test row indices")

        def compute_scaled():
            return _scale_embeddings_from_semantic_fit(
                np.asarray(train_raw), np.asarray(test_raw), fit_indices
            )

        if self.cache is None:
            train_scaled, test_scaled, scaler = compute_scaled()
        else:
            scaled_result = self.cache.resolve(
                stage="embeddings_scaled",
                item="standard-scaler",
                dependencies={
                    "train_raw": array_fingerprint(train_raw),
                    "test_raw": array_fingerprint(test_raw),
                    "fit_split": "idx_semantic_fit",
                    "fit_indices": array_fingerprint(fit_indices),
                    "sklearn": _package_versions("sklearn"),
                },
                source_fingerprint=scaling_source,
                load=_load_scaled_embeddings,
                compute=compute_scaled,
                store=_store_scaled_embeddings,
                validate=lambda value: _validate_scaled_embeddings(
                    value, prepared
                ),
                fingerprint=lambda value: {
                    "train_scaled": array_fingerprint(value[0]),
                    "test_scaled": array_fingerprint(value[1]),
                    "scaler_mean": array_fingerprint(
                        np.asarray(value[2].mean_)
                    ),
                    "scaler_scale": array_fingerprint(
                        np.asarray(value[2].scale_)
                    ),
                },
                stage_schema_version=2,
            )
            train_scaled, test_scaled, scaler = scaled_result.value

        if len(train_scaled) != len(prepared.X_train) or len(test_scaled) != len(
            prepared.X_test
        ):
            raise ValueError(
                "TabPFN embedding rows do not align with prepared data")
        if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
            raise ValueError("TabPFN embeddings contain non-finite values")
        np.savez_compressed(
            workspace / "embeddings.npz",
            train_raw=train_raw,
            test_raw=test_raw,
            train_scaled=train_scaled,
            test_scaled=test_scaled,
        )
        scaler_provenance = {
            "fit_split": "idx_semantic_fit",
            "fit_row_count": int(len(fit_indices)),
            "fit_indices_fingerprint": array_fingerprint(fit_indices),
            "mean_fingerprint": array_fingerprint(np.asarray(scaler.mean_)),
            "scale_fingerprint": array_fingerprint(np.asarray(scaler.scale_)),
        }
        _pickle_dump(workspace / "embedding_scaler.pkl", scaler)
        _write_json(
            workspace / "embedding_scaler_provenance.json", scaler_provenance
        )
        _write_json(workspace / "tabpfn_metrics.json", walkforward_metrics)
        fit = fit_holder.get("fit")
        return _EmbeddingData(
            model=fit["model"] if fit is not None else None,
            model_device=(
                fit["model_add_x_device"] if fit is not None else None
            ),
            example_add_shape=(
                fit["example_add_shape"] if fit is not None else None
            ),
            year_to_domain=domain_map,
            train_raw=np.asarray(train_raw),
            test_raw=np.asarray(test_raw),
            train_scaled=np.asarray(train_scaled),
            test_scaled=np.asarray(test_scaled),
            scaler=scaler,
            fit_time_seconds=(
                float(fit["fit_time_sec"]) if fit is not None else 0.0
            ),
            walkforward_metrics=walkforward_metrics,
            model_identity=model_identity,
            model_provider=fit_model,
        )

    def train_saes(
        self,
        prepared: _PreparedData,
        embeddings: _EmbeddingData,
        splits: Mapping[str, np.ndarray],
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> _SAEData:
        import torch

        from sae import SAE, train_sae_model
        from sae_compare import encode_sae, high_activation_profiles, train_all_saes

        fit_indices = splits["idx_semantic_fit"]
        matching_indices = splits["idx_semantic_select"]
        sae = config.sae
        fit_embeddings = embeddings.test_scaled[fit_indices]
        numerical_environment = _numerical_environment_fingerprint(
            config.accelerator.device, "numpy", "torch"
        )
        shared_hyperparameters = {
            "model_type": sae.model_type,
            "epochs": sae.epochs,
            "alpha": sae.alpha,
            "scaling_factor": sae.scaling_factor,
            "learning_rate": sae.learning_rate,
            "weight_decay": sae.weight_decay,
            "k": sae.k,
            "k_aux": sae.k_aux,
            "use_decoder_bias": True,
        }
        sae_training_source = _callable_source_fingerprint(
            train_all_saes, train_sae_model, SAE
        )
        sae_activation_source = _callable_source_fingerprint(
            encode_sae, SAE.encode
        )
        runs: list[dict[str, Any]] = []
        activations: dict[int, np.ndarray] = {}
        from progress_utils import progress_iter

        for run_index, seed in progress_iter(
            list(enumerate(sae.seeds)),
            enabled=config.show_progress,
            desc="Resolving SAE runs",
            total=len(sae.seeds),
            unit="run",
        ):
            def compute_run(seed_value=seed):
                return train_all_saes(
                    num_models=1,
                    embs=fit_embeddings,
                    alpha=sae.alpha,
                    scaling_factor=sae.scaling_factor,
                    model_type=sae.model_type,
                    k=sae.k,
                    k_aux=sae.k_aux,
                    universal_embs=None,
                    seeds=(seed_value,),
                    epochs=sae.epochs,
                    learning_rate=sae.learning_rate,
                    weight_decay=sae.weight_decay,
                    device=config.accelerator.device,
                    encoding_batch_size=sae.encoding_batch_size,
                    show_progress=config.show_progress,
                )[0]

            model_dependencies = {
                "fit_embeddings": array_fingerprint(fit_embeddings),
                "seed": int(seed),
                "hyperparameters": shared_hyperparameters,
            }
            if self.cache is None:
                trained = compute_run()
                model_key = stable_hash(model_dependencies)
            else:
                model_result = self.cache.resolve(
                    stage="sae_model",
                    item=f"seed:{seed}",
                    dependencies=model_dependencies,
                    source_fingerprint=sae_training_source,
                    environment_fingerprint=numerical_environment,
                    load=lambda directory: _load_sae_run(
                        directory,
                        data_dimension=fit_embeddings.shape[1],
                        sae_config=sae,
                    ),
                    compute=compute_run,
                    store=_store_sae_run,
                    validate=lambda value, expected_seed=seed: _validate_sae_run(
                        value, expected_seed, fit_embeddings.shape[1]
                    ),
                    fingerprint=lambda value: {
                        "model_state": _model_state_fingerprint(value["model"])
                    },
                )
                trained = model_result.value
                model_key = model_result.output_fingerprints["model_state"]

            def compute_activations(run=trained):
                return encode_sae(
                    run,
                    embeddings.test_scaled,
                    device=config.accelerator.device,
                    batch_size=sae.encoding_batch_size,
                )

            activation_dependencies = {
                "model_state": model_key,
                "test_embeddings": array_fingerprint(
                    embeddings.test_scaled
                ),
            }
            if self.cache is None:
                activation_matrix = compute_activations()
            else:
                activation_result = self.cache.resolve(
                    stage="sae_activations",
                    item=f"seed:{seed}",
                    dependencies=activation_dependencies,
                    source_fingerprint=sae_activation_source,
                    environment_fingerprint=numerical_environment,
                    load=lambda directory: _load_single_array(
                        directory / "activations.npy"
                    ),
                    compute=compute_activations,
                    store=lambda directory, value: np.save(
                        directory / "activations.npy", value
                    ),
                    validate=lambda value: _validate_activation_matrix(
                        value, len(prepared.X_test)
                    ),
                    fingerprint=lambda value: {
                        "activations": array_fingerprint(value)
                    },
                )
                activation_matrix = activation_result.value

            run = dict(trained)
            run["idx"] = run_index
            run["run_id"] = f"sae_{run_index}"
            run["seed"] = int(seed)
            run["encoded_embs"] = np.asarray(activation_matrix[fit_indices])
            profiles = high_activation_profiles(
                np.asarray(activation_matrix[matching_indices]),
                config.matching.analysis_percentiles,
            )
            run["high_activation_profiles"] = profiles
            run["high_activation_matrix"] = profiles[
                max(config.matching.analysis_percentiles)
            ]["masks"]
            runs.append(run)
            activations[run_index] = np.asarray(activation_matrix)

        _validate_activations(activations, runs, len(prepared.X_test))
        torch.save(runs, workspace / "sae_runs.pt")
        state_dir = workspace / "sae"
        state_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            torch.save(run["model"].state_dict(),
                       state_dir / f"run_{run['idx']}.pt")
        np.savez_compressed(
            workspace / "activations.npz",
            **{f"run_{run_id}": matrix for run_id, matrix in activations.items()},
        )
        np.savez_compressed(
            workspace / "high_activation_profiles.npz",
            **{
                f"run_{int(run['idx'])}_p{percentile}_{name}": np.asarray(value)
                for run in runs
                for percentile, profile in run["high_activation_profiles"].items()
                for name, value in profile.items()
            },
        )
        _write_json(
            workspace / "sae_manifest.json",
            [
                {
                    "run_id": int(run["idx"]),
                    "seed": int(run["seed"]),
                    "model_type": run["model_type"],
                    "mse": _as_float(run["mse"]),
                    "sparsity_level": float(run["sparsity_level"]),
                    "dead_neurons": int(run["dead_neurons"]),
                    "n_factors": int(activations[int(run["idx"])].shape[1]),
                }
                for run in runs
            ],
        )
        return _SAEData(runs=runs, activations=activations)

    def match(
        self,
        sae_data: _SAEData,
        config: ComparisonRunnerConfig,
        workspace: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from robustness_matching import analyze_run_pair, validate_analysis

        all_matches: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        cosine_rows: list[dict[str, Any]] = []
        overlap_rows: list[dict[str, Any]] = []
        nearest_rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        matrix_entries: list[dict[str, Any]] = []
        run_pairs = _run_pairs(len(sae_data.runs), config.matching.scope)
        matching_source = _source_files_fingerprint("robustness_matching.py")
        matrices_dir = workspace / "matching" / "matrices"
        matrices_dir.mkdir(parents=True, exist_ok=True)
        from progress_utils import progress_iter

        for left_index, right_index in progress_iter(
            run_pairs,
            enabled=config.show_progress,
            desc="Geometric run pairs",
            total=len(run_pairs),
            unit="pair",
        ):
            left_run = sae_data.runs[left_index]
            right_run = sae_data.runs[right_index]
            left_profiles = {
                int(percentile): np.asarray(profile["masks"], dtype=bool)
                for percentile, profile in left_run[
                    "high_activation_profiles"
                ].items()
            }
            right_profiles = {
                int(percentile): np.asarray(profile["masks"], dtype=bool)
                for percentile, profile in right_run[
                    "high_activation_profiles"
                ].items()
            }

            def compute_analysis():
                return analyze_run_pair(
                    np.asarray(left_run["decoder_directions"]),
                    np.asarray(right_run["decoder_directions"]),
                    left_profiles,
                    right_profiles,
                    config.matching.nearest_neighbor_top_k,
                )

            if self.cache is None:
                analysis = compute_analysis()
            else:
                pair_result = self.cache.resolve(
                    stage="matching_pair",
                    item=(
                        f"seed:{left_run['seed']}-seed:{right_run['seed']}"
                    ),
                    dependencies={
                        "left_directions": array_fingerprint(
                            np.asarray(left_run["decoder_directions"])
                        ),
                        "right_directions": array_fingerprint(
                            np.asarray(right_run["decoder_directions"])
                        ),
                        "left_profiles": {
                            str(percentile): array_fingerprint(matrix)
                            for percentile, matrix in left_profiles.items()
                        },
                        "right_profiles": {
                            str(percentile): array_fingerprint(matrix)
                            for percentile, matrix in right_profiles.items()
                        },
                        "top_k": config.matching.nearest_neighbor_top_k,
                    },
                    source_fingerprint=matching_source,
                    environment_fingerprint=stable_hash(
                        _package_versions("numpy", "scipy")
                    ),
                    load=_load_pair_matching_analysis,
                    compute=compute_analysis,
                    store=_store_pair_matching_analysis,
                    validate=validate_analysis,
                    fingerprint=lambda value: {
                        "cosine": array_fingerprint(value.cosine),
                        **{
                            f"overlap_p{percentile}": array_fingerprint(matrix)
                            for percentile, matrix in value.overlaps.items()
                        },
                        "raw_assignments_and_rankings": stable_hash(
                            _pair_matching_metadata(value)
                        ),
                    },
                    stage_schema_version=2,
                )
                analysis = pair_result.value

            matrix_path = matrices_dir / (
                f"run_{left_index}__run_{right_index}.npz"
            )
            np.savez_compressed(
                matrix_path,
                cosine=analysis.cosine,
                **{
                    f"overlap_p{percentile}": matrix
                    for percentile, matrix in analysis.overlaps.items()
                },
            )
            matrix_entry = {
                "run_i": left_index,
                "run_j": right_index,
                "shape": list(analysis.cosine.shape),
                "filename": str(matrix_path.relative_to(workspace)),
                "row_count": int(analysis.cosine.shape[0]),
                "column_count": int(analysis.cosine.shape[1]),
                "fingerprints": {
                    "cosine": array_fingerprint(analysis.cosine),
                    **{
                        f"overlap_p{percentile}": array_fingerprint(matrix)
                        for percentile, matrix in analysis.overlaps.items()
                    },
                },
            }
            matrix_entries.append(matrix_entry)
            pair_artifacts = _matching_artifact_rows(
                analysis, left_index, right_index, config.matching
            )
            cosine_rows.extend(pair_artifacts["cosine_hungarian"])
            overlap_rows.extend(pair_artifacts["overlap_hungarian"])
            nearest_rows.extend(pair_artifacts["nearest_neighbors"])
            diagnostic_rows.extend(pair_artifacts["diagnostics"])

            pair_rows, filtered, selected_percentile = _select_matching_rows(
                analysis, left_index, right_index, config.matching
            )
            matrix_entry["selector"] = {
                "criterion": config.matching.criterion,
                "threshold": (
                    config.matching.cosine_analysis_threshold
                    if config.matching.criterion == "cos_sim"
                    else config.matching.overlap_analysis_threshold
                ),
                "selected_overlap_percentile": selected_percentile,
                "threshold_qualified_factor_count": len(filtered),
            }
            all_matches.extend(pair_rows)
            if config.matching.max_pairs_per_run_pair is not None:
                filtered = sorted(
                    filtered,
                    key=lambda row: (
                        -float(row[config.matching.criterion]),
                        int(row["original_concept"]),
                    ),
                )[: config.matching.max_pairs_per_run_pair]
                filtered.sort(key=lambda row: int(row["original_concept"]))
            matrix_entry["selector"]["downstream_factor_count"] = len(filtered)
            selected.extend(filtered)

        matching_manifest = {
            "schema_version": 1,
            "scope": config.matching.scope,
            "n_runs": len(sae_data.runs),
            "expected_run_pair_count": len(run_pairs),
            "analysis_percentiles": list(config.matching.analysis_percentiles),
            "cosine_threshold": config.matching.cosine_analysis_threshold,
            "overlap_threshold": config.matching.overlap_analysis_threshold,
            "nearest_neighbor_top_k": config.matching.nearest_neighbor_top_k,
            "alternative_score_deltas": list(
                config.matching.alternative_score_deltas
            ),
            "selector_thresholds": {
                "cos_sim": config.matching.cosine_analysis_threshold,
                "overlap": config.matching.overlap_analysis_threshold,
            },
            "overlap_percentile_tie_break": "highest_percentile",
            "profile_artifact": "high_activation_profiles.npz",
            "profile_artifact_fingerprint": _file_fingerprint(
                workspace / "high_activation_profiles.npz"
            ),
            "run_pairs": matrix_entries,
        }
        _write_json(workspace / "matching" /
                    "manifest.json", matching_manifest)
        for stem, rows in (
            ("cosine_hungarian_matches", cosine_rows),
            ("overlap_hungarian_matches", overlap_rows),
            ("nearest_neighbor_candidates", nearest_rows),
            ("matching_diagnostics", diagnostic_rows),
        ):
            _write_json(workspace / "matching" / f"{stem}.json", rows)
            _write_csv(workspace / "matching" / f"{stem}.csv", rows)
        _write_json(workspace / "matches_all.json", all_matches)
        _write_json(workspace / "matched_factors.json", selected)
        _write_csv(workspace / "matches_all.csv", all_matches)
        _write_csv(workspace / "matched_factors.csv", selected)
        return all_matches, selected

    def functional(
        self,
        prepared: _PreparedData,
        embeddings: _EmbeddingData,
        sae_data: _SAEData,
        splits: Mapping[str, np.ndarray],
        matches: Sequence[Mapping[str, Any]],
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> tuple[dict[str, dict[int, dict[str, Any]]], list[dict[str, Any]]]:
        if not config.functional.enabled or not matches:
            return {}, []

        import pandas as pd

        from decision_tree import (
            get_binary_targets,
            get_rules_forced,
            get_rules_from_text,
            mask_from_rule,
            train_binary_trees,
        )
        from tcav import (
            extract_rule_conditions,
            get_rule_mask,
            get_model_gradients,
            get_tcav_scores,
            robust_tcav_significance_test,
            train_cavs_from_rules,
        )
        high_precision_source = _callable_source_fingerprint(
            train_binary_trees,
            get_binary_targets,
            get_rules_from_text,
            mask_from_rule,
        )
        forced_rule_source = _callable_source_fingerprint(
            get_rules_forced,
            get_binary_targets,
            get_rules_from_text,
            mask_from_rule,
        )
        cav_source = _callable_source_fingerprint(
            train_cavs_from_rules,
            extract_rule_conditions,
            get_rule_mask,
        )
        gradient_source = _callable_source_fingerprint(get_model_gradients)
        tcav_factor_source = _callable_source_fingerprint(
            get_tcav_scores, robust_tcav_significance_test
        )

        fit_indices = splits["idx_semantic_fit"]
        select_indices = splits["idx_semantic_select"]
        tcav_indices = splits["idx_tcav_eval"]
        factors_by_run = _matched_factors_by_run(matches)
        rule_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        cavs_by_run: dict[int, dict[int, dict[str, Any]]] = {}

        from progress_utils import progress_iter

        run_items = sorted(factors_by_run.items())
        for run_id, factor_ids in progress_iter(
            run_items,
            enabled=config.show_progress,
            desc="High-precision rules and CAVs",
            total=len(run_items),
            unit="run",
        ):
            activations = sae_data.activations[run_id]
            rules_by_percentile: dict[int, list[dict[str, Any]]] = {
                percentile: [] for percentile in (90, 80, 70, 60, 50)
            }
            for factor_id in sorted(factor_ids):
                def compute_factor_rules(current_factor=factor_id):
                    return train_binary_trees(
                        activations[fit_indices],
                        prepared.X_test[fit_indices],
                        list(prepared.feature_names),
                        model_type=config.sae.model_type,
                        max_depth=config.functional.tree_max_depth,
                        factor_ids=[current_factor],
                        min_positive_samples=(
                            config.functional.minimum_rule_samples
                        ),
                        show_progress=config.show_progress,
                        progress_desc=(
                            f"Tree rule run {run_id} factor {current_factor}"
                        ),
                    )

                if self.cache is None:
                    factor_rules = compute_factor_rules()
                else:
                    rule_result = self.cache.resolve(
                        stage="high_precision_rule",
                        item=f"run:{run_id}-factor:{factor_id}",
                        dependencies={
                            "X_fit": array_fingerprint(
                                prepared.X_test[fit_indices]
                            ),
                            "activation_fit": array_fingerprint(
                                activations[fit_indices, factor_id]
                            ),
                            "feature_names": prepared.feature_names,
                            "model_type": config.sae.model_type,
                            "max_depth": config.functional.tree_max_depth,
                            "minimum_rule_samples": (
                                config.functional.minimum_rule_samples
                            ),
                        },
                        source_fingerprint=high_precision_source,
                        environment_fingerprint=stable_hash(
                            _package_versions("numpy", "sklearn")
                        ),
                        load=lambda directory: _normalize_percentile_rules(
                            _read_json(directory / "rules.json")
                        ),
                        compute=compute_factor_rules,
                        store=lambda directory, value: _write_json(
                            directory / "rules.json", value
                        ),
                        validate=_validate_percentile_rules,
                        fingerprint=lambda value: {
                            "rules": stable_hash(value)
                        },
                    )
                    factor_rules = rule_result.value
                for percentile, rows in factor_rules.items():
                    rules_by_percentile[int(percentile)].extend(
                        dict(row) for row in rows
                    )
            best_percentile = max(
                rules_by_percentile,
                key=lambda percentile: len(rules_by_percentile[percentile]),
            )
            primary = [
                dict(rule)
                for rule in rules_by_percentile[best_percentile]
                if int(rule["Factor"]) in factor_ids
            ]
            primary_factors = {int(rule["Factor"]) for rule in primary}
            combined = [
                {**rule, "Provenance": "high_precision"}
                for rule in primary
            ]
            missing = sorted(factor_ids - primary_factors)
            if config.functional.forced_rule_fallback and missing:
                for factor_id in missing:
                    graph_dir = (
                        workspace / "decision_tree_graphs" / f"run_{run_id}"
                    )

                    def compute_forced(current_factor=factor_id):
                        empty = pd.DataFrame(
                            columns=[
                                "Factor",
                                "Rule",
                                "Class",
                                "Precision",
                                "Recall",
                                "Patients",
                                "Patients_concept",
                            ]
                        )
                        rows = get_rules_forced(
                            train_activations=activations[fit_indices],
                            X=prepared.X_test[fit_indices],
                            surviving_concepts=np.asarray(
                                [current_factor], dtype=int
                            ),
                            tree_rules_df=empty,
                            perc=best_percentile,
                            feature_names=list(prepared.feature_names),
                            model_type=config.sae.model_type,
                            graph_output_dir=graph_dir,
                        )
                        dot_path = graph_dir / f"{current_factor}.dot"
                        return {
                            "rules": [_jsonable(row) for row in rows],
                            "dot": (
                                dot_path.read_text(encoding="utf-8")
                                if dot_path.is_file()
                                else None
                            ),
                        }

                    if self.cache is None:
                        forced_value = compute_forced()
                    else:
                        forced_result = self.cache.resolve(
                            stage="forced_rule",
                            item=f"run:{run_id}-factor:{factor_id}",
                            dependencies={
                                "X_fit": array_fingerprint(
                                    prepared.X_test[fit_indices]
                                ),
                                "activation_fit": array_fingerprint(
                                    activations[fit_indices, factor_id]
                                ),
                                "feature_names": prepared.feature_names,
                                "model_type": config.sae.model_type,
                                "percentile": best_percentile,
                            },
                            source_fingerprint=forced_rule_source,
                            environment_fingerprint=stable_hash(
                                _package_versions("numpy", "sklearn")
                            ),
                            load=lambda directory: _read_json(
                                directory / "forced.json"
                            ),
                            compute=compute_forced,
                            store=lambda directory, value: _write_json(
                                directory / "forced.json", value
                            ),
                            validate=_validate_forced_rule_value,
                            fingerprint=lambda value: {
                                "forced": stable_hash(value)
                            },
                        )
                        forced_value = forced_result.value
                    if forced_value.get("dot") is not None:
                        graph_dir.mkdir(parents=True, exist_ok=True)
                        (graph_dir / f"{factor_id}.dot").write_text(
                            forced_value["dot"], encoding="utf-8"
                        )
                    combined.extend(
                        {**dict(rule), "Provenance": "forced_fallback"}
                        for rule in forced_value["rules"]
                    )

            for rule in combined:
                rule_rows.append(
                    {
                        **_jsonable(rule),
                        "run_id": run_id,
                        "percentile": int(best_percentile),
                    }
                )
            cav_rules = [
                {key: value for key, value in rule.items() if key != "Provenance"}
                for rule in combined
            ]

            def compute_cavs():
                return train_cavs_from_rules(
                    rules_per_percentile=cav_rules,
                    X_cav_train_df=prepared.test_rows.iloc[select_indices][
                        list(prepared.feature_names)
                    ].reset_index(drop=True),
                    cav_train_emb=embeddings.test_raw[select_indices],
                    cav_train_emb_encoded=activations[select_indices],
                    y_cav_train=prepared.y_test[select_indices],
                    feature_cols=list(prepared.feature_names),
                    emb_scaler=embeddings.scaler,
                    # Preserve main.py's legacy CAV negative-cohort mapping.
                    high_quantile=1.0
                    - (float(best_percentile) / 100.0),
                    min_pos_samples=config.functional.minimum_cav_samples,
                    random_state=config.seed + run_id,
                )

            if self.cache is None:
                cavs_by_run[run_id] = compute_cavs()
            else:
                cav_result = self.cache.resolve(
                    stage="cav",
                    item=f"run:{run_id}",
                    dependencies={
                        "rules": stable_hash(cav_rules),
                        "X_select": array_fingerprint(
                            prepared.X_test[select_indices]
                        ),
                        "raw_embeddings_select": array_fingerprint(
                            embeddings.test_raw[select_indices]
                        ),
                        "activations_select": array_fingerprint(
                            activations[select_indices]
                        ),
                        "outcome_select": array_fingerprint(
                            prepared.y_test[select_indices]
                        ),
                        "scaler_mean": array_fingerprint(
                            np.asarray(embeddings.scaler.mean_)
                        ),
                        "scaler_scale": array_fingerprint(
                            np.asarray(embeddings.scaler.scale_)
                        ),
                        "percentile": best_percentile,
                        "minimum_cav_samples": (
                            config.functional.minimum_cav_samples
                        ),
                        "seed": config.seed + run_id,
                    },
                    source_fingerprint=cav_source,
                    environment_fingerprint=stable_hash(
                        _package_versions("numpy", "sklearn")
                    ),
                    load=lambda directory: _pickle_load(
                        directory / "cavs.pkl"
                    ),
                    compute=compute_cavs,
                    store=lambda directory, value: _pickle_dump(
                        directory / "cavs.pkl", value
                    ),
                    validate=lambda value: _validate_cavs(
                        value, embeddings.test_raw.shape[1]
                    ),
                    fingerprint=_cav_fingerprints,
                )
                cavs_by_run[run_id] = cav_result.value
            rules_by_factor = {int(rule["Factor"]): rule for rule in combined}
            for factor_id in sorted(factor_ids):
                rule = rules_by_factor.get(factor_id)
                diagnostics.append(
                    {
                        "run_id": run_id,
                        "factor_id": factor_id,
                        "rule_status": (
                            "available" if rule is not None else "no_valid_rule"
                        ),
                        "rule_provenance": (
                            rule.get(
                                "Provenance") if rule is not None else None
                        ),
                        "cav_status": (
                            "available"
                            if factor_id in cavs_by_run[run_id]
                            else "insufficient_cav_cohorts"
                        ),
                    }
                )

        if not any(cavs_by_run.values()):
            _write_jsonl(workspace / "high_precision_rules.jsonl", rule_rows)
            _write_json(workspace / "functional.json", diagnostics)
            return {}, diagnostics

        dist_eval = np.asarray(
            [embeddings.year_to_domain[int(year)]
             for year in prepared.years_test[tcav_indices]],
            dtype=np.int64,
        )

        def compute_gradients():
            return get_model_gradients(
                model=embeddings.require_model(require_decoder=True),
                X=prepared.X_test[tcav_indices],
                dist_vec=dist_eval,
                cache_file=None,
                batch_size=config.functional.gradient_batch_size,
                device=config.accelerator.device,
                show_progress=config.show_progress,
                use_cache=False,
            )

        if self.cache is None:
            gradients = compute_gradients()
        else:
            gradient_result = self.cache.resolve(
                stage="tcav_gradients",
                item="tcav-evaluation-cohort",
                dependencies={
                    "model_identity": embeddings.model_identity,
                    "X": array_fingerprint(
                        prepared.X_test[tcav_indices]
                    ),
                    "record_keys": array_fingerprint(
                        prepared.record_keys[tcav_indices]
                    ),
                    "domain_vector": array_fingerprint(dist_eval),
                },
                source_fingerprint=gradient_source,
                environment_fingerprint=_numerical_environment_fingerprint(
                    config.accelerator.device, "numpy", "torch"
                ),
                load=lambda directory: _load_single_array(
                    directory / "gradients.npy"
                ),
                compute=compute_gradients,
                store=lambda directory, value: np.save(
                    directory / "gradients.npy", value
                ),
                validate=lambda value: _validate_gradients(
                    value,
                    len(tcav_indices),
                    embeddings.test_raw.shape[1],
                ),
                fingerprint=lambda value: {
                    "gradients": array_fingerprint(value)
                },
            )
            gradients = gradient_result.value
        with (workspace / "tcav_gradients.pkl").open("wb") as handle:
            pickle.dump(gradients, handle)
        functional: dict[str, dict[int, dict[str, Any]]] = {}
        cav_arrays: dict[str, np.ndarray] = {}
        diagnostic_lookup = {
            (int(row["run_id"]), int(row["factor_id"])): row
            for row in diagnostics
        }
        cav_items = sorted(cavs_by_run.items())
        for run_id, cavs in progress_iter(
            cav_items,
            enabled=config.show_progress,
            desc="TCAV scoring by run",
            total=len(cav_items),
            unit="run",
        ):
            factor_items = sorted(cavs.items())
            for factor_id, cav in progress_iter(
                factor_items,
                enabled=config.show_progress,
                desc=f"TCAV significance run {run_id}",
                total=len(factor_items),
                unit="factor",
                leave=False,
            ):
                def compute_tcav_factor():
                    score_value = float(
                        get_tcav_scores([cav], gradients)[factor_id]
                    )
                    result: dict[str, Any] = {"tcav_score": score_value}
                    if config.functional.significance_runs > 0:
                        try:
                            significance = robust_tcav_significance_test(
                                concept_idx=factor_id,
                                embs=embeddings.test_raw[select_indices],
                                idx_pos=np.asarray(
                                    cav["positive_idx"], dtype=int
                                ),
                                idx_neg=np.asarray(
                                    cav["negative_idx"], dtype=int
                                ),
                                model_grads=gradients,
                                scaler_emb=embeddings.scaler,
                                n_runs=config.functional.significance_runs,
                                sample_fraction=1.0,
                                rng_seed=(
                                    config.seed
                                    + run_id * 10_000
                                    + factor_id
                                ),
                            )
                            result.update(
                                {
                                    "tcav_p_value": _finite_or_none(
                                        significance.get("p_value")
                                    ),
                                    "tcav_t_stat": _finite_or_none(
                                        significance.get("t_stat")
                                    ),
                                    "tcav_effect_size": _finite_or_none(
                                        significance.get("cohens_d")
                                    ),
                                }
                            )
                        except (ValueError, FloatingPointError) as error:
                            result["significance_error"] = str(error)
                    return result

                if self.cache is None:
                    tcav_values = compute_tcav_factor()
                else:
                    tcav_result = self.cache.resolve(
                        stage="tcav_factor",
                        item=f"run:{run_id}-factor:{factor_id}",
                        dependencies={
                            "cav": array_fingerprint(
                                np.asarray(cav["CAV"])
                            ),
                            "positive_idx": array_fingerprint(
                                np.asarray(cav["positive_idx"], dtype=int)
                            ),
                            "negative_idx": array_fingerprint(
                                np.asarray(cav["negative_idx"], dtype=int)
                            ),
                            "gradients": array_fingerprint(gradients),
                            "select_embeddings": array_fingerprint(
                                embeddings.test_raw[select_indices]
                            ),
                            "scaler_mean": array_fingerprint(
                                np.asarray(embeddings.scaler.mean_)
                            ),
                            "scaler_scale": array_fingerprint(
                                np.asarray(embeddings.scaler.scale_)
                            ),
                            "significance_runs": (
                                config.functional.significance_runs
                            ),
                            "seed": (
                                config.seed + run_id * 10_000 + factor_id
                            ),
                        },
                        source_fingerprint=tcav_factor_source,
                        environment_fingerprint=stable_hash(
                            _package_versions(
                                "numpy", "scipy", "sklearn"
                            )
                        ),
                        load=lambda directory: _read_json(
                            directory / "tcav.json"
                        ),
                        compute=compute_tcav_factor,
                        store=lambda directory, value: _write_json(
                            directory / "tcav.json", value
                        ),
                        validate=_validate_tcav_factor,
                        fingerprint=lambda value: {
                            "tcav": stable_hash(value)
                        },
                    )
                    tcav_values = tcav_result.value
                score = float(tcav_values["tcav_score"])
                entry: dict[str, Any] = {
                    "CAV": np.asarray(cav["CAV"]),
                    "TCAV_score": score,
                    "Rule": cav["Rule"],
                }
                diagnostic = diagnostic_lookup[(run_id, factor_id)]
                diagnostic["tcav_score"] = score
                diagnostic.update(tcav_values)
                p_value = tcav_values.get("tcav_p_value")
                if "tcav_p_value" in tcav_values:
                    diagnostic["tcav_significant"] = (
                        p_value is not None and float(p_value) < 0.05
                    )
                functional.setdefault(str(run_id), {})[factor_id] = entry
                cav_arrays[f"run_{run_id}_factor_{factor_id}"] = np.asarray(
                    cav["CAV"]
                )

        _write_jsonl(workspace / "high_precision_rules.jsonl", rule_rows)
        _write_json(workspace / "functional.json", diagnostics)
        if cav_arrays:
            np.savez_compressed(
                workspace / "functional_cavs.npz", **cav_arrays)
        return functional, diagnostics

    def semantic(
        self,
        prepared: _PreparedData,
        sae_data: _SAEData,
        matches: Sequence[Mapping[str, Any]],
        functional: Mapping[Any, Any],
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> dict[str, Any]:
        from semantic_config import (
            SemanticExperimentConfig,
            load_clinical_groups,
        )
        from semantic_experiment import run_semantic_comparison

        semantic_path = Path(config.semantic_config_path)
        semantic = SemanticExperimentConfig.from_json(semantic_path)
        semantic = replace(
            semantic,
            runtime=replace(
                semantic.runtime,
                seed=config.seed,
                artifact_dir=str(workspace / "semantic"),
                cache=config.use_cache,
                show_progress=config.show_progress,
            ),
        )
        clinical_path = semantic.clinical_groups_path
        if clinical_path is not None:
            clinical_path = str(_resolve_path(
                semantic_path.parent, clinical_path))
        clinical_groups = load_clinical_groups(clinical_path)
        return run_semantic_comparison(
            X=prepared.X_test,
            outcome_for_stratification=prepared.y_test,
            patient_ids=prepared.patient_ids,
            feature_names=prepared.feature_names,
            activations_by_run=sae_data.activations,
            matchings=matches,
            config=semantic,
            clinical_groups=clinical_groups,
            functional_by_factor=functional,
            record_keys=prepared.record_keys,
            force=force,
            shared_cache=self.cache,
        )


def run_comparison(
    config: ComparisonRunnerConfig,
    *,
    force: bool = False,
    force_stages: Sequence[str] = (),
    adapter: DefaultComparisonAdapter | None = None,
) -> dict[str, Any]:
    """Run the complete comparison and return a compact artifact summary."""

    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Renal Feather file not found: {dataset_path}")
    semantic_path = Path(config.semantic_config_path)
    if not semantic_path.is_file():
        raise FileNotFoundError(
            f"Semantic configuration not found: {semantic_path}")

    accelerator_info = accelerator_manifest(config.accelerator.device)
    print(
        "Accelerator: "
        f"requested={accelerator_info['requested_device']}, "
        f"resolved={accelerator_info['resolved_device']}"
    )
    dataset_hash = _file_fingerprint(dataset_path)
    semantic_dependencies = _semantic_dependency_fingerprints(semantic_path)
    hash_semantic_dependencies = {
        "scientific_config_hash": semantic_dependencies[
            "scientific_config_hash"
        ],
        "clinical_groups_sha256": (
            None
            if semantic_dependencies["clinical_groups"] is None
            else semantic_dependencies["clinical_groups"]["sha256"]
        ),
    }
    source_hash = _runner_source_fingerprint()
    hash_config = _scientific_runner_config(config)
    runner_hash = stable_hash(
        hash_config,
        dataset_hash,
        hash_semantic_dependencies,
        source_hash,
    )[:20]
    workspace = Path(config.artifact_dir) / runner_hash
    workspace.mkdir(parents=True, exist_ok=True)
    summary_path = workspace / "summary.json"
    if (
        config.use_cache
        and not force
        and not force_stages
        and summary_path.exists()
    ):
        with summary_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        cached["cache_hit"] = True
        return cached

    cache_root = (
        Path(config.cache_dir)
        if config.cache_dir is not None
        else Path(config.artifact_dir) / "_cache" / "v2"
    )
    forced_groups = tuple(
        comparison_cache_group
        for comparison_cache_group in (
            (
                "prepared",
                "splits",
                "tabpfn",
                "embeddings",
                "sae",
                "activations",
                "matching",
                "functional",
                "semantic",
            )
            if force
            else tuple(force_stages)
        )
    )
    shared_cache = ComparisonCache(
        cache_root,
        enabled=config.use_cache,
        verification=config.cache_verification,
        forced_stages=forced_groups,
    )
    telemetry = StageTelemetry(
        workspace / "stage_metrics.json",
        requested_device=config.accelerator.device,
    )
    active_adapter = adapter or DefaultComparisonAdapter(shared_cache)
    with telemetry.measure("prepare"):
        prepared = active_adapter.prepare(config, workspace, force=force)
    from semantic_splits import semantic_test_subsplits

    with telemetry.measure("split"):
        def compute_splits():
            return semantic_test_subsplits(
                prepared.y_test, prepared.patient_ids, rng_seed=config.seed
            )

        split_result = shared_cache.resolve(
            stage="splits",
            item="semantic-test-subsplits",
            dependencies={
                "outcome": array_fingerprint(prepared.y_test),
                "patient_ids": array_fingerprint(prepared.patient_ids),
                "seed": config.seed,
                "fractions": (0.33, 0.335, 0.1675, 0.1675),
            },
            source_fingerprint=_source_files_fingerprint(
                "semantic_splits.py"
            ),
            load=_load_splits,
            compute=compute_splits,
            store=_store_splits,
            validate=lambda value: _validate_splits(
                value, len(prepared.y_test), prepared.patient_ids
            ),
            fingerprint=lambda value: {
                "splits": stable_hash(
                    {
                        name: indices.tolist()
                        for name, indices in sorted(value.items())
                        if name.startswith("idx_semantic")
                        or name == "idx_tcav_eval"
                    }
                )
            },
        )
        splits = split_result.value
        np.savez_compressed(
            workspace / "splits.npz",
            **{
                name: values
                for name, values in splits.items()
                if name.startswith("idx_semantic") or name == "idx_tcav_eval"
            },
        )
    with telemetry.measure("tabpfn_and_embeddings"):
        embeddings = active_adapter.embeddings(
            prepared, splits, config, workspace, force=force
        )
    with telemetry.measure("sae_training_and_encoding"):
        sae_data = active_adapter.train_saes(
            prepared, embeddings, splits, config, workspace, force=force
        )
    with telemetry.measure("geometric_matching"):
        all_matches, selected_matches = active_adapter.match(
            sae_data, config, workspace
        )
    estimate = _runtime_estimate(config, selected_matches)
    print(
        "Semantic workload: "
        f"{estimate['selected_pairs']} pairs, "
        f"{estimate['unique_factors']} unique factors, "
        f"about {estimate['tree_fits']:,} randomized-tree fits."
    )
    with telemetry.measure("high_precision_cav_tcav"):
        functional, functional_diagnostics = active_adapter.functional(
            prepared,
            embeddings,
            sae_data,
            splits,
            selected_matches,
            config,
            workspace,
            force=force or "functional" in force_stages,
        )
    with telemetry.measure("semantic_bundle"):
        _write_semantic_inputs(workspace, prepared, sae_data.activations)
    with telemetry.measure("stable_semantic_comparison"):
        semantic_result = active_adapter.semantic(
            prepared,
            sae_data,
            selected_matches,
            functional,
            config,
            workspace,
            force=force or "semantic" in force_stages,
        )
    from semantic_artifacts import environment_manifest

    tabpfn_fit_events = [
        event
        for event in shared_cache.events
        if event.stage == "tabpfn_fit"
    ]
    tabpfn_model_cache_supported = (
        None
        if not tabpfn_fit_events
        else not any(
            event.reason == "store_unsupported"
            for event in tabpfn_fit_events
        )
    )
    manifest = {
        "runner_hash": runner_hash,
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_hash,
        "semantic_dependencies": semantic_dependencies,
        "source_fingerprint": source_hash,
        "config": config.to_dict(),
        "accelerator": accelerator_info,
        "environment": environment_manifest(),
        "n_train_records": len(prepared.X_train),
        "n_test_records": len(prepared.X_test),
        "n_features": len(prepared.feature_names),
        "n_sae_runs": len(sae_data.runs),
        "n_all_matches": len(all_matches),
        "n_selected_matches": len(selected_matches),
        "functional_entries": sum(len(values) for values in functional.values()),
        "functional_diagnostics": len(functional_diagnostics),
        "record_fingerprint": array_fingerprint(prepared.record_keys),
        "workload_estimate": estimate,
        "stage_metrics": telemetry.records,
        "total_timed_seconds": sum(
            float(row["seconds"]) for row in telemetry.records.values()
        ),
        "cache": {
            **shared_cache.summary(),
            "refs_file": str(workspace / "cache_refs.json"),
            "tabpfn_model_cache_supported": (
                tabpfn_model_cache_supported
            ),
        },
    }
    shared_cache.write_refs(workspace / "cache_refs.json")
    _write_json(workspace / "runner_manifest.json", manifest)
    summary = {
        "runner_hash": runner_hash,
        "artifact_dir": str(workspace),
        "semantic_artifact_dir": semantic_result["artifact_dir"],
        "semantic_experiment_hash": semantic_result["experiment_hash"],
        "n_all_matches": len(all_matches),
        "n_selected_matches": len(selected_matches),
        "n_functional_factors": manifest["functional_entries"],
        "class_analysis_enabled": bool(
            semantic_result["manifest"]["config"]["class_analysis"]["enabled"]
        ),
        "resolved_device": accelerator_info["resolved_device"],
        "stage_metrics_file": str(workspace / "stage_metrics.json"),
        "total_timed_seconds": manifest["total_timed_seconds"],
        "cache": manifest["cache"],
        "cache_hit": False,
    }
    _write_json(summary_path, summary)
    return summary


def _scientific_runner_config(
    config: ComparisonRunnerConfig,
) -> dict[str, Any]:
    value = config.to_dict()
    for name in (
        "artifact_dir",
        "cache_dir",
        "cache_verification",
        "use_cache",
        "show_progress",
        "dataset_path",
        "semantic_config_path",
    ):
        value.pop(name, None)
    value["accelerator"].pop("device", None)
    value["tabpfn"].pop("batch_size", None)
    value["sae"].pop("encoding_batch_size", None)
    value["functional"].pop("gradient_batch_size", None)
    return value


def _source_files_fingerprint(*names: str) -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _callable_source_fingerprint(*values: Any) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(getattr(value, "__module__", "")).encode())
        digest.update(
            str(getattr(value, "__qualname__", repr(value))).encode())
        digest.update(inspect.getsource(value).encode())
    return digest.hexdigest()


def _tabpfn_checkpoint_fingerprint(model_name: str) -> dict[str, Any]:
    try:
        import tabpfn
        from importlib import resources

        candidate = Path(resources.files(tabpfn)) / "model_cache" / (
            f"{model_name}.cpkt"
        )
        return {
            "path_name": candidate.name,
            "sha256": (
                _file_fingerprint(candidate) if candidate.is_file() else None
            ),
        }
    except (ImportError, OSError, TypeError):
        return {"path_name": f"{model_name}.cpkt", "sha256": None}


def _package_versions(*names: str) -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "unavailable"
    return versions


def _numerical_environment_fingerprint(
    requested_device: str, *packages: str
) -> str:
    accelerator = accelerator_manifest(requested_device)
    accelerator.pop("requested_device", None)
    return stable_hash(accelerator, _package_versions(*packages))


def _pickle_load(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pickle_dump(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _prepared_fingerprints(prepared: _PreparedData) -> dict[str, str]:
    return {
        "X_train": array_fingerprint(prepared.X_train),
        "y_train": array_fingerprint(prepared.y_train),
        "years_train": array_fingerprint(prepared.years_train),
        "X_test": array_fingerprint(prepared.X_test),
        "y_test": array_fingerprint(prepared.y_test),
        "years_test": array_fingerprint(prepared.years_test),
        "patient_ids": array_fingerprint(prepared.patient_ids),
        "record_keys": array_fingerprint(prepared.record_keys),
        "feature_names": stable_hash(prepared.feature_names),
    }


def _validate_tabpfn_fit(value: Mapping[str, Any]) -> None:
    required = {
        "model",
        "model_add_x_device",
        "example_add_shape",
        "fit_time_sec",
        "model_source",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"TabPFN fit missing fields: {sorted(missing)}")
    if not hasattr(value["model"], "get_embeddings"):
        raise ValueError("Cached TabPFN model cannot extract embeddings")


def _store_tabpfn_fit(directory: Path, value: Mapping[str, Any]) -> None:
    if value.get("model_source") == "fallback_classifier":
        raise RuntimeError("Fallback TabPFN fits are not shared-cacheable")
    _pickle_dump(directory / "fitted_model.pkl", value)


def _load_embedding_pair(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(directory / "embeddings.npz", allow_pickle=False) as values:
        return np.asarray(values["train_raw"]), np.asarray(values["test_raw"])


def _store_embedding_pair(
    directory: Path, value: tuple[np.ndarray, np.ndarray]
) -> None:
    np.savez_compressed(
        directory / "embeddings.npz",
        train_raw=value[0],
        test_raw=value[1],
    )


def _validate_embedding_pair(
    value: tuple[np.ndarray, np.ndarray], prepared: _PreparedData
) -> None:
    train_raw, test_raw = value
    if train_raw.ndim != 2 or test_raw.ndim != 2:
        raise ValueError("Cached embeddings must be two-dimensional")
    if len(train_raw) != len(prepared.X_train) or len(test_raw) != len(
        prepared.X_test
    ):
        raise ValueError("Cached embedding rows are not aligned")
    if train_raw.shape[1] != test_raw.shape[1]:
        raise ValueError("Cached embedding dimensions differ")
    if not np.isfinite(train_raw).all() or not np.isfinite(test_raw).all():
        raise ValueError("Cached embeddings contain non-finite values")


def _validate_walkforward(value: Any) -> None:
    if not isinstance(value, list) or any(
        not isinstance(row, Mapping) for row in value
    ):
        raise ValueError("Walk-forward cache must contain metric rows")


def _scale_embeddings_from_semantic_fit(
    train_raw: np.ndarray,
    test_raw: np.ndarray,
    fit_indices: np.ndarray,
):
    """Fit one scaler on semantic-fit test rows; transform every embedding row."""

    from sklearn.preprocessing import StandardScaler

    train = np.asarray(train_raw)
    test = np.asarray(test_raw)
    indices = np.asarray(fit_indices, dtype=int)
    scaler = StandardScaler().fit(test[indices])
    return scaler.transform(train), scaler.transform(test), scaler


def _load_scaled_embeddings(directory: Path):
    with np.load(directory / "scaled.npz", allow_pickle=False) as values:
        train = np.asarray(values["train_scaled"])
        test = np.asarray(values["test_scaled"])
    scaler = _pickle_load(directory / "scaler.pkl")
    return train, test, scaler


def _store_scaled_embeddings(directory: Path, value) -> None:
    train, test, scaler = value
    np.savez_compressed(
        directory / "scaled.npz",
        train_scaled=train,
        test_scaled=test,
    )
    _pickle_dump(directory / "scaler.pkl", scaler)


def _validate_scaled_embeddings(value, prepared: _PreparedData) -> None:
    train, test, scaler = value
    _validate_embedding_pair((train, test), prepared)
    if not hasattr(scaler, "transform"):
        raise ValueError("Cached embedding scaler is invalid")


def _model_state_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(array_fingerprint(value).encode())
    return digest.hexdigest()


def _store_sae_run(directory: Path, run: Mapping[str, Any]) -> None:
    import torch

    torch.save(run["model"].state_dict(), directory / "state_dict.pt")
    metadata = {key: value for key, value in run.items() if key != "model"}
    _pickle_dump(directory / "metadata.pkl", metadata)


def _load_sae_run(
    directory: Path, *, data_dimension: int, sae_config: SAERunnerConfig
) -> dict[str, Any]:
    import torch
    from sae import SAE

    model = SAE(
        data_dimension=data_dimension,
        scaling_factor=sae_config.scaling_factor,
        use_decoder_bias=True,
        type=sae_config.model_type,
        k=sae_config.k,
        k_aux=sae_config.k_aux,
    )
    model.load_state_dict(
        torch.load(directory / "state_dict.pt", map_location="cpu")
    )
    metadata = _pickle_load(directory / "metadata.pkl")
    return {**metadata, "model": model}


def _validate_sae_run(
    run: Mapping[str, Any], expected_seed: int, data_dimension: int
) -> None:
    if int(run.get("seed", -1)) != int(expected_seed):
        raise ValueError("Cached SAE seed mismatch")
    model = run.get("model")
    if model is None or int(model.encoder.in_features) != int(data_dimension):
        raise ValueError("Cached SAE architecture mismatch")
    expected_shape = (model.num_latents, data_dimension)
    directions = np.asarray(run.get("decoder_directions"))
    if directions.shape != expected_shape:
        raise ValueError("Cached SAE directions mismatch")
    current_directions = (
        model.encoder.weight.detach().cpu().numpy()
        if run.get("model_type") == "ReLU"
        else model.decoder.weight.detach().cpu().numpy().T
    )
    if not np.array_equal(directions, current_directions):
        raise ValueError("Cached SAE directions do not match model state")


def _load_single_array(path: Path) -> np.ndarray:
    return np.asarray(np.load(path, allow_pickle=False))


def _validate_activation_matrix(value: np.ndarray, n_records: int) -> None:
    matrix = np.asarray(value)
    if matrix.ndim != 2 or len(matrix) != n_records:
        raise ValueError("Cached activations are not row-aligned")
    if not np.isfinite(matrix).all():
        raise ValueError("Cached activations contain non-finite values")


def _validate_matching_rows(rows: Any) -> None:
    required = {"original_concept", "best_pair", "cos_sim", "overlap"}
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or not required <= set(row)
        for row in rows
    ):
        raise ValueError("Cached matching rows are invalid")
    for row in rows:
        if int(row["original_concept"]) < 0 or int(row["best_pair"]) < 0:
            raise ValueError("Cached matching factor IDs must be non-negative")
        if not math.isfinite(float(row["cos_sim"])) or not math.isfinite(
            float(row["overlap"])
        ):
            raise ValueError("Cached matching scores must be finite")


def _pair_matching_metadata(analysis: Any) -> dict[str, Any]:
    return {
        "top_k": int(analysis.top_k),
        "percentiles": sorted(int(value) for value in analysis.overlaps),
        "cosine_assignment": asdict(analysis.cosine_assignment),
        "overlap_assignments": {
            str(percentile): asdict(assignment)
            for percentile, assignment in analysis.overlap_assignments.items()
        },
        "nearest_neighbors": {
            metric: [asdict(row) for row in rows]
            for metric, rows in analysis.nearest_neighbors.items()
        },
        "nearest_hungarian_gaps": {
            metric: [asdict(row) for row in rows]
            for metric, rows in analysis.nearest_hungarian_gaps.items()
        },
    }


def _store_pair_matching_analysis(directory: Path, analysis: Any) -> None:
    np.savez_compressed(
        directory / "matrices.npz",
        cosine=analysis.cosine,
        **{
            f"overlap_p{percentile}": matrix
            for percentile, matrix in analysis.overlaps.items()
        },
    )
    _write_json(directory / "raw_analysis.json",
                _pair_matching_metadata(analysis))


def _load_pair_matching_analysis(directory: Path) -> Any:
    from robustness_matching import (
        Assignment,
        NearestHungarianGap,
        NearestNeighbor,
        PairMatchingAnalysis,
    )

    metadata = _read_json(directory / "raw_analysis.json")
    with np.load(directory / "matrices.npz", allow_pickle=False) as values:
        cosine = np.asarray(values["cosine"])
        overlaps = {
            int(name.removeprefix("overlap_p")): np.asarray(values[name])
            for name in values.files
            if name.startswith("overlap_p")
        }

    def assignment(raw: Mapping[str, Any]) -> Assignment:
        return Assignment(
            pairs=tuple(tuple(int(value) for value in pair)
                        for pair in raw["pairs"]),
            left_to_right=tuple(
                None if value is None else int(value)
                for value in raw["left_to_right"]
            ),
            right_to_left=tuple(
                None if value is None else int(value)
                for value in raw["right_to_left"]
            ),
        )

    return PairMatchingAnalysis(
        cosine=cosine,
        overlaps=overlaps,
        cosine_assignment=assignment(metadata["cosine_assignment"]),
        overlap_assignments={
            int(percentile): assignment(raw)
            for percentile, raw in metadata["overlap_assignments"].items()
        },
        nearest_neighbors={
            metric: tuple(NearestNeighbor(**row) for row in rows)
            for metric, rows in metadata["nearest_neighbors"].items()
        },
        nearest_hungarian_gaps={
            metric: tuple(NearestHungarianGap(**row) for row in rows)
            for metric, rows in metadata["nearest_hungarian_gaps"].items()
        },
        top_k=int(metadata["top_k"]),
    )


def _select_matching_rows(
    analysis: Any,
    left_run: int,
    right_run: int,
    matching: MatchingRunnerConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Apply selector threshold; overlap chooses percentile with most survivors."""

    if matching.criterion == "overlap":
        percentile_choices = []
        for percentile, assignment in analysis.overlap_assignments.items():
            matrix = analysis.overlaps[percentile]
            qualifying = sum(
                matrix[left_factor, right_factor]
                >= matching.overlap_analysis_threshold
                for left_factor, right_factor in assignment.pairs
            )
            percentile_choices.append(
                (qualifying, int(percentile), assignment))
        _, selected_percentile, assignment = max(
            percentile_choices, key=lambda choice: (choice[0], choice[1])
        )
        selection_threshold = matching.overlap_analysis_threshold
        selection_field = "overlap"
    else:
        assignment = analysis.cosine_assignment
        selection_threshold = matching.cosine_analysis_threshold
        selection_field = "cos_sim"
        selected_percentile = max(
            analysis.overlaps,
            key=lambda percentile: (
                sum(
                    analysis.overlaps[percentile][left_factor, right_factor]
                    >= matching.overlap_analysis_threshold
                    for left_factor, right_factor in assignment.pairs
                ),
                int(percentile),
            ),
        )

    selection_overlap = analysis.overlaps[selected_percentile]
    rows = [
        {
            "sae_i_idx": left_run,
            "sae_j_idx": right_run,
            "original_concept": left_factor,
            "best_pair": right_factor,
            "cos_sim": float(analysis.cosine[left_factor, right_factor]),
            "overlap": float(selection_overlap[left_factor, right_factor]),
            "overlap_percentile": int(selected_percentile),
            "selection_criterion": matching.criterion,
            "selection_threshold": float(selection_threshold),
            "selection_threshold_pass": bool(
                (
                    analysis.cosine[left_factor, right_factor]
                    if selection_field == "cos_sim"
                    else selection_overlap[left_factor, right_factor]
                )
                >= selection_threshold
            ),
        }
        for left_factor, right_factor in assignment.pairs
    ]
    selected = [row for row in rows if row["selection_threshold_pass"]]
    return rows, selected, int(selected_percentile)


def _matching_artifact_rows(
    analysis: Any,
    left_run: int,
    right_run: int,
    matching: MatchingRunnerConfig,
) -> dict[str, list[dict[str, Any]]]:
    cosine_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for left_factor, right_factor in analysis.cosine_assignment.pairs:
        overlaps = {
            percentile: float(matrix[left_factor, right_factor])
            for percentile, matrix in analysis.overlaps.items()
        }
        consistency_count = sum(
            score >= matching.overlap_analysis_threshold
            for score in overlaps.values()
        )
        cosine_rows.append(
            {
                "run_i": left_run,
                "run_j": right_run,
                "factor_i": left_factor,
                "factor_j": right_factor,
                "cos_sim": float(analysis.cosine[left_factor, right_factor]),
                **{
                    f"overlap_p{percentile}": score
                    for percentile, score in overlaps.items()
                },
                "cosine_threshold_pass": bool(
                    analysis.cosine[left_factor, right_factor]
                    >= matching.cosine_analysis_threshold
                ),
                "overlap_consistency_count": consistency_count,
                "overlap_consistency_pass": bool(consistency_count >= 2),
            }
        )
    for percentile, assignment in analysis.overlap_assignments.items():
        matrix = analysis.overlaps[percentile]
        for left_factor, right_factor in assignment.pairs:
            overlap_rows.append(
                {
                    "run_i": left_run,
                    "run_j": right_run,
                    "percentile": percentile,
                    "factor_i": left_factor,
                    "factor_j": right_factor,
                    "overlap": float(matrix[left_factor, right_factor]),
                    "cos_sim": float(
                        analysis.cosine[left_factor, right_factor]
                    ),
                    "overlap_threshold_pass": bool(
                        matrix[left_factor, right_factor]
                        >= matching.overlap_analysis_threshold
                    ),
                }
            )

    nearest_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for metric, candidates in analysis.nearest_neighbors.items():
        percentile = (
            None if metric == "cosine" else int(
                metric.removeprefix("overlap_p"))
        )
        threshold = (
            matching.cosine_analysis_threshold
            if metric == "cosine"
            else matching.overlap_analysis_threshold
        )
        grouped: dict[tuple[str, int], list[Any]] = {}
        for candidate in candidates:
            grouped.setdefault(
                (candidate.source_side, candidate.source_factor), []
            ).append(candidate)
        for rows in grouped.values():
            rows.sort(key=lambda row: row.rank)
        best = {key: rows[0] for key, rows in grouped.items()}
        qualified_collisions: dict[tuple[str, int], int] = {}
        for side in ("left", "right"):
            counts: dict[int, int] = {}
            for (source_side, _), candidate in best.items():
                if source_side == side and candidate.score >= threshold:
                    counts[candidate.target_factor] = (
                        counts.get(candidate.target_factor, 0) + 1
                    )
            qualified_collisions.update(
                {(side, target): count for target, count in counts.items()}
            )

        gap_by_source = {
            (row.source_side, row.source_factor): row
            for row in analysis.nearest_hungarian_gaps[metric]
        }
        for key, rows in sorted(grouped.items()):
            source_side, source_factor = key
            best_score = rows[0].score
            target_run = right_run if source_side == "left" else left_run
            source_run = left_run if source_side == "left" else right_run
            reverse_side = "right" if source_side == "left" else "left"
            for candidate in rows:
                reverse = best.get((reverse_side, candidate.target_factor))
                reciprocal_qualified = bool(
                    candidate.rank == 1
                    and candidate.reciprocal_raw
                    and candidate.score >= threshold
                    and reverse is not None
                    and reverse.score >= threshold
                )
                qualified_collision_count = qualified_collisions.get(
                    (source_side, candidate.target_factor), 0
                )
                row = {
                    "metric": metric,
                    "percentile": percentile,
                    "source_side": source_side,
                    "source_run": source_run,
                    "target_run": target_run,
                    "source_factor": source_factor,
                    "target_factor": candidate.target_factor,
                    "rank": candidate.rank,
                    "score": candidate.score,
                    "threshold": threshold,
                    "passes_threshold": bool(candidate.score >= threshold),
                    "best_score": best_score,
                    "score_delta_from_best": best_score - candidate.score,
                    "reciprocal_raw": candidate.reciprocal_raw,
                    "reciprocal_threshold_qualified": reciprocal_qualified,
                    "target_collision_count_raw": (
                        candidate.target_collision_count_raw
                    ),
                    "target_collision_count_threshold_qualified": (
                        qualified_collision_count
                    ),
                    "target_collision_raw": bool(
                        candidate.target_collision_count_raw > 1
                    ),
                    "target_collision_threshold_qualified": bool(
                        qualified_collision_count > 1
                    ),
                }
                for delta in matching.alternative_score_deltas:
                    row[_delta_field(delta)] = bool(
                        candidate.rank > 1
                        and candidate.score >= threshold
                        and best_score - candidate.score <= delta + 1e-12
                    )
                nearest_rows.append(row)

            gap = gap_by_source[key]
            diagnostic = {
                "metric": metric,
                "percentile": percentile,
                "source_side": source_side,
                "source_run": source_run,
                "target_run": target_run,
                "source_factor": source_factor,
                "best_target": rows[0].target_factor,
                "best_score": rows[0].score,
                "second_score": rows[1].score if len(rows) > 1 else None,
                "third_score": rows[2].score if len(rows) > 2 else None,
                "best_minus_second": (
                    rows[0].score - rows[1].score if len(rows) > 1 else None
                ),
                "second_minus_third": (
                    rows[1].score - rows[2].score if len(rows) > 2 else None
                ),
                "threshold": threshold,
                "threshold_valid": bool(rows[0].score >= threshold),
                "reciprocal_raw": rows[0].reciprocal_raw,
                "target_collision_count_raw": (
                    rows[0].target_collision_count_raw
                ),
                "target_collision_count_threshold_qualified": (
                    qualified_collisions.get(
                        (source_side, rows[0].target_factor), 0
                    )
                ),
                "hungarian_target": gap.hungarian_target,
                "hungarian_score": gap.hungarian_score,
                "nearest_minus_hungarian": gap.nearest_minus_hungarian,
            }
            reverse = best.get((reverse_side, rows[0].target_factor))
            diagnostic["reciprocal_threshold_qualified"] = bool(
                rows[0].reciprocal_raw
                and rows[0].score >= threshold
                and reverse is not None
                and reverse.score >= threshold
            )
            for delta in matching.alternative_score_deltas:
                diagnostic[_delta_field(delta)] = any(
                    candidate.rank > 1
                    and candidate.score >= threshold
                    and rows[0].score - candidate.score <= delta + 1e-12
                    for candidate in rows
                )
            diagnostics.append(diagnostic)

    return {
        "cosine_hungarian": cosine_rows,
        "overlap_hungarian": overlap_rows,
        "nearest_neighbors": nearest_rows,
        "diagnostics": diagnostics,
    }


def _delta_field(delta: float) -> str:
    return f"valid_alternative_delta_{delta:.2f}".replace(".", "_")


def _normalize_percentile_rules(value: Any) -> dict[int, list[dict[str, Any]]]:
    if not isinstance(value, Mapping):
        raise ValueError("Cached percentile rules must be a mapping")
    return {
        int(percentile): [dict(row) for row in rows]
        for percentile, rows in value.items()
    }


def _validate_percentile_rules(value: Any) -> None:
    normalized = _normalize_percentile_rules(value)
    if set(normalized) != {90, 80, 70, 60, 50}:
        raise ValueError("Cached percentile rule levels are incomplete")
    required = {
        "Factor",
        "Rule",
        "Class",
        "Precision",
        "Recall",
        "Patients",
        "Patients_concept",
    }
    if any(
        not required <= set(row)
        for rows in normalized.values()
        for row in rows
    ):
        raise ValueError("Cached high-precision rule row is incomplete")


def _validate_forced_rule_value(value: Any) -> None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("rules"), list)
        or value.get("dot") is not None
        and not isinstance(value.get("dot"), str)
    ):
        raise ValueError("Cached forced-rule value is invalid")


def _validate_cavs(value: Any, embedding_dimension: int) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("Cached CAVs must be a mapping")
    for factor_id, cav in value.items():
        if not isinstance(cav, Mapping):
            raise ValueError(f"Cached CAV {factor_id} is invalid")
        vector = np.asarray(cav.get("CAV"))
        if vector.shape != (embedding_dimension,) or not np.isfinite(
            vector
        ).all():
            raise ValueError(f"Cached CAV {factor_id} has invalid direction")
        for cohort in ("positive_idx", "negative_idx"):
            if np.asarray(cav.get(cohort)).ndim != 1:
                raise ValueError(
                    f"Cached CAV {factor_id} has invalid {cohort}"
                )


def _cav_fingerprints(value: Mapping[Any, Mapping[str, Any]]) -> dict[str, str]:
    return {
        f"factor:{factor_id}": stable_hash(
            array_fingerprint(np.asarray(cav["CAV"])),
            array_fingerprint(np.asarray(cav["positive_idx"], dtype=int)),
            array_fingerprint(np.asarray(cav["negative_idx"], dtype=int)),
            cav["Rule"],
        )
        for factor_id, cav in sorted(value.items(), key=lambda item: int(item[0]))
    }


def _validate_gradients(
    value: np.ndarray, n_records: int, embedding_dimension: int
) -> None:
    gradients = np.asarray(value)
    if gradients.shape != (n_records, embedding_dimension):
        raise ValueError("Cached TCAV gradients have invalid shape")
    if not np.isfinite(gradients).all():
        raise ValueError("Cached TCAV gradients contain non-finite values")


def _validate_tcav_factor(value: Any) -> None:
    if not isinstance(value, Mapping) or "tcav_score" not in value:
        raise ValueError("Cached TCAV factor result is invalid")
    score = float(value["tcav_score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError("Cached TCAV score lies outside [0, 1]")


def _load_splits(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "splits.npz", allow_pickle=False) as values:
        return {name: np.asarray(values[name], dtype=int) for name in values.files}


def _store_splits(directory: Path, splits: Mapping[str, np.ndarray]) -> None:
    np.savez_compressed(
        directory / "splits.npz",
        **{name: np.asarray(value, dtype=int) for name, value in splits.items()},
    )


def _validate_splits(
    splits: Mapping[str, np.ndarray],
    n_records: int,
    patient_ids: np.ndarray,
) -> None:
    names = (
        "idx_semantic_fit",
        "idx_semantic_select",
        "idx_tcav_eval",
        "idx_semantic_final",
    )
    if any(name not in splits for name in names):
        raise ValueError("Cached semantic splits are incomplete")
    indices = [np.asarray(splits[name], dtype=int) for name in names]
    combined = np.concatenate(indices)
    if sorted(combined.tolist()) != list(range(n_records)):
        raise ValueError("Cached semantic splits do not partition records")
    groups = np.asarray(patient_ids)
    group_sets = [set(groups[index].tolist()) for index in indices]
    if any(
        group_sets[left] & group_sets[right]
        for left in range(len(group_sets))
        for right in range(left + 1, len(group_sets))
    ):
        raise ValueError("Cached semantic splits leak patients")


def _validate_prepared(prepared: _PreparedData) -> None:
    n_train, n_test = len(prepared.X_train), len(prepared.X_test)
    if not (
        n_train == len(prepared.y_train) == len(prepared.years_train)
        and n_test
        == len(prepared.y_test)
        == len(prepared.years_test)
        == len(prepared.patient_ids)
        == len(prepared.record_keys)
    ):
        raise ValueError("Prepared renal arrays are not row-aligned")
    if prepared.X_train.ndim != 2 or prepared.X_test.ndim != 2:
        raise ValueError("Prepared feature matrices must be two-dimensional")
    if prepared.X_train.shape[1] != len(prepared.feature_names):
        raise ValueError("Prepared feature names do not match matrix columns")
    if prepared.X_test.shape[1] != len(prepared.feature_names):
        raise ValueError("Train and test feature dimensions differ")
    if len(set(prepared.feature_names)) != len(prepared.feature_names):
        raise ValueError("Prepared feature names must be unique")
    if not np.isfinite(prepared.X_train).all() or not np.isfinite(
        prepared.X_test
    ).all():
        raise ValueError("Prepared features contain non-finite values")
    if len(np.unique(prepared.patient_ids.astype(str))) < 4:
        raise ValueError("Semantic comparison requires at least four patients")
    if len(np.unique(prepared.y_test)) < 2:
        raise ValueError("Test outcome must contain at least two classes")
    if len(set(prepared.record_keys.tolist())) != len(prepared.record_keys):
        raise ValueError("Patient-year record keys must be unique")


def _validate_activations(
    activations: Mapping[int, np.ndarray],
    runs: Sequence[Mapping[str, Any]],
    n_records: int,
) -> None:
    expected = {int(run["idx"]) for run in runs}
    if set(activations) != expected:
        raise ValueError("SAE activation run IDs do not match trained runs")
    for run_id, matrix in activations.items():
        value = np.asarray(matrix)
        if value.ndim != 2 or len(value) != n_records:
            raise ValueError(
                f"SAE run {run_id} activations are not row-aligned")
        if not np.isfinite(value).all():
            raise ValueError(
                f"SAE run {run_id} activations contain non-finite values")


def _run_pairs(n_runs: int, scope: str) -> list[tuple[int, int]]:
    if scope == "baseline":
        return [(0, index) for index in range(1, n_runs)]
    return [
        (left, right)
        for left in range(n_runs)
        for right in range(left + 1, n_runs)
    ]


def _matched_factors_by_run(
    matches: Sequence[Mapping[str, Any]],
) -> dict[int, set[int]]:
    values: dict[int, set[int]] = {}
    for row in matches:
        left_run = int(row["run_i"] if "run_i" in row else row["sae_i_idx"])
        right_run = int(row["run_j"] if "run_j" in row else row["sae_j_idx"])
        left_factor = int(
            row["factor_i"] if "factor_i" in row else row["original_concept"]
        )
        right_factor = int(
            row["factor_j"] if "factor_j" in row else row["best_pair"]
        )
        values.setdefault(left_run, set()).add(left_factor)
        values.setdefault(right_run, set()).add(right_factor)
    return values


def _runtime_estimate(
    config: ComparisonRunnerConfig,
    matches: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    factors = _matched_factors_by_run(matches)
    unique_factors = sum(len(values) for values in factors.values())
    with Path(config.semantic_config_path).open(encoding="utf-8") as handle:
        semantic_raw = json.load(handle)
    fractions = semantic_raw.get("activation_targets", {}).get(
        "positive_fractions", [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    discovery = semantic_raw.get("discovery", {})
    tree_fits = (
        unique_factors
        * len(fractions)
        * int(discovery.get("n_bootstraps", 30))
        * int(discovery.get("trees_per_bootstrap", 100))
    )
    return {
        "selected_pairs": len(matches),
        "unique_factors": unique_factors,
        "activation_thresholds": len(fractions),
        "tree_fits": tree_fits,
    }


def _write_semantic_inputs(
    workspace: Path,
    prepared: _PreparedData,
    activations: Mapping[int, np.ndarray],
) -> None:
    arrays: dict[str, np.ndarray] = {
        "X": prepared.X_test,
        "outcome": prepared.y_test,
        "patient_ids": prepared.patient_ids.astype(str),
        "feature_names": np.asarray(prepared.feature_names, dtype=str),
        "record_keys": prepared.record_keys.astype(str),
    }
    arrays.update(
        {
            f"activations_run_{run_id}": np.asarray(matrix)
            for run_id, matrix in sorted(activations.items())
        }
    )
    np.savez_compressed(workspace / "semantic_inputs.npz", **arrays)


def _file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_dependency_fingerprints(path: Path) -> dict[str, Any]:
    """Fingerprint semantic config and its external clinical taxonomy."""

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping):
        raise ValueError("Semantic configuration must be a JSON object")

    fingerprints: dict[str, Any] = {
        "config_path": str(path.resolve()),
        "config_sha256": _file_fingerprint(path),
        "scientific_config_hash": None,
        "clinical_groups": None,
    }
    scientific_raw = json.loads(json.dumps(raw))
    runtime = scientific_raw.get("runtime")
    if isinstance(runtime, dict):
        for name in ("artifact_dir", "cache", "show_progress", "n_jobs"):
            runtime.pop(name, None)
    fingerprints["scientific_config_hash"] = stable_hash(scientific_raw)
    clinical_groups_path = raw.get("clinical_groups_path")
    if clinical_groups_path is None:
        return fingerprints
    resolved = _resolve_path(path.parent, str(clinical_groups_path))
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Clinical group mapping not found: {resolved}"
        )
    fingerprints["clinical_groups"] = {
        "path": str(resolved),
        "sha256": _file_fingerprint(resolved),
    }
    return fingerprints


def _runner_source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    names = (
        "comparison_runner.py",
        "comparison_cache.py",
        "main-comparison.py",
        "runtime_acceleration.py",
        "database.py",
        "tabpfn_model.py",
        "sae.py",
        "sae_compare.py",
        "robustness_matching.py",
        "decision_tree.py",
        "tcav.py",
        "semantic_artifacts.py",
        "semantic_experiment.py",
        "semantic_rules.py",
        "stable_rule_backend.py",
        "semantic_compare.py",
        "semantic_config.py",
        "semantic_splits.py",
    )
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            _jsonable(value),
            handle,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_jsonable(row), sort_keys=True, allow_nan=False)
            )
            handle.write("\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key))
                         for key in fieldnames} for row in rows])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    return float(value)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


__all__ = [
    "AcceleratorRunnerConfig",
    "ComparisonRunnerConfig",
    "DefaultComparisonAdapter",
    "FunctionalRunnerConfig",
    "MatchingRunnerConfig",
    "SAERunnerConfig",
    "TabPFNRunnerConfig",
    "run_comparison",
]
