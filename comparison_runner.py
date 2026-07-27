"""End-to-end renal SAE comparison orchestration.

The public interface is deliberately small: load one configuration and call
``run_comparison``.  Expensive legacy stages stay behind an adapter seam so the
orchestration can be tested without training TabPFN or SAEs.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from semantic_artifacts import array_fingerprint, stable_hash
from runtime_acceleration import StageTelemetry, accelerator_manifest


@dataclass(frozen=True)
class AcceleratorRunnerConfig:
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("accelerator.device must be 'auto', 'cpu', or 'cuda'")


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
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("sae.seeds must contain at least two unique values")
        if self.model_type not in {"ReLU", "TopK"}:
            raise ValueError("sae.model_type must be 'ReLU' or 'TopK'")
        if self.epochs < 1:
            raise ValueError("sae.epochs must be positive")
        if self.alpha < 0 or self.scaling_factor <= 0:
            raise ValueError("sae alpha/scaling_factor values are invalid")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("sae optimizer values are invalid")
        if self.k < 1 or self.k_aux < 1 or self.encoding_batch_size < 1:
            raise ValueError("sae k and encoding batch values must be positive")


@dataclass(frozen=True)
class MatchingRunnerConfig:
    scope: str = "all"
    criterion: str = "cos_sim"
    minimum_score: float | None = 0.70
    max_pairs_per_run_pair: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"all", "baseline"}:
            raise ValueError("matching.scope must be 'all' or 'baseline'")
        if self.criterion not in {"cos_sim", "overlap"}:
            raise ValueError("matching.criterion must be 'cos_sim' or 'overlap'")
        if self.minimum_score is not None and not -1 <= self.minimum_score <= 1:
            raise ValueError("matching.minimum_score must be null or lie in [-1, 1]")
        if (
            self.max_pairs_per_run_pair is not None
            and self.max_pairs_per_run_pair < 1
        ):
            raise ValueError("matching.max_pairs_per_run_pair must be positive")


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
    seed: int = 42
    accelerator: AcceleratorRunnerConfig = field(
        default_factory=AcceleratorRunnerConfig
    )
    tabpfn: TabPFNRunnerConfig = field(default_factory=TabPFNRunnerConfig)
    sae: SAERunnerConfig = field(default_factory=SAERunnerConfig)
    matching: MatchingRunnerConfig = field(default_factory=MatchingRunnerConfig)
    functional: FunctionalRunnerConfig = field(default_factory=FunctionalRunnerConfig)

    def __post_init__(self) -> None:
        if not self.dataset_path or not self.semantic_config_path or not self.artifact_dir:
            raise ValueError("dataset, semantic config, and artifact paths are required")
        if not isinstance(self.use_cache, bool):
            raise ValueError("use_cache must be a boolean")

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
            "seed",
            "accelerator",
            "tabpfn",
            "sae",
            "matching",
            "functional",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown comparison config fields: {sorted(unknown)}")
        return cls(
            dataset_path=str(raw.get("dataset_path", cls.dataset_path)),
            semantic_config_path=str(
                raw.get("semantic_config_path", cls.semantic_config_path)
            ),
            artifact_dir=str(raw.get("artifact_dir", cls.artifact_dir)),
            use_cache=raw.get("use_cache", True),
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


@dataclass
class _SAEData:
    runs: list[dict[str, Any]]
    activations: dict[int, np.ndarray]


class DefaultComparisonAdapter:
    """Production adapter for expensive legacy model and filesystem stages."""

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

        cache_path = workspace / "prepared.pkl"
        if config.use_cache and not force and cache_path.exists():
            with cache_path.open("rb") as handle:
                prepared = pickle.load(handle)
            _validate_prepared(prepared)
            return prepared

        dataset_path = Path(config.dataset_path)
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Renal Feather file not found: {dataset_path}")
        frame = open_feather(str(dataset_path))
        required = {"patient_id", "date", "event"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Renal Feather missing columns: {sorted(missing)}")
        if not frame["event"].astype(str).eq("DEATH").any():
            raise ValueError("Renal Feather must contain an exact 'DEATH' event")

        preparation_config = TabPFNPrepConfig()
        preparation_config.rng_seed = config.seed
        prepared_rows = prepare_database(frame, cfg=preparation_config)
        arrays = get_tabpfn_arrays(prepared_rows)
        train_rows = prepared_rows["train_rows"].reset_index(drop=True)
        test_rows = prepared_rows["test_rows"].reset_index(drop=True)
        features = tuple(str(name) for name in prepared_rows["top_k_events"])
        patient_ids = test_rows["patient_id"].astype(str).to_numpy()
        years_test = np.asarray(arrays["years_test"], dtype=int)
        record_keys = np.asarray(
            [
                f"patient:{patient}|year:{year}"
                for patient, year in zip(patient_ids, years_test)
            ],
            dtype=str,
        )
        prepared = _PreparedData(
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
        _validate_prepared(prepared)
        with cache_path.open("wb") as handle:
            pickle.dump(prepared, handle)
        return prepared

    def embeddings(
        self,
        prepared: _PreparedData,
        config: ComparisonRunnerConfig,
        workspace: Path,
        *,
        force: bool,
    ) -> _EmbeddingData:
        from tabpfn_model import (
            EmbeddingExtractConfig,
            TabPFNEvalConfig,
            extract_embeddings_robust,
            fit_dr_tabpfn,
            flatten_embeddings,
            scale_embeddings,
            walkforward_evaluate_tabpfn,
        )

        evaluation = TabPFNEvalConfig()
        evaluation.rng_seed = config.seed
        evaluation.tabpfn_model_name = config.tabpfn.model_name
        evaluation.batch_size_predict = config.tabpfn.batch_size
        evaluation.device = config.accelerator.device
        fit = fit_dr_tabpfn(
            prepared.X_train,
            prepared.y_train,
            prepared.years_train,
            evaluation,
        )
        model = fit["model"]
        domain_map = {
            int(year): index
            for index, year in enumerate(
                sorted(
                    set(prepared.years_train.tolist())
                    | set(prepared.years_test.tolist())
                )
            )
        }
        walkforward_metrics: list[dict[str, Any]] = []
        if config.tabpfn.run_walkforward:
            walkforward = walkforward_evaluate_tabpfn(
                drift_model=model,
                test_rows=prepared.test_rows,
                top_k_events=list(prepared.feature_names),
                train_years=prepared.years_train,
                model_add_x_device=fit["model_add_x_device"],
                batch_size_predict=config.tabpfn.batch_size,
                example_add_shape=fit["example_add_shape"],
                use_cache=False,
            )
            domain_map = {
                int(year): int(index)
                for year, index in walkforward["year_to_domain_combined"].items()
            }
            walkforward_metrics = [
                _jsonable(row) for row in walkforward["results_per_year"]
            ]

        embedding_cache = workspace / "embeddings.npz"
        if config.use_cache and not force and embedding_cache.exists():
            with np.load(embedding_cache, allow_pickle=False) as cached:
                train_raw = np.asarray(cached["train_raw"])
                test_raw = np.asarray(cached["test_raw"])
        else:
            extraction = EmbeddingExtractConfig()
            extraction.batch_size = config.tabpfn.batch_size
            extraction.use_cache = False
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
            np.savez_compressed(
                embedding_cache, train_raw=train_raw, test_raw=test_raw
            )

        train_scaled, test_scaled, scaler = scale_embeddings(
            np.asarray(train_raw), np.asarray(test_raw), fit_test=False
        )
        if len(train_scaled) != len(prepared.X_train) or len(test_scaled) != len(
            prepared.X_test
        ):
            raise ValueError("TabPFN embedding rows do not align with prepared data")
        if not np.isfinite(train_scaled).all() or not np.isfinite(test_scaled).all():
            raise ValueError("TabPFN embeddings contain non-finite values")
        _write_json(workspace / "tabpfn_metrics.json", walkforward_metrics)
        return _EmbeddingData(
            model=model,
            model_device=fit["model_add_x_device"],
            example_add_shape=fit["example_add_shape"],
            year_to_domain=domain_map,
            train_raw=np.asarray(train_raw),
            test_raw=np.asarray(test_raw),
            train_scaled=np.asarray(train_scaled),
            test_scaled=np.asarray(test_scaled),
            scaler=scaler,
            fit_time_seconds=float(fit["fit_time_sec"]),
            walkforward_metrics=walkforward_metrics,
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

        from sae_compare import encode_sae_runs, train_all_saes

        model_cache = workspace / "sae_runs.pt"
        activation_cache = workspace / "activations.npz"
        if (
            config.use_cache
            and not force
            and model_cache.exists()
            and activation_cache.exists()
        ):
            runs = torch.load(model_cache, map_location="cpu")
            with np.load(activation_cache, allow_pickle=False) as cached:
                activations = {
                    int(name.removeprefix("run_")): np.asarray(cached[name])
                    for name in cached.files
                }
            _validate_activations(activations, runs, len(prepared.X_test))
            return _SAEData(runs=runs, activations=activations)

        fit_indices = splits["idx_semantic_fit"]
        matching_indices = splits["idx_semantic_select"]
        sae = config.sae
        runs = train_all_saes(
            num_models=len(sae.seeds),
            embs=embeddings.test_scaled[fit_indices],
            alpha=sae.alpha,
            scaling_factor=sae.scaling_factor,
            model_type=sae.model_type,
            k=sae.k,
            k_aux=sae.k_aux,
            universal_embs=embeddings.test_scaled[matching_indices],
            seeds=sae.seeds,
            epochs=sae.epochs,
            learning_rate=sae.learning_rate,
            weight_decay=sae.weight_decay,
            device=config.accelerator.device,
            encoding_batch_size=sae.encoding_batch_size,
        )
        activations = encode_sae_runs(
            runs,
            embeddings.test_scaled,
            device=config.accelerator.device,
            batch_size=sae.encoding_batch_size,
        )
        _validate_activations(activations, runs, len(prepared.X_test))
        torch.save(runs, model_cache)
        state_dir = workspace / "sae"
        state_dir.mkdir(parents=True, exist_ok=True)
        for run in runs:
            torch.save(run["model"].state_dict(), state_dir / f"run_{run['idx']}.pt")
        np.savez_compressed(
            activation_cache,
            **{f"run_{run_id}": matrix for run_id, matrix in activations.items()},
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
        from sae_compare import get_concepts_matching

        all_matches: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        run_pairs = _run_pairs(len(sae_data.runs), config.matching.scope)
        for left_index, right_index in run_pairs:
            frame = get_concepts_matching(
                sae_data.runs[left_index],
                sae_data.runs[right_index],
                pair_criteria=config.matching.criterion,
            )
            pair_rows = [_jsonable(row) for row in frame.to_dict(orient="records")]
            pair_rows.sort(
                key=lambda row: (
                    int(row["sae_i_idx"]),
                    int(row["sae_j_idx"]),
                    int(row["original_concept"]),
                )
            )
            all_matches.extend(pair_rows)
            filtered = [
                row
                for row in pair_rows
                if config.matching.minimum_score is None
                or float(row[config.matching.criterion])
                >= config.matching.minimum_score
            ]
            if config.matching.max_pairs_per_run_pair is not None:
                filtered = sorted(
                    filtered,
                    key=lambda row: (
                        -float(row[config.matching.criterion]),
                        int(row["original_concept"]),
                    ),
                )[: config.matching.max_pairs_per_run_pair]
                filtered.sort(key=lambda row: int(row["original_concept"]))
            selected.extend(filtered)
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

        from decision_tree import get_rules_forced, train_binary_trees
        from tcav import (
            get_model_gradients,
            get_tcav_scores,
            robust_tcav_significance_test,
            train_cavs_from_rules,
        )

        fit_indices = splits["idx_semantic_fit"]
        select_indices = splits["idx_semantic_select"]
        tcav_indices = splits["idx_tcav_eval"]
        factors_by_run = _matched_factors_by_run(matches)
        rule_rows: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        cavs_by_run: dict[int, dict[int, dict[str, Any]]] = {}

        for run_id, factor_ids in sorted(factors_by_run.items()):
            activations = sae_data.activations[run_id]
            rules_by_percentile = train_binary_trees(
                activations[fit_indices],
                prepared.X_test[fit_indices],
                list(prepared.feature_names),
                model_type=config.sae.model_type,
                max_depth=config.functional.tree_max_depth,
                factor_ids=sorted(factor_ids),
                min_positive_samples=config.functional.minimum_rule_samples,
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
                primary_frame = pd.DataFrame(
                    primary,
                    columns=[
                        "Factor",
                        "Rule",
                        "Class",
                        "Precision",
                        "Recall",
                        "Patients",
                        "Patients_concept",
                    ],
                )
                forced = get_rules_forced(
                    train_activations=activations[fit_indices],
                    X=prepared.X_test[fit_indices],
                    surviving_concepts=np.asarray(missing, dtype=int),
                    tree_rules_df=primary_frame,
                    perc=best_percentile,
                    feature_names=list(prepared.feature_names),
                    model_type=config.sae.model_type,
                    graph_output_dir=(
                        workspace / "decision_tree_graphs" / f"run_{run_id}"
                    ),
                )
                combined.extend(
                    {**dict(rule), "Provenance": "forced_fallback"}
                    for rule in forced
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
            cavs_by_run[run_id] = train_cavs_from_rules(
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
                high_quantile=1.0 - (float(best_percentile) / 100.0),
                min_pos_samples=config.functional.minimum_cav_samples,
                random_state=config.seed + run_id,
            )
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
                            rule.get("Provenance") if rule is not None else None
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
            [embeddings.year_to_domain[int(year)] for year in prepared.years_test[tcav_indices]],
            dtype=np.int64,
        )
        gradient_cache = workspace / "tcav_gradients.pkl"
        if (force or not config.use_cache) and gradient_cache.exists():
            gradient_cache.unlink()
        gradients = get_model_gradients(
            model=embeddings.model,
            X=prepared.X_test[tcav_indices],
            dist_vec=dist_eval,
            cache_file=gradient_cache,
            batch_size=config.functional.gradient_batch_size,
            device=config.accelerator.device,
        )
        functional: dict[str, dict[int, dict[str, Any]]] = {}
        cav_arrays: dict[str, np.ndarray] = {}
        diagnostic_lookup = {
            (int(row["run_id"]), int(row["factor_id"])): row
            for row in diagnostics
        }
        for run_id, cavs in sorted(cavs_by_run.items()):
            scores = get_tcav_scores(list(cavs.values()), gradients)
            for factor_id, cav in sorted(cavs.items()):
                score = float(scores[factor_id])
                entry: dict[str, Any] = {
                    "CAV": np.asarray(cav["CAV"]),
                    "TCAV_score": score,
                    "Rule": cav["Rule"],
                }
                diagnostic = diagnostic_lookup[(run_id, factor_id)]
                diagnostic["tcav_score"] = score
                if config.functional.significance_runs > 0:
                    try:
                        significance = robust_tcav_significance_test(
                            concept_idx=factor_id,
                            embs=embeddings.test_raw[select_indices],
                            idx_pos=np.asarray(cav["positive_idx"], dtype=int),
                            idx_neg=np.asarray(cav["negative_idx"], dtype=int),
                            model_grads=gradients,
                            scaler_emb=embeddings.scaler,
                            n_runs=config.functional.significance_runs,
                            sample_fraction=1.0,
                            rng_seed=config.seed + run_id * 10_000 + factor_id,
                        )
                        p_value = _finite_or_none(significance.get("p_value"))
                        diagnostic.update(
                            {
                                "tcav_p_value": p_value,
                                "tcav_t_stat": _finite_or_none(
                                    significance.get("t_stat")
                                ),
                                "tcav_effect_size": _finite_or_none(
                                    significance.get("cohens_d")
                                ),
                                "tcav_significant": (
                                    p_value is not None and p_value < 0.05
                                ),
                            }
                        )
                    except (ValueError, FloatingPointError) as error:
                        diagnostic["significance_error"] = str(error)
                functional.setdefault(str(run_id), {})[factor_id] = entry
                cav_arrays[f"run_{run_id}_factor_{factor_id}"] = np.asarray(
                    cav["CAV"]
                )

        _write_jsonl(workspace / "high_precision_rules.jsonl", rule_rows)
        _write_json(workspace / "functional.json", diagnostics)
        if cav_arrays:
            np.savez_compressed(workspace / "functional_cavs.npz", **cav_arrays)
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
            ),
        )
        clinical_path = semantic.clinical_groups_path
        if clinical_path is not None:
            clinical_path = str(_resolve_path(semantic_path.parent, clinical_path))
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
        )


def run_comparison(
    config: ComparisonRunnerConfig,
    *,
    force: bool = False,
    adapter: DefaultComparisonAdapter | None = None,
) -> dict[str, Any]:
    """Run the complete comparison and return a compact artifact summary."""

    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Renal Feather file not found: {dataset_path}")
    semantic_path = Path(config.semantic_config_path)
    if not semantic_path.is_file():
        raise FileNotFoundError(f"Semantic configuration not found: {semantic_path}")

    accelerator_info = accelerator_manifest(config.accelerator.device)
    print(
        "Accelerator: "
        f"requested={accelerator_info['requested_device']}, "
        f"resolved={accelerator_info['resolved_device']}"
    )
    dataset_hash = _file_fingerprint(dataset_path)
    semantic_dependencies = _semantic_dependency_fingerprints(semantic_path)
    source_hash = _runner_source_fingerprint()
    runner_hash = stable_hash(
        config.to_dict(),
        dataset_hash,
        semantic_dependencies,
        source_hash,
    )[:20]
    workspace = Path(config.artifact_dir) / runner_hash
    workspace.mkdir(parents=True, exist_ok=True)
    summary_path = workspace / "summary.json"
    if config.use_cache and not force and summary_path.exists():
        with summary_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        cached["cache_hit"] = True
        return cached

    telemetry = StageTelemetry(
        workspace / "stage_metrics.json",
        requested_device=config.accelerator.device,
    )
    active_adapter = adapter or DefaultComparisonAdapter()
    with telemetry.measure("prepare"):
        prepared = active_adapter.prepare(config, workspace, force=force)
    from semantic_splits import semantic_test_subsplits

    with telemetry.measure("split"):
        splits = semantic_test_subsplits(
            prepared.y_test, prepared.patient_ids, rng_seed=config.seed
        )
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
            prepared, config, workspace, force=force
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
            force=force,
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
            force=force,
        )
    from semantic_artifacts import environment_manifest

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
    }
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
        "cache_hit": False,
    }
    _write_json(summary_path, summary)
    return summary


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
            raise ValueError(f"SAE run {run_id} activations are not row-aligned")
        if not np.isfinite(value).all():
            raise ValueError(f"SAE run {run_id} activations contain non-finite values")


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
        "positive_fractions", [0.1, 0.25, 0.5]
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
        "clinical_groups": None,
    }
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
        "main-comparison.py",
        "runtime_acceleration.py",
        "database.py",
        "tabpfn_model.py",
        "sae.py",
        "sae_compare.py",
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
        writer.writerows([{key: _jsonable(row.get(key)) for key in fieldnames} for row in rows])


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
