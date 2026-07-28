"""Opt-in orchestration and CLI for cross-run SAE semantic comparison.

This module consumes raw tabular features plus activations from already-trained
SAE runs. It never changes legacy tree/CAV construction in ``main.py``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from semantic_artifacts import (
    SemanticArtifactStore,
    array_fingerprint,
    derive_seed,
    environment_manifest,
    stable_hash,
)
from comparison_cache import ComparisonCache
from semantic_compare import (
    RuleFamily,
    RuleSimilarityConfig,
    SemanticPairComparison,
    cluster_rule_families,
    compare_rule_sets_by_class,
    compare_rule_sets_symmetric,
    recurrent_representatives,
)
from semantic_config import SemanticExperimentConfig, load_clinical_groups
from semantic_rules import (
    ActivationTargetSpec,
    FittedActivationTarget,
    RuleSetSelection,
    RuleSetSelectionConfig,
    fit_activation_target,
    select_rule_set,
)
from semantic_splits import semantic_test_subsplits
from progress_utils import progress_iter
from stable_rule_backend import (
    CandidateRuleOccurrence,
    StableRuleBackendConfig,
    StableRuleDiscoveryResult,
    discover_stable_rule_candidates,
)


_SEMANTIC_SOURCE_FILES = (
    "semantic_experiment.py",
    "semantic_rules.py",
    "stable_rule_backend.py",
    "semantic_compare.py",
    "semantic_config.py",
    "semantic_splits.py",
    "semantic_artifacts.py",
    "comparison_cache.py",
)


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in _SEMANTIC_SOURCE_FILES:
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class FactorSemanticRepresentation:
    run_id: str
    factor_id: int
    target: FittedActivationTarget
    valid: bool
    reason: str | None
    families: tuple[RuleFamily, ...]
    selection: RuleSetSelection
    bootstrap_diagnostics: tuple[dict[str, Any], ...]
    candidate_diagnostics: tuple[dict[str, Any], ...]
    discovery_warnings: tuple[str, ...]
    n_candidate_occurrences: int
    n_recurrent_families: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "factor_id": self.factor_id,
            "target": self.target.to_dict(),
            "valid": self.valid,
            "reason": self.reason,
            "families": [family.to_dict() for family in self.families],
            "selection": {
                "rule_set": self.selection.rule_set.to_dict(),
                "metrics": self.selection.metrics.to_dict(),
                "feasible": self.selection.feasible,
                "reason": self.selection.reason,
                "search_method": self.selection.search_method,
                "n_candidates": self.selection.n_candidates,
                "n_evaluated_subsets": self.selection.n_evaluated_subsets,
            },
            "bootstrap_diagnostics": list(self.bootstrap_diagnostics),
            "candidate_diagnostics": list(self.candidate_diagnostics),
            "discovery_warnings": list(self.discovery_warnings),
            "n_candidate_occurrences": self.n_candidate_occurrences,
            "n_recurrent_families": self.n_recurrent_families,
        }


def _target_name(fraction: float) -> str:
    percentage = fraction * 100
    label = str(int(percentage)) if percentage.is_integer() else str(percentage).replace(".", "_")
    return f"top_{label}pct_positive"


def _selection_config(config: SemanticExperimentConfig) -> RuleSetSelectionConfig:
    objective = config.objective
    return RuleSetSelectionConfig(
        objective=objective.objective,
        min_precision=objective.min_precision,
        min_lift=objective.min_lift,
        max_rules=objective.max_rules,
        max_rule_length=objective.max_rule_length,
        min_marginal_recall=objective.min_marginal_recall,
        exhaustive_max_candidates=objective.exhaustive_candidate_limit,
        beam_width=objective.beam_width,
    )


def _backend_config(
    config: SemanticExperimentConfig,
    seed: int,
    *,
    progress_desc: str,
) -> StableRuleBackendConfig:
    discovery = config.discovery
    return StableRuleBackendConfig(
        n_bootstraps=discovery.n_bootstraps,
        trees_per_bootstrap=discovery.trees_per_bootstrap,
        max_depth=discovery.max_depth,
        min_samples_leaf=discovery.min_samples_leaf,
        max_features=discovery.max_features,
        splitter=discovery.splitter,
        positive_leaf_probability=discovery.positive_leaf_probability,
        min_positive_leaf_samples=discovery.min_positive_leaf_samples,
        bootstrap_unit="group",
        random_state=seed,
        show_progress=config.runtime.show_progress,
        progress_desc=progress_desc,
    )


def _cap_occurrences(
    occurrences: Sequence[CandidateRuleOccurrence], maximum_per_bootstrap: int
) -> tuple[CandidateRuleOccurrence, ...]:
    """Bound clustering cost using fitting/OOB diagnostics only."""

    grouped: dict[int, list[CandidateRuleOccurrence]] = {}
    for occurrence in occurrences:
        grouped.setdefault(occurrence.bootstrap_id, []).append(occurrence)
    retained: list[CandidateRuleOccurrence] = []
    for bootstrap_id in sorted(grouped):
        ranked = sorted(
            grouped[bootstrap_id],
            key=lambda item: (
                -item.oob_recall,
                -item.oob_precision,
                -item.fit_recall,
                -item.fit_precision,
                item.rule.length,
                item.rule.rule_id,
                item.tree_id,
                item.leaf_id,
            ),
        )
        unique: list[CandidateRuleOccurrence] = []
        seen_rule_ids: set[str] = set()
        for occurrence in ranked:
            if occurrence.rule.rule_id not in seen_rule_ids:
                unique.append(occurrence)
                seen_rule_ids.add(occurrence.rule.rule_id)
        retained.extend(unique[:maximum_per_bootstrap])
    return tuple(retained)


def _empty_representation(
    run_id: str,
    factor_id: int,
    target: FittedActivationTarget,
    reason: str,
    X_selection: np.ndarray,
    y_selection: np.ndarray,
) -> FactorSemanticRepresentation:
    selection = select_rule_set((), X_selection, y_selection, threshold_name=target.spec.name)
    return FactorSemanticRepresentation(
        run_id=run_id,
        factor_id=factor_id,
        target=target,
        valid=False,
        reason=reason,
        families=(),
        selection=selection,
        bootstrap_diagnostics=(),
        candidate_diagnostics=(),
        discovery_warnings=(),
        n_candidate_occurrences=0,
        n_recurrent_families=0,
    )


def _semantic_stage_source_fingerprint(*names: str) -> str:
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for name in names:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def _pickle_read(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _pickle_write(path: Path, value: Any) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _occurrence_signature(value: CandidateRuleOccurrence) -> dict[str, Any]:
    return {
        **asdict(value),
        "rule": value.rule.to_dict(),
    }


def _discovery_result_signature(
    value: StableRuleDiscoveryResult,
) -> dict[str, Any]:
    return {
        "occurrences": [
            _occurrence_signature(item) for item in value.occurrences
        ],
        "bootstrap_diagnostics": [
            asdict(item) for item in value.bootstrap_diagnostics
        ],
        "feature_names": value.feature_names,
        "n_samples": value.n_samples,
        "n_positive": value.n_positive,
        "bootstrap_unit": value.bootstrap_unit,
        "warnings": value.warnings,
    }


def _validate_bootstrap_result(value: Any) -> None:
    if not isinstance(value, StableRuleDiscoveryResult):
        raise ValueError("Cached semantic bootstrap has invalid type")
    bootstrap_ids = {
        item.bootstrap_id for item in value.bootstrap_diagnostics
    } | {item.bootstrap_id for item in value.occurrences}
    if len(bootstrap_ids) > 1:
        raise ValueError("Cached semantic bootstrap contains multiple IDs")


def _combine_bootstrap_results(
    values: Sequence[StableRuleDiscoveryResult],
    *,
    feature_names: Sequence[str],
    n_samples: int,
    n_positive: int,
) -> StableRuleDiscoveryResult:
    occurrences = tuple(
        item for value in values for item in value.occurrences
    )
    diagnostics = tuple(
        item for value in values for item in value.bootstrap_diagnostics
    )
    warnings: list[str] = []
    if any(
        "one_or_more_bootstraps_have_single_class" in value.warnings
        for value in values
    ):
        warnings.append("one_or_more_bootstraps_have_single_class")
    if not occurrences:
        warnings.append("no_valid_rule_candidates")
    units = {value.bootstrap_unit for value in values}
    if len(units) != 1:
        raise ValueError("Semantic bootstrap sampling units differ")
    return StableRuleDiscoveryResult(
        occurrences=occurrences,
        bootstrap_diagnostics=diagnostics,
        feature_names=tuple(feature_names),
        n_samples=n_samples,
        n_positive=n_positive,
        bootstrap_unit=next(iter(units)),
        warnings=tuple(warnings),
    )


def _annotate_occurrence_groups(
    occurrences: Sequence[CandidateRuleOccurrence],
    clinical_groups: Mapping[str, Sequence[str]],
) -> tuple[CandidateRuleOccurrence, ...]:
    annotated: list[CandidateRuleOccurrence] = []
    for occurrence in occurrences:
        conditions = []
        for condition in occurrence.rule.conditions:
            configured = clinical_groups.get(
                condition.feature_name,
                (f"feature:{condition.feature_name}",),
            )
            if isinstance(configured, str):
                configured = (configured,)
            conditions.append(
                replace(
                    condition,
                    clinical_groups=tuple(
                        sorted(set(str(value) for value in configured))
                    ),
                )
            )
        annotated.append(
            replace(
                occurrence,
                rule=replace(
                    occurrence.rule,
                    conditions=tuple(conditions),
                ),
            )
        )
    return tuple(annotated)


def _validate_family_result(value: Any) -> None:
    if not hasattr(value, "families") or not hasattr(
        value, "total_bootstraps"
    ):
        raise ValueError("Cached semantic families are invalid")


def _validate_selection_result(value: Any) -> None:
    if not isinstance(value, RuleSetSelection):
        raise ValueError("Cached semantic selection has invalid type")


def learn_factor_semantics(
    *,
    run_id: str | int,
    factor_id: int,
    activation_fraction: float,
    X_fit: np.ndarray,
    activations_fit: np.ndarray,
    patient_groups_fit: np.ndarray,
    X_selection: np.ndarray,
    activations_selection: np.ndarray,
    feature_names: Sequence[str],
    clinical_groups: Mapping[str, Sequence[str]],
    config: SemanticExperimentConfig,
    shared_cache: ComparisonCache | None = None,
) -> FactorSemanticRepresentation:
    """Fit one factor/threshold representation without access to final data."""

    run_name = str(run_id)
    spec = ActivationTargetSpec(_target_name(activation_fraction), activation_fraction)
    target = fit_activation_target(activations_fit, spec)
    selection_target = target.apply(activations_selection)
    if not target.valid:
        return _empty_representation(
            run_name, factor_id, target, "no_positive_fit_activations", X_selection, selection_target
        )
    fit_target = target.apply(activations_fit)
    if int(fit_target.sum()) < config.activation_targets.min_positive_samples:
        return _empty_representation(
            run_name, factor_id, target, "insufficient_high_activation_targets", X_selection, selection_target
        )
    seed = derive_seed(config.runtime.seed, "semantic", run_name, factor_id, spec.name)
    backend_config = _backend_config(
        config,
        seed,
        progress_desc=f"Rules {run_name}:{factor_id} {spec.name}",
    )
    if shared_cache is None:
        discovery = discover_stable_rule_candidates(
            X_fit,
            fit_target,
            feature_names,
            groups=patient_groups_fit,
            clinical_group_map=clinical_groups,
            config=backend_config,
        )
    else:
        bootstrap_results = []
        environment_values = environment_manifest()
        semantic_environment = stable_hash(
            {
                name: environment_values.get(name)
                for name in ("python", "numpy", "sklearn")
            }
        )
        backend_dependencies = {
            key: value
            for key, value in asdict(backend_config).items()
            if key
            not in {
                "n_bootstraps",
                "show_progress",
                "progress_desc",
            }
        }
        for bootstrap_id in range(config.discovery.n_bootstraps):
            def compute_bootstrap(current_id=bootstrap_id):
                return discover_stable_rule_candidates(
                    X_fit,
                    fit_target,
                    feature_names,
                    groups=patient_groups_fit,
                    # Group annotation belongs to family clustering so taxonomy
                    # edits do not repeat randomized-tree fitting.
                    clinical_group_map={},
                    config=backend_config,
                    bootstrap_ids=[current_id],
                )

            bootstrap_result = shared_cache.resolve(
                stage="semantic_bootstrap",
                item=(
                    f"run:{run_name}-factor:{factor_id}-"
                    f"target:{spec.name}-bootstrap:{bootstrap_id}"
                ),
                dependencies={
                    "X_fit": array_fingerprint(np.asarray(X_fit)),
                    "target_fit": array_fingerprint(
                        np.asarray(fit_target, dtype=bool)
                    ),
                    "patient_groups": array_fingerprint(
                        np.asarray(patient_groups_fit)
                    ),
                    "feature_names": tuple(feature_names),
                    "backend": backend_dependencies,
                    "bootstrap_id": bootstrap_id,
                },
                source_fingerprint=_semantic_stage_source_fingerprint(
                    "stable_rule_backend.py", "semantic_rules.py"
                ),
                environment_fingerprint=semantic_environment,
                load=lambda directory: _pickle_read(
                    directory / "bootstrap.pkl"
                ),
                compute=compute_bootstrap,
                store=lambda directory, value: _pickle_write(
                    directory / "bootstrap.pkl", value
                ),
                validate=_validate_bootstrap_result,
                fingerprint=lambda value: {
                    "bootstrap": stable_hash(
                        _discovery_result_signature(value)
                    )
                },
            )
            bootstrap_results.append(bootstrap_result.value)
        discovery = _combine_bootstrap_results(
            bootstrap_results,
            feature_names=feature_names,
            n_samples=len(X_fit),
            n_positive=int(fit_target.sum()),
        )
    occurrences = _cap_occurrences(
        discovery.occurrences, config.discovery.max_candidates_per_bootstrap
    )
    occurrences = _annotate_occurrence_groups(occurrences, clinical_groups)
    similarity_config = RuleSimilarityConfig(
        similarity_threshold=config.discovery.family_similarity_threshold,
        min_recurrence=config.discovery.min_family_recurrence,
    )

    def compute_families():
        return cluster_rule_families(
            occurrences,
            X_selection,
            similarity_config,
            total_bootstraps=config.discovery.n_bootstraps,
        )

    if shared_cache is None:
        clustering = compute_families()
    else:
        family_result = shared_cache.resolve(
            stage="semantic_families",
            item=f"run:{run_name}-factor:{factor_id}-target:{spec.name}",
            dependencies={
                "occurrences": stable_hash(
                    [_occurrence_signature(value) for value in occurrences]
                ),
                "X_selection": array_fingerprint(np.asarray(X_selection)),
                "clinical_groups": {
                    str(name): (
                        (str(values),)
                        if isinstance(values, str)
                        else tuple(str(value) for value in values)
                    )
                    for name, values in sorted(clinical_groups.items())
                },
                "similarity": asdict(similarity_config),
                "total_bootstraps": config.discovery.n_bootstraps,
            },
            source_fingerprint=_semantic_stage_source_fingerprint(
                "semantic_compare.py", "semantic_rules.py"
            ),
            environment_fingerprint=semantic_environment,
            load=lambda directory: _pickle_read(directory / "families.pkl"),
            compute=compute_families,
            store=lambda directory, value: _pickle_write(
                directory / "families.pkl", value
            ),
            validate=_validate_family_result,
            fingerprint=lambda value: {
                "families": stable_hash(
                    [family.to_dict() for family in value.families]
                )
            },
        )
        clustering = family_result.value
    families = clustering.families
    recurrent_rules = recurrent_representatives(
        families, config.discovery.min_family_recurrence
    )
    selection_config = _selection_config(config)

    def compute_selection():
        return select_rule_set(
            recurrent_rules,
            X_selection,
            selection_target,
            selection_config,
            threshold_name=spec.name,
        )

    if shared_cache is None:
        selection = compute_selection()
    else:
        selection_result = shared_cache.resolve(
            stage="semantic_selection",
            item=f"run:{run_name}-factor:{factor_id}-target:{spec.name}",
            dependencies={
                "recurrent_rules": [
                    rule.to_dict() for rule in recurrent_rules
                ],
                "X_selection": array_fingerprint(np.asarray(X_selection)),
                "selection_target": array_fingerprint(
                    np.asarray(selection_target, dtype=bool)
                ),
                "selection_config": asdict(selection_config),
                "threshold_name": spec.name,
            },
            source_fingerprint=_semantic_stage_source_fingerprint(
                "semantic_rules.py"
            ),
            environment_fingerprint=semantic_environment,
            load=lambda directory: _pickle_read(
                directory / "selection.pkl"
            ),
            compute=compute_selection,
            store=lambda directory, value: _pickle_write(
                directory / "selection.pkl", value
            ),
            validate=_validate_selection_result,
            fingerprint=lambda value: {
                "selection": stable_hash(
                    value.rule_set.to_dict(),
                    value.metrics.to_dict(),
                    value.feasible,
                    value.reason,
                )
            },
        )
        selection = selection_result.value
    recurrent_count = sum(
        family.recurrence_frequency + 1e-15 >= config.discovery.min_family_recurrence
        for family in families
    )
    reason = None if selection.feasible else selection.reason
    return FactorSemanticRepresentation(
        run_id=run_name,
        factor_id=factor_id,
        target=target,
        valid=selection.feasible,
        reason=reason,
        families=families,
        selection=selection,
        bootstrap_diagnostics=tuple(
            {
                "bootstrap_id": item.bootstrap_id,
                "seed": item.seed,
                "fit_sample_count": item.fit_sample_count,
                "fit_unique_sample_count": item.fit_unique_sample_count,
                "oob_sample_count": item.oob_sample_count,
                "positive_fit_count": item.positive_fit_count,
                "candidates_extracted": item.candidates_extracted,
            }
            for item in discovery.bootstrap_diagnostics
        ),
        candidate_diagnostics=tuple(
            {
                "rule_id": item.rule.rule_id,
                "bootstrap_id": item.bootstrap_id,
                "tree_id": item.tree_id,
                "bootstrap_seed": item.bootstrap_seed,
                "tree_seed": item.tree_seed,
                "leaf_id": item.leaf_id,
                "fit_sample_count": item.fit_sample_count,
                "fit_positive_count": item.fit_positive_count,
                "fit_selected_count": item.fit_selected_count,
                "fit_true_positive_count": item.fit_true_positive_count,
                "fit_precision": item.fit_precision,
                "fit_recall": item.fit_recall,
                "oob_sample_count": item.oob_sample_count,
                "oob_selected_count": item.oob_selected_count,
                "oob_true_positive_count": item.oob_true_positive_count,
                "oob_precision": item.oob_precision,
                "oob_recall": item.oob_recall,
            }
            for item in occurrences
        ),
        discovery_warnings=discovery.warnings,
        n_candidate_occurrences=len(occurrences),
        n_recurrent_families=int(recurrent_count),
    )


def _activation_matrix(activations: Mapping[Any, np.ndarray], run_id: Any) -> np.ndarray:
    if run_id in activations:
        return np.asarray(activations[run_id])
    run_string = str(run_id)
    for key, value in activations.items():
        if str(key) == run_string:
            return np.asarray(value)
    raise KeyError(f"Missing activations for SAE run {run_id!r}")


def _pair_fields(pair: Mapping[str, Any]) -> tuple[Any, Any, int, int]:
    aliases = (
        ("sae_i_idx", "sae_j_idx", "original_concept", "best_pair"),
        ("run_i", "run_j", "factor_i", "factor_j"),
    )
    for run_i, run_j, factor_i, factor_j in aliases:
        if all(name in pair for name in (run_i, run_j, factor_i, factor_j)):
            return pair[run_i], pair[run_j], int(pair[factor_i]), int(pair[factor_j])
    raise ValueError("Matching rows need legacy or semantic run/factor fields")


def _functional_entry(
    values: Mapping[Any, Any] | None, run_id: str, factor_id: int
) -> Mapping[str, Any] | None:
    if not values:
        return None
    for key in ((run_id, factor_id), f"{run_id}:{factor_id}"):
        if key in values:
            entry = values[key]
            return entry if isinstance(entry, Mapping) else None
    run_values = values.get(run_id)
    if isinstance(run_values, Mapping):
        entry = run_values.get(factor_id, run_values.get(str(factor_id)))
        return entry if isinstance(entry, Mapping) else None
    return None


def _functional_pair_summary(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    neutral_band: float = 0.10,
) -> dict[str, Any] | None:
    if left is None or right is None:
        return None
    result: dict[str, Any] = {}
    cav_left, cav_right = left.get("CAV", left.get("cav")), right.get("CAV", right.get("cav"))
    if cav_left is not None and cav_right is not None:
        first, second = np.asarray(cav_left, dtype=float), np.asarray(cav_right, dtype=float)
        if first.shape == second.shape and first.ndim == 1:
            denominator = np.linalg.norm(first) * np.linalg.norm(second)
            result["cav_cosine"] = float(np.dot(first, second) / denominator) if denominator else 0.0
    score_left = left.get("TCAV_score", left.get("tcav_score"))
    score_right = right.get("TCAV_score", right.get("tcav_score"))
    if score_left is not None and score_right is not None:
        score_left, score_right = float(score_left), float(score_right)

        def sign(score: float) -> int:
            return 1 if score > 0.5 + neutral_band else -1 if score < 0.5 - neutral_band else 0

        result.update({
            "tcav_i": score_left,
            "tcav_j": score_right,
            "tcav_abs_difference": abs(score_left - score_right),
            "tcav_effect_sign_i": sign(score_left),
            "tcav_effect_sign_j": sign(score_right),
            "tcav_effect_sign_agreement": sign(score_left) == sign(score_right),
        })
    return result or None


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stringify_mapping_keys(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stringify_mapping_keys(item) for item in value]
    return value


def _set_stability(models: Sequence[FactorSemanticRepresentation], extractor) -> float:
    sets = [extractor(model) for model in models]
    if len(sets) < 2:
        return 1.0
    scores = []
    for left_index in range(len(sets)):
        for right_index in range(left_index + 1, len(sets)):
            union = sets[left_index] | sets[right_index]
            scores.append(len(sets[left_index] & sets[right_index]) / len(union) if union else 1.0)
    return float(min(scores))


def _rule_set_features(model: FactorSemanticRepresentation) -> set[str]:
    return {
        condition.feature_name
        for rule in model.selection.rule_set.rules
        for condition in rule.conditions
    }


def _rule_set_groups(model: FactorSemanticRepresentation) -> set[str]:
    return {
        group
        for rule in model.selection.rule_set.rules
        for condition in rule.conditions
        for group in condition.clinical_groups
    }


def _transfer_row(transfer: SemanticPairComparison) -> dict[str, Any]:
    """Preserve the existing flattened compatibility view of pair transfer."""

    return {
        **transfer.to_dict(),
        "i_to_j": transfer.i_to_j.to_dict(),
        "j_to_i": transfer.j_to_i.to_dict(),
        "mean": transfer.mean,
        "min": transfer.minimum,
        "target_cohort_jaccard": transfer.target_cohort_jaccard,
        "exact_feature_jaccard": transfer.exact_feature_agreement.jaccard,
        "exact_feature_equal": transfer.exact_feature_equal,
        "clinical_group_jaccard": transfer.clinical_group_agreement.jaccard,
        "clinical_group_equal": transfer.clinical_group_equal,
    }


def _class_threshold_stability(
    threshold_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Summarize held-out transfer across thresholds for each outcome class."""

    by_class: dict[str, list[float]] = {}
    for threshold in threshold_rows:
        for class_result in threshold.get("class_analysis", []):
            key = str(class_result["class_value"])
            by_class.setdefault(key, []).append(
                float(class_result["transfer"]["mean"]["f2"])
            )
    return {
        class_value: {
            "transfer_f2_mean_min": min(values),
            "transfer_f2_mean_max": max(values),
            "transfer_f2_mean_range": max(values) - min(values),
        }
        for class_value, values in sorted(by_class.items())
    }


def run_semantic_comparison(
    *,
    X: np.ndarray,
    outcome_for_stratification: np.ndarray,
    patient_ids: np.ndarray,
    feature_names: Sequence[str],
    activations_by_run: Mapping[Any, np.ndarray],
    matchings: Iterable[Mapping[str, Any]],
    config: SemanticExperimentConfig,
    clinical_groups: Mapping[str, Sequence[str]] | None = None,
    functional_by_factor: Mapping[Any, Any] | None = None,
    record_keys: Sequence[Any] | None = None,
    force: bool = False,
    shared_cache: ComparisonCache | None = None,
) -> dict[str, Any]:
    """Learn per-factor semantics, evaluate matched pairs, persist artifacts."""

    if shared_cache is None:
        shared_cache = ComparisonCache(
            Path(config.runtime.artifact_dir) / "_cache" / "v2",
            enabled=config.runtime.cache,
            forced_stages=("semantic",) if force else (),
        )
    cache_event_start = len(shared_cache.events)
    X = np.asarray(X, dtype=float)
    y_stratify = np.asarray(outcome_for_stratification)
    patients = np.asarray(patient_ids)
    names = tuple(str(name) for name in feature_names)
    pairs = [dict(pair) for pair in matchings]
    if X.ndim != 2 or X.shape[1] != len(names):
        raise ValueError("X and feature_names are inconsistent")
    if len(X) != len(y_stratify) or len(X) != len(patients):
        raise ValueError("X, outcome, and patient IDs must be row-aligned")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    for run_id, matrix in activations_by_run.items():
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or len(matrix) != len(X):
            raise ValueError(f"Activations for run {run_id!r} are not aligned with X")

    group_map = dict(clinical_groups or {})
    splits = semantic_test_subsplits(y_stratify, patients, rng_seed=config.runtime.seed)
    split_fingerprint = stable_hash({name: indices.tolist() for name, indices in splits.items() if name.startswith("idx_semantic")})
    activation_fingerprints = {
        str(run): array_fingerprint(np.asarray(matrix))
        for run, matrix in sorted(activations_by_run.items(), key=lambda item: str(item[0]))
    }
    source_fingerprint = _source_fingerprint()
    hash_config = config.to_dict()
    for name in ("artifact_dir", "cache", "show_progress", "n_jobs"):
        hash_config["runtime"].pop(name, None)
    experiment_hash = stable_hash(
        hash_config,
        array_fingerprint(X),
        array_fingerprint(y_stratify),
        array_fingerprint(patients),
        activation_fingerprints,
        split_fingerprint,
        group_map,
        _stringify_mapping_keys(functional_by_factor),
        pairs,
        names,
        list(record_keys) if record_keys is not None else None,
        source_fingerprint,
    )[:20]
    store = SemanticArtifactStore(config.runtime.artifact_dir, experiment_hash)
    if config.runtime.cache and not force and store.exists("result.json"):
        cached = store.read_json("result.json")
        cached["cache_hit"] = True
        return cached

    fit_indices = splits["idx_semantic_fit"]
    selection_indices = splits["idx_semantic_select"]
    final_indices = splits["idx_semantic_final"]
    final_outcomes = y_stratify[final_indices]
    required: set[tuple[str, int]] = set()
    for pair in pairs:
        run_i, run_j, factor_i, factor_j = _pair_fields(pair)
        required.add((str(run_i), factor_i))
        required.add((str(run_j), factor_j))

    models: dict[tuple[str, int, float], FactorSemanticRepresentation] = {}
    model_tasks = [
        (run_id, factor_id, fraction)
        for run_id, factor_id in sorted(required)
        for fraction in config.activation_targets.positive_fractions
    ]
    for run_id, factor_id, fraction in progress_iter(
        model_tasks,
        enabled=config.runtime.show_progress,
        desc="Learning factor semantics",
        total=len(model_tasks),
        unit="target",
    ):
        activations = _activation_matrix(activations_by_run, run_id)
        if not 0 <= factor_id < activations.shape[1]:
            raise IndexError(f"Factor {factor_id} outside activations for run {run_id}")
        models[(run_id, factor_id, fraction)] = learn_factor_semantics(
            run_id=run_id,
            factor_id=factor_id,
            activation_fraction=fraction,
            X_fit=X[fit_indices],
            activations_fit=activations[fit_indices, factor_id],
            patient_groups_fit=patients[fit_indices],
            X_selection=X[selection_indices],
            activations_selection=activations[selection_indices, factor_id],
            feature_names=names,
            clinical_groups=group_map,
            config=config,
            shared_cache=shared_cache,
        )

    pair_results: list[dict[str, Any]] = []
    pair_iter = progress_iter(
        enumerate(pairs),
        enabled=config.runtime.show_progress,
        desc="Evaluating semantic pairs",
        total=len(pairs),
        unit="pair",
    )
    for pair_index, pair in pair_iter:
        run_i, run_j, factor_i, factor_j = _pair_fields(pair)
        run_i, run_j = str(run_i), str(run_j)
        activation_i = _activation_matrix(activations_by_run, run_i)
        activation_j = _activation_matrix(activations_by_run, run_j)
        threshold_rows: list[dict[str, Any]] = []
        models_i: list[FactorSemanticRepresentation] = []
        models_j: list[FactorSemanticRepresentation] = []
        for fraction in config.activation_targets.positive_fractions:
            model_i = models[(run_i, factor_i, fraction)]
            model_j = models[(run_j, factor_j, fraction)]
            models_i.append(model_i)
            models_j.append(model_j)
            target_i = model_i.target.apply(activation_i[final_indices, factor_i])
            target_j = model_j.target.apply(activation_j[final_indices, factor_j])
            transfer = compare_rule_sets_symmetric(
                model_i.selection.rule_set,
                target_i,
                model_j.selection.rule_set,
                target_j,
                X[final_indices],
                factor_i_id=f"{run_i}:{factor_i}",
                factor_j_id=f"{run_j}:{factor_j}",
                threshold_name=model_i.target.spec.name,
            )
            threshold_row = {
                "threshold_name": model_i.target.spec.name,
                "positive_fraction": fraction,
                "cutoff_i": model_i.target.cutoff if model_i.target.valid else None,
                "cutoff_j": model_j.target.cutoff if model_j.target.valid else None,
                "model_i_valid": model_i.valid,
                "model_j_valid": model_j.valid,
                "model_i_reason": model_i.reason,
                "model_j_reason": model_j.reason,
                "transfer": _transfer_row(transfer),
            }
            if config.class_analysis.enabled:
                class_comparisons = compare_rule_sets_by_class(
                    model_i.selection.rule_set,
                    target_i,
                    model_j.selection.rule_set,
                    target_j,
                    X[final_indices],
                    final_outcomes,
                    factor_i_id=f"{run_i}:{factor_i}",
                    factor_j_id=f"{run_j}:{factor_j}",
                    threshold_name=model_i.target.spec.name,
                )
                threshold_row["class_analysis"] = [
                    {
                        "class_value": class_result.class_value,
                        "n_samples": class_result.n_samples,
                        "n_positive_i": class_result.left_target_positive_count,
                        "n_positive_j": class_result.right_target_positive_count,
                        "valid": class_result.valid,
                        "reasons": list(class_result.reasons),
                        "transfer": _transfer_row(class_result.comparison),
                    }
                    for class_result in class_comparisons
                ]
            threshold_rows.append(threshold_row)
        f2_values = [row["transfer"]["mean"]["f2"] for row in threshold_rows]
        result = {
            "pair_id": pair_index,
            "run_i": run_i,
            "run_j": run_j,
            "factor_i": factor_i,
            "factor_j": factor_j,
            "geometry": {
                key: value for key, value in pair.items()
                if key not in {"sae_i_idx", "sae_j_idx", "original_concept", "best_pair", "run_i", "run_j", "factor_i", "factor_j"}
            },
            "functional": _functional_pair_summary(
                _functional_entry(functional_by_factor, run_i, factor_i),
                _functional_entry(functional_by_factor, run_j, factor_j),
            ),
            "thresholds": threshold_rows,
            "threshold_stability": {
                "transfer_f2_mean_min": min(f2_values) if f2_values else 0.0,
                "transfer_f2_mean_max": max(f2_values) if f2_values else 0.0,
                "transfer_f2_mean_range": max(f2_values) - min(f2_values) if f2_values else 0.0,
                "factor_i_feature_jaccard_min": _set_stability(models_i, _rule_set_features),
                "factor_j_feature_jaccard_min": _set_stability(models_j, _rule_set_features),
                "factor_i_clinical_group_jaccard_min": _set_stability(models_i, _rule_set_groups),
                "factor_j_clinical_group_jaccard_min": _set_stability(models_j, _rule_set_groups),
            },
        }
        if config.class_analysis.enabled:
            result["class_threshold_stability"] = _class_threshold_stability(
                threshold_rows
            )
        pair_results.append(result)

    if record_keys is None:
        record_keys = [f"row:{index}|patient:{patients[index]}" for index in range(len(X))]
    if len(record_keys) != len(X):
        raise ValueError("record_keys must contain one key per row")
    record_hashes = np.asarray(
        [hashlib.sha256(str(key).encode()).hexdigest() for key in record_keys], dtype="U64"
    )
    manifest = {
        "schema_version": config.schema_version,
        "experiment_hash": experiment_hash,
        "config": config.to_dict(),
        "environment": environment_manifest(),
        "source_fingerprint": source_fingerprint,
        "data_fingerprint": array_fingerprint(X),
        "outcome_fingerprint": array_fingerprint(y_stratify),
        "patient_group_fingerprint": array_fingerprint(patients),
        "activation_fingerprints": activation_fingerprints,
        "split_fingerprint": split_fingerprint,
        "clinical_groups_hash": stable_hash(group_map),
        "clinical_group_mapping": {
            "mapped_features": sum(name in group_map for name in names),
            "total_features": len(names),
            "coverage": (sum(name in group_map for name in names) / len(names)) if names else 1.0,
            "unmapped_features": [name for name in names if name not in group_map],
        },
        "n_records": len(X),
        "n_pairs": len(pairs),
        "stage_cache": {
            "root": str(shared_cache.root),
            "hits": sum(
                event.status == "hit"
                for event in shared_cache.events[cache_event_start:]
            ),
            "misses": sum(
                event.status == "miss"
                for event in shared_cache.events[cache_event_start:]
            ),
            "forced": sum(
                event.status == "forced"
                for event in shared_cache.events[cache_event_start:]
            ),
            "disabled": sum(
                event.status == "disabled"
                for event in shared_cache.events[cache_event_start:]
            ),
        },
    }
    if config.class_analysis.enabled:
        class_values, class_counts = np.unique(final_outcomes, return_counts=True)
        manifest["class_analysis"] = {
            "enabled": True,
            "class_values": [
                value.item() if isinstance(value, np.generic) else value
                for value in class_values
            ],
            "class_support": {
                str(value.item() if isinstance(value, np.generic) else value): int(count)
                for value, count in zip(class_values, class_counts)
            },
        }
    store.write_json("manifest.json", manifest)
    shared_cache.write_refs(store.root / "cache_refs.json")
    store.write_npz(
        "splits.npz",
        record_hashes=record_hashes,
        **{name: value for name, value in splits.items() if name.startswith("idx_semantic")},
    )
    model_rows = [
        models[key].to_dict()
        for key in sorted(models, key=lambda item: (item[0], item[1], item[2]))
    ]
    store.write_jsonl("semantic_rules.jsonl", model_rows)
    store.write_jsonl("pair_results.jsonl", pair_results)
    _write_flat_pair_csv(store.root / "pair_metrics.csv", pair_results)
    if config.class_analysis.enabled:
        _write_flat_class_pair_csv(
            store.root / "pair_metrics_by_class.csv", pair_results
        )
    result = {
        "schema_version": config.schema_version,
        "experiment_hash": experiment_hash,
        "artifact_dir": str(store.root),
        "cache_hit": False,
        "manifest": manifest,
        "semantic_models": model_rows,
        "pair_results": pair_results,
    }
    store.write_json("result.json", result)
    return result


def _write_flat_pair_csv(path: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for threshold in pair["thresholds"]:
            transfer = threshold["transfer"]
            rows.append({
                "pair_id": pair["pair_id"],
                "run_i": pair["run_i"],
                "run_j": pair["run_j"],
                "factor_i": pair["factor_i"],
                "factor_j": pair["factor_j"],
                "threshold_name": threshold["threshold_name"],
                "positive_fraction": threshold["positive_fraction"],
                "cos_sim": pair["geometry"].get("cos_sim"),
                "activation_overlap": pair["geometry"].get("overlap"),
                **{f"i_to_j_{key}": value for key, value in transfer["i_to_j"].items()},
                **{f"j_to_i_{key}": value for key, value in transfer["j_to_i"].items()},
                **{f"transfer_mean_{key}": value for key, value in transfer["mean"].items()},
                **{f"transfer_min_{key}": value for key, value in transfer["min"].items()},
                "target_cohort_jaccard": transfer["target_cohort_jaccard"],
                "selected_cohort_jaccard": transfer["selected_cohort_jaccard"],
                "exact_feature_jaccard": transfer["exact_feature_jaccard"],
                "clinical_group_jaccard": transfer["clinical_group_jaccard"],
            })
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_flat_class_pair_csv(
    path: Path, pairs: Sequence[Mapping[str, Any]]
) -> None:
    """Write class-stratified metrics without changing the pooled CSV schema."""

    rows: list[dict[str, Any]] = []
    for pair in pairs:
        for threshold in pair["thresholds"]:
            for class_result in threshold.get("class_analysis", []):
                transfer = class_result["transfer"]
                rows.append({
                    "pair_id": pair["pair_id"],
                    "run_i": pair["run_i"],
                    "run_j": pair["run_j"],
                    "factor_i": pair["factor_i"],
                    "factor_j": pair["factor_j"],
                    "threshold_name": threshold["threshold_name"],
                    "positive_fraction": threshold["positive_fraction"],
                    "class_value": class_result["class_value"],
                    "class_n_samples": class_result["n_samples"],
                    "class_n_positive_i": class_result["n_positive_i"],
                    "class_n_positive_j": class_result["n_positive_j"],
                    "class_valid": class_result["valid"],
                    "class_reasons": "|".join(class_result["reasons"]),
                    "cos_sim": pair["geometry"].get("cos_sim"),
                    "activation_overlap": pair["geometry"].get("overlap"),
                    **{
                        f"i_to_j_{key}": value
                        for key, value in transfer["i_to_j"].items()
                    },
                    **{
                        f"j_to_i_{key}": value
                        for key, value in transfer["j_to_i"].items()
                    },
                    **{
                        f"transfer_mean_{key}": value
                        for key, value in transfer["mean"].items()
                    },
                    **{
                        f"transfer_min_{key}": value
                        for key, value in transfer["min"].items()
                    },
                    "target_cohort_jaccard": transfer["target_cohort_jaccard"],
                    "selected_cohort_jaccard": transfer["selected_cohort_jaccard"],
                    "exact_feature_jaccard": transfer["exact_feature_jaccard"],
                    "clinical_group_jaccard": transfer["clinical_group_jaccard"],
                })
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_bundle(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], dict[str, np.ndarray], np.ndarray | None]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {"X", "outcome", "patient_ids", "feature_names"}
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"Bundle missing arrays: {sorted(missing)}")
        activations = {
            key.removeprefix("activations_run_"): np.asarray(bundle[key])
            for key in bundle.files if key.startswith("activations_run_")
        }
        if not activations:
            raise ValueError("Bundle needs at least one activations_run_<id> array")
        record_keys = np.asarray(bundle["record_keys"]) if "record_keys" in bundle.files else None
        return (
            np.asarray(bundle["X"]),
            np.asarray(bundle["outcome"]),
            np.asarray(bundle["patient_ids"]),
            tuple(str(value) for value in bundle["feature_names"]),
            activations,
            record_keys,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Semantic experiment JSON")
    parser.add_argument("--bundle", required=True, help="NPZ with X/outcome/patients/activations")
    parser.add_argument("--matches", required=True, help="JSON list of one-to-one matched factors")
    parser.add_argument("--force", action="store_true", help="Ignore matching result cache")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = SemanticExperimentConfig.from_json(config_path)
    if args.no_progress:
        config = replace(
            config,
            runtime=replace(config.runtime, show_progress=False),
        )
    clinical_path = config.clinical_groups_path
    if clinical_path is not None and not Path(clinical_path).is_absolute():
        clinical_path = str(config_path.parent / clinical_path)
    clinical_groups = load_clinical_groups(clinical_path)
    X, outcome, patients, names, activations, record_keys = _load_bundle(args.bundle)
    with Path(args.matches).open(encoding="utf-8") as handle:
        matches = json.load(handle)
    if not isinstance(matches, list):
        raise ValueError("matches JSON must contain a list")
    result = run_semantic_comparison(
        X=X,
        outcome_for_stratification=outcome,
        patient_ids=patients,
        feature_names=names,
        activations_by_run=activations,
        matchings=matches,
        config=config,
        clinical_groups=clinical_groups,
        record_keys=record_keys,
        force=args.force,
    )
    print(result["artifact_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
