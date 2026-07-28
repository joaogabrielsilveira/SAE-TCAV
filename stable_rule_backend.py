"""Deterministic candidate-rule discovery with randomized decision trees.

This module deliberately stops at candidate discovery.  Rule-family clustering,
recurrence filtering, and selection of the final OR-of-rules model belong in
``semantic_rules``.  Keeping that boundary makes the tree implementation an
optional backend rather than part of the semantic representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
if TYPE_CHECKING:
    from sklearn.tree import DecisionTreeClassifier

from semantic_rules import Condition, Rule
from progress_utils import progress_iter


BootstrapUnit = Literal["auto", "row", "group"]
MaxFeatures = int | float | Literal["sqrt", "log2"] | None


@dataclass(frozen=True)
class StableRuleBackendConfig:
    """Controls randomized-tree candidate generation.

    An outer bootstrap defines recurrence.  Multiple randomized trees inside
    each bootstrap increase candidate coverage without inflating recurrence:
    downstream code must count a family at most once per ``bootstrap_id``.
    """

    n_bootstraps: int = 30
    trees_per_bootstrap: int = 50
    max_depth: int = 3
    min_samples_leaf: int | float = 0.01
    max_features: MaxFeatures = "sqrt"
    splitter: Literal["best", "random"] = "random"
    class_weight: Literal["balanced"] | dict[int, float] | None = "balanced"
    positive_leaf_probability: float = 0.5
    min_positive_leaf_samples: int = 2
    bootstrap_unit: BootstrapUnit = "auto"
    random_state: int = 42
    show_progress: bool = False
    progress_desc: str = "Rule bootstraps"

    def __post_init__(self) -> None:
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be >= 1")
        if self.trees_per_bootstrap < 1:
            raise ValueError("trees_per_bootstrap must be >= 1")
        if self.max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        if isinstance(self.min_samples_leaf, int) and self.min_samples_leaf < 1:
            raise ValueError("integer min_samples_leaf must be >= 1")
        if isinstance(self.min_samples_leaf, float) and not 0 < self.min_samples_leaf <= 0.5:
            raise ValueError("float min_samples_leaf must be in (0, 0.5]")
        if not 0 <= self.positive_leaf_probability <= 1:
            raise ValueError("positive_leaf_probability must be in [0, 1]")
        if self.min_positive_leaf_samples < 1:
            raise ValueError("min_positive_leaf_samples must be >= 1")
        if not isinstance(self.show_progress, bool):
            raise ValueError("show_progress must be a boolean")


@dataclass(frozen=True)
class CandidateRuleOccurrence:
    """One rule occurrence from one tree and outer bootstrap."""

    rule: Rule
    source: str
    bootstrap_id: int
    tree_id: int
    bootstrap_seed: int
    tree_seed: int
    leaf_id: int
    fit_sample_count: int
    fit_positive_count: int
    fit_selected_count: int
    fit_true_positive_count: int
    fit_precision: float
    fit_recall: float
    oob_sample_count: int
    oob_selected_count: int
    oob_true_positive_count: int
    oob_precision: float
    oob_recall: float


@dataclass(frozen=True)
class BootstrapDiagnostic:
    bootstrap_id: int
    seed: int
    fit_sample_count: int
    fit_unique_sample_count: int
    oob_sample_count: int
    positive_fit_count: int
    candidates_extracted: int


@dataclass(frozen=True)
class StableRuleDiscoveryResult:
    """Candidate occurrences plus audit information for one target."""

    occurrences: tuple[CandidateRuleOccurrence, ...]
    bootstrap_diagnostics: tuple[BootstrapDiagnostic, ...]
    feature_names: tuple[str, ...]
    n_samples: int
    n_positive: int
    bootstrap_unit: Literal["row", "group"]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def candidates(self) -> tuple[Rule, ...]:
        """Unique candidates, preserving deterministic discovery order."""

        seen: set[Rule] = set()
        unique: list[Rule] = []
        for occurrence in self.occurrences:
            if occurrence.rule not in seen:
                seen.add(occurrence.rule)
                unique.append(occurrence.rule)
        return tuple(unique)


class RandomizedTreeRuleBackend:
    """Generate structured AND-rule candidates from positive tree leaves."""

    def __init__(self, config: StableRuleBackendConfig | None = None) -> None:
        self.config = config or StableRuleBackendConfig()

    def discover(
        self,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: Sequence[str],
        *,
        groups: ArrayLike | None = None,
        clinical_group_map: Mapping[str, Sequence[str]] | None = None,
        bootstrap_ids: Sequence[int] | None = None,
    ) -> StableRuleDiscoveryResult:
        """Discover rules using fitting data only.

        ``groups`` should identify patients or other sampling units.  When
        present and ``bootstrap_unit='auto'``, complete groups are sampled with
        replacement.  ECDF ranks use this same fitting matrix, never validation
        or final comparison records.
        """

        try:
            from sklearn.tree import DecisionTreeClassifier
        except ImportError as error:
            raise ImportError(
                "Stable rule discovery requires scikit-learn; install requirements-semantic.txt"
            ) from error

        X_array, y_array, names, group_array, unit = _validate_inputs(
            X, y, feature_names, groups, self.config.bootstrap_unit
        )
        condition_groups = {
            name: _clinical_groups_for(name, clinical_group_map or {}) for name in names
        }
        n_samples = X_array.shape[0]
        n_positive = int(y_array.sum())
        warnings: list[str] = []
        if n_positive == 0 or n_positive == n_samples:
            warnings.append("target_has_single_class")
            return StableRuleDiscoveryResult(
                occurrences=(),
                bootstrap_diagnostics=(),
                feature_names=names,
                n_samples=n_samples,
                n_positive=n_positive,
                bootstrap_unit=unit,
                warnings=tuple(warnings),
            )

        sorted_features = tuple(
            np.sort(X_array[:, index]) for index in range(X_array.shape[1])
        )
        selected_bootstrap_ids = (
            tuple(range(self.config.n_bootstraps))
            if bootstrap_ids is None
            else tuple(int(value) for value in bootstrap_ids)
        )
        if not selected_bootstrap_ids or any(
            value < 0 or value >= self.config.n_bootstraps
            for value in selected_bootstrap_ids
        ):
            raise ValueError(
                "bootstrap_ids must select configured bootstrap indices"
            )
        if len(set(selected_bootstrap_ids)) != len(selected_bootstrap_ids):
            raise ValueError("bootstrap_ids must be unique")
        seed_sequence = np.random.SeedSequence(self.config.random_state)
        all_bootstrap_sequences = seed_sequence.spawn(self.config.n_bootstraps)
        bootstrap_sequences = [
            (bootstrap_id, all_bootstrap_sequences[bootstrap_id])
            for bootstrap_id in selected_bootstrap_ids
        ]
        occurrences: list[CandidateRuleOccurrence] = []
        diagnostics: list[BootstrapDiagnostic] = []
        single_class_bootstraps = 0

        bootstrap_iter = progress_iter(
            bootstrap_sequences,
            enabled=self.config.show_progress,
            desc=self.config.progress_desc,
            total=len(bootstrap_sequences),
            unit="bootstrap",
            leave=False,
        )
        for bootstrap_id, bootstrap_sequence in bootstrap_iter:
            child_sequences = bootstrap_sequence.spawn(self.config.trees_per_bootstrap + 1)
            bootstrap_seed = _seed_from_sequence(child_sequences[0])
            fit_indices, oob_indices = _bootstrap_indices(
                n_samples=n_samples,
                groups=group_array if unit == "group" else None,
                seed=bootstrap_seed,
            )
            X_fit = X_array[fit_indices]
            y_fit = y_array[fit_indices]
            bootstrap_occurrence_count = 0

            # A rare target can disappear from a row bootstrap.  Record it and
            # continue; treating it as a failed rule would bias recurrence.
            if np.unique(y_fit).size < 2:
                single_class_bootstraps += 1
                diagnostics.append(
                    BootstrapDiagnostic(
                        bootstrap_id=bootstrap_id,
                        seed=bootstrap_seed,
                        fit_sample_count=len(fit_indices),
                        fit_unique_sample_count=len(np.unique(fit_indices)),
                        oob_sample_count=len(oob_indices),
                        positive_fit_count=int(y_fit.sum()),
                        candidates_extracted=0,
                    )
                )
                continue

            for tree_id, tree_sequence in enumerate(child_sequences[1:]):
                tree_seed = _seed_from_sequence(tree_sequence)
                classifier = DecisionTreeClassifier(
                    criterion="gini",
                    splitter=self.config.splitter,
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_features=self.config.max_features,
                    class_weight=self.config.class_weight,
                    random_state=tree_seed,
                )
                classifier.fit(X_fit, y_fit)
                extracted = _extract_positive_leaf_rules(
                    classifier,
                    X_fit=X_fit,
                    y_fit=y_fit,
                    feature_names=names,
                    clinical_groups_by_feature=condition_groups,
                    sorted_reference_features=sorted_features,
                    minimum_probability=self.config.positive_leaf_probability,
                    minimum_positive_samples=self.config.min_positive_leaf_samples,
                )
                for leaf_id, rule in extracted:
                    fit_mask = rule.mask(X_fit)
                    oob_mask = rule.mask(X_array[oob_indices])
                    fit_stats = _binary_selection_stats(fit_mask, y_fit)
                    oob_stats = _binary_selection_stats(oob_mask, y_array[oob_indices])
                    occurrences.append(
                        CandidateRuleOccurrence(
                            rule=rule,
                            source="randomized_tree",
                            bootstrap_id=bootstrap_id,
                            tree_id=tree_id,
                            bootstrap_seed=bootstrap_seed,
                            tree_seed=tree_seed,
                            leaf_id=leaf_id,
                            fit_sample_count=len(fit_indices),
                            fit_positive_count=int(y_fit.sum()),
                            fit_selected_count=fit_stats[0],
                            fit_true_positive_count=fit_stats[1],
                            fit_precision=fit_stats[2],
                            fit_recall=fit_stats[3],
                            oob_sample_count=len(oob_indices),
                            oob_selected_count=oob_stats[0],
                            oob_true_positive_count=oob_stats[1],
                            oob_precision=oob_stats[2],
                            oob_recall=oob_stats[3],
                        )
                    )
                    bootstrap_occurrence_count += 1

            diagnostics.append(
                BootstrapDiagnostic(
                    bootstrap_id=bootstrap_id,
                    seed=bootstrap_seed,
                    fit_sample_count=len(fit_indices),
                    fit_unique_sample_count=len(np.unique(fit_indices)),
                    oob_sample_count=len(oob_indices),
                    positive_fit_count=int(y_fit.sum()),
                    candidates_extracted=bootstrap_occurrence_count,
                )
            )

        if single_class_bootstraps:
            warnings.append("one_or_more_bootstraps_have_single_class")
        if not occurrences:
            warnings.append("no_valid_rule_candidates")
        return StableRuleDiscoveryResult(
            occurrences=tuple(occurrences),
            bootstrap_diagnostics=tuple(diagnostics),
            feature_names=names,
            n_samples=n_samples,
            n_positive=n_positive,
            bootstrap_unit=unit,
            warnings=tuple(warnings),
        )


def discover_stable_rule_candidates(
    X: ArrayLike,
    y: ArrayLike,
    feature_names: Sequence[str],
    *,
    groups: ArrayLike | None = None,
    clinical_group_map: Mapping[str, Sequence[str]] | None = None,
    config: StableRuleBackendConfig | None = None,
    bootstrap_ids: Sequence[int] | None = None,
) -> StableRuleDiscoveryResult:
    """Functional entry point for semantic-rule orchestration."""

    return RandomizedTreeRuleBackend(config).discover(
        X,
        y,
        feature_names,
        groups=groups,
        clinical_group_map=clinical_group_map,
        bootstrap_ids=bootstrap_ids,
    )


def _validate_inputs(
    X: ArrayLike,
    y: ArrayLike,
    feature_names: Sequence[str],
    groups: ArrayLike | None,
    bootstrap_unit: BootstrapUnit,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int8],
    tuple[str, ...],
    NDArray[np.object_] | None,
    Literal["row", "group"],
]:
    X_array = np.asarray(X, dtype=np.float64)
    y_raw = np.asarray(y).reshape(-1)
    names = tuple(str(name) for name in feature_names)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if X_array.shape[0] != y_raw.shape[0]:
        raise ValueError("X and y must contain the same number of rows")
    if X_array.shape[1] != len(names):
        raise ValueError("feature_names length must equal X column count")
    if X_array.shape[0] == 0:
        raise ValueError("X and y must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if not np.isfinite(X_array).all():
        raise ValueError("X must contain only finite values")
    unique_targets = set(np.unique(y_raw).tolist())
    if not unique_targets.issubset({False, True, 0, 1}):
        raise ValueError("y must be binary with values 0 and 1")
    y_array = y_raw.astype(np.int8, copy=False)

    group_array: NDArray[np.object_] | None = None
    if groups is not None:
        group_array = np.asarray(groups, dtype=object).reshape(-1)
        if len(group_array) != X_array.shape[0]:
            raise ValueError("groups must contain one value per row")
        if any(value is None for value in group_array):
            raise ValueError("groups must not contain None")
    if bootstrap_unit == "group" and group_array is None:
        raise ValueError("bootstrap_unit='group' requires groups")
    unit: Literal["row", "group"] = (
        "group"
        if bootstrap_unit == "group" or (bootstrap_unit == "auto" and group_array is not None)
        else "row"
    )
    return X_array, y_array, names, group_array, unit


def _bootstrap_indices(
    *,
    n_samples: int,
    groups: NDArray[np.object_] | None,
    seed: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    if groups is None:
        fit_indices = rng.integers(0, n_samples, size=n_samples, dtype=np.int64)
        selected = np.zeros(n_samples, dtype=bool)
        selected[np.unique(fit_indices)] = True
        return fit_indices, np.flatnonzero(~selected)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    sampled_group_positions = rng.integers(
        0, len(unique_groups), size=len(unique_groups), dtype=np.int64
    )
    rows_by_group = tuple(
        np.flatnonzero(inverse == index) for index in range(len(unique_groups))
    )
    fit_indices = np.concatenate(
        [rows_by_group[position] for position in sampled_group_positions]
    ).astype(np.int64, copy=False)
    selected_groups = np.zeros(len(unique_groups), dtype=bool)
    selected_groups[np.unique(sampled_group_positions)] = True
    oob_indices = np.flatnonzero(~selected_groups[inverse]).astype(np.int64, copy=False)
    return fit_indices, oob_indices


def _extract_positive_leaf_rules(
    classifier: DecisionTreeClassifier,
    *,
    X_fit: NDArray[np.float64],
    y_fit: NDArray[np.int8],
    feature_names: tuple[str, ...],
    clinical_groups_by_feature: Mapping[str, tuple[str, ...]],
    sorted_reference_features: tuple[NDArray[np.float64], ...],
    minimum_probability: float,
    minimum_positive_samples: int,
) -> list[tuple[int, Rule]]:
    tree = classifier.tree_
    positive_class_positions = np.flatnonzero(classifier.classes_ == 1)
    if len(positive_class_positions) != 1:
        return []
    positive_position = int(positive_class_positions[0])
    extracted: list[tuple[int, Rule]] = []
    fit_leaf_ids = classifier.apply(X_fit)

    def visit(node_id: int, path: list[tuple[int, str, float]]) -> None:
        feature_index = int(tree.feature[node_id])
        if feature_index >= 0:
            threshold = float(tree.threshold[node_id])
            visit(
                int(tree.children_left[node_id]),
                path + [(feature_index, "<=", threshold)],
            )
            visit(
                int(tree.children_right[node_id]),
                path + [(feature_index, ">", threshold)],
            )
            return

        class_counts = np.asarray(tree.value[node_id]).reshape(-1)
        total_weight = float(class_counts.sum())
        positive_probability = (
            float(class_counts[positive_position] / total_weight) if total_weight else 0.0
        )
        if positive_probability < minimum_probability:
            return
        positive_leaf_samples = int(
            np.logical_and(fit_leaf_ids == node_id, y_fit == 1).sum()
        )
        if positive_leaf_samples < minimum_positive_samples:
            return
        conditions = _canonical_conditions(
            path,
            feature_names,
            sorted_reference_features,
            clinical_groups_by_feature,
        )
        if conditions:
            extracted.append(
                (
                    node_id,
                    Rule(
                        rule_id=_rule_id(conditions),
                        conditions=conditions,
                        source=None,
                    ),
                )
            )

    visit(0, [])
    return extracted


def _canonical_conditions(
    path: list[tuple[int, str, float]],
    feature_names: tuple[str, ...],
    sorted_reference_features: tuple[NDArray[np.float64], ...],
    clinical_groups_by_feature: Mapping[str, tuple[str, ...]],
) -> tuple[Condition, ...]:
    # Repeated splits on one feature collapse to strongest lower/upper bounds.
    lower: dict[int, float] = {}
    upper: dict[int, float] = {}
    for feature_index, operator, threshold in path:
        if operator == ">":
            lower[feature_index] = max(lower.get(feature_index, -np.inf), threshold)
        else:
            upper[feature_index] = min(upper.get(feature_index, np.inf), threshold)

    conditions: list[Condition] = []
    for feature_index in sorted(set(lower) | set(upper)):
        bounds = ((">", lower.get(feature_index)), ("<=", upper.get(feature_index)))
        for operator, threshold in bounds:
            if threshold is None:
                continue
            reference = sorted_reference_features[feature_index]
            normalized_threshold = float(
                np.searchsorted(reference, threshold, side="right") / len(reference)
            )
            conditions.append(
                Condition(
                    feature_index=feature_index,
                    feature_name=feature_names[feature_index],
                    operator=operator,
                    threshold=float(threshold),
                    clinical_groups=clinical_groups_by_feature[
                        feature_names[feature_index]
                    ],
                    normalized_threshold=normalized_threshold,
                )
            )
    return tuple(conditions)


def _binary_selection_stats(
    selected: NDArray[np.bool_], y: NDArray[np.int8]
) -> tuple[int, int, float, float]:
    selected_count = int(selected.sum())
    true_positive_count = int(np.logical_and(selected, y == 1).sum())
    positive_count = int(y.sum())
    precision = true_positive_count / selected_count if selected_count else 0.0
    recall = true_positive_count / positive_count if positive_count else 0.0
    return selected_count, true_positive_count, precision, recall


def _seed_from_sequence(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _clinical_groups_for(
    feature_name: str,
    mapping: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    # Unmapped features remain distinguishable singleton groups instead of
    # silently appearing to share an empty clinical category.
    value = mapping.get(feature_name, (f"feature:{feature_name}",))
    values = (value,) if isinstance(value, str) else value
    groups = tuple(sorted(set(str(group) for group in values)))
    if any(not group for group in groups):
        raise ValueError(f"clinical groups for {feature_name!r} must be non-empty")
    return groups


def _rule_id(conditions: tuple[Condition, ...]) -> str:
    """Content-derived ID; occurrence provenance remains outside ``Rule``."""

    signature = "|".join(
        f"{condition.feature_index}:{condition.operator}:{condition.threshold:.17g}"
        for condition in conditions
    )
    return f"randomized-tree-{hashlib.sha256(signature.encode('utf-8')).hexdigest()[:16]}"
