"""Core data structures and selection logic for semantic SAE rule sets.

This module deliberately contains no rule-discovery implementation.  Discovery
backends produce :class:`Rule` objects; the functions here fit activation
targets, evaluate OR-of-ANDs rule sets, and select a constrained rule set on a
separate selection partition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable, Literal, Mapping, Sequence

import numpy as np


RuleOperator = Literal["<=", ">"]
SelectionObjective = Literal["f2", "recall"]


def _as_2d_float_array(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"X must be two-dimensional; got shape {array.shape}")
    return array


def _as_bool_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    return array.astype(bool, copy=False)


@dataclass(frozen=True)
class Condition:
    """One numeric feature condition within a conjunctive rule.

    ``feature_index`` is authoritative at evaluation time. ``feature_name`` is
    retained for readable artifacts and cross-run feature comparisons.
    ``normalized_threshold`` may hold a fitting-set ECDF rank for stable-rule
    clustering; it never changes evaluation semantics.
    """

    feature_index: int
    feature_name: str
    operator: RuleOperator
    threshold: float
    clinical_groups: tuple[str, ...] = ()
    normalized_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")
        if not self.feature_name:
            raise ValueError("feature_name must be non-empty")
        if self.operator not in ("<=", ">"):
            raise ValueError("operator must be '<=' or '>'")
        if not np.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.normalized_threshold is not None and not (
            0.0 <= self.normalized_threshold <= 1.0
        ):
            raise ValueError("normalized_threshold must lie in [0, 1]")
        object.__setattr__(self, "clinical_groups", tuple(self.clinical_groups))

    def mask(self, X: np.ndarray) -> np.ndarray:
        """Return rows satisfying this condition."""

        array = _as_2d_float_array(X)
        if self.feature_index >= array.shape[1]:
            raise IndexError(
                f"feature_index {self.feature_index} outside X with "
                f"{array.shape[1]} columns"
            )
        column = array[:, self.feature_index]
        if self.operator == "<=":
            return column <= self.threshold
        return column > self.threshold

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        result = asdict(self)
        result["clinical_groups"] = list(self.clinical_groups)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Condition":
        """Construct a condition from :meth:`to_dict` output."""

        normalized = value.get("normalized_threshold")
        return cls(
            feature_index=int(value["feature_index"]),
            feature_name=str(value["feature_name"]),
            operator=str(value["operator"]),  # type: ignore[arg-type]
            threshold=float(value["threshold"]),
            clinical_groups=tuple(str(v) for v in value.get("clinical_groups", ())),
            normalized_threshold=None if normalized is None else float(normalized),
        )


@dataclass(frozen=True)
class Rule:
    """A conjunction (AND) of feature conditions."""

    rule_id: str
    conditions: tuple[Condition, ...]
    source: str | None = None

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if not self.conditions:
            raise ValueError("a rule must contain at least one condition")

    @property
    def length(self) -> int:
        """Number of conditions in the rule."""

        return len(self.conditions)

    def mask(self, X: np.ndarray) -> np.ndarray:
        """Return rows satisfying every condition."""

        array = _as_2d_float_array(X)
        selected = np.ones(array.shape[0], dtype=bool)
        for condition in self.conditions:
            selected &= condition.mask(array)
        return selected

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "rule_id": self.rule_id,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Rule":
        """Construct a rule from :meth:`to_dict` output."""

        raw_conditions = value["conditions"]
        if not isinstance(raw_conditions, Sequence):
            raise TypeError("conditions must be a sequence")
        return cls(
            rule_id=str(value["rule_id"]),
            conditions=tuple(Condition.from_dict(item) for item in raw_conditions),  # type: ignore[arg-type]
            source=None if value.get("source") is None else str(value["source"]),
        )


@dataclass(frozen=True)
class RuleSet:
    """A disjunction (OR) of conjunctive :class:`Rule` objects."""

    rules: tuple[Rule, ...] = ()
    threshold_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules", tuple(self.rules))
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("rule IDs must be unique within a rule set")

    def mask(self, X: np.ndarray) -> np.ndarray:
        """Return union of cohorts selected by constituent rules."""

        array = _as_2d_float_array(X)
        selected = np.zeros(array.shape[0], dtype=bool)
        for rule in self.rules:
            selected |= rule.mask(array)
        return selected

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "rules": [rule.to_dict() for rule in self.rules],
            "threshold_name": self.threshold_name,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleSet":
        """Construct a rule set from :meth:`to_dict` output."""

        raw_rules = value.get("rules", ())
        if not isinstance(raw_rules, Sequence):
            raise TypeError("rules must be a sequence")
        return cls(
            rules=tuple(Rule.from_dict(item) for item in raw_rules),  # type: ignore[arg-type]
            threshold_name=(
                None
                if value.get("threshold_name") is None
                else str(value["threshold_name"])
            ),
        )


@dataclass(frozen=True)
class ActivationTargetSpec:
    """Definition of a top fraction among strictly positive activations."""

    name: str
    positive_fraction: float
    minimum_positive_samples: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("activation target name must be non-empty")
        if not 0.0 < self.positive_fraction <= 1.0:
            raise ValueError("positive_fraction must lie in (0, 1]")
        if self.minimum_positive_samples < 1:
            raise ValueError("minimum_positive_samples must be positive")


@dataclass(frozen=True)
class FittedActivationTarget:
    """Activation cutoff fitted on one designated fitting partition only."""

    spec: ActivationTargetSpec
    cutoff: float
    n_fit_samples: int
    n_positive_fit_samples: int
    valid: bool
    invalid_reason: str | None = None

    def apply(self, activations: np.ndarray) -> np.ndarray:
        """Apply frozen cutoff without refitting on evaluation activations."""

        values = np.asarray(activations, dtype=float)
        if values.ndim != 1:
            raise ValueError(
                f"activations must be one-dimensional; got shape {values.shape}"
            )
        if not self.valid:
            return np.zeros(values.shape[0], dtype=bool)
        return (values > 0.0) & (values >= self.cutoff)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "name": self.spec.name,
            "positive_fraction": self.spec.positive_fraction,
            "compatibility_H": 100.0 * (1.0 - self.spec.positive_fraction),
            "minimum_positive_samples": self.spec.minimum_positive_samples,
            "cutoff": None if not self.valid else self.cutoff,
            "n_fit_samples": self.n_fit_samples,
            "n_positive_fit_samples": self.n_positive_fit_samples,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
        }


def fit_activation_target(
    activations: np.ndarray, spec: ActivationTargetSpec
) -> FittedActivationTarget:
    """Fit a positive-only quantile cutoff.

    For example, ``positive_fraction=0.10`` uses the 90th percentile of
    strictly positive fitting activations. NaN and infinite activations are
    ignored. A factor with no finite positive fitting activation produces an
    invalid target that selects no samples when applied. The same behavior
    applies when fewer than ``spec.minimum_positive_samples`` are available.
    """

    values = np.asarray(activations, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"activations must be one-dimensional; got shape {values.shape}"
        )
    positive = values[np.isfinite(values) & (values > 0.0)]
    if positive.size < spec.minimum_positive_samples:
        return FittedActivationTarget(
            spec=spec,
            cutoff=float("inf"),
            n_fit_samples=int(values.size),
            n_positive_fit_samples=int(positive.size),
            valid=False,
            invalid_reason=(
                "no_positive_fit_activations"
                if positive.size == 0
                else "insufficient_positive_fit_activations"
            ),
        )
    cutoff = float(np.quantile(positive, 1.0 - spec.positive_fraction))
    return FittedActivationTarget(
        spec=spec,
        cutoff=cutoff,
        n_fit_samples=int(values.size),
        n_positive_fit_samples=int(positive.size),
        valid=True,
        invalid_reason=None,
    )


@dataclass(frozen=True)
class BinaryMetrics:
    """Aggregate metrics for one complete OR-of-rules cohort."""

    n_samples: int
    n_positive: int
    n_selected: int
    true_positive: int
    prevalence: float
    coverage: float
    precision: float
    recall: float
    f2: float
    lift: float
    wracc: float
    jaccard: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def cohort_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    """Jaccard similarity between two boolean cohort masks.

    Two empty cohorts are identical and therefore have similarity 1.0.
    """

    left_mask = _as_bool_vector(left, "left")
    right_mask = _as_bool_vector(right, "right")
    if left_mask.shape != right_mask.shape:
        raise ValueError("cohort masks must have equal shape")
    union = int(np.count_nonzero(left_mask | right_mask))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(left_mask & right_mask) / union)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    """Compute metrics from aggregate target and selected-cohort masks.

    Undefined precision, recall, F2, and lift use zero rather than NaN. WRAcc is
    ``coverage * (precision - prevalence)``. Jaccard compares selected and true
    cohorts.
    """

    truth = _as_bool_vector(y_true, "y_true")
    predicted = _as_bool_vector(y_pred, "y_pred")
    if truth.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal shape")

    n_samples = int(truth.size)
    n_positive = int(np.count_nonzero(truth))
    n_selected = int(np.count_nonzero(predicted))
    true_positive = int(np.count_nonzero(truth & predicted))
    prevalence = n_positive / n_samples if n_samples else 0.0
    coverage = n_selected / n_samples if n_samples else 0.0
    precision = true_positive / n_selected if n_selected else 0.0
    recall = true_positive / n_positive if n_positive else 0.0
    denominator = 4.0 * precision + recall
    f2 = 5.0 * precision * recall / denominator if denominator else 0.0
    lift = precision / prevalence if prevalence else 0.0
    wracc = coverage * (precision - prevalence)
    jaccard = cohort_jaccard(truth, predicted)
    return BinaryMetrics(
        n_samples=n_samples,
        n_positive=n_positive,
        n_selected=n_selected,
        true_positive=true_positive,
        prevalence=float(prevalence),
        coverage=float(coverage),
        precision=float(precision),
        recall=float(recall),
        f2=float(f2),
        lift=float(lift),
        wracc=float(wracc),
        jaccard=float(jaccard),
    )


def evaluate_rule_set(
    rule_set: RuleSet, X: np.ndarray, y_true: np.ndarray
) -> BinaryMetrics:
    """Evaluate union selected by ``rule_set`` against a binary target."""

    return binary_metrics(y_true, rule_set.mask(X))


@dataclass(frozen=True)
class RuleSetSelectionConfig:
    """Objective, constraints, and deterministic search limits."""

    objective: SelectionObjective = "f2"
    min_precision: float = 0.5
    min_lift: float = 1.5
    max_rules: int = 5
    max_rule_length: int = 5
    min_marginal_recall: float = 0.02
    exhaustive_max_candidates: int = 20
    beam_width: int = 64
    minimum_positive_samples: int = 1

    def __post_init__(self) -> None:
        if self.objective not in ("f2", "recall"):
            raise ValueError("objective must be 'f2' or 'recall'")
        if not 0.0 <= self.min_precision <= 1.0:
            raise ValueError("min_precision must lie in [0, 1]")
        if self.min_lift < 0.0:
            raise ValueError("min_lift must be non-negative")
        if self.max_rules < 1:
            raise ValueError("max_rules must be positive")
        if self.max_rule_length < 1:
            raise ValueError("max_rule_length must be positive")
        if not 0.0 <= self.min_marginal_recall <= 1.0:
            raise ValueError("min_marginal_recall must lie in [0, 1]")
        if self.exhaustive_max_candidates < 1:
            raise ValueError("exhaustive_max_candidates must be positive")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.minimum_positive_samples < 1:
            raise ValueError("minimum_positive_samples must be positive")


@dataclass(frozen=True)
class RuleSetSelection:
    """Result of constrained rule-set selection."""

    rule_set: RuleSet
    metrics: BinaryMetrics
    feasible: bool
    reason: str | None
    search_method: Literal["exhaustive", "beam", "none"]
    n_candidates: int
    n_evaluated_subsets: int


def _objective_key(
    indices: tuple[int, ...],
    metrics: BinaryMetrics,
    rules: Sequence[Rule],
    objective: SelectionObjective,
) -> tuple[object, ...]:
    primary = metrics.f2 if objective == "f2" else metrics.recall
    secondary = metrics.recall if objective == "f2" else metrics.f2
    ids = tuple(rules[index].rule_id for index in indices)
    total_length = sum(rules[index].length for index in indices)
    # Higher tuple wins. Negated character codes make lexical rule IDs ascending
    # at the final deterministic tie-break without relying on input order.
    lexical = tuple(tuple(-ord(char) for char in rule_id) for rule_id in ids)
    return (
        primary,
        secondary,
        metrics.precision,
        metrics.lift,
        -len(indices),
        -total_length,
        lexical,
    )


def _has_valid_addition_order(
    indices: tuple[int, ...],
    masks: Sequence[np.ndarray],
    truth: np.ndarray,
    minimum: float,
) -> bool:
    """Whether rules can be added with enough new recall after the first."""

    if len(indices) <= 1 or minimum <= 0.0:
        return True
    n_positive = int(np.count_nonzero(truth))
    if n_positive == 0:
        return False

    memo: dict[tuple[tuple[int, ...], bytes], bool] = {}

    def visit(remaining: tuple[int, ...], selected: np.ndarray, started: bool) -> bool:
        if not remaining:
            return True
        key = (remaining, selected.tobytes())
        if key in memo:
            return memo[key]
        previous_recall = np.count_nonzero(selected & truth) / n_positive
        for position, index in enumerate(remaining):
            combined = selected | masks[index]
            new_recall = np.count_nonzero(combined & truth) / n_positive
            if not started or new_recall - previous_recall + 1e-15 >= minimum:
                rest = remaining[:position] + remaining[position + 1 :]
                if visit(rest, combined, True):
                    memo[key] = True
                    return True
        memo[key] = False
        return False

    return visit(indices, np.zeros(truth.shape[0], dtype=bool), False)


def select_rule_set(
    candidates: Iterable[Rule],
    X_selection: np.ndarray,
    y_selection: np.ndarray,
    config: RuleSetSelectionConfig | None = None,
    *,
    threshold_name: str | None = None,
) -> RuleSetSelection:
    """Select a deterministic constrained OR-of-rules model.

    All objective and constraint metrics are computed on each subset's union
    mask. Candidate rules longer than ``max_rule_length`` are excluded. Search
    is exhaustive for small candidate pools and uses a deterministic beam for
    larger pools. The final held-out comparison partition must never be passed
    to this function.
    """

    selection_config = config or RuleSetSelectionConfig()
    array = _as_2d_float_array(X_selection)
    truth = _as_bool_vector(y_selection, "y_selection")
    if array.shape[0] != truth.shape[0]:
        raise ValueError("X_selection and y_selection must contain equal rows")

    eligible = sorted(
        (rule for rule in candidates if rule.length <= selection_config.max_rule_length),
        key=lambda rule: rule.rule_id,
    )
    ids = [rule.rule_id for rule in eligible]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate rule IDs must be unique")
    empty = RuleSet((), threshold_name=threshold_name)
    empty_metrics = evaluate_rule_set(empty, array, truth)
    if not eligible:
        return RuleSetSelection(
            rule_set=empty,
            metrics=empty_metrics,
            feasible=False,
            reason="no_eligible_candidates",
            search_method="none",
            n_candidates=0,
            n_evaluated_subsets=0,
        )
    n_positive = int(np.count_nonzero(truth))
    if n_positive < selection_config.minimum_positive_samples:
        return RuleSetSelection(
            rule_set=empty,
            metrics=empty_metrics,
            feasible=False,
            reason=(
                "no_positive_selection_targets"
                if n_positive == 0
                else "insufficient_positive_selection_targets"
            ),
            search_method="none",
            n_candidates=len(eligible),
            n_evaluated_subsets=0,
        )

    masks = [rule.mask(array) for rule in eligible]
    max_size = min(selection_config.max_rules, len(eligible))
    exhaustive = len(eligible) <= selection_config.exhaustive_max_candidates
    search_method: Literal["exhaustive", "beam"] = (
        "exhaustive" if exhaustive else "beam"
    )
    best: tuple[tuple[int, ...], BinaryMetrics] | None = None
    evaluated = 0

    def evaluate(indices: tuple[int, ...]) -> BinaryMetrics | None:
        nonlocal evaluated
        if not _has_valid_addition_order(
            indices, masks, truth, selection_config.min_marginal_recall
        ):
            return None
        union = np.zeros(truth.shape[0], dtype=bool)
        for index in indices:
            union |= masks[index]
        evaluated += 1
        return binary_metrics(truth, union)

    def consider(indices: tuple[int, ...], metrics: BinaryMetrics) -> None:
        nonlocal best
        if metrics.precision + 1e-15 < selection_config.min_precision:
            return
        if metrics.lift + 1e-15 < selection_config.min_lift:
            return
        if best is None or _objective_key(
            indices, metrics, eligible, selection_config.objective
        ) > _objective_key(best[0], best[1], eligible, selection_config.objective):
            best = (indices, metrics)

    if exhaustive:
        for size in range(1, max_size + 1):
            for indices in combinations(range(len(eligible)), size):
                metrics = evaluate(indices)
                if metrics is not None:
                    consider(indices, metrics)
    else:
        frontier: list[tuple[int, ...]] = [()]
        for _size in range(1, max_size + 1):
            expanded: set[tuple[int, ...]] = set()
            for state in frontier:
                start = state[-1] + 1 if state else 0
                for index in range(start, len(eligible)):
                    expanded.add(state + (index,))
            scored: list[tuple[tuple[object, ...], tuple[int, ...]]] = []
            for indices in sorted(expanded):
                metrics = evaluate(indices)
                if metrics is None:
                    continue
                consider(indices, metrics)
                scored.append(
                    (
                        _objective_key(
                            indices, metrics, eligible, selection_config.objective
                        ),
                        indices,
                    )
                )
            scored.sort(reverse=True)
            frontier = [indices for _, indices in scored[: selection_config.beam_width]]
            if not frontier:
                break

    if best is None:
        return RuleSetSelection(
            rule_set=empty,
            metrics=empty_metrics,
            feasible=False,
            reason="no_subset_satisfies_constraints",
            search_method=search_method,
            n_candidates=len(eligible),
            n_evaluated_subsets=evaluated,
        )

    chosen_indices, chosen_metrics = best
    chosen = RuleSet(
        tuple(eligible[index] for index in chosen_indices),
        threshold_name=threshold_name,
    )
    return RuleSetSelection(
        rule_set=chosen,
        metrics=chosen_metrics,
        feasible=True,
        reason=None,
        search_method=search_method,
        n_candidates=len(eligible),
        n_evaluated_subsets=evaluated,
    )


DEFAULT_ACTIVATION_TARGETS: tuple[ActivationTargetSpec, ...] = (
    ActivationTargetSpec("top_10pct_positive", 0.10),
    # ActivationTargetSpec("top_25pct_positive", 0.25),
    ActivationTargetSpec("top_20pct_positive", 0.20),
    ActivationTargetSpec("top_30pct_positive", 0.30),
    ActivationTargetSpec("top_40pct_positive", 0.40),
    ActivationTargetSpec("top_50pct_positive", 0.50),
)
