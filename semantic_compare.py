"""Stable rule-family clustering and symmetric cross-run semantic transfer.

Rule discovery and constrained OR-set selection live in separate modules.  This
module compares their structured outputs without fitting thresholds, rules, or
hyperparameters on final comparison records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import heapq
from typing import TYPE_CHECKING, Iterator, Mapping, Sequence

import numpy as np

from semantic_rules import BinaryMetrics, Rule, RuleSet, binary_metrics, cohort_jaccard
if TYPE_CHECKING:
    from stable_rule_backend import CandidateRuleOccurrence, StableRuleDiscoveryResult


@dataclass(frozen=True)
class RuleSimilarityWeights:
    """Components of stable-rule similarity; defaults sum to one."""

    cohort: float = 0.625
    exact_feature: float = 0.25
    threshold_direction: float = 0.125

    def __post_init__(self) -> None:
        values = (
            self.cohort,
            self.exact_feature,
            self.threshold_direction,
        )
        if any(value < 0.0 for value in values):
            raise ValueError("similarity weights must be non-negative")
        if not np.isclose(sum(values), 1.0):
            raise ValueError("similarity weights must sum to one")

    def to_dict(self) -> dict[str, float]:
        return {
            "cohort": self.cohort,
            "exact_feature": self.exact_feature,
            "threshold_direction": self.threshold_direction,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleSimilarityWeights":
        # Old artifacts included a clinical-group component. Discard it and
        # renormalize the remaining scientific components when deserializing.
        raw = {
            key: float(item)
            for key, item in value.items()
            if key in {"cohort", "exact_feature", "threshold_direction"}
        }
        if "clinical_group" in value and raw:
            total = sum(raw.values())
            if total <= 0:
                raise ValueError("non-clinical similarity weights must have positive mass")
            raw = {key: item / total for key, item in raw.items()}
        return cls(**raw)


@dataclass(frozen=True)
class RuleSimilarityConfig:
    """Rule equivalence and recurrence thresholds.

    ``normalized_threshold_tolerance`` is measured in fitting-set ECDF units.
    Conditions farther apart receive no threshold compatibility credit.
    """

    weights: RuleSimilarityWeights = field(default_factory=RuleSimilarityWeights)
    similarity_threshold: float = 0.70
    normalized_threshold_tolerance: float = 0.10
    min_recurrence: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must lie in [0, 1]")
        if not 0.0 < self.normalized_threshold_tolerance <= 1.0:
            raise ValueError("normalized_threshold_tolerance must lie in (0, 1]")
        if not 0.0 < self.min_recurrence <= 1.0:
            raise ValueError("min_recurrence must lie in (0, 1]")

    def to_dict(self) -> dict[str, object]:
        return {
            "weights": self.weights.to_dict(),
            "similarity_threshold": self.similarity_threshold,
            "normalized_threshold_tolerance": self.normalized_threshold_tolerance,
            "min_recurrence": self.min_recurrence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleSimilarityConfig":
        raw_weights = value.get("weights", {})
        if not isinstance(raw_weights, Mapping):
            raise TypeError("weights must be a mapping")
        return cls(
            weights=RuleSimilarityWeights.from_dict(raw_weights),
            similarity_threshold=float(value.get("similarity_threshold", 0.70)),
            normalized_threshold_tolerance=float(
                value.get("normalized_threshold_tolerance", 0.10)
            ),
            min_recurrence=float(value.get("min_recurrence", 0.50)),
        )


@dataclass(frozen=True)
class RuleSimilarity:
    total: float
    cohort_jaccard: float
    exact_feature_jaccard: float
    threshold_direction_compatibility: float
    # Legacy artifacts may contain this field. New comparisons leave it unset.
    clinical_group_jaccard: float | None = None

    def to_dict(self) -> dict[str, float]:
        result = {
            "total": self.total,
            "cohort_jaccard": self.cohort_jaccard,
            "exact_feature_jaccard": self.exact_feature_jaccard,
            "threshold_direction_compatibility": (
                self.threshold_direction_compatibility
            ),
        }
        if self.clinical_group_jaccard is not None:
            result["clinical_group_jaccard"] = self.clinical_group_jaccard
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleSimilarity":
        return cls(
            total=float(value["total"]),
            cohort_jaccard=float(value["cohort_jaccard"]),
            exact_feature_jaccard=float(value["exact_feature_jaccard"]),
            threshold_direction_compatibility=float(
                value["threshold_direction_compatibility"]
            ),
            clinical_group_jaccard=(
                None
                if "clinical_group_jaccard" not in value
                else float(value["clinical_group_jaccard"])
            ),
        )


def _set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _rule_features(rule: Rule) -> set[str]:
    return {condition.feature_name for condition in rule.conditions}


def _rule_groups(rule: Rule) -> set[str]:
    return {
        group
        for condition in rule.conditions
        for group in condition.clinical_groups
    }


def _threshold_direction_similarity(
    left: Rule, right: Rule, tolerance: float
) -> float:
    """Match direction-compatible conditions on the exact same feature."""

    if not left.conditions and not right.conditions:
        return 1.0
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_condition in enumerate(left.conditions):
        for right_index, right_condition in enumerate(right.conditions):
            if left_condition.operator != right_condition.operator:
                continue
            same_feature = left_condition.feature_name == right_condition.feature_name
            if not same_feature:
                continue
            left_threshold = left_condition.normalized_threshold
            right_threshold = right_condition.normalized_threshold
            if left_threshold is None or right_threshold is None:
                compatibility = float(
                    same_feature
                    and np.isclose(left_condition.threshold, right_condition.threshold)
                )
            else:
                difference = abs(left_threshold - right_threshold)
                compatibility = max(0.0, 1.0 - difference / tolerance)
            if compatibility:
                candidates.append((compatibility, left_index, right_index))

    # Rules are short. Greedy maximum-first matching is deterministic and stops
    # one condition receiving repeated compatibility credit.
    used_left: set[int] = set()
    used_right: set[int] = set()
    score = 0.0
    for compatibility, left_index, right_index in sorted(
        candidates, key=lambda item: (-item[0], item[1], item[2])
    ):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        score += compatibility
    return score / max(len(left.conditions), len(right.conditions))


def rule_similarity(
    left: Rule,
    right: Rule,
    X_reference: np.ndarray,
    config: RuleSimilarityConfig | None = None,
) -> RuleSimilarity:
    """Compare two rules on one fitting/reference cohort.

    ``X_reference`` must come from rule-fitting data, never the final semantic
    comparison partition.  Its only purpose is cohort-equivalence clustering.
    """

    similarity_config = config or RuleSimilarityConfig()
    array = np.asarray(X_reference)
    if array.ndim != 2:
        raise ValueError("X_reference must be two-dimensional")
    return _rule_similarity_from_masks(
        left, right, left.mask(array), right.mask(array), similarity_config
    )


def _rule_similarity_from_masks(
    left: Rule,
    right: Rule,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    similarity_config: RuleSimilarityConfig,
) -> RuleSimilarity:
    cohort = cohort_jaccard(left_mask, right_mask)
    features = _set_jaccard(_rule_features(left), _rule_features(right))
    threshold = _threshold_direction_similarity(
        left, right, similarity_config.normalized_threshold_tolerance
    )
    weights = similarity_config.weights
    total = (
        weights.cohort * cohort
        + weights.exact_feature * features
        + weights.threshold_direction * threshold
    )
    return RuleSimilarity(
        total=float(total),
        cohort_jaccard=float(cohort),
        exact_feature_jaccard=float(features),
        threshold_direction_compatibility=float(threshold),
    )


@dataclass(frozen=True)
class Recurrence:
    name: str
    bootstrap_count: int
    frequency: float

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bootstrap_count": self.bootstrap_count,
            "frequency": self.frequency,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Recurrence":
        return cls(
            name=str(value["name"]),
            bootstrap_count=int(value["bootstrap_count"]),
            frequency=float(value["frequency"]),
        )


@dataclass(frozen=True)
class DistributionSummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "standard_deviation": self.standard_deviation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DistributionSummary":
        return cls(
            count=int(value["count"]),
            minimum=float(value["minimum"]),
            maximum=float(value["maximum"]),
            mean=float(value["mean"]),
            standard_deviation=float(value["standard_deviation"]),
        )


def _distribution(values: Sequence[float], *, singleton_default: float = 0.0) -> DistributionSummary:
    if not values:
        return DistributionSummary(0, singleton_default, singleton_default, singleton_default, 0.0)
    array = np.asarray(values, dtype=float)
    return DistributionSummary(
        count=len(values),
        minimum=float(array.min()),
        maximum=float(array.max()),
        mean=float(array.mean()),
        standard_deviation=float(array.std()),
    )


@dataclass(frozen=True)
class ThresholdVariability:
    feature_name: str
    operator: str
    raw: DistributionSummary
    normalized: DistributionSummary | None

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "operator": self.operator,
            "raw": self.raw.to_dict(),
            "normalized": None if self.normalized is None else self.normalized.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ThresholdVariability":
        raw = value["raw"]
        normalized = value.get("normalized")
        if not isinstance(raw, Mapping):
            raise TypeError("raw threshold summary must be a mapping")
        if normalized is not None and not isinstance(normalized, Mapping):
            raise TypeError("normalized threshold summary must be a mapping")
        return cls(
            feature_name=str(value["feature_name"]),
            operator=str(value["operator"]),
            raw=DistributionSummary.from_dict(raw),
            normalized=(
                None
                if normalized is None
                else DistributionSummary.from_dict(normalized)
            ),
        )


@dataclass(frozen=True)
class RuleFamily:
    family_id: str
    member_rules: tuple[Rule, ...]
    representative_rule: Rule
    occurrence_count: int
    bootstrap_ids: tuple[int, ...]
    recurrence_frequency: float
    retained: bool
    feature_recurrence: dict[str, float]
    clinical_group_recurrence: dict[str, float]
    threshold_variability: tuple[ThresholdVariability, ...]
    cohort_stability: DistributionSummary

    @property
    def occurrences(self) -> tuple[tuple[int, str, int, int], ...]:
        """Compact occurrence provenance: bootstrap, rule, tree, leaf."""

        return self._occurrence_references

    @property
    def recurrence_count(self) -> int:
        return len(self.bootstrap_ids)

    @property
    def cohort_overlap_stability(self) -> float:
        return self.cohort_stability.mean

    _occurrence_references: tuple[tuple[int, str, int, int], ...] = field(
        default=(), repr=False
    )

    def to_dict(self) -> dict[str, object]:
        result = {
            "family_id": self.family_id,
            "member_rules": [rule.to_dict() for rule in self.member_rules],
            "representative_rule": self.representative_rule.to_dict(),
            "occurrence_count": self.occurrence_count,
            "bootstrap_ids": list(self.bootstrap_ids),
            "recurrence_count": self.recurrence_count,
            "recurrence_frequency": self.recurrence_frequency,
            "retained": self.retained,
            "feature_recurrence": dict(sorted(self.feature_recurrence.items())),
            "threshold_variability": [
                item.to_dict() for item in self.threshold_variability
            ],
            "cohort_stability": self.cohort_stability.to_dict(),
            "cohort_overlap_stability": self.cohort_overlap_stability,
            "occurrence_references": [list(item) for item in self.occurrences],
        }
        if self.clinical_group_recurrence:
            result["clinical_group_recurrence"] = dict(
                sorted(self.clinical_group_recurrence.items())
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleFamily":
        return cls(
            family_id=str(value["family_id"]),
            member_rules=tuple(Rule.from_dict(item) for item in value["member_rules"]),  # type: ignore[arg-type]
            representative_rule=Rule.from_dict(value["representative_rule"]),  # type: ignore[arg-type]
            occurrence_count=int(value["occurrence_count"]),
            bootstrap_ids=tuple(int(item) for item in value["bootstrap_ids"]),  # type: ignore[union-attr]
            recurrence_frequency=float(value["recurrence_frequency"]),
            retained=bool(value["retained"]),
            feature_recurrence={
                str(name): float(frequency)
                for name, frequency in value["feature_recurrence"].items()  # type: ignore[union-attr]
            },
            clinical_group_recurrence={
                str(name): float(frequency)
                for name, frequency in value.get(
                    "clinical_group_recurrence", {}
                ).items()  # type: ignore[union-attr]
            },
            threshold_variability=tuple(
                ThresholdVariability.from_dict(item)
                for item in value["threshold_variability"]  # type: ignore[arg-type]
            ),
            cohort_stability=DistributionSummary.from_dict(value["cohort_stability"]),  # type: ignore[arg-type]
            _occurrence_references=tuple(
                (int(item[0]), str(item[1]), int(item[2]), int(item[3]))
                for item in value.get("occurrence_references", ())  # type: ignore[union-attr]
            ),
        )


@dataclass(frozen=True)
class RuleFamilyClusteringResult:
    families: tuple[RuleFamily, ...]
    total_bootstraps: int
    n_occurrences: int
    n_unique_rules: int
    config: RuleSimilarityConfig

    @property
    def retained_families(self) -> tuple[RuleFamily, ...]:
        return tuple(family for family in self.families if family.retained)

    def __iter__(self) -> Iterator[RuleFamily]:
        return iter(self.families)

    def __len__(self) -> int:
        return len(self.families)

    def __getitem__(self, index: int) -> RuleFamily:
        return self.families[index]

    def to_dict(self) -> dict[str, object]:
        return {
            "families": [family.to_dict() for family in self.families],
            "total_bootstraps": self.total_bootstraps,
            "n_occurrences": self.n_occurrences,
            "n_unique_rules": self.n_unique_rules,
            "config": self.config.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "RuleFamilyClusteringResult":
        return cls(
            families=tuple(RuleFamily.from_dict(item) for item in value["families"]),  # type: ignore[arg-type]
            total_bootstraps=int(value["total_bootstraps"]),
            n_occurrences=int(value["n_occurrences"]),
            n_unique_rules=int(value["n_unique_rules"]),
            config=RuleSimilarityConfig.from_dict(value["config"]),  # type: ignore[arg-type]
        )


def _complete_link_clusters(
    similarities: np.ndarray, rule_ids: Sequence[str], threshold: float
) -> tuple[tuple[int, ...], ...]:
    """Agglomerate by maximum complete-link similarity with stable tie breaks."""

    size = len(rule_ids)
    active: dict[int, tuple[int, ...]] = {index: (index,) for index in range(size)}
    signatures: dict[int, tuple[str, ...]] = {
        index: (rule_ids[index],) for index in range(size)
    }
    pair_scores: dict[tuple[int, int], float] = {}
    queue: list[tuple[float, tuple[str, ...], tuple[str, ...], int, int]] = []
    for left in range(size):
        for right in range(left + 1, size):
            score = float(similarities[left, right])
            pair_scores[(left, right)] = score
            if score + 1e-15 >= threshold:
                heapq.heappush(
                    queue,
                    (-score, signatures[left], signatures[right], left, right),
                )

    next_id = size
    while queue:
        negative_score, _left_signature, _right_signature, left, right = heapq.heappop(queue)
        if left not in active or right not in active:
            continue
        score = pair_scores.get((min(left, right), max(left, right)), -np.inf)
        if not np.isclose(score, -negative_score):
            continue
        if score + 1e-15 < threshold:
            break
        merged_members = tuple(sorted(active.pop(left) + active.pop(right)))
        merged_signature = tuple(sorted(signatures.pop(left) + signatures.pop(right)))
        merged_id = next_id
        next_id += 1
        active[merged_id] = merged_members
        signatures[merged_id] = merged_signature
        for other in sorted(active, key=lambda item: signatures[item]):
            if other == merged_id:
                continue
            left_other = pair_scores[(min(left, other), max(left, other))]
            right_other = pair_scores[(min(right, other), max(right, other))]
            merged_score = min(left_other, right_other)
            key = (min(merged_id, other), max(merged_id, other))
            pair_scores[key] = merged_score
            if merged_score + 1e-15 >= threshold:
                first, second = sorted(
                    ((merged_id, merged_signature), (other, signatures[other])),
                    key=lambda item: item[1],
                )
                heapq.heappush(
                    queue,
                    (-merged_score, first[1], second[1], first[0], second[0]),
                )
    return tuple(
        active[index]
        for index in sorted(active, key=lambda item: signatures[item])
    )


def _family_id(rules: Sequence[Rule]) -> str:
    signature = "|".join(sorted(rule.rule_id for rule in rules))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return f"rule-family-{digest}"


def _recurrences(
    rules_and_bootstraps: Sequence[tuple[Rule, set[int]]],
    *,
    total_bootstraps: int,
    clinical_groups: bool,
) -> tuple[Recurrence, ...]:
    by_name: dict[str, set[int]] = {}
    for rule, bootstrap_ids in rules_and_bootstraps:
        names = _rule_groups(rule) if clinical_groups else _rule_features(rule)
        for name in names:
            by_name.setdefault(name, set()).update(bootstrap_ids)
    return tuple(
        Recurrence(name, len(bootstrap_ids), len(bootstrap_ids) / total_bootstraps)
        for name, bootstrap_ids in sorted(by_name.items())
    )


def _threshold_variability(rules: Sequence[Rule]) -> tuple[ThresholdVariability, ...]:
    values: dict[tuple[str, str], list[tuple[float, float | None]]] = {}
    for rule in rules:
        for condition in rule.conditions:
            values.setdefault((condition.feature_name, condition.operator), []).append(
                (condition.threshold, condition.normalized_threshold)
            )
    result: list[ThresholdVariability] = []
    for (feature_name, operator), thresholds in sorted(values.items()):
        normalized = [value for _, value in thresholds if value is not None]
        result.append(
            ThresholdVariability(
                feature_name=feature_name,
                operator=operator,
                raw=_distribution([raw for raw, _ in thresholds]),
                normalized=(
                    _distribution(normalized)
                    if len(normalized) == len(thresholds)
                    else None
                ),
            )
        )
    return tuple(result)


def cluster_rule_families(
    discovery: StableRuleDiscoveryResult | Sequence[CandidateRuleOccurrence],
    X_reference: np.ndarray,
    config: RuleSimilarityConfig | None = None,
    *,
    total_bootstraps: int | None = None,
) -> RuleFamilyClusteringResult:
    """Cluster candidate rules and retain families recurrent across bootstraps.

    Exact repeated rules are collapsed before clustering.  Recurrence and
    feature/group recurrence count each outer ``bootstrap_id`` once, regardless
    of trees or duplicate candidate occurrences.
    """

    similarity_config = config or RuleSimilarityConfig()
    if hasattr(discovery, "occurrences") and hasattr(
        discovery, "bootstrap_diagnostics"
    ):
        occurrences = tuple(discovery.occurrences)  # type: ignore[union-attr]
        inferred_bootstraps = len(discovery.bootstrap_diagnostics)  # type: ignore[union-attr]
    else:
        occurrences = tuple(discovery)
        inferred_bootstraps = (
            max((item.bootstrap_id for item in occurrences), default=-1) + 1
        )
    denominator = inferred_bootstraps if total_bootstraps is None else total_bootstraps
    if denominator < 1:
        if occurrences:
            raise ValueError("total_bootstraps must be positive")
        denominator = 1

    by_rule_id: dict[str, tuple[Rule, list[CandidateRuleOccurrence]]] = {}
    for occurrence in occurrences:
        stored = by_rule_id.get(occurrence.rule.rule_id)
        if stored is None:
            by_rule_id[occurrence.rule.rule_id] = (occurrence.rule, [occurrence])
        else:
            if stored[0] != occurrence.rule:
                raise ValueError(f"rule ID collision: {occurrence.rule.rule_id}")
            stored[1].append(occurrence)
    rules = tuple(value[0] for _, value in sorted(by_rule_id.items()))
    if not rules:
        return RuleFamilyClusteringResult(
            families=(),
            total_bootstraps=denominator,
            n_occurrences=0,
            n_unique_rules=0,
            config=similarity_config,
        )

    masks = tuple(rule.mask(X_reference) for rule in rules)
    pairwise = np.eye(len(rules), dtype=float)
    detailed: dict[tuple[int, int], RuleSimilarity] = {}
    for left in range(len(rules)):
        for right in range(left + 1, len(rules)):
            similarity = _rule_similarity_from_masks(
                rules[left], rules[right], masks[left], masks[right], similarity_config
            )
            pairwise[left, right] = pairwise[right, left] = similarity.total
            detailed[(left, right)] = similarity
    clusters = _complete_link_clusters(
        pairwise,
        [rule.rule_id for rule in rules],
        similarity_config.similarity_threshold,
    )

    families: list[RuleFamily] = []
    for members in clusters:
        member_rules = tuple(rules[index] for index in members)
        occurrences_by_rule = [
            by_rule_id[rule.rule_id][1] for rule in member_rules
        ]
        bootstraps_by_rule = [
            {item.bootstrap_id for item in rule_occurrences}
            for rule_occurrences in occurrences_by_rule
        ]
        all_bootstraps = tuple(sorted(set().union(*bootstraps_by_rule)))
        recurrence = len(all_bootstraps) / denominator

        medoid_candidates: list[tuple[float, int, str, Rule]] = []
        for local_index, rule in enumerate(member_rules):
            global_index = members[local_index]
            mean_similarity = float(pairwise[global_index, list(members)].mean())
            medoid_candidates.append(
                (-mean_similarity, rule.length, rule.rule_id, rule)
            )
        representative = min(medoid_candidates)[3]

        cohort_overlaps = [
            detailed[(min(left, right), max(left, right))].cohort_jaccard
            for position, left in enumerate(members)
            for right in members[position + 1 :]
        ]
        if not cohort_overlaps:
            cohort_overlaps = [1.0]
        rules_and_bootstraps = list(zip(member_rules, bootstraps_by_rule))
        # Weight threshold summaries once per outer bootstrap, never once per
        # tree occurrence. This preserves recurrence information without
        # treating many trees from one resample as independent evidence.
        bootstrap_weighted_rules = tuple(
            rule
            for rule, bootstrap_ids in rules_and_bootstraps
            for _bootstrap_id in sorted(bootstrap_ids)
        )
        family = RuleFamily(
            family_id=_family_id(member_rules),
            member_rules=member_rules,
            representative_rule=representative,
            occurrence_count=sum(len(items) for items in occurrences_by_rule),
            bootstrap_ids=all_bootstraps,
            recurrence_frequency=float(recurrence),
            retained=recurrence + 1e-15 >= similarity_config.min_recurrence,
            feature_recurrence={
                item.name: item.frequency
                for item in _recurrences(
                    rules_and_bootstraps,
                    total_bootstraps=denominator,
                    clinical_groups=False,
                )
            },
            clinical_group_recurrence={},
            threshold_variability=_threshold_variability(bootstrap_weighted_rules),
            cohort_stability=_distribution(cohort_overlaps, singleton_default=1.0),
            _occurrence_references=tuple(
                sorted(
                    (
                        item.bootstrap_id,
                        item.rule.rule_id,
                        item.tree_id,
                        item.leaf_id,
                    )
                    for rule_occurrences in occurrences_by_rule
                    for item in rule_occurrences
                )
            ),
        )
        families.append(family)
    families.sort(
        key=lambda family: (
            not family.retained,
            -family.recurrence_frequency,
            family.family_id,
        )
    )
    return RuleFamilyClusteringResult(
        families=tuple(families),
        total_bootstraps=denominator,
        n_occurrences=len(occurrences),
        n_unique_rules=len(rules),
        config=similarity_config,
    )


def retained_representatives(result: RuleFamilyClusteringResult) -> tuple[Rule, ...]:
    """Return one deterministic medoid from each recurrent family."""

    return tuple(family.representative_rule for family in result.retained_families)


def recurrent_representatives(
    families: RuleFamilyClusteringResult | Sequence[RuleFamily],
    minimum_recurrence: float,
) -> tuple[Rule, ...]:
    """Compatibility helper for filtering an explicit family collection."""

    if not 0.0 <= minimum_recurrence <= 1.0:
        raise ValueError("minimum_recurrence must lie in [0, 1]")
    values = families.families if isinstance(families, RuleFamilyClusteringResult) else families
    return tuple(
        family.representative_rule
        for family in values
        if family.recurrence_frequency + 1e-15 >= minimum_recurrence
    )


@dataclass(frozen=True)
class SetAgreement:
    left: tuple[str, ...]
    right: tuple[str, ...]
    intersection: tuple[str, ...]
    exact: bool
    jaccard: float

    def to_dict(self) -> dict[str, object]:
        return {
            "left": list(self.left),
            "right": list(self.right),
            "intersection": list(self.intersection),
            "exact": self.exact,
            "jaccard": self.jaccard,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SetAgreement":
        return cls(
            left=tuple(str(item) for item in value["left"]),  # type: ignore[union-attr]
            right=tuple(str(item) for item in value["right"]),  # type: ignore[union-attr]
            intersection=tuple(str(item) for item in value["intersection"]),  # type: ignore[union-attr]
            exact=bool(value["exact"]),
            jaccard=float(value["jaccard"]),
        )


def _set_agreement(left: set[str], right: set[str]) -> SetAgreement:
    return SetAgreement(
        left=tuple(sorted(left)),
        right=tuple(sorted(right)),
        intersection=tuple(sorted(left & right)),
        exact=left == right,
        jaccard=float(_set_jaccard(left, right)),
    )


@dataclass(frozen=True)
class DirectionalTransfer:
    source_factor_id: str
    target_factor_id: str
    metrics: BinaryMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "source_factor_id": self.source_factor_id,
            "target_factor_id": self.target_factor_id,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DirectionalTransfer":
        metrics = value["metrics"]
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        return cls(
            source_factor_id=str(value["source_factor_id"]),
            target_factor_id=str(value["target_factor_id"]),
            metrics=BinaryMetrics(**metrics),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SymmetricMetric:
    left_to_right: float
    right_to_left: float
    mean: float
    minimum: float

    @classmethod
    def from_values(cls, left_to_right: float, right_to_left: float) -> "SymmetricMetric":
        return cls(
            left_to_right=float(left_to_right),
            right_to_left=float(right_to_left),
            mean=float((left_to_right + right_to_left) / 2.0),
            minimum=float(min(left_to_right, right_to_left)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "left_to_right": self.left_to_right,
            "right_to_left": self.right_to_left,
            "mean": self.mean,
            "minimum": self.minimum,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SymmetricMetric":
        return cls(**{key: float(raw) for key, raw in value.items()})


@dataclass(frozen=True)
class SemanticPairComparison:
    left_factor_id: str
    right_factor_id: str
    threshold_name: str | None
    n_samples: int
    left_to_right: DirectionalTransfer
    right_to_left: DirectionalTransfer
    left_self_metrics: BinaryMetrics
    right_self_metrics: BinaryMetrics
    symmetric_metrics: dict[str, SymmetricMetric]
    exact_feature_agreement: SetAgreement
    clinical_group_agreement: SetAgreement
    selected_cohort_jaccard: float
    activation_target_jaccard: float

    @property
    def t_mean(self) -> float:
        return self.symmetric_metrics["f2"].mean

    @property
    def t_min(self) -> float:
        return self.symmetric_metrics["f2"].minimum

    @property
    def i_to_j(self) -> BinaryMetrics:
        return self.left_to_right.metrics

    @property
    def j_to_i(self) -> BinaryMetrics:
        return self.right_to_left.metrics

    @property
    def mean(self) -> dict[str, float]:
        return {name: metric.mean for name, metric in self.symmetric_metrics.items()}

    @property
    def minimum(self) -> dict[str, float]:
        return {
            name: metric.minimum for name, metric in self.symmetric_metrics.items()
        }

    @property
    def target_cohort_jaccard(self) -> float:
        return self.activation_target_jaccard

    @property
    def exact_feature_equal(self) -> bool:
        return self.exact_feature_agreement.exact

    @property
    def clinical_group_equal(self) -> bool:
        return self.clinical_group_agreement.exact

    def to_dict(self) -> dict[str, object]:
        return {
            "left_factor_id": self.left_factor_id,
            "right_factor_id": self.right_factor_id,
            "threshold_name": self.threshold_name,
            "n_samples": self.n_samples,
            "left_to_right": self.left_to_right.to_dict(),
            "right_to_left": self.right_to_left.to_dict(),
            "left_self_metrics": self.left_self_metrics.to_dict(),
            "right_self_metrics": self.right_self_metrics.to_dict(),
            "symmetric_metrics": {
                name: metric.to_dict()
                for name, metric in sorted(self.symmetric_metrics.items())
            },
            "exact_feature_agreement": self.exact_feature_agreement.to_dict(),
            "selected_cohort_jaccard": self.selected_cohort_jaccard,
            "activation_target_jaccard": self.activation_target_jaccard,
            "t_mean": self.t_mean,
            "t_min": self.t_min,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SemanticPairComparison":
        symmetric = value["symmetric_metrics"]
        if not isinstance(symmetric, Mapping):
            raise TypeError("symmetric_metrics must be a mapping")
        return cls(
            left_factor_id=str(value["left_factor_id"]),
            right_factor_id=str(value["right_factor_id"]),
            threshold_name=(
                None if value.get("threshold_name") is None else str(value["threshold_name"])
            ),
            n_samples=int(value["n_samples"]),
            left_to_right=DirectionalTransfer.from_dict(value["left_to_right"]),  # type: ignore[arg-type]
            right_to_left=DirectionalTransfer.from_dict(value["right_to_left"]),  # type: ignore[arg-type]
            left_self_metrics=BinaryMetrics(**value["left_self_metrics"]),  # type: ignore[arg-type]
            right_self_metrics=BinaryMetrics(**value["right_self_metrics"]),  # type: ignore[arg-type]
            symmetric_metrics={
                str(name): SymmetricMetric.from_dict(metric)  # type: ignore[arg-type]
                for name, metric in symmetric.items()
            },
            exact_feature_agreement=SetAgreement.from_dict(value["exact_feature_agreement"]),  # type: ignore[arg-type]
            clinical_group_agreement=SetAgreement.from_dict(
                value.get(
                    "clinical_group_agreement",
                    {
                        "left": [],
                        "right": [],
                        "intersection": [],
                        "jaccard": 1.0,
                        "exact": True,
                    },
                )  # type: ignore[arg-type]
            ),
            selected_cohort_jaccard=float(value["selected_cohort_jaccard"]),
            activation_target_jaccard=float(value["activation_target_jaccard"]),
        )


@dataclass(frozen=True)
class ClassSemanticPairComparison:
    """One frozen semantic comparison evaluated within an outcome class."""

    class_value: object
    n_samples: int
    left_target_positive_count: int
    right_target_positive_count: int
    valid: bool
    reasons: tuple[str, ...]
    comparison: SemanticPairComparison

    def to_dict(self) -> dict[str, object]:
        return {
            "class_value": self.class_value,
            "n_samples": self.n_samples,
            "left_target_positive_count": self.left_target_positive_count,
            "right_target_positive_count": self.right_target_positive_count,
            "valid": self.valid,
            "reasons": list(self.reasons),
            "comparison": self.comparison.to_dict(),
        }


def compare_semantic_pair(
    left_rule_set: RuleSet,
    right_rule_set: RuleSet,
    X_final: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    *,
    left_factor_id: str,
    right_factor_id: str,
    threshold_name: str | None = None,
) -> SemanticPairComparison:
    """Evaluate both rule-transfer directions on identical final records."""

    array = np.asarray(X_final)
    left_truth = np.asarray(left_target, dtype=bool)
    right_truth = np.asarray(right_target, dtype=bool)
    if array.ndim != 2:
        raise ValueError("X_final must be two-dimensional")
    if left_truth.ndim != 1 or right_truth.ndim != 1:
        raise ValueError("targets must be one-dimensional")
    if not (array.shape[0] == left_truth.size == right_truth.size):
        raise ValueError("X_final and both targets must contain identical rows")

    left_selected = left_rule_set.mask(array)
    right_selected = right_rule_set.mask(array)
    left_to_right_metrics = binary_metrics(right_truth, left_selected)
    right_to_left_metrics = binary_metrics(left_truth, right_selected)
    symmetric: dict[str, SymmetricMetric] = {}
    for name in ("precision", "recall", "f2", "lift", "wracc", "jaccard"):
        symmetric[name] = SymmetricMetric.from_values(
            float(getattr(left_to_right_metrics, name)),
            float(getattr(right_to_left_metrics, name)),
        )

    left_features = {
        condition.feature_name
        for rule in left_rule_set.rules
        for condition in rule.conditions
    }
    right_features = {
        condition.feature_name
        for rule in right_rule_set.rules
        for condition in rule.conditions
    }
    return SemanticPairComparison(
        left_factor_id=str(left_factor_id),
        right_factor_id=str(right_factor_id),
        threshold_name=threshold_name,
        n_samples=array.shape[0],
        left_to_right=DirectionalTransfer(
            str(left_factor_id), str(right_factor_id), left_to_right_metrics
        ),
        right_to_left=DirectionalTransfer(
            str(right_factor_id), str(left_factor_id), right_to_left_metrics
        ),
        left_self_metrics=binary_metrics(left_truth, left_selected),
        right_self_metrics=binary_metrics(right_truth, right_selected),
        symmetric_metrics=symmetric,
        exact_feature_agreement=_set_agreement(left_features, right_features),
        # Retain an empty compatibility field for legacy callers without
        # consulting clinical annotations in new semantic comparisons.
        clinical_group_agreement=_set_agreement(set(), set()),
        selected_cohort_jaccard=cohort_jaccard(left_selected, right_selected),
        activation_target_jaccard=cohort_jaccard(left_truth, right_truth),
    )


def compare_rule_sets_symmetric(
    rule_set_i: RuleSet,
    target_i: np.ndarray,
    rule_set_j: RuleSet,
    target_j: np.ndarray,
    X_final: np.ndarray,
    *,
    factor_i_id: str = "i",
    factor_j_id: str = "j",
    threshold_name: str | None = None,
) -> SemanticPairComparison:
    """Evaluate ``i`` to ``j`` and ``j`` to ``i`` without collapsing scores."""

    first = np.asarray(target_i)
    second = np.asarray(target_j)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("targets must be aligned one-dimensional arrays")
    array = np.asarray(X_final)
    if array.ndim != 2 or array.shape[0] != first.shape[0]:
        raise ValueError("X_final and targets must contain identical records")
    return compare_semantic_pair(
        rule_set_i,
        rule_set_j,
        array,
        first,
        second,
        left_factor_id=factor_i_id,
        right_factor_id=factor_j_id,
        threshold_name=threshold_name,
    )


def compare_rule_sets_by_class(
    rule_set_i: RuleSet,
    target_i: np.ndarray,
    rule_set_j: RuleSet,
    target_j: np.ndarray,
    X_final: np.ndarray,
    class_labels: np.ndarray,
    *,
    factor_i_id: str = "i",
    factor_j_id: str = "j",
    threshold_name: str | None = None,
) -> tuple[ClassSemanticPairComparison, ...]:
    """Evaluate frozen rule sets and targets within each observed final class."""

    first = np.asarray(target_i)
    second = np.asarray(target_j)
    array = np.asarray(X_final)
    labels = np.asarray(class_labels)
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("targets must be aligned one-dimensional arrays")
    if array.ndim != 2 or array.shape[0] != first.shape[0]:
        raise ValueError("X_final and targets must contain identical records")
    if labels.ndim != 1:
        raise ValueError("class_labels must be one-dimensional")
    if labels.shape[0] != array.shape[0]:
        raise ValueError(
            "class_labels, X_final, and targets must contain identical records"
        )

    try:
        class_values, class_indices = np.unique(labels, return_inverse=True)
    except TypeError as error:
        raise ValueError(
            "class_labels must contain mutually comparable scalar values"
        ) from error

    results: list[ClassSemanticPairComparison] = []
    for class_index, raw_class_value in enumerate(class_values):
        class_mask = class_indices == class_index
        class_first = first[class_mask].astype(bool, copy=False)
        class_second = second[class_mask].astype(bool, copy=False)
        left_positive_count = int(np.count_nonzero(class_first))
        right_positive_count = int(np.count_nonzero(class_second))
        reasons: list[str] = []
        if left_positive_count == 0:
            reasons.append("left_target_has_no_positive_samples")
        if right_positive_count == 0:
            reasons.append("right_target_has_no_positive_samples")

        class_value = (
            raw_class_value.item()
            if isinstance(raw_class_value, np.generic)
            else raw_class_value
        )
        comparison = compare_rule_sets_symmetric(
            rule_set_i,
            class_first,
            rule_set_j,
            class_second,
            array[class_mask],
            factor_i_id=factor_i_id,
            factor_j_id=factor_j_id,
            threshold_name=threshold_name,
        )
        results.append(
            ClassSemanticPairComparison(
                class_value=class_value,
                n_samples=int(np.count_nonzero(class_mask)),
                left_target_positive_count=left_positive_count,
                right_target_positive_count=right_positive_count,
                valid=not reasons,
                reasons=tuple(reasons),
                comparison=comparison,
            )
        )
    return tuple(results)


__all__ = [
    "ClassSemanticPairComparison",
    "DirectionalTransfer",
    "DistributionSummary",
    "Recurrence",
    "RuleFamily",
    "RuleFamilyClusteringResult",
    "RuleSimilarity",
    "RuleSimilarityConfig",
    "RuleSimilarityWeights",
    "SemanticPairComparison",
    "SetAgreement",
    "SymmetricMetric",
    "ThresholdVariability",
    "cluster_rule_families",
    "compare_rule_sets_by_class",
    "compare_rule_sets_symmetric",
    "compare_semantic_pair",
    "recurrent_representatives",
    "retained_representatives",
    "rule_similarity",
]
