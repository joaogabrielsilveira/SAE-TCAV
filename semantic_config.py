"""Configuration for cross-run SAE semantic comparison.

The semantic pipeline is opt-in.  Defaults are deliberately small enough for a
development run; publication runs should override bootstrap/tree counts in a
version-controlled JSON file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ActivationTargetConfig:
    """Positive-activation fractions represented by each binary target."""

    positive_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)
    min_positive_samples: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive_fractions", tuple(self.positive_fractions))
        if not self.positive_fractions or any(not 0 < value <= 1 for value in self.positive_fractions):
            raise ValueError("positive_fractions must be in (0, 1]")
        if self.min_positive_samples < 1:
            raise ValueError("min_positive_samples must be positive")


@dataclass(frozen=True)
class RuleObjectiveConfig:
    objective: str = "f2"
    min_precision: float = 0.50
    min_lift: float = 1.50
    max_rules: int = 5
    max_rule_length: int = 3
    min_marginal_recall: float = 0.02
    exhaustive_candidate_limit: int = 20
    beam_width: int = 64

    def __post_init__(self) -> None:
        if self.objective not in {"f2", "recall"}:
            raise ValueError("objective must be 'f2' or 'recall'")
        if not 0 <= self.min_precision <= 1 or self.min_lift < 0:
            raise ValueError("Invalid precision or lift constraint")
        if self.max_rules < 1 or self.max_rule_length < 1:
            raise ValueError("Rule count and length limits must be positive")


@dataclass(frozen=True)
class DiscoveryConfig:
    backend: str = "randomized_tree"
    n_bootstraps: int = 30
    trees_per_bootstrap: int = 100
    max_depth: int = 3
    min_samples_leaf: float = 0.01
    max_features: str | int | float | None = "sqrt"
    splitter: str = "random"
    positive_leaf_probability: float = 0.50
    min_positive_leaf_samples: int = 2
    max_candidates_per_bootstrap: int = 20
    min_family_recurrence: float = 0.50
    family_similarity_threshold: float = 0.70

    def __post_init__(self) -> None:
        if self.backend != "randomized_tree":
            raise ValueError("Only backend='randomized_tree' is currently supported")
        if self.n_bootstraps < 1 or self.trees_per_bootstrap < 1:
            raise ValueError("Bootstrap and tree counts must be positive")
        if not 0 < self.min_family_recurrence <= 1:
            raise ValueError("min_family_recurrence must be in (0, 1]")
        if not 0 <= self.family_similarity_threshold <= 1:
            raise ValueError("family_similarity_threshold must be in [0, 1]")
        if self.max_candidates_per_bootstrap < 1:
            raise ValueError("max_candidates_per_bootstrap must be positive")
        if self.splitter not in {"best", "random"}:
            raise ValueError("splitter must be 'best' or 'random'")
        if not 0 <= self.positive_leaf_probability <= 1:
            raise ValueError("positive_leaf_probability must be in [0, 1]")


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    n_jobs: int = 1
    cache: bool = True
    artifact_dir: str = "stats/semantic"

    def __post_init__(self) -> None:
        if self.n_jobs != 1:
            raise ValueError("Only n_jobs=1 is supported for deterministic v1 execution")


@dataclass(frozen=True)
class ClassAnalysisConfig:
    """Additive held-out evaluation stratified by observed outcome class."""

    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("class_analysis.enabled must be a boolean")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ClassAnalysisConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("class_analysis must be a JSON object")
        unknown = set(raw) - {"enabled"}
        if unknown:
            raise ValueError(f"Unknown class_analysis fields: {sorted(unknown)}")
        return cls(enabled=raw.get("enabled", True))


@dataclass(frozen=True)
class SemanticExperimentConfig:
    schema_version: str = "1.0"
    activation_targets: ActivationTargetConfig = field(default_factory=ActivationTargetConfig)
    objective: RuleObjectiveConfig = field(default_factory=RuleObjectiveConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    class_analysis: ClassAnalysisConfig = field(default_factory=ClassAnalysisConfig)
    clinical_groups_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SemanticExperimentConfig":
        known = {
            "schema_version",
            "activation_targets",
            "objective",
            "discovery",
            "runtime",
            "class_analysis",
            "clinical_groups_path",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown semantic config fields: {sorted(unknown)}")
        return cls(
            schema_version=str(raw.get("schema_version", "1.0")),
            activation_targets=ActivationTargetConfig(**raw.get("activation_targets", {})),
            objective=RuleObjectiveConfig(**raw.get("objective", {})),
            discovery=DiscoveryConfig(**raw.get("discovery", {})),
            runtime=RuntimeConfig(**raw.get("runtime", {})),
            class_analysis=ClassAnalysisConfig.from_dict(raw.get("class_analysis", {})),
            clinical_groups_path=raw.get("clinical_groups_path"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "SemanticExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def load_clinical_groups(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    """Load feature-to-group mapping; unmapped features remain singleton groups."""

    if path is None:
        return {}
    with Path(path).open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Clinical group mapping must be a JSON object")
    result: dict[str, tuple[str, ...]] = {}
    for feature, groups in raw.items():
        if isinstance(groups, str):
            groups = [groups]
        if not groups or not all(isinstance(group, str) and group for group in groups):
            raise ValueError(f"Invalid clinical groups for feature {feature!r}")
        result[str(feature)] = tuple(sorted(set(groups)))
    return result
