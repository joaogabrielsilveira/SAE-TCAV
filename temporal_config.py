"""Configuration for leakage-safe reference-year temporal experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Mapping
import warnings


def _ordered_unique(values, name: str, lower: float, upper: float):
    normalized = tuple(type(lower)(value) for value in values)
    if (
        not normalized
        or normalized != tuple(sorted(set(normalized)))
        or any(value < lower or value > upper for value in normalized)
    ):
        raise ValueError(f"{name} must be ordered, unique, and lie in [{lower}, {upper}]")
    return normalized


@dataclass(frozen=True)
class TemporalSupportConfig:
    context_deaths: int = 20
    context_survivors: int = 50
    t0_deaths: int = 10
    t0_survivors: int = 30
    factor_positive_activations: int = 20
    target_high_records_per_role: int = 10
    selected_rule_records: int = 10
    cav_positive_records: int = 50
    cav_negative_records: int = 50

    def __post_init__(self) -> None:
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("all temporal support thresholds must be positive")


@dataclass(frozen=True)
class TemporalMatchingConfig:
    cosine_analysis_thresholds: tuple[float, ...] = (0.50, 0.60, 0.70, 0.80)
    overlap_analysis_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80)
    overlap_percentiles: tuple[int, ...] = (70, 80, 90)
    headline_cosine_threshold: float = 0.60
    headline_overlap_threshold: float = 0.70

    def __post_init__(self) -> None:
        cosine = _ordered_unique(
            self.cosine_analysis_thresholds,
            "cosine_analysis_thresholds",
            -1.0,
            1.0,
        )
        overlap = _ordered_unique(
            self.overlap_analysis_thresholds,
            "overlap_analysis_thresholds",
            0.0,
            1.0,
        )
        percentiles = _ordered_unique(
            self.overlap_percentiles, "overlap_percentiles", 0, 100
        )
        object.__setattr__(self, "cosine_analysis_thresholds", cosine)
        object.__setattr__(self, "overlap_analysis_thresholds", overlap)
        object.__setattr__(self, "overlap_percentiles", percentiles)
        if self.headline_cosine_threshold not in cosine:
            raise ValueError("headline cosine threshold must be configured")
        if self.headline_overlap_threshold not in overlap:
            raise ValueError("headline overlap threshold must be configured")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TemporalMatchingConfig":
        values = dict(raw)
        migrations = (
            ("cosine_analysis_threshold", "cosine_analysis_thresholds"),
            ("overlap_analysis_threshold", "overlap_analysis_thresholds"),
        )
        for legacy, plural in migrations:
            if legacy in values:
                if plural in values:
                    raise ValueError(f"cannot specify both {legacy} and {plural}")
                warnings.warn(
                    f"{legacy} is deprecated; use {plural}",
                    DeprecationWarning,
                    stacklevel=2,
                )
                values[plural] = (values.pop(legacy),)
                headline = (
                    "headline_cosine_threshold"
                    if legacy.startswith("cosine")
                    else "headline_overlap_threshold"
                )
                values.setdefault(headline, values[plural][0])
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown temporal matching fields: {sorted(unknown)}")
        return cls(**values)


@dataclass(frozen=True)
class TemporalRetentionConfig:
    minimum_t0_active: int = 10
    minimum_future_active: int = 5
    activation_prevalence_floor: float = 0.005
    minimum_evaluation_records: int = 30

    def __post_init__(self) -> None:
        if self.minimum_t0_active < 1 or self.minimum_future_active < 1:
            raise ValueError("active-count thresholds must be positive")
        if not 0 <= self.activation_prevalence_floor <= 1:
            raise ValueError("activation_prevalence_floor must lie in [0, 1]")
        if self.minimum_evaluation_records < 1:
            raise ValueError("minimum_evaluation_records must be positive")


@dataclass(frozen=True)
class TemporalRobustnessConfig:
    schema_version: str = "2.0"
    dataset_path: str = "tidy_event_data.feather"
    artifact_dir: str = "stats/temporal_robustness"
    comparison_config_path: str = "comparison_runner.example.json"
    semantic_config_path: str = "semantic_experiment.example.json"
    use_cache: bool = True
    show_progress: bool = True
    device: str = "auto"
    force: bool = False
    reference_years: tuple[int, ...] = tuple(range(2007, 2016))
    patient_split_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    sae_seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    activation_positive_fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5)
    feature_selection_max_year: int | None = None
    maximum_split_attempts: int = 100
    bootstrap_replicates: int = 2000
    bootstrap_seed: int = 42
    support: TemporalSupportConfig = field(default_factory=TemporalSupportConfig)
    matching: TemporalMatchingConfig = field(default_factory=TemporalMatchingConfig)
    retention: TemporalRetentionConfig = field(default_factory=TemporalRetentionConfig)

    def __post_init__(self) -> None:
        for name in ("reference_years", "patient_split_seeds", "sae_seeds"):
            values = tuple(int(value) for value in getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must contain unique values")
            object.__setattr__(self, name, values)
        fractions = tuple(float(value) for value in self.activation_positive_fractions)
        if (
            not fractions
            or fractions != tuple(sorted(set(fractions)))
            or any(not 0 < value <= 1 for value in fractions)
        ):
            raise ValueError("activation_positive_fractions must be ordered, unique, and in (0, 1]")
        object.__setattr__(self, "activation_positive_fractions", fractions)
        if self.maximum_split_attempts < len(self.patient_split_seeds):
            raise ValueError("maximum_split_attempts cannot be smaller than requested splits")
        if self.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        for name in ("use_cache", "show_progress", "force"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if (
            self.feature_selection_max_year is not None
            and self.feature_selection_max_year > min(self.reference_years)
        ):
            raise ValueError("feature vocabulary uses a year later than earliest reference year")

    @property
    def canonical_sae_seed(self) -> int:
        return self.sae_seeds[0]

    def development_profile(self) -> "TemporalRobustnessConfig":
        """Return predeclared two-by-two development profile."""

        return replace(
            self,
            patient_split_seeds=self.patient_split_seeds[:2],
            sae_seeds=self.sae_seeds[:2],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "TemporalRobustnessConfig":
        if not isinstance(raw, Mapping):
            raise ValueError("temporal configuration must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown temporal config fields: {sorted(unknown)}")
        values = dict(raw)
        values["support"] = TemporalSupportConfig(**values.get("support", {}))
        values["matching"] = TemporalMatchingConfig.from_dict(values.get("matching", {}))
        values["retention"] = TemporalRetentionConfig(**values.get("retention", {}))
        return cls(**values)

    @classmethod
    def from_json(cls, path: str | Path) -> "TemporalRobustnessConfig":
        config_path = Path(path).resolve()
        with config_path.open(encoding="utf-8") as handle:
            config = cls.from_dict(json.load(handle))
        return replace(
            config,
            dataset_path=str((config_path.parent / config.dataset_path).resolve())
            if not Path(config.dataset_path).is_absolute()
            else config.dataset_path,
            artifact_dir=str((config_path.parent / config.artifact_dir).resolve())
            if not Path(config.artifact_dir).is_absolute()
            else config.artifact_dir,
            comparison_config_path=str(
                (config_path.parent / config.comparison_config_path).resolve()
            ) if not Path(config.comparison_config_path).is_absolute()
            else config.comparison_config_path,
            semantic_config_path=str(
                (config_path.parent / config.semantic_config_path).resolve()
            ) if not Path(config.semantic_config_path).is_absolute()
            else config.semantic_config_path,
        )
