"""Declarative definitions of the prespecified metric profiles."""

from __future__ import annotations

from dataclasses import dataclass


CORE_METRICS = (
    "u_f2",
    "u_jaccard",
    "u_prevalence",
    "u_activation",
    "u_feature_association",
)
EXTENDED_METRICS = CORE_METRICS + ("u_tcav",)


@dataclass(frozen=True)
class MetricProfile:
    name: str
    metrics: tuple[str, ...]
    activation_target: float | None = None
    dimensionality_role: str = "primary"


METRIC_PROFILES = {
    "core": MetricProfile("core", CORE_METRICS),
    "p50_tcav_extended": MetricProfile(
        "p50_tcav_extended", EXTENDED_METRICS, .5, "audited_sensitivity"
    ),
}


def profile_metrics(profile: str) -> tuple[str, ...]:
    try:
        return METRIC_PROFILES[profile].metrics
    except KeyError as error:
        raise ValueError(f"unknown metric profile: {profile}") from error
