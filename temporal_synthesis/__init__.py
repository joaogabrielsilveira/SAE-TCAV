"""Maintainable components for downstream temporal metric synthesis.

The package is deliberately downstream-only.  In particular, it must never
import enrichment, CRI, CAV, TCAV, matching, SAE, or model builders.
"""

from .config import MetricSynthesisConfig, MetricSynthesisRuntimeConfig
from .dimensionality import bootstrap_stability, dimensionality, metric_quality
from .profiles import CORE_METRICS, EXTENDED_METRICS, METRIC_PROFILES, profile_metrics

__all__ = [
    "CORE_METRICS",
    "EXTENDED_METRICS",
    "METRIC_PROFILES",
    "MetricSynthesisConfig",
    "MetricSynthesisRuntimeConfig",
    "bootstrap_stability",
    "dimensionality",
    "metric_quality",
    "profile_metrics",
]
