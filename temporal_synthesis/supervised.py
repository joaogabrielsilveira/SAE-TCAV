"""Reference-cluster uncertainty for already-computed OOF predictions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def reference_cluster_interval(values: Sequence[Mapping[str, Any]], *, value_name: str,
                               repetitions: int, seed: int) -> tuple[float | None, float | None]:
    """Bootstrap reference clusters, preserving all within-reference rows."""
    groups: dict[int, list[float]] = {}
    for row in values:
        value = row.get(value_name)
        if value is not None and np.isfinite(value):
            groups.setdefault(int(row["reference_year"]), []).append(float(value))
    references = sorted(groups)
    if not references or repetitions < 1:
        return None, None
    per_reference = np.asarray([np.mean(groups[reference]) for reference in references])
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(per_reference, size=(repetitions, len(per_reference)), replace=True), axis=1)
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def paired_reference_sign_flip(differences: Sequence[float], *, repetitions: int, seed: int) -> float | None:
    """Two-sided paired randomization test over reference-level loss differences."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    observed = abs(float(values.mean()))
    if len(values) <= 16:
        signs = 1 - 2 * ((np.arange(2 ** len(values))[:, None] >> np.arange(len(values))) & 1)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice((-1, 1), size=(max(1, repetitions), len(values)))
    permuted = np.abs(np.mean(signs * values, axis=1))
    return float((1 + np.sum(permuted >= observed - 1e-15)) / (len(permuted) + 1))


def compare_oof_models(predictions: Sequence[Mapping[str, Any]], *, candidate: str,
                       baseline: str = "performance_history", repetitions: int = 1000,
                       seed: int = 42) -> dict[str, Any]:
    """Compare paired OOF absolute errors at the reference-cluster level."""
    identity = ("reference_year", "patient_split_seed", "cohort_view", "activation_target",
                "temporal_distance", "target_year")
    def key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return tuple(row.get(name) for name in identity)
    baseline_rows = {key(row): row for row in predictions if row.get("model") == baseline and row.get("absolute_error") is not None}
    candidate_rows = {key(row): row for row in predictions if row.get("model") == candidate and row.get("absolute_error") is not None}
    paired = []
    for row_key in sorted(set(baseline_rows) & set(candidate_rows), key=str):
        candidate_row, baseline_row = candidate_rows[row_key], baseline_rows[row_key]
        paired.append({"reference_year": int(candidate_row["reference_year"]),
                       "improvement": float(baseline_row["absolute_error"] - candidate_row["absolute_error"])})
    references = sorted({row["reference_year"] for row in paired})
    differences = [float(np.mean([row["improvement"] for row in paired if row["reference_year"] == reference])) for reference in references]
    low, high = reference_cluster_interval(paired, value_name="improvement", repetitions=repetitions, seed=seed)
    return {"candidate_model": candidate, "baseline_model": baseline, "paired_row_count": len(paired),
            "paired_reference_count": len(references), "mean_improvement": None if not differences else float(np.mean(differences)),
            "bootstrap_95_low": low, "bootstrap_95_high": high,
            "paired_sign_flip_p_value": paired_reference_sign_flip(differences, repetitions=repetitions, seed=seed)}
