"""Artifact-stage lead/lag and exploratory change-point analysis."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def orient_degradation(metric_name: str, values: Sequence[float], baseline: float) -> np.ndarray:
    """Orient each metric separately; never create a composite score."""

    data = np.asarray(values, dtype=float)
    lower_is_degradation = {
        "macro_f1", "death_f1", "rule_transfer", "target_prediction_jaccard",
        "common_patient_jaccard", "cav_cosine", "direction_agreement",
    }
    higher_is_degradation = {
        "dead_fraction", "underused_fraction", "absolute_smd", "tcav_difference",
    }
    if metric_name in lower_is_degradation:
        return float(baseline) - data
    if metric_name in higher_is_degradation:
        return data - float(baseline)
    raise ValueError(f"degradation orientation not declared for {metric_name!r}")


def lead_lag_rows(
    rows: Sequence[Mapping[str, object]], concept_metric: str, *, lag: int = 1
) -> list[dict[str, object]]:
    """Align split-aggregated concept degradation with later F1 degradation."""

    if lag not in {1, 2}:
        raise ValueError("lead/lag supports one-year primary and two-year sensitivity")
    by_group = {}
    for row in rows:
        key = (int(row["reference_year"]), int(row["patient_split_seed"]))
        by_group.setdefault(key, {})[int(row["temporal_distance"])] = row
    aligned = []
    for (reference, split), timeline in by_group.items():
        for distance, source in timeline.items():
            target = timeline.get(distance + lag)
            previous = timeline.get(distance + lag - 1)
            if target is None or previous is None:
                continue
            values = (source.get(concept_metric), target.get("f1_degradation"), previous.get("f1_degradation"))
            if any(value is None or not np.isfinite(value) for value in values):
                continue
            aligned.append(
                {
                    "reference_year": reference,
                    "patient_split_seed": split,
                    "temporal_distance": distance + lag,
                    "lag": lag,
                    "concept_metric": concept_metric,
                    "lagged_concept_degradation": float(values[0]),
                    "f1_degradation": float(values[1]),
                    "previous_f1_degradation": float(values[2]),
                    "contemporaneous_concept_degradation": float(target.get(concept_metric, np.nan)),
                }
            )
    return aligned


def fit_lead_lag_regression(
    rows: Sequence[Mapping[str, object]], *, replicates: int = 2000, seed: int = 42
) -> dict[str, object]:
    """Fit controlled longitudinal OLS; cluster-bootstrap reference/split groups."""

    if len(rows) < 5:
        return {"valid": False, "failure_reason": "insufficient_lead_lag_rows"}
    columns = (
        "lagged_concept_degradation", "temporal_distance",
        "previous_f1_degradation", "contemporaneous_concept_degradation",
    )
    X = np.column_stack(
        [np.ones(len(rows))]
        + [np.asarray([float(row[column]) for row in rows]) for column in columns]
    )
    y = np.asarray([float(row["f1_degradation"]) for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        return {"valid": False, "failure_reason": "nonfinite_lead_lag_inputs"}
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault((int(row["reference_year"]), int(row["patient_split_seed"])), []).append(index)
    keys = tuple(groups)
    rng = np.random.default_rng(seed)
    samples = []
    if len(keys) >= 2:
        for _ in range(replicates):
            chosen = rng.integers(0, len(keys), len(keys))
            indices = np.concatenate([groups[keys[position]] for position in chosen])
            if len(indices) >= X.shape[1]:
                samples.append(float(np.linalg.lstsq(X[indices], y[indices], rcond=None)[0][1]))
    interval = np.percentile(samples, [2.5, 97.5]) if samples else (np.nan, np.nan)
    return {
        "valid": True,
        "failure_reason": None,
        "row_count": len(rows),
        "cluster_count": len(keys),
        "lag_coefficient": float(coefficients[1]),
        "lag_coefficient_lower_95": float(interval[0]),
        "lag_coefficient_upper_95": float(interval[1]),
        "control_coefficients": {
            name: float(value) for name, value in zip(columns[1:], coefficients[2:])
        },
        "exploratory": True,
    }


def first_sustained_crossing(
    distances: Sequence[int], values: Sequence[float], threshold: float
) -> int | None:
    ordered = sorted(zip(distances, values))
    for (distance, value), (next_distance, next_value) in zip(ordered, ordered[1:]):
        if next_distance == distance + 1 and value >= threshold and next_value >= threshold:
            return int(distance)
    return None


def first_interval_departure(
    distances: Sequence[int], lower: Sequence[float], upper: Sequence[float],
    baseline_interval: tuple[float, float],
) -> int | None:
    baseline_lower, baseline_upper = baseline_interval
    ordered = sorted(zip(distances, lower, upper))
    for distance, cell_lower, cell_upper in ordered:
        if distance < 2:
            continue
        if cell_lower > baseline_upper or cell_upper < baseline_lower:
            return int(distance)
    return None


def segmented_breakpoint(distances: Sequence[int], values: Sequence[float]) -> dict[str, object]:
    x = np.asarray(distances, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 5:
        return {"valid": False, "failure_reason": "insufficient_segmented_observations"}

    def bic(design):
        residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        rss = max(float(residual @ residual), np.finfo(float).tiny)
        return len(y) * np.log(rss / len(y)) + design.shape[1] * np.log(len(y))

    no_break_bic = bic(np.column_stack([np.ones(len(x)), x]))
    candidates = []
    unique = np.unique(x)
    for breakpoint in unique[2:-2]:
        design = np.column_stack([np.ones(len(x)), x, np.maximum(0, x - breakpoint)])
        candidates.append((bic(design), float(breakpoint)))
    if not candidates:
        return {"valid": False, "failure_reason": "no_eligible_breakpoint"}
    segmented_bic, breakpoint = min(candidates)
    return {
        "valid": True,
        "breakpoint": breakpoint,
        "no_break_bic": no_break_bic,
        "segmented_bic": segmented_bic,
        "bic_improvement": no_break_bic - segmented_bic,
        "segmented_preferred": segmented_bic < no_break_bic,
        "exploratory": True,
    }


def aggregate_breakpoint_replicated(rows: Sequence[Mapping[str, object]]) -> bool:
    """Require ±1-year recurrence in 3 SAE seeds, splits, and references."""

    valid = [row for row in rows if row.get("breakpoint") is not None]
    for center in sorted({int(round(float(row["breakpoint"]))) for row in valid}):
        nearby = [row for row in valid if abs(float(row["breakpoint"]) - center) <= 1]
        if (
            len({int(row["member_sae_seed"]) for row in nearby}) >= 3
            and len({int(row["patient_split_seed"]) for row in nearby}) >= 3
            and len({int(row["reference_year"]) for row in nearby}) >= 3
        ):
            return True
    return False


def individual_factor_breakpoint_replicated(
    rows: Sequence[Mapping[str, object]], *, tolerance: int = 1
) -> list[dict[str, object]]:
    """Replicate only among SAE members sharing one exact factor-family UID."""

    by_family = {}
    for row in rows:
        uid = str(row["factor_family_uid"])
        parts = uid.split("/")
        if len(parts) != 4:
            raise ValueError("factor_family_uid must encode reference/split/canonical seed/factor")
        if int(row["reference_year"]) != int(parts[0]) or int(row["patient_split_seed"]) != int(parts[1]):
            raise ValueError("factor breakpoint row crosses experiment identity")
        if row.get("breakpoint") is not None:
            by_family.setdefault(uid, []).append(row)
    results = []
    for uid, family_rows in sorted(by_family.items()):
        replicated = False
        center = None
        for candidate in sorted({int(round(float(row["breakpoint"]))) for row in family_rows}):
            nearby = [row for row in family_rows if abs(float(row["breakpoint"]) - candidate) <= tolerance]
            if len({int(row["member_sae_seed"]) for row in nearby}) >= 3:
                replicated = True
                center = candidate
                break
        results.append(
            {
                "factor_family_uid": uid,
                "replicated": replicated,
                "breakpoint": center,
                "scope": "within_factor_family_only",
                "exploratory": True,
            }
        )
    return results
