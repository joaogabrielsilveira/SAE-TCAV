"""Canonical activation targets and deterministic per-source target winners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from semantic_rules import ActivationTargetSpec, FittedActivationTarget, fit_activation_target


def fit_canonical_targets(
    activations: Sequence[float], positive_fractions: Sequence[float], *, minimum_positive_samples: int = 20
) -> dict[float, FittedActivationTarget]:
    values = np.asarray(activations, dtype=float)
    return {
        float(fraction): fit_activation_target(
            values,
            ActivationTargetSpec(
                name=f"A={float(fraction):g}",
                positive_fraction=float(fraction),
                minimum_positive_samples=minimum_positive_samples,
            ),
        )
        for fraction in positive_fractions
    }


def compatibility_percentile(positive_fraction: float) -> float:
    return 100.0 * (1.0 - float(positive_fraction))


def activation_target_support(
    target: FittedActivationTarget,
    rule_discovery_activations: Sequence[float],
    rule_selection_activations: Sequence[float],
    *,
    minimum_high_records: int = 10,
) -> dict[str, object]:
    discovery_count = int(
        np.count_nonzero(target.apply(np.asarray(rule_discovery_activations, dtype=float)))
    )
    selection_count = int(
        np.count_nonzero(target.apply(np.asarray(rule_selection_activations, dtype=float)))
    )
    valid = (
        target.valid
        and discovery_count >= minimum_high_records
        and selection_count >= minimum_high_records
    )
    if not target.valid:
        reason = target.invalid_reason
    elif discovery_count < minimum_high_records:
        reason = "insufficient_rule_discovery_high_activation_records"
    elif selection_count < minimum_high_records:
        reason = "insufficient_rule_selection_high_activation_records"
    else:
        reason = None
    return {
        "valid": valid,
        "failure_reason": reason,
        "rule_discovery_high_count": discovery_count,
        "rule_selection_high_count": selection_count,
    }


def select_activation_winner(
    candidates: Sequence[Mapping[str, object]], *, rule_source: str
) -> list[dict[str, object]]:
    """Mark one primary A and preserve valid/invalid secondary rows."""

    if rule_source not in {"high_precision", "semantic"}:
        raise ValueError("rule_source must be high_precision or semantic")
    rows = [dict(row) for row in candidates]
    eligible = [
        row for row in rows
        if bool(row.get("valid", True)) and not bool(row.get("forced_fallback", False))
    ]
    if rule_source == "high_precision":
        eligible = [
            row for row in eligible
            if float(row.get("precision", 0)) >= 0.90
            and float(row.get("recall", 0)) >= 0.25
            and int(row.get("selected_count", 0)) >= int(row.get("minimum_support", 1))
        ]

        def key(row):
            return (
                float(row.get("f2", 0)),
                float(row.get("precision", 0)),
                float(row.get("lift", 0)),
                -int(row.get("condition_count", 10**9)),
                -float(row["activation_target"]),
            )
    else:
        def key(row):
            return (
                float(row.get("f2", 0)),
                float(row.get("precision", 0)),
                float(row.get("lift", 0)),
                -int(row.get("rule_count", 10**9)),
                -int(row.get("condition_count", 10**9)),
                -float(row["activation_target"]),
            )
    winner = max(eligible, key=key) if eligible else None
    for row in rows:
        row["rule_source"] = rule_source
        row["target_role"] = "primary" if row is winner else "secondary"
        row["compatibility_H"] = compatibility_percentile(float(row["activation_target"]))
        if bool(row.get("forced_fallback", False)):
            row["eligible_for_primary_cav"] = False
            row.setdefault("failure_reason", "forced_fallback_diagnostic_only")
        else:
            row["eligible_for_primary_cav"] = row is winner
    return rows


def validate_shared_target_masks(
    fitted_target: FittedActivationTarget,
    activations: Sequence[float],
    high_precision_mask: Sequence[bool],
    semantic_mask: Sequence[bool],
) -> None:
    expected = fitted_target.apply(np.asarray(activations, dtype=float))
    for name, mask in (
        ("high_precision", high_precision_mask), ("semantic", semantic_mask)
    ):
        if not np.array_equal(expected, np.asarray(mask, dtype=bool)):
            raise ValueError(f"{name} path did not reuse canonical activation target mask")
