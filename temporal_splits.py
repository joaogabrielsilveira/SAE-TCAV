"""Reference-year-only patient roles and deterministic support retries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ROLE_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("tabpfn_context", 0.50),
    ("sae_discovery", 0.125),
    ("rule_discovery", 0.125),
    ("rule_selection_cav", 0.125),
    ("t0_evaluation", 0.125),
)
FITTING_ROLES = frozenset(name for name, _ in ROLE_FRACTIONS[:-1])


@dataclass(frozen=True)
class ReferenceSplit:
    requested_seed: int
    effective_seed: int
    attempt: int
    roles: Mapping[str, np.ndarray]
    support: Mapping[str, Mapping[str, int]]


def _largest_remainder(size: int, fractions: np.ndarray) -> np.ndarray:
    exact = fractions * size
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[: size - int(counts.sum())]] += 1
    return counts


def split_reference_patients(
    patient_ids: Sequence[object],
    outcomes: Sequence[int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Assign reference patients to five disjoint roles, stratified only by T=0."""

    patients = np.asarray(patient_ids).astype(str)
    y = np.asarray(outcomes, dtype=int)
    if patients.ndim != 1 or y.ndim != 1 or len(patients) != len(y):
        raise ValueError("patient_ids and outcomes must be aligned one-dimensional arrays")
    if np.any((y != 0) & (y != 1)):
        raise ValueError("outcomes must be binary")
    unique, inverse = np.unique(patients, return_inverse=True)
    patient_outcome = np.zeros(len(unique), dtype=int)
    for index in range(len(unique)):
        values = np.unique(y[inverse == index])
        if len(values) != 1:
            raise ValueError("reference patient has conflicting outcomes")
        patient_outcome[index] = values[0]

    fractions = np.asarray([fraction for _, fraction in ROLE_FRACTIONS])
    assignments = np.full(len(unique), -1, dtype=int)
    rng = np.random.default_rng(int(seed))
    for outcome in (0, 1):
        members = np.flatnonzero(patient_outcome == outcome)
        rng.shuffle(members)
        counts = _largest_remainder(len(members), fractions)
        start = 0
        for role_index, count in enumerate(counts):
            assignments[members[start : start + count]] = role_index
            start += count
    if np.any(assignments < 0):
        raise AssertionError("patient assignment incomplete")

    roles = {
        role: np.flatnonzero(assignments[inverse] == role_index)
        for role_index, (role, _) in enumerate(ROLE_FRACTIONS)
    }
    role_patients = [set(patients[indices]) for indices in roles.values()]
    if any(
        role_patients[left] & role_patients[right]
        for left in range(len(role_patients))
        for right in range(left + 1, len(role_patients))
    ):
        raise AssertionError("patient leakage between reference roles")
    if sum(len(indices) for indices in roles.values()) != len(patients):
        raise AssertionError("reference split lost records")
    return roles


def role_support(
    roles: Mapping[str, np.ndarray], outcomes: Sequence[int], patient_ids: Sequence[object]
) -> dict[str, dict[str, int]]:
    y = np.asarray(outcomes, dtype=int)
    patients = np.asarray(patient_ids).astype(str)
    result = {}
    for role, indices in roles.items():
        idx = np.asarray(indices, dtype=int)
        result[role] = {
            "records": int(len(idx)),
            "patients": int(len(np.unique(patients[idx]))),
            "deaths": int(np.count_nonzero(y[idx] == 1)),
            "survivors": int(np.count_nonzero(y[idx] == 0)),
        }
    return result


def reference_support_failures(support, thresholds) -> list[dict[str, object]]:
    requirements = {
        "tabpfn_context": (thresholds.context_deaths, thresholds.context_survivors),
        "t0_evaluation": (thresholds.t0_deaths, thresholds.t0_survivors),
    }
    failures = []
    for role, (minimum_deaths, minimum_survivors) in requirements.items():
        observed = support[role]
        if observed["deaths"] < minimum_deaths or observed["survivors"] < minimum_survivors:
            failures.append(
                {
                    "reason": "insufficient_class_support",
                    "role": role,
                    "observed_deaths": observed["deaths"],
                    "observed_survivors": observed["survivors"],
                    "required_deaths": minimum_deaths,
                    "required_survivors": minimum_survivors,
                }
            )
    return failures


def generate_valid_reference_splits(
    patient_ids: Sequence[object],
    outcomes: Sequence[int],
    requested_seeds: Sequence[int],
    thresholds,
    *,
    maximum_attempts: int = 100,
) -> tuple[list[ReferenceSplit], list[dict[str, object]]]:
    """Try configured seeds first, then deterministic derived replacements."""

    requested = tuple(int(seed) for seed in requested_seeds)
    valid: list[ReferenceSplit] = []
    attempts: list[dict[str, object]] = []
    used: set[int] = set()
    for attempt in range(maximum_attempts):
        effective_seed = (
            requested[attempt]
            if attempt < len(requested)
            else int(np.random.SeedSequence([requested[-1], attempt]).generate_state(1)[0])
        )
        requested_seed = effective_seed
        if effective_seed in used:
            continue
        used.add(effective_seed)
        roles = split_reference_patients(patient_ids, outcomes, seed=effective_seed)
        support = role_support(roles, outcomes, patient_ids)
        failures = reference_support_failures(support, thresholds)
        attempts.append(
            {
                "attempt": attempt,
                "requested_seed": requested_seed,
                "effective_seed": effective_seed,
                "valid": not failures,
                "failures": failures,
                "support": support,
            }
        )
        if not failures:
            valid.append(ReferenceSplit(requested_seed, effective_seed, attempt, roles, support))
            if len(valid) == len(requested):
                break
    return valid, attempts
