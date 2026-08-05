"""Reference-relative domains and exhaustive future provenance cohorts."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from temporal_splits import FITTING_ROLES


PROVENANCE_COHORTS = (
    "returning_t0",
    "returning_fitting",
    "new_entrant",
    "prior_nonreference_returner",
)
COHORT_VIEWS = ("all_comer", "pipeline_unseen", *PROVENANCE_COHORTS)


def relative_domain_map(reference_year: int, years: Sequence[int]) -> dict[int, int]:
    mapping = {int(year): int(year) - int(reference_year) for year in sorted(set(years))}
    if reference_year not in mapping:
        mapping[int(reference_year)] = 0
    return dict(sorted(mapping.items()))


def relative_domains(reference_year: int, years: Sequence[int]) -> np.ndarray:
    values = np.asarray(years, dtype=int)
    domains = values - int(reference_year)
    if np.any(domains < 0):
        raise ValueError("temporal evaluation cannot use years before reference year")
    return domains


def assert_aligned_domains(
    expected: Sequence[int], **stage_domains: Sequence[int]
) -> None:
    expected_values = np.asarray(expected, dtype=int)
    for stage, values in stage_domains.items():
        actual = np.asarray(values, dtype=int)
        if actual.shape != expected_values.shape or not np.array_equal(actual, expected_values):
            raise ValueError(f"{stage} domain IDs do not align with records")


def role_patient_sets(
    reference_patient_ids: Sequence[object], roles: Mapping[str, Sequence[int]]
) -> dict[str, set[str]]:
    patients = np.asarray(reference_patient_ids).astype(str)
    return {
        role: set(patients[np.asarray(indices, dtype=int)].tolist())
        for role, indices in roles.items()
    }


def assign_future_provenance(
    patient_ids: Sequence[object],
    first_eligible_year: Mapping[object, int],
    reference_year: int,
    reference_role_patients: Mapping[str, set[str]],
) -> np.ndarray:
    """Assign exactly one required provenance category to each future row."""

    patients = np.asarray(patient_ids).astype(str)
    fitting = set().union(
        *(reference_role_patients.get(role, set()) for role in FITTING_ROLES)
    )
    t0 = set(reference_role_patients.get("t0_evaluation", set()))
    if fitting & t0:
        raise ValueError("reference fitting and T=0 patient sets overlap")
    first = {str(patient): int(year) for patient, year in first_eligible_year.items()}
    labels = []
    for patient in patients:
        if patient in t0:
            label = "returning_t0"
        elif patient in fitting:
            label = "returning_fitting"
        elif patient not in first:
            raise ValueError(f"missing first eligible year for patient {patient!r}")
        elif first[patient] > reference_year:
            label = "new_entrant"
        elif first[patient] < reference_year:
            label = "prior_nonreference_returner"
        else:
            raise ValueError("reference-year patient lacks a reference role")
        labels.append(label)
    result = np.asarray(labels, dtype=str)
    if not np.isin(result, PROVENANCE_COHORTS).all():
        raise AssertionError("future provenance partition is incomplete")
    return result


def cohort_masks(provenance: Sequence[str]) -> dict[str, np.ndarray]:
    labels = np.asarray(provenance).astype(str)
    masks = {name: labels == name for name in PROVENANCE_COHORTS}
    masks["all_comer"] = np.ones(len(labels), dtype=bool)
    masks["pipeline_unseen"] = masks["returning_t0"] | masks["new_entrant"] | masks[
        "prior_nonreference_returner"
    ]
    partition_total = sum(mask.astype(int) for name, mask in masks.items() if name in PROVENANCE_COHORTS)
    if not np.all(partition_total == 1):
        raise AssertionError("future provenance is not exhaustive and mutually exclusive")
    return masks


def cohort_baseline_kind(cohort_view: str) -> str:
    if cohort_view == "returning_t0":
        return "paired"
    if cohort_view in COHORT_VIEWS:
        return "benchmark"
    raise ValueError(f"unknown cohort view {cohort_view!r}")


def metric_delta(
    future_value: float | None,
    *,
    complete_t0_value: float | None,
    paired_t0_value: float | None,
    cohort_view: str,
) -> dict[str, float | bool | None]:
    """Keep paired and benchmark differences semantically separate."""

    result: dict[str, float | bool | None] = {
        "paired_delta": None,
        "benchmark_delta": None,
        "delta_not_defined": False,
    }
    if future_value is None or not np.isfinite(future_value):
        result["delta_not_defined"] = True
    elif cohort_view == "returning_t0":
        if paired_t0_value is None or not np.isfinite(paired_t0_value):
            result["delta_not_defined"] = True
        else:
            result["paired_delta"] = float(future_value - paired_t0_value)
    elif complete_t0_value is None or not np.isfinite(complete_t0_value):
        result["delta_not_defined"] = True
    else:
        result["benchmark_delta"] = float(future_value - complete_t0_value)
    return result
