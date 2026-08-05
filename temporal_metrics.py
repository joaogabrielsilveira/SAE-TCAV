"""Support-aware temporal performance, retention, similarity, and uncertainty."""

from __future__ import annotations

from statistics import NormalDist
from typing import Callable, Mapping, Sequence

import numpy as np


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    truth = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    if truth.shape != pred.shape or truth.ndim != 1:
        raise ValueError("classification inputs must be aligned vectors")
    scores = []
    for label in (0, 1):
        tp = int(np.count_nonzero((truth == label) & (pred == label)))
        fp = int(np.count_nonzero((truth != label) & (pred == label)))
        fn = int(np.count_nonzero((truth == label) & (pred != label)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    deaths = int(np.count_nonzero(truth == 1))
    survivors = int(np.count_nonzero(truth == 0))
    return {
        "sample_count": int(len(truth)),
        "death_count": deaths,
        "survivor_count": survivors,
        "prevalence": deaths / len(truth) if len(truth) else float("nan"),
        "macro_f1": float(np.mean(scores)),
        "death_f1": float(scores[1]),
    }


def support_aware_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], *, minimum_deaths: int, minimum_survivors: int
) -> dict[str, object]:
    metrics = classification_metrics(y_true, y_pred)
    valid = metrics["death_count"] >= minimum_deaths and metrics["survivor_count"] >= minimum_survivors
    if not valid:
        metrics["macro_f1"] = None
        metrics["death_f1"] = None
    return {
        **metrics,
        "valid": bool(valid),
        "failure_reason": None if valid else "insufficient_class_support",
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total < 1:
        return float("nan"), float("nan")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie in [0, total]")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return float(max(0, center - margin)), float(min(1, center + margin))


def prevalence_retention(
    t0_active: int,
    t0_total: int,
    future_active: int,
    future_total: int,
    config,
) -> dict[str, object]:
    p0 = t0_active / t0_total if t0_total else 0.0
    pt = future_active / future_total if future_total else 0.0
    rho = pt / p0 if p0 > 0 else None
    if t0_active < config.minimum_t0_active or p0 <= 0:
        status = "insufficient_reference_support"
    elif future_total < config.minimum_evaluation_records or (
        0 < future_active < config.minimum_future_active
    ):
        status = "insufficient_future_support"
    elif pt == 0 or (
        pt < config.activation_prevalence_floor
        and future_active >= config.minimum_future_active
    ):
        status = "dead_absent"
    elif rho < 0.75:
        status = "underused"
    elif rho <= 1.50:
        status = "stable"
    else:
        status = "overused"
    return {
        "t0_active_count": int(t0_active),
        "t0_denominator": int(t0_total),
        "future_active_count": int(future_active),
        "future_denominator": int(future_total),
        "t0_prevalence": p0,
        "future_prevalence": pt,
        "prevalence_ratio": rho,
        "t0_prevalence_interval": wilson_interval(t0_active, t0_total),
        "future_prevalence_interval": wilson_interval(future_active, future_total),
        "status": status,
    }


def patient_bootstrap_ratio_interval(
    t0_patient_active: Mapping[object, bool],
    future_patient_active: Mapping[object, bool],
    *,
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    patients = np.asarray(sorted(set(t0_patient_active) | set(future_patient_active), key=str), dtype=object)
    if not len(patients):
        return float("nan"), float("nan")
    t0 = np.asarray([bool(t0_patient_active.get(patient, False)) for patient in patients])
    future = np.asarray([bool(future_patient_active.get(patient, False)) for patient in patients])
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(replicates):
        indices = rng.integers(0, len(patients), size=len(patients))
        denominator = float(np.mean(t0[indices]))
        if denominator > 0:
            ratios.append(float(np.mean(future[indices]) / denominator))
    if not ratios:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.percentile(ratios, [2.5, 97.5]))


def target_prediction_jaccard(target: Sequence[bool], prediction: Sequence[bool]) -> float:
    left = np.asarray(target, dtype=bool)
    right = np.asarray(prediction, dtype=bool)
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("within-year masks must be aligned vectors")
    union = int(np.count_nonzero(left | right))
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def common_patient_jaccard(
    t0_patient_ids: Sequence[object],
    t0_selected: Sequence[bool],
    future_patient_ids: Sequence[object],
    future_selected: Sequence[bool],
    *,
    minimum_common_patients: int = 30,
) -> dict[str, object]:
    def reduce(ids, selected):
        result = {}
        for patient, value in zip(np.asarray(ids).astype(str), np.asarray(selected, dtype=bool)):
            result[patient] = result.get(patient, False) or bool(value)
        return result

    left = reduce(t0_patient_ids, t0_selected)
    right = reduce(future_patient_ids, future_selected)
    common = sorted(set(left) & set(right))
    union = sum(left[patient] or right[patient] for patient in common)
    valid = len(common) >= minimum_common_patients and union > 0
    return {
        "common_patient_count": len(common),
        "jaccard": (
            sum(left[patient] and right[patient] for patient in common) / union
            if valid
            else None
        ),
        "valid": valid,
        "failure_reason": None if valid else (
            "insufficient_common_patients" if len(common) < minimum_common_patients else "empty_union"
        ),
    }


def feature_distribution_shift(
    t0: np.ndarray, future: np.ndarray, *, t0_selected=None, future_selected=None
) -> dict[str, float | int | None]:
    reference = np.asarray(t0, dtype=float)
    test = np.asarray(future, dtype=float)
    if reference.ndim != 2 or test.ndim != 2 or reference.shape[1] != test.shape[1]:
        raise ValueError("feature matrices must be two-dimensional with equal columns")

    def summarize(left, right, prefix):
        if len(left) == 0 or len(right) == 0:
            return {f"{prefix}median_abs_smd": None, f"{prefix}p90_abs_smd": None}
        mean0 = np.nanmean(left, axis=0)
        scale0 = np.nanstd(left, axis=0, ddof=1)
        diff = np.abs(np.nanmean(right, axis=0) - mean0)
        smd = np.divide(diff, scale0, out=np.full(diff.shape, np.nan), where=scale0 > 0)
        finite = smd[np.isfinite(smd)]
        return {
            f"{prefix}median_abs_smd": float(np.median(finite)) if len(finite) else None,
            f"{prefix}p90_abs_smd": float(np.percentile(finite, 90)) if len(finite) else None,
        }

    result = summarize(reference, test, "")
    if t0_selected is not None and future_selected is not None:
        left_mask = np.asarray(t0_selected, dtype=bool)
        right_mask = np.asarray(future_selected, dtype=bool)
        result.update(summarize(reference[left_mask], test[right_mask], "selected_"))
        result["t0_selected_count"] = int(np.count_nonzero(left_mask))
        result["future_selected_count"] = int(np.count_nonzero(right_mask))
    return result


def percentile_interval(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "mean": None, "median": None, "std": None, "lower_95": None, "upper_95": None}
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "lower_95": float(np.percentile(finite, 2.5)),
        "upper_95": float(np.percentile(finite, 97.5)),
    }


def bootstrap_interval(
    values: Sequence[float], *, replicates: int = 2000, seed: int = 42, statistic: Callable = np.mean
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = [statistic(finite[rng.integers(0, len(finite), len(finite))]) for _ in range(replicates)]
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def hierarchical_bootstrap_interval(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    *,
    split_key: str = "patient_split_seed",
    member_key: str = "member_sae_seed",
    family_key: str = "factor_family_uid",
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Aggregate families within split, then resample splits and SAE members."""

    nested = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not np.isfinite(value):
            continue
        split = row[split_key]
        member = row[member_key]
        family = row[family_key]
        nested.setdefault(split, {}).setdefault(member, {}).setdefault(family, []).append(float(value))
    splits = tuple(sorted(nested, key=str))
    if not splits:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled_splits = rng.integers(0, len(splits), len(splits))
        split_values = []
        for position in sampled_splits:
            members = nested[splits[position]]
            member_keys = tuple(sorted(members, key=str))
            sampled_members = rng.integers(0, len(member_keys), len(member_keys))
            member_values = []
            for member_position in sampled_members:
                families = members[member_keys[member_position]]
                family_values = [np.mean(values) for values in families.values()]
                member_values.append(float(np.mean(family_values)))
            split_values.append(float(np.mean(member_values)))
        estimates.append(float(np.mean(split_values)))
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def validate_performance_rows_not_sae_duplicated(
    rows: Sequence[Mapping[str, object]],
) -> None:
    """F1 lives once per reference/split/test/cohort, never once per SAE join."""

    key_fields = (
        "reference_year", "patient_split_seed", "test_year", "cohort_view"
    )
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"duplicate performance cell: {key}")
        seen.add(key)
        if row.get("member_sae_seed") is not None:
            raise ValueError("performance row must not carry an SAE member key")


def triangular_matrix(rows, metric: str, reference_years: Sequence[int], test_years: Sequence[int]):
    references = tuple(sorted(set(int(year) for year in reference_years)))
    tests = tuple(sorted(set(int(year) for year in test_years)))
    values = np.full((len(references), len(tests)), np.nan)
    support = np.zeros(values.shape, dtype=int)
    grouped = {}
    for row in rows:
        key = (int(row["reference_year"]), int(row["test_year"]))
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouped.setdefault(key, []).append(float(value))
    for row_index, reference in enumerate(references):
        for column_index, test in enumerate(tests):
            if test < reference:
                continue
            cell = grouped.get((reference, test), [])
            support[row_index, column_index] = len(cell)
            if cell:
                values[row_index, column_index] = np.mean(cell)
    return values, support
