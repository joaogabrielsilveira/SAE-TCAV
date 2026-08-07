"""Parent-level artifact summaries; no model fitting or configuration tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from artifact_storage import read_artifact

from temporal_analysis import (
    fit_lead_lag_regression,
    first_sustained_crossing,
    lead_lag_rows,
    segmented_breakpoint,
)
from temporal_metrics import (
    bootstrap_interval,
    percentile_interval,
)


def build_parent_reports(root: Path, successful, config) -> dict[str, list[dict]]:
    tables = _load(successful)
    performance = tables.get("performance", [])
    factors = tables.get("factor_year_metrics", [])
    tcav = tables.get("tcav", [])
    return {
        "distance_summaries": _distance(performance, factors, tcav),
        "uncertainty": _uncertainty(performance, factors, config),
        "lead_lag": _lead_lag(performance, factors, config),
        "change_points": _change_points(performance, factors),
        "triangular_matrices": _matrices(performance, factors, tcav),
    }


def _load(successful):
    tables = {}
    for entry in successful:
        manifest_path = Path(entry["manifest"])
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        for name, descriptor in manifest.get("artifacts", {}).items():
            try:
                value = read_artifact(directory, descriptor)
            except (OSError, ValueError, TypeError):
                # Legacy manifests sometimes named a CSV even when the JSON
                # table was the typed canonical copy.
                legacy = directory / f"{name}.json"
                if not legacy.is_file():
                    continue
                value = json.loads(legacy.read_text())
            if isinstance(value, list):
                tables.setdefault(name, []).extend(value)
    return tables


def _groups(rows, fields):
    result = {}
    for row in rows:
        result.setdefault(tuple(row.get(field) for field in fields), []).append(row)
    return result


def _distance(performance, factors, tcav):
    output = []
    specs = (
        (performance, ("temporal_distance", "cohort_view"), ("macro_f1", "death_f1"), "performance"),
        (factors, ("temporal_distance", "cohort_view", "rule_source", "target_role"), ("f2", "prevalence_ratio"), "concept"),
        (tcav, ("temporal_distance", "cohort_view", "rule_source", "target_role"), ("tcav",), "tcav"),
    )
    for rows, fields, metrics, source in specs:
        for key, values in _groups(rows, fields).items():
            for metric in metrics:
                summary = percentile_interval([row.get(metric, np.nan) for row in values])
                output.append({
                    **dict(zip(fields, key)), "metric": metric,
                    "source": source, **summary,
                })
    return output


def _uncertainty(performance, factors, config):
    output = []
    p_fields = ("reference_year", "test_year", "cohort_view")
    for key, rows in _groups(performance, p_fields).items():
        for metric in ("macro_f1", "death_f1"):
            values = [row.get(metric, np.nan) for row in rows]
            lower, upper = bootstrap_interval(
                values, replicates=config.bootstrap_replicates,
                seed=config.bootstrap_seed,
            )
            output.append({
                **dict(zip(p_fields, key)), "metric": metric,
                "lower_95": lower, "upper_95": upper,
                "valid_repetition_count": len([value for value in values if value is not None]),
                "resampling_unit": "patient_split",
            })
    f_fields = (
        "reference_year", "test_year", "cohort_view", "rule_source",
        "activation_target", "target_role",
    )
    for key, rows in _groups(factors, f_fields).items():
        for metric in ("f2", "prevalence_ratio"):
            try:
                lower, upper = _hierarchical_bootstrap_interval(
                    rows, metric, replicates=config.bootstrap_replicates,
                    seed=config.bootstrap_seed,
                )
            except KeyError:
                lower, upper = np.nan, np.nan
            output.append({
                **dict(zip(f_fields, key)), "metric": metric,
                "lower_95": lower, "upper_95": upper,
                "valid_repetition_count": len(rows),
                "resampling_unit": "patient_split_then_sae_member",
                "exploratory": len({row.get("factor_family_uid") for row in rows}) < 10,
            })
    return output


def _hierarchical_bootstrap_interval(
    rows: Sequence[Mapping[str, object]],
    value_key: str,
    *,
    split_key: str = "patient_split_seed",
    member_key: str = "member_sae_seed",
    family_key: str = "factor_family_uid",
    replicates: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Result-equivalent parent bootstrap with invariant means precomputed."""

    nested = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not np.isfinite(value):
            continue
        split = row[split_key]
        member = row[member_key]
        family = row[family_key]
        nested.setdefault(split, {}).setdefault(member, {}).setdefault(
            family, []
        ).append(float(value))
    splits = tuple(sorted(nested, key=str))
    if not splits:
        return float("nan"), float("nan")

    # Family and member means do not depend on a bootstrap draw. Computing
    # them once preserves the original operation order and RNG sequence while
    # avoiding millions of identical reductions for large parent reports.
    member_means = []
    for split in splits:
        members = nested[split]
        values = []
        for member in sorted(members, key=str):
            families = members[member]
            family_values = [np.mean(items) for items in families.values()]
            values.append(float(np.mean(family_values)))
        member_means.append(np.asarray(values, dtype=float))

    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(replicates):
        sampled_splits = rng.integers(0, len(splits), len(splits))
        split_values = []
        for position in sampled_splits:
            values = member_means[position]
            sampled_members = rng.integers(0, len(values), len(values))
            split_values.append(float(np.mean(values[sampled_members])))
        estimates.append(float(np.mean(split_values)))
    return tuple(
        float(value) for value in np.percentile(estimates, [2.5, 97.5])
    )


def _lead_lag(performance, factors, config):
    f1 = {}
    for row in performance:
        if row.get("cohort_view") == "all_comer" and row.get("macro_f1") is not None:
            key = (int(row["reference_year"]), int(row["patient_split_seed"]))
            f1.setdefault(key, {})[int(row["temporal_distance"])] = float(row["macro_f1"])
    concepts = {}
    for row in factors:
        if row.get("cohort_view") != "all_comer" or row.get("target_role") != "primary":
            continue
        key = (int(row["reference_year"]), int(row["patient_split_seed"]), int(row["temporal_distance"]))
        value = row.get("prevalence_ratio")
        if value is not None:
            concepts.setdefault(key, []).append(1.0 - float(value))
    timeline = []
    for (reference, split), values in f1.items():
        if 0 not in values:
            continue
        baseline = values[0]
        for distance, value in values.items():
            concept = concepts.get((reference, split, distance), [])
            if concept:
                timeline.append({
                    "reference_year": reference, "patient_split_seed": split,
                    "temporal_distance": distance,
                    "f1_degradation": baseline-value,
                    "prevalence_degradation": float(np.mean(concept)),
                })
    output = []
    for lag in (1, 2):
        aligned = lead_lag_rows(timeline, "prevalence_degradation", lag=lag)
        result = fit_lead_lag_regression(
            aligned, replicates=config.bootstrap_replicates,
            seed=config.bootstrap_seed + lag,
        )
        output.append({
            "concept_metric": "prevalence_degradation", "lag": lag,
            "cohort_view": "all_comer", "rule_source": "semantic",
            "matching_view": "union", **result,
        })
    return output


def _change_points(performance, factors):
    output = []
    specs = ((performance, "macro_f1", False), (factors, "prevalence_ratio", True))
    for rows, metric, concept in specs:
        grouped = _groups(rows, ("reference_year",))
        for (reference,), values in grouped.items():
            by_distance = {}
            for row in values:
                if row.get("cohort_view") != "all_comer" or row.get(metric) is None:
                    continue
                by_distance.setdefault(int(row["temporal_distance"]), []).append(float(row[metric]))
            distances = sorted(by_distance)
            means = [float(np.mean(by_distance[distance])) for distance in distances]
            if concept:
                oriented = [1-value for value in means]
                threshold = 0.25
            elif means:
                oriented = [means[0]-value for value in means]
                threshold = 0.05
            else:
                oriented = []
                threshold = 0.05
            segmented = segmented_breakpoint(distances, oriented)
            output.append({
                "reference_year": reference, "metric": metric,
                "sustained_crossing": first_sustained_crossing(distances, oriented, threshold),
                **segmented,
            })
    return output


def _matrices(performance, factors, tcav):
    output = []
    for rows, metrics, source in (
        (performance, ("macro_f1", "death_f1"), "performance"),
        (factors, ("f2", "prevalence_ratio"), "concept"),
        (tcav, ("tcav",), "tcav"),
    ):
        for metric in metrics:
            for key, values in _groups(rows, ("reference_year", "test_year", "cohort_view" )).items():
                finite = [float(row[metric]) for row in values if row.get(metric) is not None and np.isfinite(row[metric])]
                output.append({
                    "reference_year": key[0], "test_year": key[1], "cohort_view": key[2],
                    "metric": metric, "source": source,
                    "value": float(np.mean(finite)) if finite else None,
                    "support": len(finite),
                })
    return output
