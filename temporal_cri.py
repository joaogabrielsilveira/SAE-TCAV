"""MAUT-inspired Conceptual Robustness Index derived artifacts.

Consumes completed unified enrichment.  This module is deliberately downstream
of enrichment: CRI changes never invalidate temporal experiments, TCAV work, or
the primary original-system performance artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION, atomic_write_json, atomic_write_jsonl_gzip,
    canonical_json, file_sha256, read_artifact, validate_descriptor,
)
from temporal_unified_analysis import UnifiedAnalysisConfig, select_headline_factor_rows
from temporal_unified_enrichment import load_enrichment


@dataclass(frozen=True)
class CRIAnalysisConfig:
    """Fixed, interpretable CRI choices.  No Death-F1 tuning belongs here."""
    rule_source: str = "semantic"
    matching_view: str = "intersection"
    target_role: str = "primary"
    reference_floor: float = 1e-6
    degradation_threshold: float = 0.50
    tau_quantile: float = 0.90
    tau_anchor_utility: float = 0.50
    minimum_tau_samples: int = 20
    tau_fallback: float = 1.0
    equal_weights: tuple[float, float, float, float] = (.25, .25, .25, .25)
    learned_weight_step: float = .05
    prediction_clip: tuple[float, float] = (0.0, 1.0)

    def __post_init__(self) -> None:
        if self.matching_view != "intersection" or self.rule_source != "semantic":
            raise ValueError("first CRI version is semantic/intersection only")
        if self.target_role not in {"primary", "all"}:
            raise ValueError("target_role must be primary or all")
        if not 0 < self.reference_floor < 1:
            raise ValueError("reference_floor must lie in (0, 1)")
        if not 0 < self.degradation_threshold < 1:
            raise ValueError("degradation_threshold must lie in (0, 1)")
        if not 0 < self.tau_quantile < 1 or not 0 < self.tau_anchor_utility < 1:
            raise ValueError("tau calibration values must lie in (0, 1)")
        if self.minimum_tau_samples < 2 or self.tau_fallback <= 0:
            raise ValueError("tau support is invalid")
        if len(self.equal_weights) != 4 or any(value < 0 for value in self.equal_weights):
            raise ValueError("four nonnegative utility weights are required")
        if not np.isclose(sum(self.equal_weights), 1.0):
            raise ValueError("utility weights must sum to one")
        if not 0 < self.learned_weight_step <= 1 or not np.isclose(1 / self.learned_weight_step, round(1 / self.learned_weight_step)):
            raise ValueError("learned_weight_step must divide one")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MEMBER_KEYS = (
    "reference_year", "patient_split_seed", "factor_family_uid", "member_sae_seed",
    "member_factor_id", "activation_target",
)
FAMILY_KEYS = ("reference_year", "patient_split_seed", "factor_family_uid", "activation_target")
SYSTEM_KEYS = ("reference_year", "patient_split_seed", "cohort_view", "activation_target", "temporal_distance")


def _finite(value: object) -> bool:
    return value is not None and bool(np.isfinite(value))


def _key(row: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def _ratio_utility(ratio: object, tau: float) -> float | None:
    if not _finite(ratio) or float(ratio) <= 0:
        return None
    return float(np.exp(-abs(np.log(float(ratio))) / tau))


def _preservation(value: object, reference: object, floor: float) -> float | None:
    if not _finite(value) or not _finite(reference) or float(reference) <= floor:
        return None
    return float(min(1.0, (float(value) + np.finfo(float).eps) / (float(reference) + np.finfo(float).eps)))


def _geometric(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(values))))


def _headline_rows(rows: Sequence[Mapping[str, Any]], unified: UnifiedAnalysisConfig, config: CRIAnalysisConfig) -> list[dict[str, Any]]:
    selected = select_headline_factor_rows(rows, unified)
    return [dict(row) for row in selected if row.get("matching_view") == config.matching_view
            and row.get("rule_source") == config.rule_source
            and (config.target_role == "all" or row.get("target_role") == config.target_role)]


def _reference_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("cohort_view") != "all_comer" or int(row.get("temporal_distance", -1)) != 0:
            continue
        key = _key(row, MEMBER_KEYS)
        prior = output.setdefault(key, row)
        if prior is not row and any(prior.get(name) != row.get(name) for name in ("f2", "jaccard", "prevalence_ratio", "activation_magnitude")):
            raise ValueError(f"duplicate nonidentical d0 member row: {key}")
    return output


def _reference_eligible(row: Mapping[str, Any], floor: float) -> tuple[bool, str | None]:
    if row.get("status") == "insufficient_reference_support":
        return False, "insufficient_reference_support"
    for name in ("f2", "jaccard"):
        if not _finite(row.get(name)) or float(row[name]) <= floor:
            return False, f"invalid_reference_{name}"
    prevalence = row.get("prevalence_ratio")
    # d0 ratio is normally 1.  Require a real positive baseline ratio and magnitude.
    if not _finite(prevalence) or float(prevalence) <= 0:
        return False, "invalid_reference_prevalence"
    magnitude = row.get("activation_magnitude")
    if not _finite(magnitude) or float(magnitude) <= floor:
        return False, "invalid_reference_activation_magnitude"
    return True, None


def build_family_universe(rows: Sequence[Mapping[str, Any]], unified: UnifiedAnalysisConfig, config: CRIAnalysisConfig) -> list[dict[str, Any]]:
    """Freeze semantic/intersection exact members at all-comer d0."""
    reference = _reference_rows(_headline_rows(rows, unified, config))
    output = []
    for key, row in sorted(reference.items(), key=lambda item: str(item[0])):
        eligible, reason = _reference_eligible(row, config.reference_floor)
        output.append({**{name: value for name, value in zip(MEMBER_KEYS, key)},
                       "reference_status": row.get("status"), "reference_f2": row.get("f2"),
                       "reference_jaccard": row.get("jaccard"),
                       "reference_prevalence_ratio": row.get("prevalence_ratio"),
                       "reference_activation_magnitude": row.get("activation_magnitude"),
                       "eligible": eligible, "ineligible_reason": reason})
    return output


def _tau(values: Sequence[float], config: CRIAnalysisConfig) -> tuple[float, int, bool]:
    finite = np.asarray([value for value in values if np.isfinite(value) and value > 0], dtype=float)
    if len(finite) < config.minimum_tau_samples:
        return config.tau_fallback, int(len(finite)), True
    deviation = np.abs(np.log(finite / np.median(finite)))
    q = float(np.quantile(deviation, config.tau_quantile))
    if q <= np.finfo(float).eps:
        return config.tau_fallback, int(len(finite)), True
    return float(q / -np.log(config.tau_anchor_utility)), int(len(finite)), False


def calibrate_taus(rows: Sequence[Mapping[str, Any]], config: CRIAnalysisConfig) -> list[dict[str, Any]]:
    """Use raw reference retraining member dispersion; never Death F1."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("cohort_view") == "all_comer" and int(row.get("temporal_distance", -1)) == 0:
            grouped.setdefault((row.get("activation_target"),), []).append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        # Relative deviations within a family around its retraining median.
        by_family: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
        for row in values:
            by_family.setdefault(_key(row, FAMILY_KEYS), []).append(row)
        ratios: dict[str, list[float]] = {"prevalence": [], "activation": []}
        for members in by_family.values():
            for name, target in (("t0_prevalence", "prevalence"), ("activation_magnitude", "activation")):
                raw = [float(row[name]) for row in members if _finite(row.get(name)) and float(row[name]) > 0]
                if raw:
                    ratios[target].extend(float(value / np.median(raw)) for value in raw)
        for metric, values_ in ratios.items():
            tau, count, fallback = _tau(values_, config)
            output.append({"activation_target": key[0], "metric": metric, "tau": tau,
                           "reference_member_count": count, "used_fallback": fallback})
    return output


def _tau_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[Any, str], float]:
    return {(row["activation_target"], row["metric"]): float(row["tau"]) for row in rows}


def compute_member_utilities(
    rows: Sequence[Mapping[str, Any]], universe: Sequence[Mapping[str, Any]], taus: Sequence[Mapping[str, Any]],
    unified: UnifiedAnalysisConfig, config: CRIAnalysisConfig,
) -> list[dict[str, Any]]:
    selected = _headline_rows(rows, unified, config)
    index = {_key(row, MEMBER_KEYS + ("cohort_view", "temporal_distance")): row for row in selected}
    family_times: dict[tuple[Any, ...], dict[tuple[Any, int], Mapping[str, Any]]] = {}
    for row in selected:
        family_times.setdefault(_key(row, FAMILY_KEYS), {})[
            (row.get("cohort_view"), int(row["temporal_distance"]))
        ] = row
    tau_map = _tau_map(taus)
    output = []
    for member in universe:
        member_key = _key(member, MEMBER_KEYS)
        for (cohort, distance), representative in family_times.get(_key(member, FAMILY_KEYS), {}).items():
            row = index.get(member_key + (cohort, distance))
            status = None if row is None else row.get("status")
            available = bool(member["eligible"])
            reason = member.get("ineligible_reason")
            if available and (row is None or status in {"insufficient_reference_support", "insufficient_future_support", None}):
                available, reason = False, "technical_or_insufficient_support"
            if available:
                u_f = _preservation(row.get("f2"), member.get("reference_f2"), config.reference_floor)
                u_j = _preservation(row.get("jaccard"), member.get("reference_jaccard"), config.reference_floor)
                if status == "dead_absent":
                    u_p = u_a = 0.0
                else:
                    u_p = _ratio_utility(row.get("prevalence_ratio"), tau_map[(member["activation_target"], "prevalence")])
                    base = float(member["reference_activation_magnitude"])
                    current = row.get("activation_magnitude")
                    u_a = _ratio_utility(None if not _finite(current) else float(current) / base, tau_map[(member["activation_target"], "activation")])
                available = all(value is not None for value in (u_f, u_j, u_p, u_a))
                if not available:
                    reason = "technical_or_invalid_metric"
            else:
                u_f = u_j = u_p = u_a = None
            arithmetic = None if not available else float(np.dot(config.equal_weights, [u_f, u_j, u_p, u_a]))
            geometric = None if not available else _geometric([u_f, u_j, u_p, u_a])
            output.append({**{name: value for name, value in zip(MEMBER_KEYS, member_key)},
                           "cohort_view": cohort, "temporal_distance": distance,
                           "test_year": representative.get("test_year"), "status": status, "member_eligible": member["eligible"],
                           "utility_available": available, "missing_reason": reason,
                           "u_f2": u_f, "u_jaccard": u_j, "u_prevalence": u_p, "u_activation": u_a,
                           "semantic_cohort_utility": None if not available else float((u_f + u_j) / 2),
                           "activation_utility": None if not available else float((u_p + u_a) / 2),
                           "cri_arithmetic": arithmetic, "cri_geometric": geometric})
    return output


def aggregate_family_scores(universe: Sequence[Mapping[str, Any]], members: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in universe:
        expected.setdefault(_key(row, FAMILY_KEYS), []).append(row)
    by_time: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in members:
        by_time.setdefault(_key(row, FAMILY_KEYS + ("cohort_view", "temporal_distance")), []).append(row)
    output = []
    for key, member_rows in sorted(by_time.items(), key=lambda item: str(item[0])):
        family_key = key[:len(FAMILY_KEYS)]
        frozen = expected[family_key]
        eligible_count = sum(bool(row["eligible"]) for row in frozen)
        available = [row for row in member_rows if row["utility_available"]]
        complete = eligible_count > 0 and len(available) == eligible_count
        def mean(name): return float(np.mean([row[name] for row in available])) if complete else None
        output.append({**{name: value for name, value in zip(FAMILY_KEYS + ("cohort_view", "temporal_distance"), key)},
                       "test_year": member_rows[0].get("test_year"), "frozen_member_count": len(frozen),
                       "eligible_member_count": eligible_count, "available_member_count": len(available),
                       "family_complete": complete, "family_missing_reason": None if complete else "member_missing_or_ineligible",
                       "u_f2": mean("u_f2"), "u_jaccard": mean("u_jaccard"),
                       "u_prevalence": mean("u_prevalence"), "u_activation": mean("u_activation"),
                       "semantic_cohort_utility": mean("semantic_cohort_utility"), "activation_utility": mean("activation_utility"),
                       "cri_arithmetic": mean("cri_arithmetic"), "cri_geometric": mean("cri_geometric")})
    return output


def summarize_system_scores(families: Sequence[Mapping[str, Any]], config: CRIAnalysisConfig) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in families: grouped.setdefault(_key(row, SYSTEM_KEYS), []).append(row)
    output = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        values = [float(row["cri_arithmetic"]) for row in rows if row["family_complete"]]
        denominator = len(rows); observed = len(values)
        output.append({**dict(zip(SYSTEM_KEYS, key)), "frozen_family_count": denominator,
                       "complete_family_count": observed, "coverage": observed / denominator if denominator else 0.0,
                       "median_cri": float(np.median(values)) if values else None,
                       "q25_cri": float(np.quantile(values, .25)) if values else None,
                       "median_geometric_cri": float(np.median([row["cri_geometric"] for row in rows if row["family_complete"]])) if values else None,
                       "median_semantic_cohort_utility": float(np.median([row["semantic_cohort_utility"] for row in rows if row["family_complete"]])) if values else None,
                       "median_activation_utility": float(np.median([row["activation_utility"] for row in rows if row["family_complete"]])) if values else None,
                       "fraction_below_threshold_observed": float(np.mean(np.asarray(values) < config.degradation_threshold)) if values else None,
                       "fraction_below_threshold_lower": sum(value < config.degradation_threshold for value in values) / denominator if denominator else None,
                       "fraction_below_threshold_upper": (sum(value < config.degradation_threshold for value in values) + denominator - observed) / denominator if denominator else None})
    return output


def _ols_predict(train: Sequence[Mapping[str, Any]], test: Mapping[str, Any], fields: Sequence[str]) -> tuple[float | None, list[float] | None]:
    usable = [row for row in train if _finite(row.get("target")) and all(_finite(row.get(name)) for name in fields)]
    if len(usable) < len(fields) + 2: return None, None
    refs = sorted({row["reference_year"] for row in usable})
    weights = np.asarray([1 / sum(item["reference_year"] == row["reference_year"] for item in usable) for row in usable], dtype=float)
    x = np.asarray([[1.0] + [float(row[name]) for name in fields] for row in usable])
    y = np.asarray([float(row["target"]) for row in usable])
    beta, _, rank, _ = np.linalg.lstsq(x * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)
    if rank < x.shape[1] or not all(_finite(test.get(name)) for name in fields): return None, None
    return float(np.clip(np.dot([1.0] + [float(test[name]) for name in fields], beta), 0, 1)), beta.tolist()


def build_prediction_rows(system: Sequence[Mapping[str, Any]], performance: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    perf = {(row.get("reference_year"), row.get("patient_split_seed"), row.get("cohort_view"), row.get("temporal_distance")): row for row in performance}
    by_trajectory: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = {}
    for row in system: by_trajectory.setdefault(_key(row, ("reference_year", "patient_split_seed", "cohort_view", "activation_target")), {})[int(row["temporal_distance"])] = row
    output=[]
    for key, timeline in by_trajectory.items():
        ref, split, cohort, target = key
        for distance, cri in timeline.items():
            now = perf.get((ref, split, cohort, distance)); previous = perf.get((ref, split, cohort, distance - 1)); future = perf.get((ref, split, cohort, distance + 1)); previous_cri = timeline.get(distance - 1)
            if not now or not previous or not future: continue
            if not _finite(now.get("death_f1")) or not _finite(previous.get("death_f1")) or not _finite(future.get("death_f1")): continue
            output.append({"reference_year": ref, "patient_split_seed": split, "cohort_view": cohort, "activation_target": target,
                           "temporal_distance": distance, "test_year": now.get("test_year"), "target": future["death_f1"],
                           "f1_d": now["death_f1"], "delta_f1_d": float(now["death_f1"] - previous["death_f1"]),
                           "cri_d": cri.get("median_cri"), "delta_cri_d": None if not _finite(cri.get("median_cri")) or previous_cri is None or not _finite(previous_cri.get("median_cri")) else float(cri["median_cri"] - previous_cri["median_cri"]),
                           "coverage": cri.get("coverage"), "selected_variant": now.get("selected_variant")})
    return output


def _simplex_weights(step: float) -> list[tuple[float, float, float, float]]:
    units = int(round(1 / step))
    return [tuple(value / units for value in values) for values in itertools.product(range(units + 1), repeat=4) if sum(values) == units]


def _system_with_weights(families: Sequence[Mapping[str, Any]], weights: Sequence[float], config: CRIAnalysisConfig) -> list[dict[str, Any]]:
    altered = []
    for row in families:
        value = None if not row["family_complete"] else float(np.dot(weights, [row["u_f2"], row["u_jaccard"], row["u_prevalence"], row["u_activation"]]))
        altered.append({**row, "cri_arithmetic": value})
    return summarize_system_scores(altered, config)


def evaluate_learned_weights(
    families: Sequence[Mapping[str, Any]], performance: Sequence[Mapping[str, Any]], config: CRIAnalysisConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Nested-LOYO exploratory weights; selections never see outer reference."""
    candidates = _simplex_weights(config.learned_weight_step)
    candidate_rows = {weights: build_prediction_rows(_system_with_weights(families, weights, config), performance) for weights in candidates}
    output: list[dict[str, Any]] = []; audits: list[dict[str, Any]] = []
    strata = {(row["cohort_view"], row["activation_target"]) for rows in candidate_rows.values() for row in rows}
    equal = tuple(config.equal_weights)
    for cohort, target in sorted(strata, key=str):
        references = sorted({int(row["reference_year"]) for row in candidate_rows[equal] if row["cohort_view"] == cohort and row["activation_target"] == target})
        for held_out in references:
            candidate_scores = []
            for weights, rows in candidate_rows.items():
                inner = [row for row in rows if row["cohort_view"] == cohort and row["activation_target"] == target and int(row["reference_year"]) != held_out]
                inner_predictions, _, inner_metrics = evaluate_loyo(inner)
                metric = next((row for row in inner_metrics if row["model"] == "performance_history_plus_cri"), None)
                if metric is not None: candidate_scores.append((float(metric["mae_macro_reference"]), weights))
            if not candidate_scores: continue
            # Deterministic tie rule: equal-distance first, then lexical simplex order.
            selected = min(candidate_scores, key=lambda item: (item[0], sum((a-b)**2 for a, b in zip(item[1], equal)), item[1]))[1]
            rows = candidate_rows[selected]
            train = [row for row in rows if row["cohort_view"] == cohort and row["activation_target"] == target and int(row["reference_year"]) != held_out]
            test = [row for row in rows if row["cohort_view"] == cohort and row["activation_target"] == target and int(row["reference_year"]) == held_out]
            for row in test:
                prediction, beta = _ols_predict(train, row, ("f1_d", "delta_f1_d", "temporal_distance", "cri_d", "delta_cri_d"))
                output.append({**dict(row), "held_out_reference_year": held_out, "model": "performance_history_plus_cri_learned_weights",
                               "learned_weights": list(selected), "prediction": prediction,
                               "absolute_error": None if prediction is None else abs(float(row["target"]) - prediction),
                               "squared_error": None if prediction is None else (float(row["target"]) - prediction) ** 2,
                               "coefficients": beta})
            audits.append({"cohort_view": cohort, "activation_target": target, "held_out_reference_year": held_out,
                           "selected_weights": list(selected), "candidate_count": len(candidate_scores),
                           "inner_reference_years": [year for year in references if year != held_out]})
    return output, audits


def evaluate_loyo(prediction_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out every split and distance from one reference system together."""
    predictions: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    groups: dict[tuple[Any, Any], list[Mapping[str, Any]]] = {}
    for row in prediction_rows:
        groups.setdefault((row["cohort_view"], row["activation_target"]), []).append(row)
    for (cohort, target), rows in sorted(groups.items(), key=lambda item: str(item[0])):
        references = sorted({int(row["reference_year"]) for row in rows})
        for held_out in references:
            train = [row for row in rows if int(row["reference_year"]) != held_out]
            test = [row for row in rows if int(row["reference_year"]) == held_out]
            for row in test:
                persistence = float(row["f1_d"])
                history, history_beta = _ols_predict(train, row, ("f1_d", "delta_f1_d", "temporal_distance"))
                cri, cri_beta = _ols_predict(train, row, ("f1_d", "delta_f1_d", "temporal_distance", "cri_d", "delta_cri_d"))
                for model, value, beta in (
                    ("persistence", persistence, None),
                    ("performance_history", history, history_beta),
                    ("performance_history_plus_cri", cri, cri_beta),
                ):
                    predictions.append({**dict(row), "held_out_reference_year": held_out,
                                        "model": model, "prediction": value,
                                        "absolute_error": None if value is None else abs(float(row["target"]) - value),
                                        "squared_error": None if value is None else (float(row["target"]) - value) ** 2,
                                        "coefficients": beta})
            folds.append({"cohort_view": cohort, "activation_target": target,
                          "held_out_reference_year": held_out,
                          "train_reference_years": [year for year in references if year != held_out],
                          "train_row_count": len(train), "test_row_count": len(test),
                          "entire_reference_held_out": not any(int(row["reference_year"]) == held_out for row in train)})
    metrics = []
    by_metric: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in predictions:
        if row["prediction"] is not None:
            by_metric.setdefault((row["cohort_view"], row["activation_target"], row["model"]), []).append(row)
    for key, rows in sorted(by_metric.items(), key=lambda item: str(item[0])):
        errors = np.asarray([row["absolute_error"] for row in rows], dtype=float)
        squared = np.asarray([row["squared_error"] for row in rows], dtype=float)
        per_reference = [np.mean([item["absolute_error"] for item in rows if item["reference_year"] == year]) for year in sorted({item["reference_year"] for item in rows})]
        metrics.append({"cohort_view": key[0], "activation_target": key[1], "model": key[2],
                        "oof_row_count": len(rows), "oof_reference_count": len(per_reference),
                        "mae_micro": float(np.mean(errors)), "rmse_micro": float(np.sqrt(np.mean(squared))),
                        "mae_macro_reference": float(np.mean(per_reference)),
                        "rmse_macro_reference": float(np.sqrt(np.mean([item["squared_error"] for item in rows])))})
    baseline = {(row["cohort_view"], row["activation_target"], row["model"]): row for row in metrics}
    for row in metrics:
        persistence = baseline.get((row["cohort_view"], row["activation_target"], "persistence"))
        history = baseline.get((row["cohort_view"], row["activation_target"], "performance_history"))
        row["mae_improvement_vs_persistence"] = None if persistence is None else float(persistence["mae_macro_reference"] - row["mae_macro_reference"])
        row["mae_improvement_vs_performance_history"] = None if history is None else float(history["mae_macro_reference"] - row["mae_macro_reference"])
    return predictions, folds, metrics


def summarize_oof_predictions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report micro diagnostics and reference-year-macro primary error."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("prediction") is not None:
            grouped.setdefault((row["cohort_view"], row["activation_target"], row["model"]), []).append(row)
    output = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        absolute = np.asarray([row["absolute_error"] for row in values], dtype=float)
        squared = np.asarray([row["squared_error"] for row in values], dtype=float)
        per_reference = [np.mean([row["absolute_error"] for row in values if row["reference_year"] == ref]) for ref in sorted({row["reference_year"] for row in values})]
        output.append({"cohort_view": key[0], "activation_target": key[1], "model": key[2],
                       "oof_row_count": len(values), "oof_reference_count": len(per_reference),
                       "mae_micro": float(np.mean(absolute)), "rmse_micro": float(np.sqrt(np.mean(squared))),
                       "mae_macro_reference": float(np.mean(per_reference)), "rmse_macro_reference": float(np.sqrt(np.mean(squared)))})
    return output


def cri_hash(enrichment_manifest: Path, config: CRIAnalysisConfig) -> str:
    payload = canonical_json({"enrichment_manifest_sha256": file_sha256(enrichment_manifest),
                              "config": config.to_dict(), "cri_source_sha256": file_sha256(Path(__file__))})
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def build_cri_analysis(
    enrichment_manifest_path: str | Path,
    unified_config: UnifiedAnalysisConfig | None = None,
    config: CRIAnalysisConfig | None = None,
) -> Path:
    """Write deterministic CRI tables below completed enrichment's parent."""
    enrichment_manifest = Path(enrichment_manifest_path)
    manifest = json.loads(enrichment_manifest.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError("unified enrichment is incomplete")
    config = config or CRIAnalysisConfig()
    if unified_config is None:
        upstream_config = dict(manifest["config"])
        upstream_config["activation_targets"] = tuple(upstream_config["activation_targets"])
        upstream_config["cohorts"] = tuple(upstream_config["cohorts"])
        unified_config = UnifiedAnalysisConfig(**upstream_config)
    identifier = cri_hash(enrichment_manifest, config)
    output_root = enrichment_manifest.parent.parent / f"cri_{identifier}"
    output_manifest = output_root / "manifest.json"
    if output_manifest.is_file():
        existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            for descriptor in existing.get("artifacts", {}).values(): validate_descriptor(output_root, descriptor)
            return output_manifest
    source = load_enrichment(enrichment_manifest)
    factor_rows = source.get("headline_factor_metrics", [])
    universe = build_family_universe(factor_rows, unified_config, config)
    headline = _headline_rows(factor_rows, unified_config, config)
    taus = calibrate_taus(headline, config)
    members = compute_member_utilities(factor_rows, universe, taus, unified_config, config)
    families = aggregate_family_scores(universe, members)
    system = summarize_system_scores(families, config)
    primary_performance = source.get(
        "primary_performance", source.get("selected_performance", [])
    )
    prediction_rows = build_prediction_rows(system, primary_performance)
    predictions, folds, metrics = evaluate_loyo(prediction_rows)
    learned_predictions, learned_audits = evaluate_learned_weights(
        families, primary_performance, config
    )
    learned_metrics = summarize_oof_predictions(learned_predictions)
    products = {
        "cri_family_universe": universe, "cri_tau_calibration": taus,
        "cri_member_utilities": members, "cri_family_scores": families,
        "cri_system_summaries": system, "cri_prediction_rows": prediction_rows,
        "cri_loyo_predictions": predictions, "cri_loyo_folds": folds, "cri_loyo_metrics": metrics,
        "cri_learned_weight_predictions": learned_predictions, "cri_learned_weight_folds": learned_audits,
        "cri_learned_weight_metrics": learned_metrics,
    }
    descriptors = {}
    for name, rows in products.items():
        descriptor = atomic_write_jsonl_gzip(output_root / f"{name}.jsonl.gz", rows)
        descriptor["path"] = f"{name}.jsonl.gz"; descriptors[name] = descriptor
    output = {"schema_version": "1.0", "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
              "complete": True, "cri_hash": identifier, "enrichment_manifest": str(enrichment_manifest),
              "enrichment_manifest_sha256": file_sha256(enrichment_manifest),
              "cri_source_sha256": file_sha256(Path(__file__)), "unified_config": unified_config.to_dict(),
              "config": config.to_dict(), "artifacts": descriptors}
    atomic_write_json(output_manifest, output, compact=False)
    return output_manifest


def load_cri_analysis(manifest_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(manifest_path); manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True: raise RuntimeError("CRI analysis is incomplete")
    return {name: read_artifact(path.parent, descriptor) for name, descriptor in manifest.get("artifacts", {}).items()}
