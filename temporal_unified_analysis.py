"""Deterministic analysis helpers for the unified temporal robustness report.

This module deliberately sits outside the temporal runner's scientific source
fingerprint.  It consumes a completed immutable parent and produces derived
reporting artifacts; importing it cannot change an experiment that is still
running.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from artifact_storage import (
    canonical_json,
    file_sha256,
    read_artifact,
    validate_descriptor,
)


COHORTS = ("all_comer", "pipeline_unseen")
HEADLINE_VIEWS = ("cosine_qualified", "intersection")


@dataclass(frozen=True)
class UnifiedAnalysisConfig:
    parent_hash: str = "5fd57eb7b61700cda81e"
    cosine_threshold: float = 0.60
    overlap_threshold: float = 0.70
    overlap_percentile: int = 70
    recurrence_min: float = 0.50
    eligibility_activation_target: float = 0.50
    eligibility_minimum_t0_active: int = 10
    activation_targets: tuple[float, ...] = (0.10, 0.30, 0.50)
    cohorts: tuple[str, ...] = COHORTS
    tcav_repetitions: int = 15
    tcav_fdr_alpha: float = 0.05
    tcav_neutral_lower: float = 0.40
    tcav_neutral_upper: float = 0.60
    bootstrap_repetitions: int = 1000
    bootstrap_seed: int = 42
    quiescent_seconds: int = 60

    def __post_init__(self) -> None:
        if not self.parent_hash:
            raise ValueError("parent_hash must be pinned")
        if self.cohorts != COHORTS:
            raise ValueError("the unified estimand is restricted to all_comer and pipeline_unseen")
        if self.activation_targets != tuple(sorted(set(self.activation_targets))):
            raise ValueError("activation targets must be ordered and unique")
        if not 0 <= self.recurrence_min < 1:
            raise ValueError("recurrence_min must lie in [0, 1)")
        if self.eligibility_minimum_t0_active < 1:
            raise ValueError("eligibility support must be positive")
        if self.tcav_repetitions < 2 or self.bootstrap_repetitions < 1:
            raise ValueError("TCAV and bootstrap repetition counts are invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_fingerprint() -> str:
    return file_sha256(Path(__file__))


def enrichment_hash(parent_manifest: Path, config: UnifiedAnalysisConfig) -> str:
    payload = canonical_json({
        "parent_manifest_sha256": file_sha256(parent_manifest),
        "config": config.to_dict(),
        "analysis_source_sha256": source_fingerprint(),
    })
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def validate_completed_parent(
    artifact_root: str | Path,
    config: UnifiedAnalysisConfig,
    *,
    now: float | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Validate identity, checksums, experiment coverage, and quiescence."""

    parent_root = Path(artifact_root) / config.parent_hash
    manifest_path = parent_root / "parent_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"Pinned temporal parent {config.parent_hash} is not complete: "
            "parent_manifest.json is absent"
        )
    parent = json.loads(manifest_path.read_text(encoding="utf-8"))
    if parent.get("complete") is not True or parent.get("runner_hash") != config.parent_hash:
        raise RuntimeError("pinned temporal parent is incomplete or has the wrong identity")
    failed = parent.get("failed_experiments", [])
    skipped = parent.get("skipped_references", [])
    if failed or skipped:
        raise RuntimeError(
            f"pinned temporal parent is not complete: {len(failed)} failed experiments "
            f"and {len(skipped)} skipped references"
        )
    parent_config = parent.get("config", {})
    expected = len(parent_config.get("reference_years", ())) * len(
        parent_config.get("patient_split_seeds", ())
    )
    successful = parent.get("successful_experiments", [])
    if expected < 1 or len(successful) != expected:
        raise RuntimeError(
            f"pinned temporal parent has {len(successful)} successful experiments; "
            f"expected {expected}"
        )
    for row in successful:
        split_manifest = Path(row["manifest"])
        if not split_manifest.is_file() or file_sha256(split_manifest) != row.get(
            "manifest_fingerprint"
        ):
            raise RuntimeError(f"split manifest changed or is missing: {split_manifest}")
        split = json.loads(split_manifest.read_text(encoding="utf-8"))
        if split.get("complete") is not True:
            raise RuntimeError(f"split is incomplete: {split_manifest}")
    for descriptor in parent.get("aggregate_artifacts", {}).values():
        validate_descriptor(parent_root, descriptor)

    if config.quiescent_seconds:
        excluded = parent_root / "derived"
        mtimes = [
            path.stat().st_mtime
            for path in parent_root.rglob("*")
            if path.is_file() and excluded not in path.parents
        ]
        age = (time.time() if now is None else now) - max(mtimes, default=0.0)
        if age < config.quiescent_seconds:
            raise RuntimeError(
                f"pinned temporal parent is still changing (latest write {age:.1f}s ago)"
            )
    return manifest_path, parent


def load_parent_tables(parent_root: Path, parent: Mapping[str, Any]) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    for name, descriptor in parent.get("aggregate_artifacts", {}).items():
        value = read_artifact(parent_root, descriptor)
        if isinstance(value, list):
            tables[name] = value
    return tables


def _same_number(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(float(left), float(right), rtol=0, atol=1e-12))


def _headline_recurrence(
    rows: Sequence[Mapping[str, Any]],
    config: UnifiedAnalysisConfig,
    view: str,
) -> dict[str, Mapping[str, Any]]:
    selected = {}
    for row in rows:
        if row.get("matching_view") != view:
            continue
        if not _same_number(row.get("cosine_threshold"), config.cosine_threshold):
            continue
        if view == "intersection" and (
            int(row.get("overlap_percentile", -1)) != config.overlap_percentile
            or not _same_number(row.get("overlap_threshold"), config.overlap_threshold)
        ):
            continue
        uid = str(row["factor_family_uid"])
        if uid in selected:
            raise ValueError(f"duplicate headline recurrence row for {uid}/{view}")
        selected[uid] = row
    return selected


def exact_member_join(
    factor_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    rule_rows: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Attach factor IDs without inferring them from row order or proximity."""

    members: dict[tuple[str, int], set[int]] = {}
    for row in family_rows:
        key = (str(row["factor_family_uid"]), int(row["member_sae_seed"]))
        factor = int(row["member_factor_id"])
        members.setdefault(key, set()).add(factor)
    rules = {}
    for row in rule_rows:
        key = (
            str(row["factor_family_uid"]), int(row["member_sae_seed"]),
            str(row["rule_source"]), float(row["activation_target"]),
            str(row["target_role"]),
        )
        factor = int(row["member_factor_id"])
        previous = rules.setdefault(key, factor)
        if previous != factor:
            raise ValueError(f"ambiguous exact rule member mapping for {key}")
    output = []
    for source in factor_rows:
        row = dict(source)
        key = (str(row["factor_family_uid"]), int(row["member_sae_seed"]))
        raw_target = row.get("activation_target")
        rule_key = key + (
            str(row.get("rule_source")),
            None if raw_target is None else float(raw_target),
            str(row.get("target_role")),
        )
        factor = rules.get(rule_key)
        candidates = members.get(key, set())
        if factor is None and len(candidates) == 1:
            factor = next(iter(candidates))
        if factor is None and candidates:
            raise ValueError(
                f"factor-year row requires its exact rule member because family mappings "
                f"are ambiguous: {rule_key}"
            )
        if factor is None:
            raise ValueError(f"factor-year row has no exact family member: {key}")
        existing = row.get("member_factor_id")
        if existing is not None and int(existing) != factor:
            raise ValueError(f"factor-year member mismatch for {key}")
        row["member_factor_id"] = factor
        output.append(row)
    return output


def build_family_ladder(
    family_rows: Sequence[Mapping[str, Any]],
    recurrence_rows: Sequence[Mapping[str, Any]],
    support_rows: Sequence[Mapping[str, Any]],
    rule_rows: Sequence[Mapping[str, Any]],
    cav_rows: Sequence[Mapping[str, Any]],
    tcav_rows: Sequence[Mapping[str, Any]],
    config: UnifiedAnalysisConfig,
) -> list[dict[str, Any]]:
    canonical = {
        str(row["factor_family_uid"]): row
        for row in family_rows
        if row.get("matching_criterion") == "canonical_identity"
    }
    support = {
        str(row["factor_family_uid"]): int(row.get("t0_active_count", 0))
        for row in support_rows
        if _same_number(row.get("activation_target"), config.eligibility_activation_target)
        and int(row.get("member_sae_seed", -1))
        == int(row.get("canonical_sae_seed", row.get("member_sae_seed", -2)))
    }
    cosine = _headline_recurrence(recurrence_rows, config, "cosine_qualified")
    intersection = _headline_recurrence(recurrence_rows, config, "intersection")
    semantic = {
        str(row["factor_family_uid"])
        for row in rule_rows
        if row.get("rule_source") == "semantic" and row.get("valid") is True
    }
    high_precision = {
        str(row["factor_family_uid"])
        for row in rule_rows
        if row.get("rule_source") == "high_precision" and row.get("valid") is True
    }
    cav_ready = {
        str(row["factor_family_uid"])
        for row in cav_rows
        if row.get("valid") is True
    }
    tcav_valid = {
        str(row["factor_family_uid"])
        for row in tcav_rows
        if row.get("tcav_valid") is True
    }
    output = []
    for uid, identity in sorted(canonical.items()):
        eligible = support.get(uid, 0) >= config.eligibility_minimum_t0_active
        geometric = cosine.get(uid, {})
        matched = eligible and int(geometric.get("pass_count", 0)) >= 1
        recurrent = matched and float(geometric.get("recurrence", 0.0)) > config.recurrence_min
        consensus_row = intersection.get(uid, {})
        consensus = recurrent and float(consensus_row.get("recurrence", 0.0)) > config.recurrence_min
        interpretable = consensus and uid in semantic
        dual_source = interpretable and uid in high_precision
        flags = (eligible, matched, recurrent, consensus, interpretable, dual_source)
        if any(right and not left for left, right in zip(flags, flags[1:])):
            raise AssertionError(f"family ladder is not nested for {uid}")
        output.append({
            "reference_year": identity.get("reference_year"),
            "patient_split_seed": identity.get("patient_split_seed"),
            "factor_family_uid": uid,
            "canonical_sae_seed": identity.get("canonical_sae_seed"),
            "canonical_factor_id": identity.get("canonical_factor_id"),
            "t0_active_count_p50": support.get(uid, 0),
            "eligible": eligible,
            "matched": matched,
            "recurrent": recurrent,
            "consensus": consensus,
            "interpretable": interpretable,
            "dual_source": dual_source,
            "cav_ready": uid in cav_ready,
            "tcav_valid": uid in tcav_valid,
            "geometric_pass_count": geometric.get("pass_count", 0),
            "geometric_comparison_count": geometric.get("comparison_count", 0),
            "geometric_recurrence": geometric.get("recurrence", 0.0),
        })
    return output


def select_headline_factor_rows(
    rows: Sequence[Mapping[str, Any]], config: UnifiedAnalysisConfig
) -> list[dict[str, Any]]:
    """Select only recurrent cosine/intersection rows and preserve strata."""

    output = []
    for source in rows:
        row = dict(source)
        view = row.get("matching_view")
        if row.get("cohort_view") not in config.cohorts or view not in HEADLINE_VIEWS:
            continue
        if row.get("activation_target") not in config.activation_targets:
            continue
        if not _same_number(row.get("cosine_threshold"), config.cosine_threshold):
            continue
        if view == "intersection" and (
            int(row.get("overlap_percentile", -1)) != config.overlap_percentile
            or not _same_number(row.get("overlap_threshold"), config.overlap_threshold)
        ):
            continue
        if float(row.get("geometric_factor_recurrence", 0.0)) <= config.recurrence_min:
            continue
        output.append(row)
    return output


def choose_f1_variant(
    performance_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep original-system F1 primary and audit balanced sensitivity triggers.

    Two consecutive aggregate zero death-F1 distances still trigger generation
    of the balanced-context experiment.  They never replace the original
    system's rows: concept and performance measures in the primary analysis
    must describe the same fitted system.
    """

    original = [row for row in performance_rows if row.get("variant") == "original"]
    balanced = [row for row in performance_rows if row.get("variant") == "balanced_context"]
    triggers: dict[str, list[tuple[int, int]]] = {}
    for cohort in COHORTS:
        by_distance: dict[int, list[float]] = {}
        for row in original:
            if row.get("cohort_view") != cohort:
                continue
            value = row.get("death_f1")
            if value is not None and np.isfinite(value):
                by_distance.setdefault(int(row["temporal_distance"]), []).append(float(value))
        zero = [
            distance
            for distance, values in sorted(by_distance.items())
            if values and float(np.mean(values)) == 0.0
        ]
        triggers[cohort] = [
            (left, right) for left, right in zip(zero, zero[1:]) if right == left + 1
        ]
    selected = []
    audit = []
    for cohort in COHORTS:
        sensitivity_triggered = bool(triggers[cohort])
        cohort_rows = [
            dict(row) for row in original if row.get("cohort_view") == cohort
        ]
        for row in cohort_rows:
            row["selected_variant"] = row.pop("variant")
            selected.append(row)
        balanced_row_count = sum(
            row.get("cohort_view") == cohort for row in balanced
        )
        audit.append({
            "cohort_view": cohort,
            # Kept for backward-compatible audit readers. A primary fallback is
            # now forbidden; the trigger controls only the sensitivity fit.
            "fallback_triggered": False,
            "sensitivity_triggered": sensitivity_triggered,
            "selected_variant": "original",
            "primary_variant": "original",
            "sensitivity_variant": (
                "balanced_context" if sensitivity_triggered else None
            ),
            "balanced_sensitivity_available": bool(balanced_row_count),
            "balanced_sensitivity_row_count": balanced_row_count,
            "trigger_distance_pairs": [list(pair) for pair in triggers[cohort]],
            "selected_row_count": len(cohort_rows),
        })
    return selected, audit


def benjamini_hochberg(p_values: Sequence[float | None]) -> list[float | None]:
    finite = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    result: list[float | None] = [None] * len(p_values)
    if not finite:
        return result
    ordered = sorted(finite, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 1.0
    count = len(ordered)
    for rank in range(count, 0, -1):
        _, value = ordered[rank - 1]
        running = min(running, value * count / rank)
        adjusted[rank - 1] = running
    for (index, _), value in zip(ordered, adjusted):
        result[index] = float(min(1.0, value))
    return result


def paired_sign_flip_pvalue(actual: Sequence[float], random: Sequence[float]) -> float | None:
    left = np.asarray(actual, dtype=float)
    right = np.asarray(random, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    if len(differences) < 2:
        return None
    observed = abs(float(np.mean(differences)))
    if len(differences) <= 20:
        estimates = []
        for bits in range(1 << len(differences)):
            signs = np.asarray([1 if bits & (1 << index) else -1 for index in range(len(differences))])
            estimates.append(abs(float(np.mean(differences * signs))))
        return float((sum(value >= observed - 1e-15 for value in estimates) + 1) / (len(estimates) + 1))
    rng = np.random.default_rng(42)
    estimates = [abs(float(np.mean(differences * rng.choice((-1, 1), len(differences))))) for _ in range(100000)]
    return float((sum(value >= observed - 1e-15 for value in estimates) + 1) / (len(estimates) + 1))


def summarize_tcav_repetitions(
    rows: Sequence[Mapping[str, Any]], config: UnifiedAnalysisConfig
) -> list[dict[str, Any]]:
    keys = (
        "reference_year", "patient_split_seed", "factor_family_uid",
        "member_sae_seed", "member_factor_id", "activation_target",
        "rule_source", "target_role", "test_year", "temporal_distance",
        "cohort_view",
    )
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        actual = [row.get("actual_tcav") for row in values]
        random = [row.get("random_tcav") for row in values]
        finite_actual = np.asarray([value for value in actual if value is not None and np.isfinite(value)], dtype=float)
        output.append({
            **dict(zip(keys, key)),
            "tcav": float(np.mean(finite_actual)) if len(finite_actual) else None,
            "tcav_std": float(np.std(finite_actual, ddof=1)) if len(finite_actual) > 1 else None,
            "repetition_count": len(finite_actual),
            "p_value": paired_sign_flip_pvalue(actual, random),
        })
    correction_fields = (
        "reference_year", "patient_split_seed", "test_year", "cohort_view",
        "activation_target", "rule_source",
    )
    correction_groups: dict[tuple[Any, ...], list[int]] = {}
    for index, row in enumerate(output):
        correction_groups.setdefault(tuple(row.get(field) for field in correction_fields), []).append(index)
    for indices in correction_groups.values():
        adjusted = benjamini_hochberg([output[index]["p_value"] for index in indices])
        for index, q_value in zip(indices, adjusted):
            score = output[index]["tcav"]
            output[index]["q_value"] = q_value
            output[index]["tcav_valid"] = bool(
                q_value is not None
                and q_value < config.tcav_fdr_alpha
                and score is not None
                and (score < config.tcav_neutral_lower or score > config.tcav_neutral_upper)
            )
    return output


def _finite(value: object) -> bool:
    return value is not None and np.isfinite(value)


def paired_temporal_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    strata: Sequence[str],
    pair_keys: Sequence[str],
    config: UnifiedAnalysisConfig,
    stability_reference: float | None = None,
    percentage_points: bool = False,
) -> list[dict[str, Any]]:
    """Pair units first, then summarize adjacent and cumulative changes."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        value = row.get(metric)
        if _finite(value):
            grouped.setdefault(tuple(row.get(field) for field in strata), []).append(row)
    output = []
    for stratum, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        units: dict[tuple[Any, ...], dict[int, float]] = {}
        for row in values:
            unit = tuple(row.get(field) for field in pair_keys)
            distance = int(row["temporal_distance"])
            value = float(row[metric])
            existing = units.setdefault(unit, {}).get(distance)
            if existing is not None and not np.isclose(existing, value):
                raise ValueError(f"duplicate nonidentical paired value for {unit}/{distance}/{metric}")
            units[unit][distance] = value
        distances = sorted({distance for timeline in units.values() for distance in timeline})
        comparisons = [(left, right, "adjacent") for left, right in zip(distances, distances[1:])]
        if distances:
            comparisons.extend((distances[0], right, "cumulative") for right in distances[1:])
        for left, right, kind in comparisons:
            pairs = [
                (unit, timeline[left], timeline[right])
                for unit, timeline in units.items()
                if left in timeline and right in timeline
            ]
            if not pairs:
                continue
            previous = np.asarray([pair[1] for pair in pairs], dtype=float)
            current = np.asarray([pair[2] for pair in pairs], dtype=float)
            deltas = current - previous
            if percentage_points:
                deltas = deltas * 100.0
            rng = np.random.default_rng(
                config.bootstrap_seed + left * 1009 + right * 9173 + (0 if kind == "adjacent" else 1)
            )
            cluster_positions = [
                pair_keys.index(name)
                for name in ("reference_year", "patient_split_seed")
                if name in pair_keys
            ]
            nested: dict[tuple[Any, ...], list[float]] = {}
            nested_previous: dict[tuple[Any, ...], list[float]] = {}
            nested_current: dict[tuple[Any, ...], list[float]] = {}
            for (unit, prior, after), delta in zip(pairs, deltas):
                cluster = tuple(unit[position] for position in cluster_positions)
                nested.setdefault(cluster, []).append(float(delta))
                nested_previous.setdefault(cluster, []).append(float(prior))
                nested_current.setdefault(cluster, []).append(float(after))
            clusters = tuple(sorted(nested, key=str))
            estimates = []
            for _ in range(config.bootstrap_repetitions):
                sampled_clusters = rng.integers(0, len(clusters), len(clusters))
                cluster_means = []
                for position in sampled_clusters:
                    values_in_cluster = np.asarray(nested[clusters[position]], dtype=float)
                    sampled_values = values_in_cluster[
                        rng.integers(0, len(values_in_cluster), len(values_in_cluster))
                    ]
                    cluster_means.append(float(np.mean(sampled_values)))
                estimates.append(float(np.mean(cluster_means)))
            lower, upper = np.percentile(estimates, [2.5, 97.5])
            previous_cluster_means = np.asarray([
                np.mean(nested_previous[cluster]) for cluster in clusters
            ])
            current_cluster_means = np.asarray([
                np.mean(nested_current[cluster]) for cluster in clusters
            ])
            nonzero = previous_cluster_means != 0
            relative = np.divide(
                current_cluster_means - previous_cluster_means,
                previous_cluster_means,
                out=np.full(len(previous_cluster_means), np.nan),
                where=nonzero,
            )
            row = {
                **dict(zip(strata, stratum)),
                "metric": metric,
                "delta_kind": kind,
                "from_distance": left,
                "to_distance": right,
                "previous_mean": float(np.mean(previous_cluster_means)),
                "current_mean": float(np.mean(current_cluster_means)),
                "delta": float(np.mean([np.mean(nested[cluster]) for cluster in clusters])),
                "relative_change": float(np.nanmean(relative)) if np.any(np.isfinite(relative)) else None,
                "paired_support": len(pairs),
                "fraction_negative": float(np.mean(deltas < 0)),
                "fraction_positive": float(np.mean(deltas > 0)),
                "lower_95": float(lower),
                "upper_95": float(upper),
                "ci_excludes_zero": bool(lower > 0 or upper < 0),
            }
            if stability_reference is not None:
                movement = np.abs(current - stability_reference) - np.abs(previous - stability_reference)
                nested_movement: dict[tuple[Any, ...], list[float]] = {}
                for (unit, _, _), value in zip(pairs, movement):
                    cluster = tuple(unit[position] for position in cluster_positions)
                    nested_movement.setdefault(cluster, []).append(float(value))
                row["stability_deviation_change"] = float(np.mean([
                    np.mean(nested_movement[cluster]) for cluster in clusters
                ]))
            output.append(row)
    return output


def normalize_first_finite(
    rows: Sequence[Mapping[str, Any]], *, metric: str, group_fields: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        groups.setdefault(tuple(row.get(field) for field in group_fields), []).append(row)
    output = []
    for values in groups.values():
        values.sort(key=lambda row: int(row["temporal_distance"]))
        baseline = next((float(row[metric]) for row in values if _finite(row.get(metric))), None)
        usable = baseline is not None and baseline != 0
        for row in values:
            value = row.get(metric)
            row["normalized_value"] = float(value) / baseline if usable and _finite(value) else None
            row["normalization_baseline"] = baseline
            row["normalization_available"] = usable
            output.append(row)
    return output


def delta_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    strata = sorted({
        (row.get("metric"), row.get("cohort_view"), row.get("matching_view"),
         row.get("rule_source"), row.get("activation_target"))
        for row in rows
    }, key=str)
    output = []
    for key in strata:
        selected = [
            row for row in rows
            if (row.get("metric"), row.get("cohort_view"), row.get("matching_view"),
                row.get("rule_source"), row.get("activation_target")) == key
        ]
        adjacent = [row for row in selected if row.get("delta_kind") == "adjacent"]
        cumulative = [row for row in selected if row.get("delta_kind") == "cumulative"]
        worst = min(adjacent, key=lambda row: row["delta"], default=None)
        directional = next((row for row in sorted(adjacent, key=lambda row: row["to_distance"]) if row["ci_excludes_zero"]), None)
        furthest = max(cumulative, key=lambda row: row["to_distance"], default=None)
        output.append({
            "metric": key[0], "cohort_view": key[1], "matching_view": key[2],
            "rule_source": key[3], "activation_target": key[4],
            "largest_adjacent_deterioration": None if worst is None else worst["delta"],
            "largest_adjacent_to_distance": None if worst is None else worst["to_distance"],
            "first_directional_to_distance": None if directional is None else directional["to_distance"],
            "furthest_cumulative_change": None if furthest is None else furthest["delta"],
            "minimum_paired_support": min((row["paired_support"] for row in selected), default=0),
            "insufficient_paired_support": any(row["paired_support"] < 2 for row in selected),
        })
    return output
