"""Build immutable derived artifacts for ``robustness_analysis_unified.ipynb``.

The base temporal run is read-only.  Expensive additions reuse its retained
activations, embeddings, gradients, role assignments, and fitted semantics.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from artifact_storage import (
    ARTIFACT_SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_jsonl_gzip,
    file_sha256,
    read_artifact,
    validate_descriptor,
)
from temporal_unified_analysis import (
    UnifiedAnalysisConfig,
    build_family_ladder,
    choose_f1_variant,
    delta_evidence,
    enrichment_hash,
    load_parent_tables,
    paired_temporal_deltas,
    select_headline_factor_rows,
    summarize_tcav_repetitions,
    validate_completed_parent,
)


def prevalence_instability(
    t0_active: int, t0_total: int, future_active: int, future_total: int,
) -> float:
    """Symmetric, finite log-scale movement in activation prevalence."""

    reference = (float(t0_active) + 0.5) / (float(t0_total) + 1.0)
    current = (float(future_active) + 0.5) / (float(future_total) + 1.0)
    return float(abs(np.log(current / reference)))


def feature_association_cosine(
    reference_features: np.ndarray,
    reference_activations: np.ndarray,
    current_features: np.ndarray,
    current_activations: np.ndarray,
    *,
    minimum_records: int = 30,
) -> dict[str, object]:
    """Compare factor-to-feature Pearson association vectors across cohorts."""

    reference_features = np.asarray(reference_features, dtype=float)
    current_features = np.asarray(current_features, dtype=float)
    reference_activations = np.asarray(reference_activations, dtype=float)
    current_activations = np.asarray(current_activations, dtype=float)
    result = {
        "feature_association_cosine": None,
        "feature_association_valid": False,
        "feature_association_failure_reason": None,
        "feature_association_common_feature_count": 0,
        "feature_association_reference_records": int(len(reference_activations)),
        "feature_association_current_records": int(len(current_activations)),
    }
    if len(reference_activations) < minimum_records:
        result["feature_association_failure_reason"] = "insufficient_reference_records"
        return result
    if len(current_activations) < minimum_records:
        result["feature_association_failure_reason"] = "insufficient_current_records"
        return result
    if (
        reference_features.ndim != 2 or current_features.ndim != 2
        or reference_features.shape[1] != current_features.shape[1]
        or reference_features.shape[0] != len(reference_activations)
        or current_features.shape[0] != len(current_activations)
    ):
        raise ValueError("association inputs must be aligned feature matrices and activations")

    def associations(features: np.ndarray, activations: np.ndarray) -> np.ndarray:
        valid = np.isfinite(features) & np.isfinite(activations)[:, None]
        count = valid.sum(axis=0, dtype=float)
        left = np.where(valid, features, 0.0)
        right = np.where(valid, activations[:, None], 0.0)
        sum_left = left.sum(axis=0)
        sum_right = right.sum(axis=0)
        numerator = (left * right).sum(axis=0) - sum_left * sum_right / np.maximum(count, 1)
        left_ss = (left * left).sum(axis=0) - sum_left * sum_left / np.maximum(count, 1)
        right_ss = (right * right).sum(axis=0) - sum_right * sum_right / np.maximum(count, 1)
        denominator = np.sqrt(np.maximum(left_ss, 0.0) * np.maximum(right_ss, 0.0))
        return np.divide(
            numerator, denominator, out=np.full(features.shape[1], np.nan),
            where=(count >= 2) & (denominator > 0),
        )

    reference = associations(reference_features, reference_activations)
    current = associations(current_features, current_activations)
    common = np.isfinite(reference) & np.isfinite(current)
    result["feature_association_common_feature_count"] = int(common.sum())
    if common.sum() < 2:
        result["feature_association_failure_reason"] = "insufficient_common_finite_features"
        return result
    left, right = reference[common], current[common]
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if np.isclose(denominator, 0.0):
        result["feature_association_failure_reason"] = "zero_association_vector_norm"
        return result
    result["feature_association_cosine"] = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    result["feature_association_valid"] = True
    return result


def _split_table(split_root: Path, name: str) -> list[dict[str, Any]]:
    manifest = json.loads((split_root / "manifest.json").read_text(encoding="utf-8"))
    descriptor = manifest.get("artifacts", {}).get(name)
    if descriptor is None:
        return []
    value = read_artifact(split_root, descriptor)
    if not isinstance(value, list):
        raise ValueError(f"split artifact {name!r} is not a table")
    return value


def _roles(split_root: Path) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = {}
    for row in _split_table(split_root, "reference_roles"):
        grouped.setdefault(str(row["role"]), []).append(int(row["row_index"]))
    return {name: np.asarray(sorted(indices), dtype=int) for name, indices in grouped.items()}


def _local_roles(global_roles: Mapping[str, np.ndarray], evaluation: np.ndarray) -> dict[str, np.ndarray]:
    lookup = {int(index): position for position, index in enumerate(evaluation)}
    return {
        role: np.asarray([lookup[int(index)] for index in indices], dtype=int)
        for role, indices in global_roles.items()
    }


def _activation_runs(split_root: Path, sae_seeds: Sequence[int]) -> dict[int, np.ndarray]:
    with np.load(split_root / "activations.npz", allow_pickle=False) as values:
        return {
            int(seed): np.asarray(values[f"run_{run}"])
            for run, seed in enumerate(sae_seeds)
        }


def _cohort_masks_for_split(population, reference_year: int, global_roles, evaluation):
    from temporal_cohorts import assign_future_provenance, cohort_masks

    patients = population.patient_ids[evaluation].astype(str)
    years = population.years[evaluation].astype(int)
    role_patients = {
        role: set(population.patient_ids[indices].astype(str))
        for role, indices in global_roles.items()
    }
    local = _local_roles(global_roles, evaluation)
    output = {}
    for year in sorted(set(years)):
        year_mask = years == year
        if year == reference_year:
            mask = np.zeros(len(evaluation), dtype=bool)
            mask[local["t0_evaluation"]] = True
            output[(year, "all_comer")] = mask
            continue
        labels = assign_future_provenance(
            patients[year_mask], population.first_eligible_year,
            reference_year, role_patients,
        )
        for cohort, selected in cohort_masks(labels).items():
            if cohort not in {"all_comer", "pipeline_unseen"}:
                continue
            mask = np.zeros(len(evaluation), dtype=bool)
            mask[np.flatnonzero(year_mask)] = selected
            output[(year, cohort)] = mask
    return output


def activation_support_rows(
    split_root: Path,
    population,
    parent_config: Mapping[str, Any],
    analysis_config: UnifiedAnalysisConfig,
) -> list[dict[str, Any]]:
    from temporal_rules import fit_canonical_targets

    split_manifest = json.loads((split_root / "manifest.json").read_text())
    reference_year = int(split_manifest["reference_year"])
    split_seed = int(split_manifest["patient_split_seed"])
    sae_seeds = tuple(int(value) for value in parent_config["sae_seeds"])
    canonical_seed = sae_seeds[0]
    evaluation = np.flatnonzero(population.years >= reference_year)
    roles = _local_roles(_roles(split_root), evaluation)
    activations = _activation_runs(split_root, sae_seeds)[canonical_seed]
    families = {
        int(row["canonical_factor_id"]): row
        for row in _split_table(split_root, "factor_families")
        if row.get("matching_criterion") == "canonical_identity"
    }
    output = []
    for factor, identity in sorted(families.items()):
        targets = fit_canonical_targets(
            activations[roles["sae_discovery"], factor],
            analysis_config.activation_targets,
            minimum_positive_samples=1,
        )
        for fraction, target in targets.items():
            values = activations[roles["t0_evaluation"], factor]
            active = target.apply(values) if target.cutoff is not None else np.zeros(len(values), dtype=bool)
            output.append({
                "reference_year": reference_year,
                "patient_split_seed": split_seed,
                "factor_family_uid": identity["factor_family_uid"],
                "canonical_sae_seed": canonical_seed,
                "member_sae_seed": canonical_seed,
                "member_factor_id": factor,
                "activation_target": float(fraction),
                "cutoff": target.cutoff,
                "target_valid": bool(target.valid),
                "t0_active_count": int(np.count_nonzero(active)),
                "t0_denominator": int(len(values)),
            })
    return output


def activation_magnitude_rows(
    split_root: Path,
    population,
    parent_config: Mapping[str, Any],
    analysis_config: UnifiedAnalysisConfig,
) -> list[dict[str, Any]]:
    from semantic_rules import binary_metrics
    from temporal_config import TemporalRetentionConfig
    from temporal_metrics import prevalence_retention

    reference_year = int(split_root.parent.name.removeprefix("reference_"))
    split_seed = int(split_root.name.removeprefix("split_"))
    sae_seeds = tuple(int(value) for value in parent_config["sae_seeds"])
    evaluation = np.flatnonzero(population.years >= reference_year)
    global_roles = _roles(split_root)
    masks = _cohort_masks_for_split(population, reference_year, global_roles, evaluation)
    runs = _activation_runs(split_root, sae_seeds)
    recurrence = _split_table(split_root, "matching_recurrence")
    rules = [
        row for row in _split_table(split_root, "rules")
        if row.get("valid") is True
        and float(row.get("activation_target")) in analysis_config.activation_targets
        and row.get("factor_family_uid") is not None
    ]
    retention_config = TemporalRetentionConfig(**parent_config["retention"])
    association_cache: dict[tuple[int, int, int, str], dict[str, object]] = {}
    headline_views: dict[str, list[dict[str, Any]]] = {}
    for row in recurrence:
        candidate = {
            "factor_family_uid": row["factor_family_uid"],
            "cohort_view": "all_comer",
            "activation_target": analysis_config.activation_targets[0],
            "matching_view": row["matching_view"],
            "cosine_threshold": row.get("cosine_threshold"),
            "overlap_percentile": row.get("overlap_percentile"),
            "overlap_threshold": row.get("overlap_threshold"),
            "geometric_factor_recurrence": row.get("recurrence"),
        }
        if select_headline_factor_rows([candidate], analysis_config):
            headline_views.setdefault(str(row["factor_family_uid"]), []).append(row)
    output = []
    for rule in rules:
        views = headline_views.get(str(rule["factor_family_uid"]), [])
        if not views or rule.get("cutoff") is None:
            continue
        seed = int(rule["member_sae_seed"])
        factor = int(rule["member_factor_id"])
        values = runs[seed][:, factor]
        target = (
            np.isfinite(values) & (values > 0)
            & (values >= float(rule["cutoff"]))
        )
        selected = _rule_object(rule, population.feature_names).mask(
            population.X[evaluation]
        )
        t0_mask = masks[(reference_year, "all_comer")]
        t0_active = int(np.count_nonzero(target[t0_mask]))
        t0_values = values[t0_mask & target]
        t0_mean = float(np.mean(t0_values)) if len(t0_values) else None
        for (year, cohort), mask in masks.items():
            association_key = (seed, factor, int(year), str(cohort))
            association = association_cache.get(association_key)
            if association is None:
                association = feature_association_cosine(
                    population.X[evaluation][t0_mask], values[t0_mask],
                    population.X[evaluation][mask], values[mask],
                    minimum_records=retention_config.minimum_evaluation_records,
                )
                association_cache[association_key] = association
            current_values = values[mask & target]
            current_mean = float(np.mean(current_values)) if len(current_values) else None
            magnitude_ratio = (
                current_mean / t0_mean
                if current_mean is not None and t0_mean is not None and t0_mean != 0
                else None
            )
            metrics = binary_metrics(target[mask], selected[mask]).to_dict()
            retention = prevalence_retention(
                t0_active, int(np.count_nonzero(t0_mask)),
                int(np.count_nonzero(target[mask])), int(np.count_nonzero(mask)),
                retention_config,
            )
            instability = prevalence_instability(
                t0_active, int(np.count_nonzero(t0_mask)),
                int(np.count_nonzero(target[mask])), int(np.count_nonzero(mask)),
            )
            for view in views:
                output.append({
                    "reference_year": reference_year,
                    "patient_split_seed": split_seed,
                    "test_year": int(year),
                    "temporal_distance": int(year - reference_year),
                    "factor_family_uid": rule["factor_family_uid"],
                    "member_sae_seed": seed,
                    "member_factor_id": factor,
                    "cohort_view": cohort,
                    "rule_source": rule["rule_source"],
                    "activation_target": float(rule["activation_target"]),
                    "target_role": rule["target_role"],
                    "matching_view": view["matching_view"],
                    "cosine_threshold": view.get("cosine_threshold"),
                    "overlap_percentile": view.get("overlap_percentile"),
                    "overlap_threshold": view.get("overlap_threshold"),
                    "geometric_factor_recurrence": view["recurrence"],
                    **metrics,
                    **retention,
                    "prevalence_instability": instability,
                    **association,
                    "activation_magnitude": current_mean,
                    "t0_activation_magnitude": t0_mean,
                    "activation_magnitude_ratio": magnitude_ratio,
                    "activation_magnitude_support": int(len(current_values)),
                })
    return output


class _FixedMaskRule:
    def __init__(self, mask: np.ndarray):
        self._mask = np.asarray(mask, dtype=bool)

    def mask(self, features):
        if len(features) != len(self._mask):
            raise ValueError("fixed random-concept mask does not align")
        return self._mask


def _rule_object(row: Mapping[str, Any], feature_names: Sequence[str]):
    if row["rule_source"] == "semantic":
        from semantic_rules import RuleSet

        return RuleSet.from_dict(json.loads(row["rule_text"]))
    from temporal_production import _TextRule

    return _TextRule(row["rule_text"], feature_names)


def tcav_repetition_rows(
    split_root: Path,
    population,
    parent_config: Mapping[str, Any],
    analysis_config: UnifiedAnalysisConfig,
) -> list[dict[str, Any]]:
    from temporal_cav import rule_cohort_mask, temporal_tcav, train_temporal_cav

    gradient_path = split_root / "semantic_gradients.pkl"
    if not gradient_path.is_file():
        return []
    with gradient_path.open("rb") as handle:
        gradients = np.asarray(pickle.load(handle))
    with np.load(split_root / "embeddings.npz", allow_pickle=False) as stored:
        embeddings = np.asarray(stored["test_raw"], dtype=float)
    reference_year = int(split_root.parent.name.removeprefix("reference_"))
    split_seed = int(split_root.name.removeprefix("split_"))
    sae_seeds = tuple(int(value) for value in parent_config["sae_seeds"])
    evaluation = np.flatnonzero(population.years >= reference_year)
    global_roles = _roles(split_root)
    local = _local_roles(global_roles, evaluation)
    selection = local["rule_selection_cav"]
    masks = _cohort_masks_for_split(population, reference_year, global_roles, evaluation)
    runs = _activation_runs(split_root, sae_seeds)
    recurrence = _split_table(split_root, "matching_recurrence")
    consensus = {
        row["factor_family_uid"]
        for row in recurrence
        if row.get("matching_view") == "intersection"
        and np.isclose(float(row.get("cosine_threshold")), analysis_config.cosine_threshold)
        and int(row.get("overlap_percentile")) == analysis_config.overlap_percentile
        and np.isclose(float(row.get("overlap_threshold")), analysis_config.overlap_threshold)
        and float(row.get("recurrence", 0.0)) > analysis_config.recurrence_min
    }
    rules = [
        row for row in _split_table(split_root, "rules")
        if row.get("rule_source") == "semantic"
        and row.get("valid") is True
        and row.get("factor_family_uid") in consensus
        and float(row.get("activation_target")) in analysis_config.activation_targets
    ]
    features = population.X[evaluation]
    patients = population.patient_ids[evaluation]
    output = []
    for rule_row in rules:
        seed = int(rule_row["member_sae_seed"])
        factor = int(rule_row["member_factor_id"])
        activations = runs[seed][:, factor]
        rule = _rule_object(rule_row, population.feature_names)
        positive_mask = rule_cohort_mask(rule, features[selection])
        positive_count = int(np.count_nonzero(positive_mask))
        for repetition in range(analysis_config.tcav_repetitions):
            seed_value = split_seed * 1_000_000 + seed * 10_000 + factor * 31 + repetition
            actual = train_temporal_cav(
                embeddings=embeddings[selection], features=features[selection],
                activations=activations[selection], rule=rule,
                activation_target=float(rule_row["activation_target"]),
                rule_source="semantic", patient_ids=patients[selection],
                minimum_positive=int(parent_config["support"]["cav_positive_records"]),
                minimum_negative=int(parent_config["support"]["cav_negative_records"]),
                seed=seed_value,
            )
            rng = np.random.default_rng(seed_value + 7_919)
            random_mask = np.zeros(len(selection), dtype=bool)
            if positive_count:
                random_mask[rng.choice(len(selection), size=min(positive_count, len(selection)), replace=False)] = True
            random_cav = train_temporal_cav(
                embeddings=embeddings[selection], features=features[selection],
                activations=activations[selection], rule=_FixedMaskRule(random_mask),
                activation_target=float(rule_row["activation_target"]),
                rule_source="random_control", patient_ids=patients[selection],
                minimum_positive=int(parent_config["support"]["cav_positive_records"]),
                minimum_negative=int(parent_config["support"]["cav_negative_records"]),
                seed=seed_value + 7_919,
            )
            if not actual["valid"] or not random_cav["valid"]:
                continue
            for (year, cohort), mask in masks.items():
                if not np.count_nonzero(mask):
                    continue
                output.append({
                    "reference_year": reference_year,
                    "patient_split_seed": split_seed,
                    "factor_family_uid": rule_row["factor_family_uid"],
                    "member_sae_seed": seed,
                    "member_factor_id": factor,
                    "activation_target": float(rule_row["activation_target"]),
                    "rule_source": "semantic",
                    "target_role": rule_row["target_role"],
                    "test_year": year,
                    "temporal_distance": year - reference_year,
                    "cohort_view": cohort,
                    "repetition": repetition,
                    "actual_tcav": temporal_tcav(actual["cav"], gradients[mask])["tcav"],
                    "random_tcav": temporal_tcav(random_cav["cav"], gradients[mask])["tcav"],
                })
    return output


def _balanced_predictions(population, reference_year, split_seed, context, evaluation, parent_config):
    import torch
    from tabpfn_model import TabPFNEvalConfig, fit_dr_tabpfn, make_dist_tensor

    outcomes = population.outcomes[context]
    deaths = context[outcomes == 1]
    survivors = context[outcomes == 0]
    count = min(len(deaths), len(survivors))
    if count < 1:
        raise ValueError("balanced context requires both outcome classes")
    rng = np.random.default_rng(split_seed)
    balanced = np.sort(np.concatenate([
        rng.choice(deaths, count, replace=False),
        rng.choice(survivors, count, replace=False),
    ]))
    evaluation_config = TabPFNEvalConfig()
    evaluation_config.rng_seed = split_seed
    evaluation_config.tabpfn_model_name = json.loads(
        Path(parent_config["comparison_config_path"]).read_text()
    )["tabpfn"]["model_name"]
    evaluation_config.batch_size_predict = 512
    evaluation_config.device = parent_config["device"]
    evaluation_config.show_progress = False
    fitted = fit_dr_tabpfn(
        population.X[balanced], population.outcomes[balanced],
        population.years[balanced], evaluation_config,
    )
    model = fitted["model"]
    predictions = np.empty(len(evaluation), dtype=int)
    years = population.years[evaluation]
    for start in range(0, len(evaluation), 512):
        end = min(start + 512, len(evaluation))
        domains = years[start:end].astype(int) - int(reference_year)
        dist = make_dist_tensor(
            domains, fitted["model_add_x_device"], fitted["example_add_shape"]
        )
        values = population.X[evaluation[start:end]].astype(np.float32)
        if torch.device(fitted["model_add_x_device"]).type == "cpu":
            probability = model.predict_proba(values, additional_x={"dist_shift_domain": dist})
        else:
            with torch.no_grad():
                probability = model.predict_proba(
                    torch.as_tensor(values, device=fitted["model_add_x_device"]),
                    additional_x={"dist_shift_domain": dist},
                )
        if isinstance(probability, torch.Tensor):
            probability = probability.detach().cpu().numpy()
        predictions[start:end] = np.argmax(np.asarray(probability), axis=1)
    return predictions, {
        "balanced_context_count": int(len(balanced)),
        "balanced_context_deaths": int(count),
        "balanced_context_survivors": int(count),
    }


def performance_variant_rows(
    parent_tables: Mapping[str, list[dict]],
    split_roots: Sequence[Path],
    population,
    parent_config: Mapping[str, Any],
    analysis_config: UnifiedAnalysisConfig,
) -> list[dict[str, Any]]:
    original = [
        {**row, "variant": "original"}
        for row in parent_tables.get("performance", [])
        if row.get("cohort_view") in analysis_config.cohorts
    ]
    # Determine the trigger without fitting the balanced model unnecessarily.
    probe = list(original)
    for cohort in analysis_config.cohorts:
        probe.append({
            "variant": "balanced_context", "cohort_view": cohort,
            "temporal_distance": 0, "macro_f1": 0.0, "death_f1": 0.0,
        })
    try:
        _, audit = choose_f1_variant(probe)
    except ValueError:
        audit = []
    triggered = any(row["fallback_triggered"] for row in audit)
    if not triggered:
        return original

    from temporal_config import TemporalRobustnessConfig
    from temporal_production import _performance_rows

    temporal_config = TemporalRobustnessConfig.from_dict(parent_config)
    balanced_rows = []
    for split_root in split_roots:
        split_manifest = json.loads((split_root / "manifest.json").read_text())
        reference_year = int(split_manifest["reference_year"])
        split_seed = int(split_manifest["patient_split_seed"])
        global_roles = _roles(split_root)
        evaluation = np.flatnonzero(population.years >= reference_year)
        predictions, counts = _balanced_predictions(
            population, reference_year, split_seed,
            global_roles["tabpfn_context"], evaluation, parent_config,
        )
        rows = _performance_rows(
            population, evaluation, global_roles, predictions,
            reference_year, split_seed, temporal_config,
        )
        balanced_rows.extend({
            **row, **counts, "variant": "balanced_context"
        } for row in rows if row.get("cohort_view") in analysis_config.cohorts)
    return original + balanced_rows


def _tcav_headline_views(rows, recurrence_rows, config):
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    for row in recurrence_rows:
        by_family.setdefault(str(row["factor_family_uid"]), []).append(row)
    output = []
    for source in rows:
        for view in by_family.get(str(source["factor_family_uid"]), []):
            if view.get("matching_view") not in {"cosine_qualified", "intersection"}:
                continue
            candidate = {
                **source,
                "matching_view": view["matching_view"],
                "cosine_threshold": view.get("cosine_threshold"),
                "overlap_percentile": view.get("overlap_percentile"),
                "overlap_threshold": view.get("overlap_threshold"),
                "geometric_factor_recurrence": view.get("recurrence"),
            }
            if select_headline_factor_rows([candidate], config):
                output.append(candidate)
    return output


def _status_composition(rows):
    fields = (
        "reference_year", "patient_split_seed", "temporal_distance", "test_year",
        "cohort_view", "matching_view", "rule_source", "activation_target", "status",
    )
    denominator_fields = fields[:-1]
    counts: dict[tuple[Any, ...], int] = {}
    denominators: dict[tuple[Any, ...], int] = {}
    seen = set()
    for row in rows:
        identity = tuple(row.get(field) for field in denominator_fields) + (
            row.get("factor_family_uid"), row.get("member_sae_seed"),
            row.get("member_factor_id"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        key = tuple(row.get(field) for field in fields)
        denominator = key[:-1]
        counts[key] = counts.get(key, 0) + 1
        denominators[denominator] = denominators.get(denominator, 0) + 1
    return [
        {
            **dict(zip(fields, key)), "status_proportion": count / denominators[key[:-1]],
            "status_family_count": count, "status_denominator": denominators[key[:-1]],
        }
        for key, count in sorted(counts.items(), key=lambda item: str(item[0]))
    ]


def _all_deltas(performance, factors, tcav, status, config):
    output = []
    output.extend(paired_temporal_deltas(
        performance, metric="macro_f1", strata=("cohort_view",),
        pair_keys=("reference_year", "patient_split_seed", "selected_variant"),
        config=config,
    ))
    output.extend(paired_temporal_deltas(
        performance, metric="death_f1", strata=("cohort_view",),
        pair_keys=("reference_year", "patient_split_seed", "selected_variant"),
        config=config,
    ))
    factor_strata = ("cohort_view", "matching_view", "rule_source", "activation_target")
    factor_pairs = (
        "reference_year", "patient_split_seed", "factor_family_uid",
        "member_sae_seed", "member_factor_id", "target_role",
    )
    for metric in (
        "f2", "prevalence_ratio", "jaccard", "activation_magnitude_ratio",
        "feature_association_cosine",
    ):
        output.extend(paired_temporal_deltas(
            factors, metric=metric, strata=factor_strata, pair_keys=factor_pairs,
            config=config, stability_reference=(1.0 if metric in {"prevalence_ratio", "activation_magnitude_ratio"} else None),
        ))
    output.extend(paired_temporal_deltas(
        tcav, metric="tcav",
        strata=("cohort_view", "matching_view", "rule_source", "activation_target"),
        pair_keys=factor_pairs, config=config,
    ))
    output.extend(paired_temporal_deltas(
        status, metric="status_proportion",
        strata=("cohort_view", "matching_view", "rule_source", "activation_target", "status"),
        pair_keys=("reference_year", "patient_split_seed"),
        config=config, percentage_points=True,
    ))
    return output


def build_unified_enrichment(
    artifact_root: str | Path = "stats/temporal_robustness",
    config: UnifiedAnalysisConfig | None = None,
) -> Path:
    config = config or UnifiedAnalysisConfig()
    parent_manifest_path, parent = validate_completed_parent(artifact_root, config)
    parent_root = parent_manifest_path.parent
    base_identifier = enrichment_hash(parent_manifest_path, config)
    enrichment_source_sha256 = file_sha256(Path(__file__))
    identifier = hashlib.sha256(
        f"{base_identifier}:{enrichment_source_sha256}".encode()
    ).hexdigest()[:20]
    output_root = parent_root / "derived" / identifier
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") is True:
            for descriptor in manifest.get("artifacts", {}).values():
                validate_descriptor(output_root, descriptor)
            return manifest_path

    from temporal_config import TemporalRobustnessConfig
    from temporal_production import ProductionTemporalAdapter

    parent_config = parent["config"]
    temporal_config = TemporalRobustnessConfig.from_dict(parent_config)
    adapter = ProductionTemporalAdapter()
    population = adapter.load_population(temporal_config)
    population.validate()
    split_roots = [Path(row["manifest"]).parent for row in parent["successful_experiments"]]
    tables = load_parent_tables(parent_root, parent)

    support = []
    magnitude = []
    tcav_repetitions = []
    for split_root in split_roots:
        support.extend(activation_support_rows(
            split_root, population, parent_config, config
        ))
        magnitude.extend(activation_magnitude_rows(
            split_root, population, parent_config, config
        ))
        tcav_repetitions.extend(tcav_repetition_rows(
            split_root, population, parent_config, config
        ))
    tcav_summary = summarize_tcav_repetitions(tcav_repetitions, config)
    tcav_headline = _tcav_headline_views(
        tcav_summary, tables.get("matching_recurrence", []), config
    )
    tcav_valid_headline = [
        row for row in tcav_headline if row.get("tcav_valid") is True
    ]
    performance_variants = performance_variant_rows(
        tables, split_roots, population, parent_config, config
    )
    selected_performance, fallback_audit = choose_f1_variant(performance_variants)
    for row in fallback_audit:
        balanced = [
            item for item in performance_variants
            if item.get("variant") == "balanced_context"
            and item.get("cohort_view") == row["cohort_view"]
        ]
        for field in (
            "balanced_context_count", "balanced_context_deaths",
            "balanced_context_survivors",
        ):
            values = sorted({item.get(field) for item in balanced if item.get(field) is not None})
            row[field] = values
    ladder = build_family_ladder(
        tables.get("factor_families", []), tables.get("matching_recurrence", []),
        support, tables.get("rules", []), tables.get("cavs", []), tcav_summary,
        config,
    )
    status = _status_composition(magnitude)
    deltas = _all_deltas(
        selected_performance, magnitude, tcav_valid_headline, status, config
    )
    evidence = delta_evidence(deltas)
    products = {
        "activation_support": support,
        "headline_factor_metrics": magnitude,
        "performance_variants": performance_variants,
        "selected_performance": selected_performance,
        "f1_fallback_audit": fallback_audit,
        "tcav_repetitions": tcav_repetitions,
        "tcav_significance": tcav_headline,
        "family_ladder": ladder,
        "status_composition": status,
        "paired_temporal_deltas": deltas,
        "delta_evidence": evidence,
    }
    artifacts = {}
    for name, rows in products.items():
        path = output_root / f"{name}.jsonl.gz"
        descriptor = atomic_write_jsonl_gzip(path, rows)
        descriptor["path"] = path.name
        artifacts[name] = descriptor
    manifest = {
        "schema_version": "1.0",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "complete": True,
        "enrichment_hash": identifier,
        "parent_hash": config.parent_hash,
        "parent_manifest": str(parent_manifest_path),
        "parent_manifest_sha256": file_sha256(parent_manifest_path),
        "parent_source_fingerprints": parent.get("source_fingerprints"),
        "analysis_source_sha256": file_sha256(
            Path(__file__).with_name("temporal_unified_analysis.py")
        ),
        "enrichment_source_sha256": enrichment_source_sha256,
        "config": config.to_dict(),
        "artifacts": artifacts,
    }
    atomic_write_json(manifest_path, manifest, compact=False)
    return manifest_path


def load_enrichment(manifest_path: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise RuntimeError("unified enrichment is incomplete")
    return {
        name: read_artifact(path.parent, descriptor)
        for name, descriptor in manifest.get("artifacts", {}).items()
    }


if __name__ == "__main__":
    print(build_unified_enrichment())
