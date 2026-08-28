"""Production adapter connecting temporal orchestration to TabPFN/SAE stages."""

from __future__ import annotations

from dataclasses import replace
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from comparison_cache import ComparisonCache
from semantic_artifacts import stable_hash
from temporal_cav import temporal_tcav, train_temporal_cav
from temporal_cohorts import (
    assign_future_provenance,
    cohort_masks,
    metric_delta,
    role_patient_sets,
)
from temporal_matching import build_canonical_factor_views
from temporal_metrics import (
    classification_metrics,
    prevalence_retention,
    support_aware_classification_metrics,
)
from temporal_robustness import TemporalPopulation
from temporal_rules import select_activation_winner


class ProductionTemporalAdapter:
    """Fit independent reference systems using existing production model stages."""

    def __init__(self) -> None:
        self._source_prepared = None
        self._base_config = None

    def _configure(self, config):
        from comparison_runner import ComparisonRunnerConfig

        base = ComparisonRunnerConfig.from_json(config.comparison_config_path)
        base = replace(
            base,
            dataset_path=config.dataset_path,
            semantic_config_path=config.semantic_config_path,
            artifact_dir=config.artifact_dir,
            use_cache=config.use_cache,
            show_progress=config.show_progress,
            accelerator=replace(base.accelerator, device=config.device),
        )
        self._base_config = base
        return base

    def _population_from_prepared(self, prepared) -> TemporalPopulation:
        from comparison_runner import _validate_prepared

        _validate_prepared(prepared)
        self._source_prepared = prepared
        patients = prepared.patient_ids.astype(str)
        years = prepared.years_test.astype(int)
        first = {
            patient: int(np.min(years[patients == patient]))
            for patient in np.unique(patients)
        }
        return TemporalPopulation(
            X=np.asarray(prepared.X_test),
            outcomes=np.asarray(prepared.y_test),
            years=years,
            patient_ids=patients,
            feature_names=tuple(prepared.feature_names),
            first_eligible_year=first,
            record_keys=np.asarray(prepared.record_keys).astype(str),
            feature_selection_max_year=int(np.max(prepared.years_train)),
        )

    def load_population(self, config) -> TemporalPopulation:
        from comparison_runner import DefaultComparisonAdapter

        base = self._configure(config)
        cache = ComparisonCache(
            Path(config.artifact_dir) / "_population_cache",
            enabled=config.use_cache,
            forced_stages=("prepared",) if config.force else (),
        )
        workspace = Path(config.artifact_dir) / "_population"
        workspace.mkdir(parents=True, exist_ok=True)
        prepared = DefaultComparisonAdapter(cache).prepare(
            base, workspace, force=config.force
        )
        return self._population_from_prepared(prepared)

    def load_retained_population(self, config, artifact_root, expected_fingerprints):
        """Load the exact prepared population retained with an immutable parent."""
        from temporal_robustness import _fingerprint_population

        root = Path(artifact_root)
        candidates = [root / "_population" / "prepared.pkl"]
        candidates.extend(sorted(
            (root / "_population_cache" / "prepared").glob("*/prepared.pkl")
        ))
        mismatches = []
        for path in candidates:
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                prepared = pickle.load(handle)
            population = self._population_from_prepared(prepared)
            actual = _fingerprint_population(population)
            if actual == expected_fingerprints:
                self._configure(config)
                return population
            mismatches.append(str(path))
        raise RuntimeError(
            "no retained prepared population matches the temporal parent fingerprints; "
            f"checked {len(mismatches)} candidate(s) under {root}"
        )

    def run_reference_experiment(
        self,
        *,
        population,
        reference_year,
        split,
        global_roles,
        evaluation_indices,
        domain_map,
        config,
        workspace,
    ) -> Mapping[str, Any]:
        if self._source_prepared is None or self._base_config is None:
            raise RuntimeError("load_population must run before reference experiments")
        from comparison_runner import (
            DefaultComparisonAdapter,
            _PreparedData,
        )
        from robustness_matching import analyze_run_pair

        local = {int(global_index): position for position, global_index in enumerate(evaluation_indices)}
        role_local = {
            role: np.asarray([local[int(index)] for index in indices], dtype=int)
            for role, indices in global_roles.items()
        }
        context = np.asarray(global_roles["tabpfn_context"], dtype=int)
        source = self._source_prepared
        prepared = _PreparedData(
            train_rows=source.test_rows.iloc[context].reset_index(drop=True),
            test_rows=source.test_rows.iloc[evaluation_indices].reset_index(drop=True),
            feature_names=population.feature_names,
            X_train=population.X[context],
            y_train=population.outcomes[context],
            years_train=population.years[context],
            X_test=population.X[evaluation_indices],
            y_test=population.outcomes[evaluation_indices],
            years_test=population.years[evaluation_indices],
            patient_ids=population.patient_ids[evaluation_indices],
            record_keys=population.record_keys[evaluation_indices],
            domain_reference_year=int(reference_year),
        )
        runner_config = replace(
            self._base_config,
            artifact_dir=str(workspace / "lower_stages"),
            cache_dir=str(workspace / "cache"),
            use_cache=config.use_cache,
            show_progress=config.show_progress,
            seed=split.effective_seed,
            accelerator=replace(self._base_config.accelerator, device=config.device),
            tabpfn=replace(self._base_config.tabpfn, run_walkforward=False),
            sae=replace(self._base_config.sae, seeds=config.sae_seeds),
            matching=replace(
                self._base_config.matching,
                scope="all",
                analysis_percentiles=config.matching.overlap_percentiles,
                cosine_analysis_threshold=config.matching.headline_cosine_threshold,
                overlap_analysis_threshold=config.matching.headline_overlap_threshold,
            ),
            functional=replace(
                self._base_config.functional,
                minimum_rule_samples=config.support.selected_rule_records,
                minimum_cav_samples=max(
                    config.support.cav_positive_records,
                    config.support.cav_negative_records,
                ),
                significance_runs=0,
            ),
        )
        splits = {
            "idx_semantic_fit": role_local["sae_discovery"],
            "idx_semantic_select": role_local["rule_selection_cav"],
            "idx_semantic_final": role_local["t0_evaluation"],
            "idx_tcav_eval": np.arange(len(evaluation_indices), dtype=int),
            "idx_matching_fit": role_local["rule_discovery"],
            "idx_matching_apply": role_local["rule_selection_cav"],
            "idx_rule_discovery": role_local["rule_discovery"],
            "idx_rule_selection_cav": role_local["rule_selection_cav"],
        }
        cache = ComparisonCache(
            workspace / "cache",
            enabled=config.use_cache,
            forced_stages=(
                "tabpfn", "embeddings", "sae", "activations", "matching",
                "functional", "semantic",
            ) if config.force else (),
        )
        lower = DefaultComparisonAdapter(cache)
        embeddings = lower.embeddings(prepared, splits, runner_config, workspace, force=config.force)
        sae_data = lower.train_saes(prepared, embeddings, splits, runner_config, workspace, force=config.force)

        canonical_run = sae_data.runs[0]
        analyses = {}
        for member_run in sae_data.runs[1:]:
            analyses[int(member_run["seed"])] = analyze_run_pair(
                np.asarray(canonical_run["decoder_directions"]),
                np.asarray(member_run["decoder_directions"]),
                _profile_masks(canonical_run),
                _profile_masks(member_run),
                runner_config.matching.nearest_neighbor_top_k,
            )
        views = build_canonical_factor_views(
            reference_year=reference_year,
            patient_split_seed=split.effective_seed,
            canonical_sae_seed=config.canonical_sae_seed,
            sae_seeds=config.sae_seeds,
            canonical_factor_count=sae_data.activations[0].shape[1],
            analyses_by_member_seed=analyses,
            cosine_thresholds=config.matching.cosine_analysis_thresholds,
            overlap_percentiles=config.matching.overlap_percentiles,
            overlap_thresholds=config.matching.overlap_analysis_thresholds,
        )
        union_matches = _union_matches(views["threshold_membership"], config.sae_seeds)
        functional = {}
        functional_diagnostics = [{
            "status": "replaced_by_temporal_rule_source_cavs",
            "significance_gating": False,
        }]
        semantic = _run_semantic(
            prepared, sae_data.activations, union_matches, functional,
            splits, config, workspace, cache,
        )
        predictions = _predict(embeddings, prepared, runner_config)
        performance = _performance_rows(
            population, evaluation_indices, global_roles, predictions,
            reference_year, split.effective_seed, config,
        )
        rules, semantic_models = _normalize_rules(
            semantic["semantic_models"], views["family_members"], config
        )
        semantic_selection_diagnostics = _semantic_selection_diagnostic_rows(
            semantic_models,
            views["family_members"],
            reference_year=int(reference_year),
            patient_split_seed=int(split.effective_seed),
            config=config,
        )
        rules.extend(_high_precision_rules(
            sae_data.activations, prepared, splits, union_matches,
            views["family_members"], runner_config, config,
        ))
        for row in rules:
            row["reference_year"] = int(reference_year)
            row["patient_split_seed"] = int(split.effective_seed)
        factor_year = _factor_year_rows(
            semantic_models, sae_data.activations, prepared, role_local,
            population.first_eligible_year, reference_year, split.effective_seed,
            config,
        )
        factor_year.extend(_high_precision_factor_year_rows(
            rules, sae_data.activations, prepared, role_local,
            population.first_eligible_year, reference_year,
            split.effective_seed, config,
        ))
        factor_year = _join_threshold_views(
            factor_year, views["recurrence"]
        )
        cavs, tcav_rows = _semantic_cavs(
            semantic_models, sae_data.activations, embeddings, prepared,
            role_local, reference_year, split.effective_seed, config, workspace,
            rules, population.first_eligible_year,
        )
        tcav_rows = _join_threshold_views(
            tcav_rows, views["recurrence"]
        )
        return {
            "stage_domains": {
                name: prepared.years_test - reference_year
                for name in ("predictions", "embeddings", "gradients", "activations")
            },
            "performance": performance,
            "factor_families": views["family_members"],
            "threshold_membership": views["threshold_membership"],
            "matching_recurrence": views["recurrence"],
            "overlap_percentile_winners": views["overlap_percentile_winners"],
            "matched_factors": _anchor_matches(
                views["threshold_membership"], config
            ),
            "reference_threshold_rankings": _reference_rankings(
                views["recurrence"], rules
            ),
            "post_hoc_future_sensitivity": [],
            "rules": rules,
            "semantic_selection_diagnostics": semantic_selection_diagnostics,
            "factor_year_metrics": factor_year,
            "cavs": cavs,
            "tcav": tcav_rows,
            "functional_diagnostics": functional_diagnostics,
            "semantic_result": {
                "artifact_dir": semantic["artifact_dir"],
                "experiment_hash": semantic["experiment_hash"],
            },
        }


def _profile_masks(run):
    return {
        int(percentile): np.asarray(profile["masks"], dtype=bool)
        for percentile, profile in run["high_activation_profiles"].items()
    }


def _union_matches(rows, sae_seeds):
    seed_to_run = {int(seed): index for index, seed in enumerate(sae_seeds)}
    matches = {}
    for row in rows:
        if not row.get("qualified"):
            continue
        view = str(row["matching_view"])
        target = (
            row.get("cosine_member_factor_id")
            if view.startswith("cosine") else row.get("overlap_member_factor_id")
        )
        if target is None:
            continue
        canonical_factor = int(row["canonical_factor_id"])
        member_run = seed_to_run[int(row["member_sae_seed"])]
        key = (0, member_run, canonical_factor, int(target))
        matches[key] = {
            "run_i": 0, "run_j": member_run,
            "factor_i": canonical_factor, "factor_j": int(target),
            "matching_view_union": True,
        }
    return [matches[key] for key in sorted(matches)]


def _anchor_matches(rows, config):
    output = {}
    for row in rows:
        if (
            row.get("matching_view") != "cosine_qualified"
            or not row.get("qualified")
            or float(row["cosine_threshold"]) != config.matching.headline_cosine_threshold
        ):
            continue
        key = (
            row["factor_family_uid"], int(row["member_sae_seed"]),
            int(row["cosine_member_factor_id"]),
        )
        output[key] = {
            "factor_family_uid": key[0], "member_sae_seed": key[1],
            "member_factor_id": key[2], "matching_criterion": "cosine",
            "matching_threshold": config.matching.headline_cosine_threshold,
        }
    return [output[key] for key in sorted(output)]


def _reference_rankings(recurrence, rules):
    from temporal_selection import rank_reference_configurations

    groups = {}
    for row in recurrence:
        key = (
            row.get("matching_view"), row.get("cosine_threshold"),
            row.get("overlap_percentile"), row.get("overlap_threshold"),
        )
        groups.setdefault(key, []).append(row)
    availability = sum(bool(row.get("valid")) for row in rules)
    quality = [float(row.get("f2", 0)) for row in rules if row.get("valid")]
    candidates = []
    for key, rows in groups.items():
        candidates.append({
            "configuration_id": "/".join("none" if value is None else str(value) for value in key),
            "matching_view": key[0], "cosine_threshold": key[1],
            "overlap_percentile": key[2], "overlap_threshold": key[3],
            "recurrent_factor_count": sum(bool(row["recurrent"]) for row in rows),
            "median_recurrence": float(np.median([row["recurrence"] for row in rows])),
            "raw_matching_score": float(np.mean([row["recurrence"] for row in rows])),
            "matching_agreement": float(np.mean([row["pass_count"] > 0 for row in rows])),
            "reference_rule_availability": availability,
            "reference_rule_quality": float(np.mean(quality)) if quality else 0.0,
        })
    return rank_reference_configurations(candidates)


def _run_semantic(prepared, activations, matches, functional, splits, config, workspace, cache):
    from semantic_config import SemanticExperimentConfig
    from semantic_experiment import run_semantic_comparison

    semantic = SemanticExperimentConfig.from_json(config.semantic_config_path)
    semantic = replace(
        semantic,
        activation_targets=replace(
            semantic.activation_targets,
            positive_fractions=config.activation_positive_fractions,
        ),
        runtime=replace(
            semantic.runtime,
            seed=int(prepared.domain_reference_year),
            artifact_dir=str(workspace / "semantic"),
            cache=config.use_cache,
            show_progress=config.show_progress,
        ),
    )
    predefined = {
        "idx_semantic_fit": splits["idx_rule_discovery"],
        "idx_semantic_select": splits["idx_rule_selection_cav"],
        "idx_semantic_final": splits["idx_semantic_final"],
    }
    return run_semantic_comparison(
        X=prepared.X_test,
        outcome_for_stratification=prepared.y_test,
        patient_ids=prepared.patient_ids,
        feature_names=prepared.feature_names,
        activations_by_run=activations,
        matchings=matches,
        config=semantic,
        functional_by_factor=functional,
        record_keys=prepared.record_keys,
        force=config.force,
        shared_cache=cache,
        predefined_splits=predefined,
    )


def _predict(embeddings, prepared, runner_config):
    import torch
    from tabpfn_model import make_dist_tensor

    model = embeddings.require_model()
    result = np.empty(len(prepared.X_test), dtype=int)
    batch = runner_config.tabpfn.batch_size
    for start in range(0, len(result), batch):
        end = min(start + batch, len(result))
        domains = np.asarray(
            [embeddings.year_to_domain[int(year)] for year in prepared.years_test[start:end]],
            dtype=np.int64,
        )
        dist = make_dist_tensor(domains, embeddings.model_device, embeddings.example_add_shape)
        values = prepared.X_test[start:end].astype(np.float32)
        if torch.device(embeddings.model_device).type == "cpu":
            probability = model.predict_proba(values, additional_x={"dist_shift_domain": dist})
        else:
            with torch.no_grad():
                probability = model.predict_proba(
                    torch.as_tensor(values, device=embeddings.model_device),
                    additional_x={"dist_shift_domain": dist},
                )
        if isinstance(probability, torch.Tensor):
            probability = probability.detach().cpu().numpy()
        result[start:end] = np.argmax(np.asarray(probability), axis=1)
    return result


def _performance_rows(population, evaluation_indices, global_roles, predictions, reference_year, split_seed, config):
    eval_patients = population.patient_ids[evaluation_indices].astype(str)
    eval_years = population.years[evaluation_indices]
    eval_outcomes = population.outcomes[evaluation_indices]
    role_sets = {
        role: set(population.patient_ids[indices].astype(str))
        for role, indices in global_roles.items()
    }
    t0_global = np.asarray(global_roles["t0_evaluation"], dtype=int)
    eval_lookup = {int(index): position for position, index in enumerate(evaluation_indices)}
    t0_local = np.asarray([eval_lookup[int(index)] for index in t0_global], dtype=int)
    baseline = classification_metrics(eval_outcomes[t0_local], predictions[t0_local])
    rows = []
    for year in sorted(set(eval_years)):
        provenance_counts = {name: 0 for name in (
            "returning_t0", "returning_fitting", "new_entrant",
            "prior_nonreference_returner",
        )}
        if year == reference_year:
            masks = {"all_comer": np.isin(np.arange(len(eval_years)), t0_local)}
        else:
            year_mask = eval_years == year
            labels = assign_future_provenance(
                eval_patients[year_mask], population.first_eligible_year,
                reference_year, role_sets,
            )
            masks = {
                name: _expand_mask(year_mask, mask)
                for name, mask in cohort_masks(labels).items()
            }
            provenance_counts.update({
                name: int(np.count_nonzero(labels == name))
                for name in provenance_counts
            })
        for cohort, mask in masks.items():
            metrics = support_aware_classification_metrics(
                eval_outcomes[mask], predictions[mask],
                minimum_deaths=config.support.t0_deaths,
                minimum_survivors=config.support.t0_survivors,
            )
            paired = None
            if cohort == "returning_t0" and year != reference_year:
                returning = set(eval_patients[mask])
                paired_mask = np.isin(eval_patients, list(returning)) & np.isin(np.arange(len(eval_years)), t0_local)
                paired = classification_metrics(eval_outcomes[paired_mask], predictions[paired_mask])
            row = {
                "reference_year": int(reference_year), "test_year": int(year),
                "temporal_distance": int(year-reference_year),
                "patient_split_seed": int(split_seed), "cohort_view": cohort,
                "patient_count": int(len(np.unique(eval_patients[mask]))),
                **{
                    f"provenance_{name}_count": count
                    for name, count in provenance_counts.items()
                },
                **metrics,
            }
            for metric in ("macro_f1", "death_f1"):
                row.update({f"{metric}_{key}": value for key, value in metric_delta(
                    metrics[metric], complete_t0_value=baseline[metric],
                    paired_t0_value=None if paired is None else paired[metric],
                    cohort_view=cohort,
                ).items()})
            rows.append(row)
    return rows


def _expand_mask(container, selected):
    result = np.zeros(len(container), dtype=bool)
    result[np.flatnonzero(container)] = selected
    return result


def _normalize_rules(models, family_members, config):
    identity = {}
    for member in family_members:
        identity.setdefault(
            (int(member["member_sae_seed"]), int(member["member_factor_id"])),
            member["factor_family_uid"],
        )
    normalized = []
    seed_by_run = {index: seed for index, seed in enumerate(config.sae_seeds)}
    for model in models:
        metrics = model["selection"]["metrics"]
        rule_set = model["selection"]["rule_set"]
        run = int(model["run_id"])
        factor = int(model["factor_id"])
        target = model["target"]
        normalized.append({
            "factor_family_uid": identity.get((seed_by_run[run], factor)),
            "member_sae_seed": seed_by_run[run], "member_factor_id": factor,
            "rule_source": "semantic", "activation_target": target["positive_fraction"],
            "compatibility_H": target["compatibility_H"], "cutoff": target["cutoff"],
            "rule_text": json.dumps(rule_set, sort_keys=True),
            "precision": metrics["precision"], "recall": metrics["recall"],
            "f2": metrics["f2"], "lift": metrics["lift"],
            "target_prevalence": metrics["prevalence"],
            "prediction_prevalence": metrics["coverage"],
            "cohort_size": metrics["n_samples"],
            "rule_count": len(rule_set.get("rules", [])),
            "condition_count": sum(len(rule.get("conditions", [])) for rule in rule_set.get("rules", [])),
            "semantic_family_recurrence": max(
                [family["recurrence_frequency"] for family in model["families"]], default=0.0
            ),
            "valid": model["valid"], "failure_reason": model["reason"],
        })
    output = []
    for key in sorted({(row["member_sae_seed"], row["member_factor_id"]) for row in normalized}):
        candidates = [row for row in normalized if (row["member_sae_seed"], row["member_factor_id"]) == key]
        output.extend(select_activation_winner(candidates, rule_source="semantic"))
    lookup = {
        (row["member_sae_seed"], row["member_factor_id"], row["activation_target"]): row
        for row in output
    }
    for model in models:
        run = int(model["run_id"])
        model["normalized"] = lookup[(seed_by_run[run], int(model["factor_id"]), model["target"]["positive_fraction"])]
    return output, models


def _semantic_selection_diagnostic_rows(
    models,
    family_members,
    *,
    reference_year,
    patient_split_seed,
    config,
):
    """Flatten one diagnostic row for every semantic factor-target attempt."""

    identity = {}
    for member in family_members:
        identity.setdefault(
            (int(member["member_sae_seed"]), int(member["member_factor_id"])),
            member["factor_family_uid"],
        )
    seed_by_run = {index: seed for index, seed in enumerate(config.sae_seeds)}
    rescue_names = (
        "max_rule_length",
        "max_rules",
        "min_marginal_recall",
        "min_precision",
        "min_lift",
    )
    rows = []
    for model in models:
        run = int(model["run_id"])
        factor = int(model["factor_id"])
        member_seed = int(seed_by_run[run])
        diagnostics = dict(model.get("selection_diagnostics", {}))
        selection = model.get("selection", {})
        selection_details = selection.get("diagnostics", {})
        for name in (
            "n_input_candidates",
            "n_eligible_candidates",
            "n_excluded_by_rule_length",
            "n_positive_selection_targets",
        ):
            diagnostics.setdefault(name, int(selection_details.get(name, 0)))
        for name in rescue_names:
            diagnostics.setdefault(f"rescued_without_{name}", False)
            diagnostics.setdefault(f"ablation_{name}_applicable", False)
            diagnostics.setdefault(f"ablation_{name}_evaluated_subsets", 0)
        diagnostics.setdefault("ablation_eligible", False)
        diagnostics.setdefault("funnel_stage", "unknown")
        rows.append({
            "reference_year": int(reference_year),
            "patient_split_seed": int(patient_split_seed),
            "factor_family_uid": identity.get((member_seed, factor)),
            "member_sae_seed": member_seed,
            "member_factor_id": factor,
            "activation_target": model["target"]["positive_fraction"],
            "target_name": model["target"]["name"],
            "valid": bool(model.get("valid", False)),
            "failure_reason": model.get("reason"),
            **diagnostics,
        })
    return rows


def _high_precision_rules(activations, prepared, splits, matches, family_members, runner_config, config):
    from decision_tree import mask_from_rule, train_binary_trees
    from semantic_rules import binary_metrics
    from temporal_rules import fit_canonical_targets
    from comparison_runner import _matched_factors_by_run

    discovery = splits["idx_rule_discovery"]
    selection = splits["idx_rule_selection_cav"]
    factors = _matched_factors_by_run(matches)
    identity = {}
    for member in family_members:
        identity.setdefault(
            (int(member["member_sae_seed"]), int(member["member_factor_id"])),
            member["factor_family_uid"],
        )
    rows = []
    for run, factor_ids in sorted(factors.items()):
        candidates = train_binary_trees(
            np.asarray(activations[run])[discovery], prepared.X_test[discovery],
            list(prepared.feature_names), model_type=runner_config.sae.model_type,
            max_depth=runner_config.functional.tree_max_depth,
            factor_ids=sorted(factor_ids),
            min_positive_samples=config.support.target_high_records_per_role,
            show_progress=config.show_progress,
            progress_desc=f"Temporal high-precision run {run}",
        )
        for factor in sorted(factor_ids):
            targets = fit_canonical_targets(
                np.asarray(activations[run])[discovery, factor],
                config.activation_positive_fractions,
                minimum_positive_samples=config.support.factor_positive_activations,
            )
            candidate_by_percentile = {
                int(percentile): next(
                    (row for row in values if int(row["Factor"]) == factor), None
                ) for percentile, values in candidates.items()
            }
            factor_rows = []
            for fraction, target in targets.items():
                percentile = int(round(100 * (1-fraction)))
                candidate = candidate_by_percentile.get(percentile)
                if candidate is None:
                    factor_rows.append({
                        "factor_family_uid": identity.get((config.sae_seeds[run], factor)),
                        "member_sae_seed": config.sae_seeds[run], "member_factor_id": factor,
                        "activation_target": fraction, "cutoff": target.cutoff if target.valid else None,
                        "rule_text": None, "precision": 0.0, "recall": 0.0, "f2": 0.0,
                        "lift": 0.0, "selected_count": 0,
                        "minimum_support": config.support.selected_rule_records,
                        "condition_count": 0, "valid": False,
                        "failure_reason": "no_valid_reference_candidate",
                    })
                    continue
                prediction = mask_from_rule(
                    candidate["Rule"], prepared.X_test[selection],
                    list(prepared.feature_names),
                )
                metrics = binary_metrics(
                    target.apply(np.asarray(activations[run])[selection, factor]), prediction
                )
                valid = (
                    target.valid and metrics.precision >= 0.90 and metrics.recall >= 0.25
                    and metrics.n_selected >= config.support.selected_rule_records
                )
                factor_rows.append({
                    "factor_family_uid": identity.get((config.sae_seeds[run], factor)),
                    "member_sae_seed": config.sae_seeds[run], "member_factor_id": factor,
                    "activation_target": fraction, "cutoff": target.cutoff if target.valid else None,
                    "rule_text": candidate["Rule"], "precision": metrics.precision,
                    "recall": metrics.recall, "f2": metrics.f2, "lift": metrics.lift,
                    "target_prevalence": metrics.prevalence,
                    "prediction_prevalence": metrics.coverage,
                    "cohort_size": metrics.n_samples, "selected_count": metrics.n_selected,
                    "minimum_support": config.support.selected_rule_records,
                    "condition_count": len(candidate["Rule"].split(" AND ")),
                    "valid": valid,
                    "failure_reason": None if valid else "selection_constraints_failed",
                })
            rows.extend(select_activation_winner(factor_rows, rule_source="high_precision"))
    return rows


def _factor_year_rows(models, activations, prepared, role_local, first_year, reference_year, split_seed, config):
    from semantic_rules import RuleSet, binary_metrics

    t0 = role_local["t0_evaluation"]
    rows = []
    for model in models:
        normalized = model["normalized"]
        if not model["valid"] or model["target"]["cutoff"] is None:
            continue
        run = int(model["run_id"]); factor = int(model["factor_id"])
        values = np.asarray(activations[run])[:, factor]
        target = (values > 0) & (values >= float(model["target"]["cutoff"]))
        selected = RuleSet.from_dict(model["selection"]["rule_set"]).mask(prepared.X_test)
        t0_active = int(np.count_nonzero(target[t0]))
        for year in sorted(set(prepared.years_test)):
            year_mask = prepared.years_test == year
            masks = {
                "all_comer": (
                    np.isin(np.arange(len(prepared.years_test)), t0)
                    if year == reference_year else year_mask
                )
            }
            if year > reference_year:
                role_sets = {
                    role: set(prepared.patient_ids[indices].astype(str))
                    for role, indices in role_local.items()
                    if role in {
                        "tabpfn_context", "sae_discovery", "rule_discovery",
                        "rule_selection_cav", "t0_evaluation",
                    }
                }
                labels = assign_future_provenance(
                    prepared.patient_ids[year_mask], first_year, reference_year, role_sets
                )
                masks.update({name: _expand_mask(year_mask, mask) for name, mask in cohort_masks(labels).items()})
            for cohort, mask in masks.items():
                metrics = binary_metrics(target[mask], selected[mask]).to_dict()
                retention = prevalence_retention(
                    t0_active, len(t0), int(np.count_nonzero(target[mask])), int(np.count_nonzero(mask)),
                    config.retention,
                )
                rows.append({
                    "reference_year": reference_year, "test_year": int(year),
                    "temporal_distance": int(year-reference_year), "patient_split_seed": split_seed,
                    "factor_family_uid": normalized["factor_family_uid"],
                    "member_sae_seed": normalized["member_sae_seed"], "cohort_view": cohort,
                    "rule_source": "semantic", "activation_target": normalized["activation_target"],
                    "target_role": normalized["target_role"], **metrics, **retention,
                })
    return rows


def _high_precision_factor_year_rows(rules, activations, prepared, role_local, first_year, reference_year, split_seed, config):
    from decision_tree import mask_from_rule
    from semantic_rules import binary_metrics

    seed_to_run = {int(seed): index for index, seed in enumerate(config.sae_seeds)}
    t0 = role_local["t0_evaluation"]
    role_sets = {
        role: set(prepared.patient_ids[indices].astype(str))
        for role, indices in role_local.items()
        if role in {
            "tabpfn_context", "sae_discovery", "rule_discovery",
            "rule_selection_cav", "t0_evaluation",
        }
    }
    rows = []
    for rule in rules:
        if rule.get("rule_source") != "high_precision" or not rule.get("valid"):
            continue
        run = seed_to_run[int(rule["member_sae_seed"])]
        factor = int(rule["member_factor_id"])
        values = np.asarray(activations[run])[:, factor]
        target = (values > 0) & (values >= float(rule["cutoff"]))
        selected = mask_from_rule(
            rule["rule_text"], prepared.X_test, list(prepared.feature_names)
        )
        t0_active = int(np.count_nonzero(target[t0]))
        for year in sorted(set(prepared.years_test)):
            year_mask = prepared.years_test == year
            if year == reference_year:
                masks = {
                    "all_comer": np.isin(
                        np.arange(len(prepared.years_test)), t0
                    )
                }
            else:
                labels = assign_future_provenance(
                    prepared.patient_ids[year_mask], first_year,
                    reference_year, role_sets,
                )
                masks = {
                    name: _expand_mask(year_mask, mask)
                    for name, mask in cohort_masks(labels).items()
                }
            for cohort, mask in masks.items():
                metrics = binary_metrics(target[mask], selected[mask]).to_dict()
                retention = prevalence_retention(
                    t0_active, len(t0), int(np.count_nonzero(target[mask])),
                    int(np.count_nonzero(mask)), config.retention,
                )
                rows.append({
                    "reference_year": reference_year, "test_year": int(year),
                    "temporal_distance": int(year-reference_year),
                    "patient_split_seed": split_seed,
                    "factor_family_uid": rule["factor_family_uid"],
                    "member_sae_seed": rule["member_sae_seed"],
                    "cohort_view": cohort, "rule_source": "high_precision",
                    "activation_target": rule["activation_target"],
                    "target_role": rule["target_role"], **metrics, **retention,
                })
    return rows


def _join_threshold_views(rows, recurrence):
    membership = {}
    for item in recurrence:
        if item.get("recurrent"):
            membership.setdefault(item["factor_family_uid"], []).append(item)
    output = []
    for row in rows:
        views = membership.get(row.get("factor_family_uid"), [])
        for view in views:
            output.append({
                **row,
                "matching_view": view["matching_view"],
                "cosine_threshold": view.get("cosine_threshold"),
                "overlap_percentile": view.get("overlap_percentile"),
                "overlap_threshold": view.get("overlap_threshold"),
                "geometric_factor_recurrence": view["recurrence"],
            })
    return output


def _semantic_cavs(models, activations, embeddings, prepared, role_local, reference_year, split_seed, config, workspace, rules, first_year):
    from semantic_rules import RuleSet
    from tcav import get_model_gradients

    selection = role_local["rule_selection_cav"]
    cavs = []
    trained = []
    for model in models:
        if not model["valid"]:
            continue
        normalized = model["normalized"]
        run = int(model["run_id"]); factor = int(model["factor_id"])
        fitted = train_temporal_cav(
            embeddings=embeddings.test_raw[selection], features=prepared.X_test[selection],
            activations=np.asarray(activations[run])[selection, factor],
            rule=RuleSet.from_dict(model["selection"]["rule_set"]),
            activation_target=float(model["target"]["positive_fraction"]),
            rule_source="semantic", patient_ids=prepared.patient_ids[selection],
            minimum_positive=config.support.cav_positive_records,
            minimum_negative=config.support.cav_negative_records,
            seed=split_seed + run * 1000 + factor,
        )
        row = {key: value for key, value in fitted.items() if key not in {"positive_indices", "negative_indices", "cav"}}
        row.update({
            "reference_year": reference_year, "patient_split_seed": split_seed,
            "factor_family_uid": normalized["factor_family_uid"],
            "member_sae_seed": normalized["member_sae_seed"], "member_factor_id": factor,
            "target_role": normalized["target_role"],
            "cav": None if fitted["cav"] is None else fitted["cav"].tolist(),
        })
        cavs.append(row)
        if fitted["valid"]:
            trained.append((row, np.asarray(fitted["cav"])))
    seed_to_run = {int(seed): index for index, seed in enumerate(config.sae_seeds)}
    for rule_row in rules:
        if rule_row.get("rule_source") != "high_precision" or not rule_row.get("valid"):
            continue
        seed = int(rule_row["member_sae_seed"])
        run = seed_to_run[seed]
        factor = int(rule_row["member_factor_id"])
        fitted = train_temporal_cav(
            embeddings=embeddings.test_raw[selection],
            features=prepared.X_test[selection],
            activations=np.asarray(activations[run])[selection, factor],
            rule=_TextRule(rule_row["rule_text"], prepared.feature_names),
            activation_target=float(rule_row["activation_target"]),
            rule_source="high_precision",
            patient_ids=prepared.patient_ids[selection],
            minimum_positive=config.support.cav_positive_records,
            minimum_negative=config.support.cav_negative_records,
            seed=split_seed + run * 1000 + factor,
        )
        row = {
            key: value for key, value in fitted.items()
            if key not in {"positive_indices", "negative_indices", "cav"}
        }
        row.update({
            "reference_year": reference_year, "patient_split_seed": split_seed,
            "factor_family_uid": rule_row["factor_family_uid"],
            "member_sae_seed": seed, "member_factor_id": factor,
            "target_role": rule_row["target_role"],
            "cav": None if fitted["cav"] is None else fitted["cav"].tolist(),
        })
        cavs.append(row)
        if fitted["valid"]:
            trained.append((row, np.asarray(fitted["cav"])))
    if not trained:
        return cavs, []
    domains = np.asarray([embeddings.year_to_domain[int(year)] for year in prepared.years_test])
    gradients = get_model_gradients(
        model=embeddings.require_model(require_decoder=True), dist_vec=domains,
        X=prepared.X_test, cache_file=workspace / "semantic_gradients.pkl",
        batch_size=512, device=config.device, show_progress=config.show_progress,
        use_cache=config.use_cache and not config.force,
    )
    rows = []
    role_sets = {
        role: set(prepared.patient_ids[indices].astype(str))
        for role, indices in role_local.items()
        if role in {
            "tabpfn_context", "sae_discovery", "rule_discovery",
            "rule_selection_cav", "t0_evaluation",
        }
    }
    for cav_row, direction in trained:
        for year in sorted(set(prepared.years_test)):
            year_mask = prepared.years_test == year
            if year == reference_year:
                masks = {
                    "all_comer": np.isin(
                        np.arange(len(prepared.years_test)),
                        role_local["t0_evaluation"],
                    )
                }
            else:
                labels = assign_future_provenance(
                    prepared.patient_ids[year_mask], first_year,
                    reference_year, role_sets,
                )
                masks = {
                    name: _expand_mask(year_mask, mask)
                    for name, mask in cohort_masks(labels).items()
                }
            for cohort, mask in masks.items():
                if not np.count_nonzero(mask):
                    continue
                score = temporal_tcav(direction, gradients[mask])
                rows.append({
                    **{key: cav_row[key] for key in (
                        "reference_year", "patient_split_seed", "factor_family_uid",
                        "member_sae_seed", "member_factor_id", "activation_target", "target_role",
                    )},
                    "test_year": int(year), "temporal_distance": int(year-reference_year),
                    "cohort_view": cohort,
                    "rule_source": cav_row["rule_source"], **score,
                })
    return cavs, rows


class _TextRule:
    def __init__(self, text, feature_names):
        self.text = str(text)
        self.feature_names = list(feature_names)

    def mask(self, X):
        from decision_tree import mask_from_rule

        return mask_from_rule(self.text, np.asarray(X), self.feature_names)
