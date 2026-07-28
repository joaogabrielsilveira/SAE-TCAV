import csv
from dataclasses import replace

import numpy as np

import semantic_experiment
from comparison_cache import ComparisonCache
from semantic_config import SemanticExperimentConfig
from semantic_rules import Condition, Rule
from stable_rule_backend import (
    BootstrapDiagnostic,
    CandidateRuleOccurrence,
    StableRuleDiscoveryResult,
)


def _small_config(artifact_dir, *, class_analysis_enabled=None):
    raw = {
        "activation_targets": {
            "positive_fractions": [0.5, 1.0],
            "min_positive_samples": 1,
        },
        "objective": {
            "objective": "f2",
            "min_precision": 0.0,
            "min_lift": 0.0,
            "max_rules": 2,
            "max_rule_length": 2,
            "min_marginal_recall": 0.0,
            "exhaustive_candidate_limit": 10,
            "beam_width": 4,
        },
        "discovery": {
            "n_bootstraps": 2,
            "trees_per_bootstrap": 1,
            "max_depth": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "splitter": "best",
            "positive_leaf_probability": 0.5,
            "min_positive_leaf_samples": 1,
            "max_candidates_per_bootstrap": 3,
            "min_family_recurrence": 0.5,
            "family_similarity_threshold": 0.7,
        },
        "runtime": {
            "seed": 9,
            "n_jobs": 1,
            "cache": True,
            "artifact_dir": str(artifact_dir),
        },
    }
    if class_analysis_enabled is not None:
        raw["class_analysis"] = {"enabled": class_analysis_enabled}
    return SemanticExperimentConfig.from_dict(raw)


def _install_stable_discovery(monkeypatch):
    def fake_discovery(
        X_fit,
        y_fit,
        feature_names,
        *,
        groups,
        clinical_group_map,
        config,
        bootstrap_ids=None,
    ):
        selected_ids = (
            range(2) if bootstrap_ids is None else bootstrap_ids
        )
        rule = Rule(
            "stable-signal",
            (Condition(0, "signal", ">", 0.0, ("signal_group",), 0.5),),
        )
        occurrences = tuple(
            CandidateRuleOccurrence(
                rule=rule,
                source="test",
                bootstrap_id=bootstrap_id,
                tree_id=0,
                bootstrap_seed=100 + bootstrap_id,
                tree_seed=200 + bootstrap_id,
                leaf_id=1,
                fit_sample_count=len(X_fit),
                fit_positive_count=int(y_fit.sum()),
                fit_selected_count=int((X_fit[:, 0] > 0).sum()),
                fit_true_positive_count=int(((X_fit[:, 0] > 0) & y_fit).sum()),
                fit_precision=1.0,
                fit_recall=1.0,
                oob_sample_count=0,
                oob_selected_count=0,
                oob_true_positive_count=0,
                oob_precision=0.0,
                oob_recall=0.0,
            )
            for bootstrap_id in selected_ids
        )
        diagnostics = tuple(
            BootstrapDiagnostic(
                bootstrap_id,
                100 + bootstrap_id,
                len(X_fit),
                len(X_fit),
                0,
                int(y_fit.sum()),
                1,
            )
            for bootstrap_id in selected_ids
        )
        return StableRuleDiscoveryResult(
            occurrences,
            diagnostics,
            tuple(feature_names),
            len(X_fit),
            int(y_fit.sum()),
            "group",
        )

    monkeypatch.setattr(
        semantic_experiment, "discover_stable_rule_candidates", fake_discovery
    )


def _multiclass_inputs():
    n = 90
    row_ids = np.arange(n, dtype=float)
    outcome = np.arange(n) % 3
    signal = ((np.arange(n) * 7) % 23 - 11).astype(float) / 5.0
    positive_signal = signal - signal.min() + 0.1
    return {
        "X": np.column_stack((signal, row_ids)),
        "outcome_for_stratification": outcome,
        "patient_ids": np.arange(n),
        "feature_names": ("signal", "row_id"),
        "activations_by_run": {
            "0": positive_signal.reshape(-1, 1),
            "1": (1.5 * positive_signal + 0.3).reshape(-1, 1),
        },
        "matchings": [
            {
                "sae_i_idx": 0,
                "sae_j_idx": 1,
                "original_concept": 0,
                "best_pair": 0,
                "cos_sim": 0.91,
                "overlap": 0.72,
            }
        ],
        "clinical_groups": {"signal": ("signal_group",)},
    }


def test_end_to_end_uses_fit_only_then_shared_final_records(monkeypatch, tmp_path):
    n = 48
    row_id = np.arange(n, dtype=float)
    signal = np.linspace(-2, 2, n)
    X = np.column_stack((signal, row_id))
    patients = np.arange(n)
    outcome = (signal > 0).astype(int)
    activations = {
        "0": np.maximum(signal + 2.1, 0).reshape(-1, 1),
        "1": np.maximum(2 * signal + 4.3, 0).reshape(-1, 1),
    }
    config = SemanticExperimentConfig.from_dict({
        "activation_targets": {"positive_fractions": [0.5], "min_positive_samples": 1},
        "objective": {
            "objective": "f2", "min_precision": 0.0, "min_lift": 0.0,
            "max_rules": 2, "max_rule_length": 2, "min_marginal_recall": 0.0,
            "exhaustive_candidate_limit": 10, "beam_width": 4,
        },
        "discovery": {
            "n_bootstraps": 2, "trees_per_bootstrap": 1, "max_depth": 2,
            "min_samples_leaf": 1, "max_features": None, "splitter": "best",
            "positive_leaf_probability": 0.5, "min_positive_leaf_samples": 1,
            "max_candidates_per_bootstrap": 3, "min_family_recurrence": 0.5,
            "family_similarity_threshold": 0.7,
        },
        "runtime": {"seed": 9, "n_jobs": 1, "cache": True, "artifact_dir": str(tmp_path)},
    })
    expected_splits = semantic_experiment.semantic_test_subsplits(outcome, patients, rng_seed=9)
    expected_fit_rows = set(row_id[expected_splits["idx_semantic_fit"]])
    expected_final_rows = set(row_id[expected_splits["idx_semantic_final"]])
    discovery_rows = []

    def fake_discovery(
        X_fit,
        y_fit,
        feature_names,
        *,
        groups,
        clinical_group_map,
        config,
        bootstrap_ids=None,
    ):
        discovery_rows.append(set(X_fit[:, 1]))
        selected_ids = (
            range(2) if bootstrap_ids is None else bootstrap_ids
        )
        rule = Rule(
            "stable-signal",
            (Condition(0, "signal", ">", 0.0, ("signal_group",), 0.5),),
        )
        occurrences = tuple(
            CandidateRuleOccurrence(
                rule=rule, source="test", bootstrap_id=bootstrap_id, tree_id=0,
                bootstrap_seed=100 + bootstrap_id, tree_seed=200 + bootstrap_id,
                leaf_id=1, fit_sample_count=len(X_fit), fit_positive_count=int(y_fit.sum()),
                fit_selected_count=int((X_fit[:, 0] > 0).sum()),
                fit_true_positive_count=int(((X_fit[:, 0] > 0) & y_fit).sum()),
                fit_precision=1.0, fit_recall=1.0, oob_sample_count=0,
                oob_selected_count=0, oob_true_positive_count=0,
                oob_precision=0.0, oob_recall=0.0,
            )
            for bootstrap_id in selected_ids
        )
        diagnostics = tuple(
            BootstrapDiagnostic(bootstrap_id, 100 + bootstrap_id, len(X_fit), len(X_fit), 0, int(y_fit.sum()), 1)
            for bootstrap_id in selected_ids
        )
        return StableRuleDiscoveryResult(
            occurrences, diagnostics, tuple(feature_names), len(X_fit), int(y_fit.sum()), "group"
        )

    monkeypatch.setattr(semantic_experiment, "discover_stable_rule_candidates", fake_discovery)
    functional = {
        ("0", 0): {"CAV": np.array([1.0, 0.0]), "TCAV_score": 0.8},
        ("1", 0): {"CAV": np.array([0.9, 0.1]), "TCAV_score": 0.75},
    }
    result = semantic_experiment.run_semantic_comparison(
        X=X,
        outcome_for_stratification=outcome,
        patient_ids=patients,
        feature_names=("signal", "row_id"),
        activations_by_run=activations,
        matchings=[{
            "sae_i_idx": 0, "sae_j_idx": 1, "original_concept": 0,
            "best_pair": 0, "cos_sim": 0.91, "overlap": 0.72,
        }],
        config=config,
        clinical_groups={"signal": ("signal_group",)},
        functional_by_factor=functional,
    )
    assert discovery_rows and all(rows == expected_fit_rows for rows in discovery_rows)
    assert all(not rows & expected_final_rows for rows in discovery_rows)
    threshold = result["pair_results"][0]["thresholds"][0]
    assert threshold["cutoff_i"] != threshold["cutoff_j"]
    assert "i_to_j" in threshold["transfer"]
    assert "j_to_i" in threshold["transfer"]
    assert result["pair_results"][0]["functional"]["tcav_effect_sign_agreement"] is True
    assert (tmp_path / result["experiment_hash"] / "semantic_rules.jsonl").exists()
    cached = semantic_experiment.run_semantic_comparison(
        X=X, outcome_for_stratification=outcome, patient_ids=patients,
        feature_names=("signal", "row_id"), activations_by_run=activations,
        matchings=[{"sae_i_idx": 0, "sae_j_idx": 1, "original_concept": 0, "best_pair": 0, "cos_sim": 0.91, "overlap": 0.72}],
        config=config, clinical_groups={"signal": ("signal_group",)}, functional_by_factor=functional,
    )
    assert cached["cache_hit"] is True


def test_class_analysis_is_additive_and_uses_only_final_class_rows(
    monkeypatch, tmp_path
):
    _install_stable_discovery(monkeypatch)
    inputs = _multiclass_inputs()
    config = _small_config(tmp_path)
    assert config.class_analysis.enabled is True

    result = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=config,
    )
    final_indices = semantic_experiment.semantic_test_subsplits(
        inputs["outcome_for_stratification"],
        inputs["patient_ids"],
        rng_seed=config.runtime.seed,
    )["idx_semantic_final"]
    final_classes = inputs["outcome_for_stratification"][final_indices]
    expected_values = sorted(np.unique(final_classes).tolist())
    expected_support = {
        str(value): int(np.count_nonzero(final_classes == value))
        for value in expected_values
    }

    pair = result["pair_results"][0]
    assert pair["thresholds"][0]["transfer"]["n_samples"] == len(final_indices)
    for threshold in pair["thresholds"]:
        class_rows = threshold["class_analysis"]
        assert [row["class_value"] for row in class_rows] == expected_values
        assert sum(row["n_samples"] for row in class_rows) == len(final_indices)
        for row in class_rows:
            class_value = row["class_value"]
            expected_count = expected_support[str(class_value)]
            assert row["n_samples"] == expected_count
            assert row["transfer"]["n_samples"] == expected_count
            assert row["transfer"]["i_to_j"]["n_samples"] == expected_count
            assert row["transfer"]["j_to_i"]["n_samples"] == expected_count
            assert row["n_positive_i"] == row["transfer"]["j_to_i"]["n_positive"]
            assert row["n_positive_j"] == row["transfer"]["i_to_j"]["n_positive"]
            assert row["transfer"]["i_to_j"]["prevalence"] == (
                row["n_positive_j"] / expected_count
            )
            assert row["transfer"]["j_to_i"]["prevalence"] == (
                row["n_positive_i"] / expected_count
            )

    assert result["manifest"]["class_analysis"] == {
        "enabled": True,
        "class_values": expected_values,
        "class_support": expected_support,
    }
    for class_value in expected_values:
        f2_values = [
            next(
                row["transfer"]["mean"]["f2"]
                for row in threshold["class_analysis"]
                if row["class_value"] == class_value
            )
            for threshold in pair["thresholds"]
        ]
        stability = pair["class_threshold_stability"][str(class_value)]
        assert stability == {
            "transfer_f2_mean_min": min(f2_values),
            "transfer_f2_mean_max": max(f2_values),
            "transfer_f2_mean_range": max(f2_values) - min(f2_values),
        }

    artifact_dir = tmp_path / result["experiment_hash"]
    with (artifact_dir / "pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
        pooled_rows = list(csv.DictReader(handle))
    with (artifact_dir / "pair_metrics_by_class.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        class_rows = list(csv.DictReader(handle))
    assert len(pooled_rows) == len(config.activation_targets.positive_fractions)
    assert "class_value" not in pooled_rows[0]
    assert len(class_rows) == len(pooled_rows) * len(expected_values)
    assert {int(row["class_value"]) for row in class_rows} == set(expected_values)

    cached = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=config,
    )
    assert cached["cache_hit"] is True
    assert cached["pair_results"] == result["pair_results"]
    assert cached["manifest"]["class_analysis"] == result["manifest"]["class_analysis"]


def test_progress_toggle_reuses_semantic_result_cache(monkeypatch, tmp_path):
    _install_stable_discovery(monkeypatch)
    inputs = _multiclass_inputs()
    config = _small_config(tmp_path)
    quiet_config = replace(
        config,
        runtime=replace(config.runtime, show_progress=False),
    )

    first = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=quiet_config,
    )
    cached = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=replace(
            quiet_config,
            runtime=replace(quiet_config.runtime, show_progress=True),
        ),
    )

    assert cached["experiment_hash"] == first["experiment_hash"]
    assert cached["cache_hit"] is True


def test_objective_change_reuses_bootstraps_and_families(tmp_path):
    config = _small_config(tmp_path / "semantic-output")
    signal = np.linspace(-2.0, 2.0, 60)
    X = np.column_stack((signal, signal**2))
    activations = signal - signal.min() + 0.1
    cache_root = tmp_path / "shared-cache"

    def learn(current_config, cache):
        return semantic_experiment.learn_factor_semantics(
            run_id="0",
            factor_id=0,
            activation_fraction=0.5,
            X_fit=X[:40],
            activations_fit=activations[:40],
            patient_groups_fit=np.arange(40),
            X_selection=X[40:],
            activations_selection=activations[40:],
            feature_names=("signal", "squared"),
            clinical_groups={"signal": ("clinical",)},
            config=current_config,
            shared_cache=cache,
        )

    first_cache = ComparisonCache(cache_root)
    learn(config, first_cache)
    changed_config = replace(
        config,
        objective=replace(config.objective, min_precision=0.1),
    )
    second_cache = ComparisonCache(cache_root)
    learn(changed_config, second_cache)

    statuses = {
        event.stage: event.status for event in second_cache.events
    }
    assert statuses["semantic_bootstrap"] == "hit"
    assert statuses["semantic_families"] == "hit"
    assert statuses["semantic_selection"] == "miss"

    increased_config = replace(
        changed_config,
        discovery=replace(
            changed_config.discovery,
            n_bootstraps=3,
        ),
    )
    third_cache = ComparisonCache(cache_root)
    learn(increased_config, third_cache)
    bootstrap_statuses = [
        event.status
        for event in third_cache.events
        if event.stage == "semantic_bootstrap"
    ]
    assert bootstrap_statuses.count("hit") == 2
    assert bootstrap_statuses.count("miss") == 1

    taxonomy_cache = ComparisonCache(cache_root)
    semantic_experiment.learn_factor_semantics(
        run_id="0",
        factor_id=0,
        activation_fraction=0.5,
        X_fit=X[:40],
        activations_fit=activations[:40],
        patient_groups_fit=np.arange(40),
        X_selection=X[40:],
        activations_selection=activations[40:],
        feature_names=("signal", "squared"),
        clinical_groups={"signal": ("renamed-clinical-group",)},
        config=increased_config,
        shared_cache=taxonomy_cache,
    )
    assert all(
        event.status == "hit"
        for event in taxonomy_cache.events
        if event.stage == "semantic_bootstrap"
    )
    assert next(
        event
        for event in taxonomy_cache.events
        if event.stage == "semantic_families"
    ).status == "miss"


def test_disabling_class_analysis_preserves_pooled_output_and_omits_additions(
    monkeypatch, tmp_path
):
    _install_stable_discovery(monkeypatch)
    inputs = _multiclass_inputs()
    enabled_config = _small_config(tmp_path / "enabled")
    disabled_config = _small_config(
        tmp_path / "disabled", class_analysis_enabled=False
    )

    enabled = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=enabled_config,
    )
    disabled = semantic_experiment.run_semantic_comparison(
        **inputs,
        config=disabled_config,
    )

    enabled_pair = enabled["pair_results"][0]
    disabled_pair = disabled["pair_results"][0]
    assert [row["transfer"] for row in enabled_pair["thresholds"]] == [
        row["transfer"] for row in disabled_pair["thresholds"]
    ]
    assert enabled_pair["threshold_stability"] == disabled_pair["threshold_stability"]
    assert all("class_analysis" not in row for row in disabled_pair["thresholds"])
    assert "class_threshold_stability" not in disabled_pair
    assert "class_analysis" not in disabled["manifest"]

    enabled_dir = tmp_path / "enabled" / enabled["experiment_hash"]
    disabled_dir = tmp_path / "disabled" / disabled["experiment_hash"]
    assert (
        enabled_dir / "pair_metrics.csv"
    ).read_text() == (
        disabled_dir / "pair_metrics.csv"
    ).read_text()
    assert (enabled_dir / "pair_metrics_by_class.csv").exists()
    assert not (disabled_dir / "pair_metrics_by_class.csv").exists()
