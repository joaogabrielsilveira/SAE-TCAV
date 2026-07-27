import numpy as np

import semantic_experiment
from semantic_config import SemanticExperimentConfig
from semantic_rules import Condition, Rule
from stable_rule_backend import (
    BootstrapDiagnostic,
    CandidateRuleOccurrence,
    StableRuleDiscoveryResult,
)


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

    def fake_discovery(X_fit, y_fit, feature_names, *, groups, clinical_group_map, config):
        discovery_rows.append(set(X_fit[:, 1]))
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
            for bootstrap_id in range(2)
        )
        diagnostics = tuple(
            BootstrapDiagnostic(bootstrap_id, 100 + bootstrap_id, len(X_fit), len(X_fit), 0, int(y_fit.sum()), 1)
            for bootstrap_id in range(2)
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
