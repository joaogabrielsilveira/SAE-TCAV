import numpy as np

from stable_rule_backend import (
    StableRuleBackendConfig,
    _bootstrap_indices,
    discover_stable_rule_candidates,
)


def _config(**overrides):
    values = {
        "n_bootstraps": 6,
        "trees_per_bootstrap": 4,
        "max_depth": 2,
        "min_samples_leaf": 2,
        "max_features": None,
        "splitter": "best",
        "positive_leaf_probability": 0.5,
        "min_positive_leaf_samples": 1,
        "random_state": 9182,
    }
    values.update(overrides)
    return StableRuleBackendConfig(**values)


def test_discovery_is_deterministic_including_provenance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] > 0.15).astype(int)

    first = discover_stable_rule_candidates(
        X, y, ("signal", "noise_a", "noise_b"), config=_config()
    )
    second = discover_stable_rule_candidates(
        X, y, ("signal", "noise_a", "noise_b"), config=_config()
    )

    assert first == second
    assert first.occurrences
    assert first.bootstrap_diagnostics


def test_selected_bootstraps_match_same_ids_from_complete_discovery():
    rng = np.random.default_rng(14)
    X = rng.normal(size=(70, 3))
    y = (X[:, 0] - X[:, 1] > 0).astype(int)
    config = _config(n_bootstraps=7, trees_per_bootstrap=3)

    complete = discover_stable_rule_candidates(
        X, y, ("a", "b", "noise"), config=config
    )
    selected = discover_stable_rule_candidates(
        X,
        y,
        ("a", "b", "noise"),
        config=config,
        bootstrap_ids=[2, 5],
    )

    assert selected.occurrences == tuple(
        item for item in complete.occurrences if item.bootstrap_id in {2, 5}
    )
    assert selected.bootstrap_diagnostics == tuple(
        item
        for item in complete.bootstrap_diagnostics
        if item.bootstrap_id in {2, 5}
    )


def test_progress_reporting_does_not_change_discovery_result():
    rng = np.random.default_rng(18)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)

    quiet = discover_stable_rule_candidates(
        X,
        y,
        ("signal", "noise"),
        config=_config(show_progress=False),
    )
    visible = discover_stable_rule_candidates(
        X,
        y,
        ("signal", "noise"),
        config=_config(show_progress=True),
    )

    assert visible == quiet


def test_group_bootstrap_keeps_group_level_oob_and_recurrence_provenance():
    # Equal-size groups make partial-group OOB leakage observable: every OOB
    # count must be divisible by rows per group.
    groups = np.repeat(np.arange(20), 3)
    signal = np.repeat(np.linspace(-2, 2, 20), 3)
    X = np.column_stack((signal, np.tile([-1.0, 0.0, 1.0], 20)))
    y = (signal > 0).astype(int)
    result = discover_stable_rule_candidates(
        X,
        y,
        ("signal", "within_group_noise"),
        groups=groups,
        config=_config(n_bootstraps=8, trees_per_bootstrap=5),
    )

    assert result.bootstrap_unit == "group"
    assert all(item.fit_sample_count == len(X) for item in result.bootstrap_diagnostics)
    assert all(item.oob_sample_count % 3 == 0 for item in result.bootstrap_diagnostics)
    assert result.occurrences
    assert all(0 <= item.bootstrap_id < 8 for item in result.occurrences)
    assert all(0 <= item.tree_id < 5 for item in result.occurrences)

    # Family recurrence must count distinct outer bootstraps, never tree-level
    # occurrences. This fixture deliberately yields repeated occurrences.
    by_rule = {}
    for occurrence in result.occurrences:
        by_rule.setdefault(occurrence.rule.rule_id, []).append(occurrence)
    assert any(
        len(items) > len({item.bootstrap_id for item in items})
        for items in by_rule.values()
    )


def test_group_bootstrap_never_splits_variable_size_groups():
    groups = np.array(
        ["patient_a"] * 2
        + ["patient_b"] * 3
        + ["patient_c"] * 4
        + ["patient_d"] * 5,
        dtype=object,
    )

    fit, oob = _bootstrap_indices(
        n_samples=len(groups), groups=groups, seed=71234
    )
    repeated_fit, repeated_oob = _bootstrap_indices(
        n_samples=len(groups), groups=groups, seed=71234
    )

    np.testing.assert_array_equal(fit, repeated_fit)
    np.testing.assert_array_equal(oob, repeated_oob)
    fit_counts = np.bincount(fit, minlength=len(groups))
    oob_mask = np.zeros(len(groups), dtype=bool)
    oob_mask[oob] = True
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        # Sampling one group k times repeats every row exactly k times. OOB
        # membership likewise applies to every row or no row in that group.
        assert len(set(fit_counts[rows].tolist())) == 1
        assert len(set(oob_mask[rows].tolist())) == 1
        assert not (fit_counts[rows[0]] > 0 and oob_mask[rows[0]])


def test_legacy_clinical_groups_argument_is_ignored():
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 30).astype(int)
    result = discover_stable_rule_candidates(
        X,
        y,
        ("creatinine",),
        clinical_group_map={"creatinine": ("renal", "laboratory")},
        config=_config(n_bootstraps=3, trees_per_bootstrap=2, max_depth=1),
    )

    assert result.occurrences
    for occurrence in result.occurrences:
        assert occurrence.rule.conditions[0].clinical_groups == ()


def test_default_backend_uses_all_features_and_allows_zero_leaf_support():
    config = StableRuleBackendConfig()

    assert config.max_depth == 15
    assert config.max_features is None
    assert config.min_positive_leaf_samples == 0


def test_negative_leaf_support_is_invalid():
    import pytest

    with pytest.raises(ValueError, match=">= 0"):
        StableRuleBackendConfig(min_positive_leaf_samples=-1)


def test_single_class_and_rare_targets_have_explicit_audit_results():
    X = np.arange(30, dtype=float).reshape(-1, 1)
    empty = discover_stable_rule_candidates(
        X, np.zeros(30, dtype=int), ("x",), config=_config()
    )
    assert empty.occurrences == ()
    assert empty.bootstrap_diagnostics == ()
    assert empty.warnings == ("target_has_single_class",)

    rare_y = np.zeros(30, dtype=int)
    rare_y[-1] = 1
    rare = discover_stable_rule_candidates(
        X,
        rare_y,
        ("x",),
        config=_config(n_bootstraps=20, trees_per_bootstrap=1),
    )
    assert len(rare.bootstrap_diagnostics) == 20
    assert "one_or_more_bootstraps_have_single_class" in rare.warnings
    assert all(item.positive_fit_count >= 0 for item in rare.bootstrap_diagnostics)


def test_two_class_target_with_no_extractable_rule_is_explicit():
    X = np.arange(60, dtype=float).reshape(-1, 1)
    y = (X[:, 0] >= 30).astype(int)
    result = discover_stable_rule_candidates(
        X,
        y,
        ("x",),
        config=_config(
            n_bootstraps=5,
            trees_per_bootstrap=3,
            min_positive_leaf_samples=100,
        ),
    )

    assert result.n_positive == 30
    assert result.occurrences == ()
    assert len(result.bootstrap_diagnostics) == 5
    assert all(item.candidates_extracted == 0 for item in result.bootstrap_diagnostics)
    assert all(item.leaves_considered > 0 for item in result.bootstrap_diagnostics)
    assert all(
        item.leaves_rejected_positive_support > 0
        for item in result.bootstrap_diagnostics
    )
    assert "no_valid_rule_candidates" in result.warnings
