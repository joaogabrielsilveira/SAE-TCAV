import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

import concept_subgroup_drift_pilot as pilot
import prospective_audit as pa
import prospective_graded_trees as gt


FEATURES = [f"x{i}" for i in range(4)]


def _tree_data(rows=240, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(rows, 4))
    y = np.maximum(X[:, 0], 0.0) + 0.1 * rng.normal(size=rows)
    patients = np.repeat(np.arange(rows // 4).astype(str), 4)
    return X, y, patients


def test_dead_factors_are_dropped_and_scale_is_frozen_on_discovery_rows():
    discovery = np.array([[0.0, 2.0, 0.0],
                          [0.0, 0.0, 0.0],
                          [0.0, 4.0, 0.0],
                          [0.0, 0.0, 0.0]])
    evaluation = np.array([[9.0, 1.0, 5.0],
                           [9.0, 0.0, 5.0]])
    assert gt.live_factor_mask(discovery).tolist() == [False, True, False]
    table = gt.factor_scale_table(discovery)
    assert table["mu_plus"].iloc[1] == pytest.approx(3.0)
    assert np.isnan(table["mu_plus"].iloc[0]) and np.isnan(table["mu_plus"].iloc[2])
    scaled = gt.normalize_activations(evaluation[:, [1]], table["mu_plus"].to_numpy()[[1]])
    assert scaled.ravel().tolist() == pytest.approx([1.0 / 3.0, 0.0])


def test_normalize_refuses_dead_factor_scale():
    with pytest.raises(ValueError, match="finite and positive"):
        gt.normalize_activations(np.ones((2, 1)), np.array([np.nan]))


def test_matched_bias_hits_the_reference_active_share_exactly():
    scores = np.random.default_rng(1).normal(size=200)
    for share in (0.0, 0.37, 0.5, 1.0):
        bias = gt.matched_bias(scores, share)
        assert (scores + bias > 0).sum() == round(share * 200)


def test_sparsity_matched_random_factors_match_each_candidate_share_on_fit_rows():
    Z = np.random.default_rng(2).normal(size=(150, 6))
    shares = np.array([0.1, 0.4, 0.5])
    factors, biases = gt.sparsity_matched_random_factors(Z, Z, shares, np.random.default_rng(3))
    assert factors.shape == (150, 3) and (factors >= 0).all()
    assert len(biases) == 3
    assert (factors > 0).mean(axis=0) == pytest.approx(shares)


def test_control_sets_have_one_factor_per_candidate_and_match_the_candidate_shares():
    Z = np.random.default_rng(13).normal(size=(120, 5))
    shares = np.array([0.25, 0.5, 0.1])
    fit_rows = np.arange(120) < 80
    controls, biases = gt.control_factor_sets(Z, fit_rows, shares, seed=5)
    assert set(controls) == {"random_biased", "random_unbiased", "pca"}
    assert all(values.shape == (120, 3) for values in controls.values())
    assert len(biases) == 3
    assert (controls["random_biased"][fit_rows] > 0).mean(axis=0) == pytest.approx(shares)


def test_candidate_columns_pick_the_medoid_factor_of_each_candidate():
    codes = {42: np.arange(12, dtype=float).reshape(3, 4), 135: np.arange(12, 24, dtype=float).reshape(3, 4)}
    candidates = [{"medoid": [135, 2]}, {"medoid": [42, 0]}]
    columns = gt.candidate_columns(codes, candidates)
    assert columns.tolist() == [[14.0, 0.0], [18.0, 4.0], [22.0, 8.0]]


def test_unbiased_random_and_pca_controls_have_requested_count_and_nonnegative_values():
    Z = np.random.default_rng(4).normal(size=(80, 5))
    unbiased = gt.unbiased_random_factors(Z, 4, np.random.default_rng(6))
    pca = gt.pca_factors(Z, Z, 4)
    assert unbiased.shape == (80, 4) and (unbiased >= 0).all()
    assert pca.shape == (80, 4) and (pca >= 0).all()


def test_ccp_alpha_tie_chooses_the_larger_alpha():
    alphas = np.array([0.0, 0.01, 0.02, 0.05])
    error = np.array([1.0, 1.0 + 5e-7, 1.4, 2.0])
    assert gt.choose_ccp_alpha(alphas, error) == pytest.approx(0.01)


def test_ccp_alpha_outside_tie_chooses_the_minimum_error():
    alphas = np.array([0.0, 0.01, 0.02])
    error = np.array([1.0, 0.5, 0.9])
    assert gt.choose_ccp_alpha(alphas, error) == pytest.approx(0.01)


def test_constant_target_gives_a_single_leaf_with_constant_status():
    X, _, patients = _tree_data()
    fitted = gt.fit_graded_tree(X, np.full(len(X), 0.7), patients)
    assert fitted["status"] == "constant"
    assert fitted["tree"].get_n_leaves() == 1


def test_fitted_tree_respects_depth_and_minimum_leaf_size():
    X, y, patients = _tree_data()
    fitted = gt.fit_graded_tree(X, y, patients)
    tree = fitted["tree"]
    assert fitted["status"] == "ok"
    assert tree.get_depth() <= gt.TREE_MAX_DEPTH
    leaves = gt.leaf_nodes(tree)
    assert (tree.tree_.n_node_samples[leaves] >= gt.TREE_MIN_SAMPLES_LEAF).all()


def test_patient_grouped_folds_never_split_a_patient():
    _, _, patients = _tree_data()
    folds = gt.patient_grouped_folds(patients)
    assert len(folds) == gt.CV_FOLDS
    for train, test in folds:
        assert not set(patients[train]) & set(patients[test])
        assert len(train) + len(test) == len(patients)


def test_leaf_table_is_sorted_by_numeric_leaf_value():
    X, y, patients = _tree_data()
    tree = gt.fit_graded_tree(X, y, patients)["tree"]
    table = gt.leaf_table(tree, FEATURES)
    assert table["mean"].is_monotonic_increasing
    assert table["n"].sum() == len(X)


def test_reference_baseline_uses_evaluation_rows_only():
    X, y, patients = _tree_data()
    tree = gt.fit_graded_tree(X, y, patients)["tree"]
    X_eval, y_eval = X[:60] + 0.2, y[:60] + 3.0
    baseline = gt.reference_leaf_baseline(tree, X_eval, y_eval)
    leaf_of_eval = tree.apply(X_eval)
    for _, row in baseline.iterrows():
        inside = leaf_of_eval == row["leaf"]
        assert row["n"] == inside.sum()
        if inside.any():
            assert row["positive_mean"] == pytest.approx(y_eval[inside & (y_eval > 0)].mean())
    fit_baseline = gt.leaf_summary(tree.apply(X), y, gt.leaf_nodes(tree))
    assert not np.allclose(baseline["positive_mean"], fit_baseline["positive_mean"], equal_nan=True)


def test_leaf_with_at_most_one_row_reports_sem_intervalo():
    leaf_ids = np.array([1, 1, 2])
    y = np.array([0.0, 2.0, 0.0])
    table = gt.leaf_summary(leaf_ids, y, [1, 2, 3])
    status = dict(zip(table["leaf"], table["status"]))
    assert status[1] == "ok"
    assert status[2] == "sem intervalo"
    assert status[3] == "média indefinida"
    assert np.isnan(table.loc[table["leaf"] == 3, "positive_mean"].iloc[0])
    assert table.loc[table["leaf"] == 3, "n"].iloc[0] == 0


def test_probe_fit_rows_are_disjoint_from_tree_fit_rows():
    row_role = np.array(["tabpfn_context", "sae_discovery", "rule_discovery", "cav_fit", "evaluation", "evaluation"])
    year = np.array([2009, 2009, 2009, 2009, 2009, 2010])
    masks = gt.role_masks(row_role, year, 2009)
    assert not (masks["tree_fit"] & masks["probe_fit"]).any()
    assert masks["probe_fit"].tolist() == [False, False, False, True, False, False]
    assert masks["reference_eval"].tolist() == [False, False, False, False, True, False]
    assert masks["family_fit"].tolist() == [False, False, True, True, False, False]


def test_assert_disjoint_refuses_shared_rows():
    shared = np.array([True, False, True])
    with pytest.raises(ValueError, match="overlap"):
        gt.assert_disjoint(shared, shared)


def test_survivor_bridge_rows_follow_the_prospective_cohort_row_by_row():
    # a is alive on 31 Dec 2009 and dies in 2010; b dies in 2009; c first appears in 2010 and is alive that year.
    bridge_patient = np.array(["a", "a", "b", "c"])
    bridge_year = np.array([2009, 2010, 2009, 2010])
    cohort_patient = np.array(["a", "c"])
    cohort_year = np.array([2009, 2010])
    mask = gt.survivor_row_mask(bridge_patient, bridge_year, cohort_patient, cohort_year)
    assert mask.tolist() == [True, False, False, True]


def test_metrics_on_perfect_and_constant_predictions():
    y = np.array([0.0, 0.0, 1.0, 2.0, 3.0])
    perfect = gt.year_factor_metrics(y, y.copy(), reference_mean=1.0)
    assert perfect["skill"] == pytest.approx(1.0)
    assert perfect["hurdle_auroc"] == pytest.approx(1.0)
    assert perfect["calibration_slope"] == pytest.approx(1.0)
    constant = gt.year_factor_metrics(y, np.full(len(y), 0.5), reference_mean=1.0)
    assert np.isnan(constant["hurdle_auroc"])
    assert np.isnan(constant["spearman_patient"])
    assert np.isnan(constant["calibration_slope"])


def test_weighted_leaf_regression_recovers_a_line():
    held = np.array([0.0, 1.0, 2.0, 3.0])
    observed = 0.5 + 2.0 * held
    slope, intercept = gt.weighted_leaf_regression(observed, held, np.array([5, 1, 3, 2]))
    assert slope == pytest.approx(2.0)
    assert intercept == pytest.approx(0.5)


def test_paired_draws_share_one_patient_count_per_draw():
    patients = np.array(["p1", "p1", "p2", "p3", "p3", "p3"])
    universe, counts = pa.patient_draw_counts(patients, draws=8, seed=11)
    codes = pa.row_patient_codes(patients, universe)
    y = np.array([0.0, 1.0, 0.0, 2.0, 3.0, 0.0])
    point, draws = gt.paired_draw_statistics(lambda y: {"mean": y.mean()}, {"y": y}, codes, counts)
    assert point["mean"] == pytest.approx(y.mean())
    expected = [y[np.repeat(np.arange(len(y)), counts[b, codes])].mean() for b in range(8)]
    assert draws["mean"].to_numpy() == pytest.approx(expected)


def test_undefined_draw_share_is_reported():
    point = pd.Series({"auroc": 0.7})
    draws = pd.DataFrame({"auroc": [0.6, np.nan, 0.8, np.nan]})
    table = gt.summarize_paired_draws(point, draws)
    assert table.loc["auroc", "undefined_share"] == pytest.approx(0.5)
    assert table.loc["auroc", "valid_draws"] == 2


def test_structural_stability_returns_one_row_per_bootstrap():
    X, y, patients = _tree_data()
    fitted = gt.fit_graded_tree(X, y, patients)
    prediction = fitted["tree"].predict(X[:50])
    table = gt.structural_stability(X, y, patients, fitted["ccp_alpha"], X[:50], prediction, FEATURES,
                                    draws=4, seed=9)
    assert len(table) == 4
    assert set(table["root_feature"]) <= set(FEATURES) | {"constant"}


def test_cav_probe_recovers_the_linear_direction_in_raw_units():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(200, 5)) * np.array([1.0, 2.0, 0.5, 4.0, 1.0])
    direction = np.array([0.5, 0.0, -1.0, 0.0, 0.25])
    probe = gt.fit_cav_probe(Z, Z @ direction)
    weights = gt.raw_probe_weights(probe)
    assert gt.cosine(weights, direction) == pytest.approx(1.0, abs=1e-3)
    energy = gt.component_energy_curve(weights, Z)
    assert energy[-1] == pytest.approx(1.0)
    assert np.all(np.diff(energy) >= -1e-12)


def test_gradient_angle_and_mean_directional_derivative():
    weights = np.array([1.0, 0.0])
    assert gt.head_gradient_angle_degrees(weights, np.array([0.0, 1.0])) == pytest.approx(90.0)
    gradients = np.array([[2.0, 0.0], [0.0, 5.0]])
    assert gt.mean_directional_derivative(gradients, weights) == pytest.approx(1.0)


def test_family_candidate_set_finds_the_shared_factor_across_seeds():
    rng = np.random.default_rng(8)
    shared = rng.random(size=(60, 1)) + 0.5
    codes = {}
    for seed in (42, 135, 215):
        noise = 0.001 * rng.random(size=(60, 1))
        codes[seed] = np.hstack([shared + noise, np.zeros((60, 2))])
    live = {seed: gt.live_factor_mask(codes[seed]) for seed in codes}
    result = gt.family_candidate_set(codes, live, "unit", "cpu")
    assert result["selection"]["cap"] == 1
    candidates = result["selection"]["candidates"]
    assert len(candidates) == 1
    assert candidates[0]["family_id"].startswith("unit/")
    assert candidates[0]["medoid"][1] == 0


def test_binary_rule_arm_forwards_candidate_activations_to_the_sweep(monkeypatch):
    captured = {}

    def fake_sweep(X, patients, feature_names, candidates, activations, sweep, rule, **kwargs):
        captured.update(activations=activations, kwargs=kwargs)
        return {"frozen": []}

    monkeypatch.setattr(pilot, "run_threshold_sweep", fake_sweep)
    candidates = [{"family_id": "a", "medoid": [42, 0]}, {"family_id": "b", "medoid": [42, 1]}]
    result = gt.run_binary_rule_arm(np.zeros((6, 2)), np.array(list("abcdef")), ["f1", "f2"], candidates,
                                    np.arange(12, dtype=float).reshape(6, 2), [0.1], {"rule": 1},
                                    reference_year=2009, split_seed=42, workers=1,
                                    rule_source="semantic", recurrence=None)
    assert result == {"frozen": []}
    assert captured["activations"].shape == (2, 6)
    assert captured["kwargs"]["reference_year"] == 2009


def test_binary_rule_arm_refuses_misaligned_activations():
    candidates = [{"family_id": "a", "medoid": [42, 0]}]
    with pytest.raises(ValueError, match="one column per candidate"):
        gt.run_binary_rule_arm(np.zeros((3, 2)), np.array(list("abc")), ["f1", "f2"], candidates,
                               np.zeros((3, 2)), [0.1], {}, reference_year=2009, split_seed=42, workers=1,
                               rule_source="semantic", recurrence=None)


def test_read_only_bridge_cache_loads_saved_units_without_writing(tmp_path):
    config = {"reference_year_candidates": [2009], "t0_only_reference_years": [], "sae_seeds": [42, 135, 215],
              "epochs": 100, "disk_guard_gib": 0, "device": "cpu"}
    writer = pilot.UnitCache(tmp_path, config, "synthetic-source")
    payload = {"row_index": np.arange(4), "embeddings": np.ones((4, 3), dtype=np.float32),
               "reference_year": 2009, "roles_identity": "roles-x"}
    writer.run("P/2009", lambda: payload)
    directory = writer.directory
    before = sorted(path.as_posix() for path in directory.rglob("*"))

    cache = gt.open_bridge_cache(directory)
    loaded = cache.load("P/2009")
    assert loaded["reference_year"] == 2009
    assert loaded["roles_identity"] == "roles-x"
    assert np.array_equal(loaded["row_index"], np.arange(4))
    assert sorted(path.as_posix() for path in directory.rglob("*")) == before
    with pytest.raises(PermissionError):
        cache.run("P/2009", lambda: payload)


def test_bridge_predictor_output_must_match_the_reference_roles():
    predictor = {"row_index": np.arange(3), "embeddings": np.zeros((3, 2), dtype=np.float32),
                 "reference_year": 2009, "roles_identity": "other-roles"}
    year = np.array([2009, 2010, 2011])
    with pytest.raises(ValueError, match="different roles"):
        gt.checked_bridge_embeddings(predictor, {"identity": "roles-x", "reference_year": 2009}, year, 2009)


def test_reference_sae_fit_and_encode_round_trip_on_cpu():
    Z = np.random.default_rng(12).normal(size=(60, 6)).astype(np.float32)
    dictionaries = gt.fit_reference_saes(Z, "cpu", seeds=[42])
    codes = gt.encode_reference_saes(dictionaries, Z, "cpu")
    assert set(codes) == {42}
    assert codes[42].shape[0] == 60
    assert (codes[42] >= 0).all() and np.isfinite(codes[42]).all()


def test_duration_features_use_last_month_active_months_and_total_count():
    timing = pd.DataFrame({"last_month": [3.0, 12.0], "active_months": [2.0, 5.0]})
    X = np.array([[1.0, 2.0], [0.0, 4.0]])
    assert gt.DURATION_FEATURES == ("last_month", "active_months", "total_count")
    assert gt.duration_features(timing, X).tolist() == [[3.0, 2.0, 3.0], [12.0, 5.0, 4.0]]


def test_evaluation_rows_run_from_the_reference_year_to_the_last_outcome_year():
    role = np.array(["evaluation", "evaluation", "rule_discovery", "evaluation", "evaluation"])
    year = np.array([2009, 2013, 2009, 2008, 2014])
    assert gt.evaluation_year_mask(role, year, 2009).tolist() == [True, True, False, False, False]



def test_seed_member_columns_take_each_family_member_in_its_own_seed():
    families = [{"id": "f0", "kind": "full", "members": [[42, 1], [135, 2], [215, 0]], "medoid": [42, 1]}]
    candidates = [{"family_id": "f0", "medoid": [42, 1]}]
    codes = {42: np.arange(6, dtype=float).reshape(2, 3),
             135: np.arange(6, 12, dtype=float).reshape(2, 3),
             215: np.arange(12, 18, dtype=float).reshape(2, 3)}
    members = gt.seed_member_columns(codes, families, candidates)
    assert set(members) == {42, 135, 215}
    assert members[42].tolist() == [[1.0], [4.0]]
    assert members[135].tolist() == [[8.0], [11.0]]
    assert members[215].tolist() == [[12.0], [15.0]]


def _single_feature_rule(threshold=0.5):
    from semantic_rules import Condition, Rule, RuleSet
    condition = Condition(feature_index=0, feature_name="x0", operator=">", threshold=threshold)
    return RuleSet(rules=(Rule(rule_id="r1", conditions=(condition,)),))


def test_frozen_rule_metrics_give_active_share_given_rule_and_activation_auroc():
    entry = {"rule_set": _single_feature_rule().to_dict(), "cutoff": 1.0}
    X = np.array([[1.0], [1.0], [0.0], [0.0]])
    activation = np.array([2.0, 1.5, 0.2, 0.0])
    metrics = gt.frozen_rule_metrics(entry, X, activation)
    assert metrics["rule_share"] == pytest.approx(0.5)
    assert metrics["active_share"] == pytest.approx(0.5)
    assert metrics["active_given_rule"] == pytest.approx(1.0)
    assert metrics["auroc_activation_rule"] == pytest.approx(1.0)


def test_frozen_rule_metrics_are_undefined_when_the_rule_selects_no_rows():
    entry = {"rule_set": _single_feature_rule().to_dict(), "cutoff": 1.0}
    metrics = gt.frozen_rule_metrics(entry, np.zeros((3, 1)), np.array([2.0, 0.0, 3.0]))
    assert metrics["rule_share"] == 0.0
    assert np.isnan(metrics["active_given_rule"])
    assert np.isnan(metrics["auroc_activation_rule"])


def test_frozen_rule_year_rows_pair_patients_across_years():
    entry = {"rule_set": _single_feature_rule().to_dict(), "cutoff": 1.0}
    X = np.array([[1.0], [0.0], [1.0], [0.0], [1.0], [0.0]])
    activation = np.array([2.0, 0.0, 0.5, 0.0, 3.0, 0.0])
    years = np.array([2010, 2010, 2010, 2010, 2011, 2011])
    patients = np.array(["p1", "p2", "p3", "p4", "p1", "p3"])
    universe = np.array(["p1", "p2", "p3", "p4"])
    table = gt.frozen_rule_year_rows(entry, X, activation, years, patients, universe, draws=10, seed=5)
    assert set(table["input_year"]) == {2010, 2011}
    assert set(table["metric"]) == {"rule_share", "active_share", "active_given_rule", "auroc_activation_rule"}
    assert {"estimate", "ci_low", "ci_high", "valid_draws", "undefined_share"} <= set(table.columns)


def _double(value):
    return 2 * value


def test_process_map_keeps_job_order():
    assert gt.map_in_processes(_double, [3, 1, 2], workers=2) == [6, 2, 4]


def _worker_blas_thread_counts(_):
    import threadpoolctl
    return [entry["num_threads"] for entry in threadpoolctl.threadpool_info()]


def test_process_pool_workers_use_one_blas_thread():
    results = gt.map_in_processes(_worker_blas_thread_counts, [0, 1], workers=2)
    for thread_counts in results:
        assert thread_counts and set(thread_counts) == {1}


def test_leaf_regression_on_mean_over_all_rows_keeps_the_level_of_a_mostly_inactive_leaf():
    # Reference leaf means E[y | leaf] are 1, 2, 3. In year t, leaf 1 has 60 rows with one active row of value 3
    # (mean 0.05, positive mean 3); leaves 2 and 3 have 20 rows each, with means 2 and 3.
    leaf = np.array([1] * 60 + [2] * 20 + [3] * 20)
    y = np.concatenate([[3.0], np.zeros(59), np.full(20, 2.0), np.full(20, 3.0)])
    slope, intercept = gt._leaf_regression(leaf, y, [1, 2, 3], np.array([1.0, 2.0, 3.0]))
    assert slope == pytest.approx(98.2 / 64.0)
    assert intercept == pytest.approx(-1.425)


def test_leaf_interval_rows_blank_the_interval_when_the_row_count_behind_a_mean_is_at_most_one():
    leaves = [1, 2]
    reference_leaf, reference_y = np.array([1, 1, 1, 2, 2]), np.array([0.0, 1.0, 2.0, 0.0, 3.0])
    reference_patients = np.array(["a", "b", "c", "a", "b"])
    reference = gt.leaf_summary(reference_leaf, reference_y, leaves)
    columns = {"y": np.array([0.0, 4.0, 0.0]), "leaf": np.array([1, 1, 2])}
    patients = np.array(["a", "b", "c"])
    universe = np.array(["a", "b", "c"])
    _, counts = pa.patient_draw_counts(universe, draws=10, seed=1)
    reference_draws = gt.leaf_draws({"y": reference_y, "leaf": reference_leaf}, leaves, reference_patients,
                                    universe, counts)
    table = gt.leaf_interval_rows(columns, patients, leaves, reference, reference_draws, universe, counts,
                                  comparison="actual", input_year=2010)
    rows = table.set_index(["leaf", "quantity"])
    assert len(rows) == len(leaves) * len(gt.LEAF_QUANTITIES)
    assert rows.loc[(1, "active_share"), "status"] == "ok"
    assert rows.loc[(1, "active_share"), "ci_low"] <= rows.loc[(1, "active_share"), "ci_high"]
    assert rows.loc[(1, "mean"), "estimate"] == pytest.approx(2.0)
    assert rows.loc[(1, "mean"), "reference_value"] == pytest.approx(1.0)
    assert rows.loc[(1, "mean"), "delta_estimate"] == pytest.approx(1.0)
    assert rows.loc[(1, "mean"), "valid_draws"] == 10
    assert rows.loc[(1, "leaf_share"), "delta_estimate"] == pytest.approx(2 / 3 - 3 / 5)
    assert rows.loc[(1, "positive_mean"), "status"] == "sem intervalo"
    assert np.isnan(rows.loc[(1, "positive_mean"), "ci_low"])
    assert rows.loc[(1, "positive_mean"), "estimate"] == pytest.approx(4.0)
    assert rows.loc[(1, "positive_mean"), "delta_estimate"] == pytest.approx(2.5)
    assert rows.loc[(2, "leaf_share"), "status"] == "ok"
    assert rows.loc[(2, "mean"), "status"] == "sem intervalo"
    assert np.isnan(rows.loc[(2, "mean"), "ci_high"])
    assert rows.loc[(2, "mean"), "estimate"] == pytest.approx(0.0)
    assert rows.loc[(2, "mean"), "delta_estimate"] == pytest.approx(-1.5)


def test_candidate_leaf_table_has_one_row_per_comparison_year_leaf_and_quantity():
    rng = np.random.default_rng(31)
    X_fit = rng.normal(size=(240, 4))
    y_fit = np.maximum(X_fit[:, 0] + 0.1 * rng.normal(size=240), 0.0)
    patients_fit = np.repeat(np.array([f"f{i}" for i in range(60)]), 4)
    X_ref = rng.normal(size=(80, 4))
    y_ref = np.maximum(X_ref[:, 0] + 0.1 * rng.normal(size=80), 0.0)
    patients_ref = np.repeat(np.array([f"r{i}" for i in range(20)]), 4)
    X_eval = rng.normal(size=(120, 4))
    y_eval = np.maximum(X_eval[:, 0] + 0.1 * rng.normal(size=120), 0.0)
    y_held = np.maximum(X_eval[:, 0] + 0.5 + 0.1 * rng.normal(size=120), 0.0)
    years_eval = np.repeat([2010, 2011], 60)
    patients_eval = np.tile(np.repeat(np.array([f"e{i}" for i in range(15)]), 4), 2)
    job = {"variant": "prospective", "reference_year": 2009, "arm": "main", "candidate": "f0",
           "feature_names": FEATURES, "fit_X": X_fit, "fit_y": y_fit, "fit_patients": patients_fit,
           "reference_X": X_ref, "reference_y": y_ref, "reference_patients": patients_ref,
           "evaluation_X": X_eval, "evaluation_y": y_eval,
           "held_y": y_held, "evaluation_year": years_eval, "evaluation_patients": patients_eval,
           "universe": np.unique(np.concatenate([patients_fit, patients_ref, patients_eval])),
           "draws": 20, "bootstrap_seed": 7, "structural_draws": 2, "probe": None}
    result = gt.evaluate_candidate(job)
    leaves = result["leaves"]
    leaf_count = int(result["summary"]["leaves"].iloc[0])
    assert len(leaves) == 2 * 2 * leaf_count * len(gt.LEAF_QUANTITIES)
    assert not leaves.duplicated(["comparison", "input_year", "leaf", "quantity"]).any()
    assert set(leaves["comparison"]) == {"actual", "held"}
    assert {"n", "n_active", "estimate", "ci_low", "ci_high", "valid_draws", "undefined_share", "reference_value",
            "delta_estimate", "delta_ci_low", "delta_ci_high", "status"} <= set(leaves.columns)
    blank = leaves["status"] != "ok"
    assert leaves.loc[blank, ["ci_low", "ci_high"]].isna().all().all()


def test_paired_leaf_deltas_are_exactly_zero_when_the_year_rows_are_the_reference_rows():
    rng = np.random.default_rng(4)
    leaf = rng.integers(1, 4, 200)
    y = np.maximum(rng.normal(0.3 * leaf, 1.0), 0.0)
    leaves = [1, 2, 3]
    patients = np.array([f"p{i}" for i in range(200)])
    universe, counts = pa.patient_draw_counts(patients, draws=50, seed=3)
    columns = {"y": y, "leaf": leaf}
    reference = gt.leaf_summary(leaf, y, leaves)
    reference_draws = gt.leaf_draws(columns, leaves, patients, universe, counts)
    table = gt.leaf_interval_rows(columns, patients, leaves, reference, reference_draws, universe, counts,
                                  comparison="actual", input_year=2009)
    assert (table["delta_status"] == "ok").all()
    assert (table["delta_estimate"] == 0).all()
    assert (table["delta_ci_low"] == 0).all()
    assert (table["delta_ci_high"] == 0).all()


def test_paired_leaf_delta_follows_a_real_shift_on_the_same_patients():
    rng = np.random.default_rng(4)
    leaf = rng.integers(1, 4, 200)
    y = np.maximum(rng.normal(0.3 * leaf, 1.0), 0.0)
    leaves = [1, 2, 3]
    patients = np.array([f"p{i}" for i in range(200)])
    universe, counts = pa.patient_draw_counts(patients, draws=50, seed=3)
    reference = gt.leaf_summary(leaf, y, leaves)
    reference_draws = gt.leaf_draws({"y": y, "leaf": leaf}, leaves, patients, universe, counts)
    table = gt.leaf_interval_rows({"y": y + 1.0, "leaf": leaf}, patients, leaves, reference, reference_draws,
                                  universe, counts, comparison="actual", input_year=2010)
    rows = table.set_index(["leaf", "quantity"])
    for leaf_id in leaves:
        mean_row = rows.loc[(leaf_id, "mean")]
        assert mean_row["delta_estimate"] == pytest.approx(1.0)
        assert mean_row["delta_ci_low"] == pytest.approx(1.0)
        assert mean_row["delta_ci_high"] == pytest.approx(1.0)


def test_paired_leaf_delta_is_blank_when_the_reference_value_rests_on_one_row():
    leaves = [1, 2]
    reference_leaf = np.array([1, 2, 2, 2])
    reference_y = np.array([1.0, 0.0, 2.0, 0.0])
    reference_patients = np.array(["a", "b", "c", "d"])
    universe = np.array(["a", "b", "c", "d"])
    _, counts = pa.patient_draw_counts(universe, draws=20, seed=2)
    reference = gt.leaf_summary(reference_leaf, reference_y, leaves)
    reference_draws = gt.leaf_draws({"y": reference_y, "leaf": reference_leaf}, leaves, reference_patients,
                                    universe, counts)
    columns = {"y": np.array([0.0, 4.0, 2.0, 0.0, 3.0, 1.0]), "leaf": np.array([1, 1, 1, 2, 2, 2])}
    table = gt.leaf_interval_rows(columns, universe[[0, 1, 2, 3, 0, 1]], leaves, reference, reference_draws,
                                  universe, counts, comparison="actual", input_year=2010)
    rows = table.set_index(["leaf", "quantity"])
    assert rows.loc[(1, "mean"), "reference_n"] == 1
    assert rows.loc[(1, "mean"), "reference_n_active"] == 1
    assert rows.loc[(1, "mean"), "status"] == "ok"
    assert rows.loc[(1, "mean"), "delta_status"] == "sem intervalo"
    assert rows.loc[(1, "mean"), "delta_estimate"] == pytest.approx(2.0 - 1.0)
    assert np.isnan(rows.loc[(1, "mean"), "delta_ci_low"])
    assert np.isnan(rows.loc[(1, "mean"), "delta_ci_high"])
    assert rows.loc[(2, "mean"), "reference_n"] == 3
    assert rows.loc[(2, "mean"), "delta_status"] == "ok"
    assert not np.isnan(rows.loc[(2, "mean"), "delta_ci_low"])


def test_zero_row_leaf_is_undefined_and_one_row_leaf_is_sem_intervalo():
    leaves = [1, 2, 3]
    reference_leaf = np.array([1, 1, 2, 2, 3, 3])
    reference_y = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
    reference_patients = np.array(["a", "b", "c", "d", "a", "b"])
    universe = np.array(["a", "b", "c", "d"])
    _, counts = pa.patient_draw_counts(universe, draws=20, seed=2)
    reference = gt.leaf_summary(reference_leaf, reference_y, leaves)
    reference_draws = gt.leaf_draws({"y": reference_y, "leaf": reference_leaf}, leaves, reference_patients,
                                    universe, counts)
    columns = {"y": np.array([0.0, 1.0, 2.0, 3.0]), "leaf": np.array([1, 1, 1, 2])}
    table = gt.leaf_interval_rows(columns, universe, leaves, reference, reference_draws, universe, counts,
                                  comparison="actual", input_year=2010)
    rows = table.set_index(["leaf", "quantity"])
    for quantity in ("active_share", "mean", "positive_mean"):
        assert rows.loc[(3, quantity), "status"] == "média indefinida"
        assert np.isnan(rows.loc[(3, quantity), "estimate"])
        assert rows.loc[(3, quantity), "delta_status"] == "média indefinida"
        assert np.isnan(rows.loc[(3, quantity), "delta_estimate"])
    assert rows.loc[(2, "mean"), "status"] == "sem intervalo"
    assert rows.loc[(2, "mean"), "estimate"] == pytest.approx(3.0)
    assert np.isnan(rows.loc[(2, "mean"), "ci_low"])
    assert rows.loc[(2, "positive_mean"), "status"] == "sem intervalo"
    assert rows.loc[(2, "positive_mean"), "estimate"] == pytest.approx(3.0)


def test_delta_draws_are_undefined_when_the_leaf_is_empty_in_year_t_or_in_the_reference():
    leaves = [1, 2]
    universe = np.array(["a", "b", "c", "d"])
    _, counts = pa.patient_draw_counts(universe, draws=200, seed=9)
    reference_columns = {"y": np.array([1.0, 2.0, 1.0, 0.0]), "leaf": np.array([1, 1, 1, 2])}
    reference_patients = np.array(["a", "c", "d", "b"])
    year_columns = {"y": np.array([1.0, 0.0, 2.0, 0.0]), "leaf": np.array([1, 1, 2, 2])}
    year_patients = np.array(["a", "b", "c", "d"])
    reference = gt.leaf_summary(reference_columns["leaf"], reference_columns["y"], leaves)
    reference_draws = gt.leaf_draws(reference_columns, leaves, reference_patients, universe, counts)
    table = gt.leaf_interval_rows(year_columns, year_patients, leaves, reference, reference_draws, universe, counts,
                                  comparison="actual", input_year=2010)
    _, year_draws = gt.leaf_draws(year_columns, leaves, year_patients, universe, counts)
    _, reference_only_draws = reference_draws
    name = gt.leaf_metric_name("mean", 2)
    undefined = year_draws[name].isna().to_numpy() | reference_only_draws[name].isna().to_numpy()
    assert undefined.any()
    row = table.set_index(["leaf", "quantity"]).loc[(2, "mean")]
    assert row["delta_undefined_share"] == pytest.approx(undefined.mean())
    assert row["delta_valid_draws"] == int((~undefined).sum())


def test_candidate_leaves_name_each_leaf_by_its_path_conditions():
    job = _candidate_job()
    leaves = gt.evaluate_candidate(job)["leaves"]
    tree = gt.fit_graded_tree(job["fit_X"], job["fit_y"], job["fit_patients"])["tree"]
    expected = gt.leaf_table(tree, FEATURES).set_index("leaf")["conditions"]
    named = leaves.drop_duplicates("leaf").set_index("leaf")["conditions"]
    assert named.notna().all()
    pd.testing.assert_series_equal(named.sort_index(), expected.sort_index(), check_names=False)


def test_sparsity_match_table_reports_target_and_achieved_active_share_per_control_factor():
    Z = np.random.default_rng(21).normal(size=(120, 5))
    shares = np.array([0.25, 0.5, 0.1])
    fit_rows = np.arange(120) < 80
    controls, _ = gt.control_factor_sets(Z, fit_rows, shares, seed=5)
    table = gt.sparsity_match_table(controls["random_biased"], fit_rows, shares)
    assert table["target_active_share"].tolist() == pytest.approx(shares.tolist())
    assert table["achieved_active_share"].tolist() == pytest.approx(shares.tolist())
    assert table["gap"].tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_bridge_dictionaries_forward_to_the_cached_loader(monkeypatch):
    seen = {}

    def fake_loader(cache, r, device):
        seen.update(cache=cache, r=r, device=device)
        return {42: "dictionary"}

    monkeypatch.setattr(pilot, "load_reference_dictionaries", fake_loader)
    assert gt.load_bridge_dictionaries("cache-object", 2010, "cpu") == {42: "dictionary"}
    assert seen == {"cache": "cache-object", "r": 2010, "device": "cpu"}


def _candidate_job(draws=20):
    X, y, patients = _tree_data(rows=240)
    years = np.repeat([2010, 2011], 40)
    return {"variant": "prospective", "reference_year": 2009, "arm": "main", "candidate": "c1",
            "feature_names": FEATURES, "fit_X": X[:120], "fit_y": y[:120], "fit_patients": patients[:120],
            "reference_X": X[120:160], "reference_y": y[120:160], "reference_patients": patients[120:160],
            "evaluation_X": X[160:240], "evaluation_y": y[160:240] + 0.1, "evaluation_year": years,
            "evaluation_patients": patients[160:240], "held_y": y[160:240] + 0.3, "universe": np.unique(patients),
            "draws": draws, "bootstrap_seed": pa.BOOTSTRAP_SEED, "structural_draws": 4, "probe": None}


def _held_metrics(job):
    metrics = gt.evaluate_candidate(job)["year_metrics"]
    return metrics.pivot_table(index=["input_year", "metric"], columns="comparison", values="estimate",
                               dropna=False)


def test_held_variant_equals_actual_when_held_activation_is_the_actual_one():
    job = _candidate_job()
    job["held_y"] = job["evaluation_y"].copy()
    wide = _held_metrics(job)
    np.testing.assert_allclose(wide["held"].to_numpy(dtype=float), wide["actual"].to_numpy(dtype=float),
                               equal_nan=True)


def test_held_variant_uses_the_held_activation_on_the_same_rows():
    job = _candidate_job()
    job["held_y"] = job["evaluation_y"] + 1.0
    wide = _held_metrics(job)
    skill = wide.xs("skill", level="metric")
    assert not np.allclose(skill["held"].to_numpy(dtype=float), skill["actual"].to_numpy(dtype=float))


def test_held_variant_is_absent_without_held_activations():
    job = _candidate_job()
    job["held_y"] = None
    metrics = gt.evaluate_candidate(job)["year_metrics"]
    assert set(metrics["comparison"]) == {"actual"}


def test_held_activations_must_align_with_evaluation_rows():
    job = _candidate_job()
    job["held_y"] = job["evaluation_y"][:10]
    with pytest.raises(ValueError, match="Held activations"):
        gt.evaluate_candidate(job)


def test_candidate_job_returns_summary_year_rows_and_leaf_table():
    result = gt.evaluate_candidate(_candidate_job())
    assert set(result) == {"summary", "year_metrics", "leaves", "probe_rows"}
    summary = result["summary"].iloc[0]
    assert summary["candidate"] == "c1" and summary["tree_status"] in {"ok", "constant"}
    assert set(result["year_metrics"]["input_year"]) == {2010, 2011}
    assert set(result["year_metrics"]["comparison"]) == {"actual", "held"}
    metrics = set(result["year_metrics"]["metric"])
    assert {"skill", "hurdle_auroc", "calibration_slope", "spearman_patient", "leaf_regression_slope"} <= metrics
    assert set(result["leaves"]["status"]) <= {"ok", gt.SMALL_LEAF_STATUS}
    assert result["probe_rows"].empty


def test_candidate_job_with_probe_adds_probe_metrics_sanity_geometry_and_energy():
    job = _candidate_job()
    rng = np.random.default_rng(21)
    Z = rng.normal(size=(240, 6))
    direction = rng.normal(size=6)
    job["probe"] = {"Z_fit": Z[:60], "X_fit": job["fit_X"][:60], "y_fit": job["fit_y"][:60],
                    "Z_eval": Z[160:240], "Z_reference": Z[120:160], "gradients": rng.normal(size=(80, 6)),
                    "direction": direction}
    rows = gt.evaluate_candidate(job)["probe_rows"]
    assert set(rows["section"]) == {"probe_metrics", "sanity", "geometry", "energy"}
    sanity = set(rows.loc[rows["section"] == "sanity", "metric"])
    assert {"cosine_tree_direction", "cosine_activation_direction", "r2_reference_activation"} <= sanity
    geometry = set(rows.loc[rows["section"] == "geometry", "metric"])
    assert geometry == {"gradient_angle_degrees", "mean_directional_derivative"}
    energy = rows.loc[rows["section"] == "energy"].sort_values("metric")
    assert len(energy) == 6
    assert energy.loc[energy["metric"] == "cumulative_energy_pc6", "estimate"].iloc[0] == pytest.approx(1.0)


def test_frozen_rule_metrics_are_undefined_on_a_draw_with_no_rows():
    import warnings
    entry = {"rule_set": _single_feature_rule().to_dict(), "cutoff": 1.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = gt.frozen_rule_metrics(entry, np.zeros((0, 1)), np.zeros(0))
    assert np.isnan(metrics["rule_share"]) and np.isnan(metrics["active_share"])
    assert np.isnan(metrics["active_given_rule"]) and np.isnan(metrics["auroc_activation_rule"])
