import hashlib

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import concept_candidates as cc
import concept_regions as cr


def concept_rows(seed=0, n=60):
    """Synthetic cav_fit rows: 12 rule positives, unequal feature scales, eligible negatives at activation 0."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)) * np.array([0.5, 1.0, 2.0, 4.0])
    rule = np.zeros(n, bool)
    rule[rng.choice(n, 12, replace=False)] = True
    X[rule] += 1.0
    activation = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.uniform(1.0, 2.0, n))
    activation[rule] = rng.uniform(1.0, 2.0, rule.sum())
    patients = np.arange(n)
    return X, rule, activation, patients


def setting(position, auroc, status="measured"):
    return {"position": position, "selection_auroc": auroc, "status": status, "pipeline": None}


def test_gamma_scale_matches_sklearn():
    X, rule, _, _ = concept_rows()
    for matrix in (X, StandardScaler().fit_transform(X)):
        model = SVC(kernel="rbf", gamma="scale").fit(matrix, rule.astype(int))
        assert cr.gamma_scale(matrix) == pytest.approx(model._gamma, rel=1e-12)


def test_region_weights_equal_class_weights():
    X, rule, activation, patients = concept_rows()
    fit = cr.fit_region(X, rule, activation, patients, "literal", "raw", 1.0, 1.0)
    rows = cc.pool_rows(rule, activation)["literal"]
    labels = rule[rows].astype(int)
    weights = cc.class_weights(labels, np.ones(rows.sum()), cc.pool_mass(rule, patients, rows))
    reference = SVC(kernel="rbf", C=1.0, gamma=cr.gamma_scale(X[rows])).fit(X[rows], labels, sample_weight=weights)
    np.testing.assert_allclose(fit["pipeline"].decision_function(X), reference.decision_function(X),
                               rtol=1e-12, atol=1e-12)


def test_eligible_scaler_mean_is_eligible_fit_mean():
    X, rule, activation, patients = concept_rows()
    eligible = cc.pool_rows(rule, activation)["eligible"]
    fit = cr.fit_region(X, rule, activation, patients, "eligible", "standardised", 1.0, 1.0)
    scaler = fit["pipeline"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, X[eligible].mean(0))
    assert not np.allclose(scaler.mean_, X.mean(0))


def test_scaler_sees_fit_rows_only():
    X, rule, activation, patients = concept_rows()
    eligible = cc.pool_rows(rule, activation)["eligible"]
    before = cr.fit_region(X, rule, activation, patients, "eligible", "standardised", 1.0, 1.0)
    changed = X.copy()
    changed[~eligible] += 100.0
    after = cr.fit_region(changed, rule, activation, patients, "eligible", "standardised", 1.0, 1.0)
    np.testing.assert_array_equal(after["pipeline"].named_steps["scale"].mean_,
                                  before["pipeline"].named_steps["scale"].mean_)
    np.testing.assert_allclose(after["pipeline"].decision_function(X[eligible]),
                               before["pipeline"].decision_function(X[eligible]), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pool", ["literal", "eligible"])
def test_fit_linear_equals_declared_fit(pool):
    X, rule, activation, patients = concept_rows()
    declared = cc.declared_fit(X, rule, activation, patients, pool)
    fit = cr.fit_linear(X, rule, activation, patients, pool, "raw", 0.1)
    assert fit["status"] == declared["status"] == "measured"
    model = fit["pipeline"].named_steps["classifier"]
    np.testing.assert_array_equal(model.coef_[0], declared["coef"])
    assert model.intercept_[0] == declared["intercept"]
    assert fit["support"] == declared["support"]


def test_choose_max_wins():
    settings = [setting((0, 2, 2), 0.90), setting((0, 3, 2), 0.95), setting((1, 2, 1), 0.92)]
    choice = cr.choose_setting(settings, cr.REGION_ANCHOR)
    assert choice["setting"] is settings[1]
    assert choice["rule"] == "max"
    assert choice["tied"] == 1


def test_tie_goes_to_anchor():
    anchor = setting((0, 2, 2), 0.95)
    settings = [setting((0, 1, 1), 0.95), anchor, setting((1, 2, 2), 0.95), setting((0, 2, 4), 0.90)]
    choice = cr.choose_setting(settings, cr.REGION_ANCHOR)
    assert choice["setting"] is anchor
    assert choice["rule"] == "tie: anchor"
    assert choice["tied"] == 3


def test_tie_without_anchor_order():
    nearest_beats_raw = [setting((1, 2, 2), 0.95), setting((0, 4, 4), 0.95)]
    assert cr.choose_setting(nearest_beats_raw, cr.REGION_ANCHOR)["setting"] is nearest_beats_raw[0]

    raw_then_smaller_c = [setting((1, 2, 2), 0.95), setting((0, 3, 2), 0.95), setting((0, 2, 1), 0.95)]
    choice = cr.choose_setting(raw_then_smaller_c, cr.REGION_ANCHOR)
    assert choice["setting"] is raw_then_smaller_c[2]
    assert choice["rule"] == "tie: nearest grid setting"

    smaller_m = [setting((0, 3, 3), 0.95), setting((0, 3, 1), 0.95)]
    assert cr.choose_setting(smaller_m, cr.REGION_ANCHOR)["setting"] is smaller_m[1]


def test_all_undefined_uses_labelled_default():
    anchor = setting((0, 2, 2), np.nan)
    settings = [setting((0, 1, 1), np.nan), anchor, setting((1, 4, 0), np.nan)]
    choice = cr.choose_setting(settings, cr.REGION_ANCHOR)
    assert choice["setting"] is anchor
    assert choice["rule"] == "default: selection undefined"


def test_warned_settings_excluded():
    warned = setting((0, 2, 2), 0.99, status="failed")
    measured = setting((0, 3, 2), 0.90)
    choice = cr.choose_setting([warned, measured], cr.REGION_ANCHOR)
    assert choice["setting"] is measured
    assert choice["excluded"] == 1


def test_fit_rows_reject_heldout_roles():
    roles = np.array(["cav_fit", "concept_selection", "evaluation", "rule_discovery", "cav_fit"])
    year = np.full(len(roles), 2011)
    is_entrant = np.zeros(len(roles), bool)
    cc.check_fit_rows(np.array([0, 4]), roles, year, is_entrant, 2011, cr.FIT_ROLES)
    for position in (1, 2, 3):
        with pytest.raises(ValueError):
            cc.check_fit_rows(np.array([position]), roles, year, is_entrant, 2011, cr.FIT_ROLES)
    with pytest.raises(ValueError):
        cc.check_fit_rows(np.array([0]), roles, np.full(len(roles), 2012), is_entrant, 2011, cr.FIT_ROLES)


def test_selection_auroc_nan_without_both_classes():
    X, rule, activation, patients = concept_rows()
    probe = cr.fit_linear(X, rule, activation, patients, "literal", "raw", 0.1)["pipeline"]
    assert np.isnan(cr.selection_auroc(probe, X, np.zeros(len(X), bool)))
    assert cr.selection_auroc(probe, X, rule) == pytest.approx(cc.auroc(rule, probe.decision_function(X)))


def test_fit_concept_full_surface():
    X, rule, activation, patients = concept_rows()
    X_selection, rule_selection, _, _ = concept_rows(seed=1, n=40)
    result = cr.fit_concept(X, rule, activation, patients, X_selection, rule_selection, "literal")
    assert len(result["tuning"]) == 60
    assert sum(row["probe"] == "region" for row in result["tuning"]) == 50
    assert sum(row["probe"] == "linear" for row in result["tuning"]) == 10
    chosen = [row for row in result["tuning"] if row["chosen"]]
    assert sorted(row["probe"] for row in chosen) == ["linear", "region"]
    fixed = cr.fit_linear(X, rule, activation, patients, "literal", "raw", cc.CAV_C)["pipeline"]
    np.testing.assert_array_equal(result["fixed_linear"].named_steps["classifier"].coef_,
                                  fixed.named_steps["classifier"].coef_)


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "object.bin"
    path.write_bytes(b"abc")
    assert cr.sha256(path) == hashlib.sha256(b"abc").hexdigest()


HAND_LABELS = np.array([1, 1, 0, 0, 1], bool)
HAND_SCORES = np.array([0.5, -0.1, 0.2, -0.3, 1.0])
HAND_IN_A = np.array([1, 0, 1, 1, 0], bool)
HAND_WEIGHTS = np.array([[1, 1, 1, 1, 1], [2, 1, 1, 1, 1]], float)


def test_decision_scores_linear_and_region():
    X, rule, activation, patients = concept_rows()
    declared = cc.declared_fit(X, rule, activation, patients, "literal")
    linear = cr.fit_linear(X, rule, activation, patients, "literal", "raw", cc.CAV_C)["pipeline"]
    replay = cr.decision_scores((declared["coef"], declared["intercept"]), X)
    np.testing.assert_allclose(replay, X @ declared["coef"] + declared["intercept"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(replay, cr.decision_scores(linear, X), rtol=1e-12, atol=1e-12)
    region = cr.fit_region(X, rule, activation, patients, "literal", "raw", 1.0, 1.0)["pipeline"]
    np.testing.assert_array_equal(cr.decision_scores(region, X), region.decision_function(X))


def test_shared_patient_same_count_across_builds_and_years():
    population = np.array(["p2", "p1", "p3", "p4", "p1"])
    count_matrix = cc.patient_weights(np.unique(population))
    build_a = cr.draw_counts_for_rows(count_matrix, population, np.array(["p1", "p3"]))
    build_b = cr.draw_counts_for_rows(count_matrix, population, np.array(["p4", "p1", "p2"]))
    assert build_a.shape == (len(cc.BOOTSTRAP_SEEDS), 2)
    np.testing.assert_array_equal(build_a[:, 0], build_b[:, 1])
    np.testing.assert_array_equal(build_a[:, 1], count_matrix[:, 2])


def test_recall_membership_jaccard_hand_counts():
    np.testing.assert_allclose(cr.recall_by_draw(HAND_LABELS, HAND_SCORES, HAND_WEIGHTS), [2 / 3, 3 / 4])
    result = cr.membership(HAND_LABELS, HAND_SCORES, HAND_WEIGHTS, HAND_IN_A)
    np.testing.assert_allclose(result["members"], [3, 4])
    np.testing.assert_allclose(result["recall"], [2 / 3, 3 / 4])
    np.testing.assert_allclose(result["precision"], [2 / 3, 3 / 4])
    np.testing.assert_allclose(result["p_a_given_member"], [2 / 3, 3 / 4])
    other = np.array([0.1, 0.2, -0.2, -0.3, 0.4])
    np.testing.assert_allclose(cr.members_jaccard(HAND_SCORES, other, HAND_WEIGHTS), [1 / 2, 3 / 5])


def test_empty_sets_undefined():
    assert np.isnan(cr.recall_by_draw(np.zeros(5, bool), HAND_SCORES, HAND_WEIGHTS)).all()
    no_members = -np.ones(5)
    result = cr.membership(HAND_LABELS, no_members, HAND_WEIGHTS, HAND_IN_A)
    np.testing.assert_array_equal(result["members"], [0, 0])
    assert np.isnan(result["precision"]).all()
    assert np.isnan(result["p_a_given_member"]).all()
    assert np.isnan(cr.members_jaccard(no_members, no_members, HAND_WEIGHTS)).all()


def test_score_quantiles_by_group():
    scores = np.arange(10, dtype=float)
    labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], bool)
    levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    result = cr.score_quantiles(scores, labels)
    np.testing.assert_allclose(result["in rule"], np.quantile(np.arange(5.0), levels))
    np.testing.assert_allclose(result["not in rule"], np.quantile(np.arange(5.0, 10.0), levels))
    assert np.isnan(cr.score_quantiles(scores, np.zeros(10, bool))["in rule"]).all()


def test_median_difference_pairs_by_draw_nanmedian():
    region = np.array([[0.9, 0.8, np.nan], [0.7, np.nan, np.nan], [1.0, 0.6, np.nan], [0.5, 0.5, np.nan]])
    linear = np.array([[0.8, 0.8, 0.1], [0.6, 0.5, 0.1], [0.1, np.nan, 0.1], [0.5, 0.1, 0.1]])
    result = cr.median_by_draw(region - linear)
    np.testing.assert_allclose(result["median"][:2], [0.1, 0.2])
    assert np.isnan(result["median"][2])
    np.testing.assert_array_equal(result["defined"], [4, 2, 0])
    np.testing.assert_allclose(result["undefined_fraction"], [0.0, 0.5, 1.0])


def test_identical_interval_counted_separately():
    draws = 50
    differences = np.vstack([
        np.zeros(draws),
        np.full(draws, 0.5),
        np.linspace(0.1, 0.3, draws),
        np.linspace(-0.2, 0.2, draws),
        np.linspace(-0.3, -0.1, draws),
        np.full(draws, np.nan),
    ])
    assert cr.concept_positions(differences) == ["identical", "identical", "above 0", "spans 0", "below 0", "undefined"]


def test_changes_from_reference():
    reference = {"region": np.array([0.8, 0.9]), "linear": np.array([0.7, 0.6])}
    later = {"region": np.array([0.6, 0.9]), "linear": np.array([0.65, 0.5])}
    change = cr.changes_from_reference(reference, later)
    np.testing.assert_allclose(change["region"], [-0.2, 0.0])
    np.testing.assert_allclose(change["linear"], [-0.05, -0.1])
    np.testing.assert_allclose(change["region - linear"], [-0.15, 0.1])


def test_later_year_scoring_rows_evaluation_only():
    row_role = np.array(["evaluation", "rule_discovery", "evaluation", "cav_fit", "evaluation", "evaluation"])
    year = np.array([2012, 2012, 2013, 2013, 2014, 2015])
    mask = cr.scoring_rows(row_role, year, (2012, 2013, 2014))
    np.testing.assert_array_equal(mask, [True, False, True, False, True, False])


def test_coverage_row_for_unsupported():
    X, rule, activation, patients = concept_rows()
    populations = cc.scoring_populations(rule, activation, patients, cutoff=0.0)
    row = cr.coverage_row("last_3", 2012, "concept", "literal", "restricted", populations["restricted"],
                          {"region": "unsupported", "linear": "unsupported"})
    assert row["population"] == "restricted"
    assert row["rows"] == len(populations["restricted"]["rows"])
    assert row["positive_patients"] == 12
    assert row["negative_patients"] == int(np.sum(~rule & (activation == 0)))
    assert row["region_status"] == row["linear_status"] == "unsupported"


def test_agreement_given_rule_hand_counts():
    np.testing.assert_allclose(cr.agreement_given_rule(HAND_LABELS, HAND_IN_A, HAND_WEIGHTS), [1 / 3, 1 / 2])
    assert np.isnan(cr.agreement_given_rule(np.zeros(5, bool), HAND_IN_A, HAND_WEIGHTS)).all()


def test_restricted_cutoff_is_minimum_finite_activation():
    assert cr.restricted_cutoff(np.array([np.nan, 0.5, 2.0, 0.2])) == 0.2
    assert np.isnan(cr.restricted_cutoff(np.array([np.nan, np.nan])))


def test_fit_concept_task_matches_fit_concept():
    X, rule, activation, patients = concept_rows()
    X_selection, rule_selection, _, _ = concept_rows(seed=1, n=40)
    direct = cr.fit_concept(X, rule, activation, patients, X_selection, rule_selection, "literal")
    task = cr.fit_concept_task(X, rule, activation, patients, X_selection, rule_selection, "literal")
    assert [(r["probe"], r["position"], r["chosen"]) for r in direct["tuning"]] == \
        [(r["probe"], r["position"], r["chosen"]) for r in task["tuning"]]
    assert direct["region"]["rule"] == task["region"]["rule"]
    np.testing.assert_array_equal(direct["fixed_linear"].named_steps["classifier"].coef_,
                                  task["fixed_linear"].named_steps["classifier"].coef_)


def test_declared_matches_saved_exactly():
    saved = {"status": "measured", "support": [3, 4], "coef": np.array([0.5, -1.0]), "intercept": 0.25}
    assert cr.declared_matches({**saved, "coef": saved["coef"].copy()}, saved)
    assert not cr.declared_matches({**saved, "coef": np.array([0.5, -1.0 + 1e-15])}, saved)
    unsupported = {"status": "unsupported", "support": [1, 4]}
    assert cr.declared_matches(unsupported, {"status": "unsupported", "support": [1, 4]})
    assert not cr.declared_matches(unsupported, saved)


def test_median_by_draw_counts_defined_concepts():
    values = np.array([[0.2, np.nan], [0.4, np.nan], [0.6, 0.5]])
    result = cr.median_by_draw(values)
    np.testing.assert_allclose(result["median"], [0.4, 0.5])
    np.testing.assert_array_equal(result["defined"], [3, 1])
    np.testing.assert_allclose(result["undefined_fraction"], [0.0, 2 / 3])
    assert np.isnan(cr.median_by_draw(np.full((2, 1), np.nan))["median"]).all()


def test_row_mask_keeps_references_of_one_build_apart():
    rows = pd.DataFrame({
        "group": ["sae", "sae", "sae", "sae", "random"],
        "build": ["last_3"] * 5,
        "reference": [2010, 2010, 2011, 2011, 2010],
        "year": [2010, 2011, 2011, 2012, 2010],
        "years_since_reference": [0, 1, 0, 1, 0],
        "stage": ["reference", "transfer", "reference", "transfer", "reference"],
        "panel": ["primary"] * 5,
    })
    np.testing.assert_array_equal(cr.row_mask(rows, build="last_3", reference=2010), [True, True, False, False, False])
    np.testing.assert_array_equal(cr.row_mask(rows, build="last_3", reference=2011, years_since=1),
                                  [False, False, False, True, False])
    np.testing.assert_array_equal(cr.row_mask(rows, build="last_3"), [True, True, True, True, False])
    np.testing.assert_array_equal(cr.row_mask(rows, panel="primary", stage="reference", group="random"),
                                  [False, False, False, False, True])


def test_position_counts_with_chance_per_side():
    differences = np.vstack([
        np.zeros(50),
        np.full(50, 0.5),
        np.linspace(0.1, 0.3, 50),
        np.linspace(-0.2, 0.2, 50),
        np.linspace(-0.3, -0.1, 50),
        np.full(50, np.nan),
    ])
    counts = cr.position_counts(differences)
    assert {k: v for k, v in counts.items() if k != "chance each side"} == \
        {"above 0": 1, "below 0": 1, "spans 0": 1, "identical": 2, "undefined": 1}
    assert counts["chance each side"] == pytest.approx(0.075)


def test_unknown_preprocessing_raises():
    X, rule, activation, patients = concept_rows()
    with pytest.raises(ValueError):
        cr.fit_region(X, rule, activation, patients, "literal", "whitened", 1.0, 1.0)
    with pytest.raises(ValueError):
        cr.fit_linear(X, rule, activation, patients, "literal", "whitened", 0.1)
