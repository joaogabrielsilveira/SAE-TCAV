import sys
import types

import numpy as np
import pytest

import concept_candidates as cc


def pca_encoder_stub(fit, count):
    """Same body as concept_study.alternative_bases.pca_encoder, used when that package is not on this machine."""
    mean = fit.mean(0)
    vectors = np.linalg.svd(fit - mean, full_matrices=False)[2][:(count + 1) // 2]

    def encode(z):
        scores = (z - mean) @ vectors.T
        factors = np.empty((len(z), 2 * scores.shape[1]), np.float32)
        factors[:, 0::2], factors[:, 1::2] = np.maximum(scores, 0), np.maximum(-scores, 0)
        return factors[:, :count]
    return encode


@pytest.fixture
def alternative_bases(monkeypatch):
    try:
        import concept_study.alternative_bases as module
        return module
    except ImportError:
        package = types.ModuleType("concept_study")
        module = types.ModuleType("concept_study.alternative_bases")
        module.pca_encoder = pca_encoder_stub
        package.alternative_bases = module
        monkeypatch.setitem(sys.modules, "concept_study", package)
        monkeypatch.setitem(sys.modules, "concept_study.alternative_bases", module)
        return module


def test_group_score_scaled_on_sae_rows():
    fit = np.array([[0.0, 0.0], [2.0, 10.0], [4.0, 20.0], [2.0, 10.0]])
    match = np.array([[1.0, 5.0], [3.0, 0.0]])
    result = cc.group_score(match, fit)
    sd = fit.std(0)
    np.testing.assert_allclose(result["scale"], sd)
    np.testing.assert_allclose(result["score"], (match / sd).mean(1))
    assert result["left_out"] == 0


def test_zero_sd_member_left_out():
    fit = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 2.0], [5.0, 0.0, 0.0]])
    match = np.array([[2.0, 7.0, 4.0], [0.0, 1.0, 2.0]])
    result = cc.group_score(match, fit)
    assert result["left_out"] == 1
    assert list(result["used"]) == [True, False, True]
    np.testing.assert_allclose(result["score"], (match[:, [0, 2]] / fit.std(0)[[0, 2]]).mean(1))
    none = cc.group_score(match[:, [1]], fit[:, [1]])
    assert none["score"] is None and none["left_out"] == 1


def test_member_correlations_list_negative_members():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(40)
    match = np.stack([base, base + 0.1 * rng.standard_normal(40), -base], axis=1)
    used = np.ones(3, bool)
    score = match.mean(1) + 0.3 * base
    result = cc.member_correlations(match, score, used)
    np.testing.assert_allclose(result["with_score"][0], np.corrcoef(base, score)[0, 1])
    np.testing.assert_allclose(result["between"][0, 2], -1.0)
    assert result["negative_with_score"] == [2]
    assert result["negative_pairs"] == [(0, 2), (1, 2)]


def test_pca_both_signs_fit_rows(alternative_bases):
    rng = np.random.default_rng(1)
    fit = rng.standard_normal((30, 6)) * np.array([5, 3, 1, 1, 1, 1]) + 100.0
    other = rng.standard_normal((10, 6)) + 50.0
    encode = cc.pca_candidates(fit, 4)
    factors = encode(other)
    assert factors.shape == (10, 4)
    assert (factors >= 0).all()
    np.testing.assert_allclose(encode(fit).mean(0), alternative_bases.pca_encoder(fit, 4)(fit).mean(0))
    centred = fit - fit.mean(0)
    axis = np.linalg.svd(centred, full_matrices=False)[2][0]
    scores = (other - fit.mean(0)) @ axis
    np.testing.assert_allclose(factors[:, 0], np.maximum(scores, 0), atol=1e-4)
    np.testing.assert_allclose(factors[:, 1], np.maximum(-scores, 0), atol=1e-4)


def candidate(source, name, count):
    return {"source": source, "id": name, "positive_count": count}


def test_matched_count_pool_rank_and_cap():
    candidates = [candidate("sae", "s1", 5), candidate("sae", "s2", 9),
                  candidate("group", "g1", 9), candidate("pca", "p1", 7),
                  candidate("random", "r1", 100), candidate("pca", "p2", 1)]
    pool = cc.matched_count_pool(candidates, 4)
    assert [c["id"] for c in pool] == ["s2", "g1", "p1", "s1"]
    assert all(c["source"] != "random" for c in pool)
    assert len(cc.matched_count_pool(candidates, 50)) == 5


def test_candidate_row_support_and_fields():
    activation = {"match": np.array([0.0, 1.0, 2.0, 3.0]), "evaluation": np.array([0.0, 0.0, 1.0])}
    patients = {"match": np.array(["a", "b", "b", "c"]), "evaluation": np.array(["x", "y", "z"])}
    row = cc.candidate_row("last_3", 2011, "group", 42, 0, {"family": "2011/42/7", "members": [3, 5]},
                           1.5, "+", activation, patients, {"negative_with_score": [1]})
    assert row["positive_rows_match"] == 3 and row["positive_patients_match"] == 2
    assert row["positive_rows_evaluation"] == 1 and row["positive_patients_evaluation"] == 1
    assert row["source"] == "group" and row["reference"] == 2011 and row["members"] == [3, 5]
    assert row["negative_with_score"] == [1]


def roles():
    return {"rows": np.arange(6),
            "row_role": np.array(["sae_discovery", "sae_discovery", "evaluation", "concept_selection",
                                  "sae_discovery", "sae_discovery"]),
            "year": np.array([2011, 2011, 2011, 2011, 2012, 2011]),
            "is_entrant": np.array([False, False, False, False, False, True])}


@pytest.mark.parametrize("rows,reason", [([2], "evaluation"), ([3], "concept_selection"),
                                         ([4], "later"), ([5], "entrant")])
def test_role_guard_rejects_heldout_rows(rows, reason):
    r = roles()
    with pytest.raises(ValueError, match=reason):
        cc.check_fit_rows(rows, r["row_role"], r["year"], r["is_entrant"], 2011, ("sae_discovery",))
    cc.check_fit_rows([0, 1], r["row_role"], r["year"], r["is_entrant"], 2011, ("sae_discovery",))


def test_tree_tie_prefers_fewer_leaves():
    attempts = [{"leaves": 8, "tune_mae": 1.0}, {"leaves": 5, "tune_mae": 1.0 + 5e-7},
                {"leaves": 3, "tune_mae": 1.0 + 2e-6}]
    assert cc.choose_leaves(attempts) == 5
    assert cc.choose_leaves([{"leaves": 3, "tune_mae": 2.0}, {"leaves": 8, "tune_mae": 1.0}]) == 8


def test_tree_panel_fits_and_scores_on_tune_rows():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = np.where(X[:, 0] > 0, 3.0, 0.0) + 0.05 * rng.standard_normal(200)
    panel = cc.tree_panel(X[:100], y[:100], X[100:], y[100:])
    assert panel["leaves"] in (3, 5, 8)
    assert set(panel["tune_mae"]) == {"tree", "constant", "ridge", "boosting"}
    assert panel["tune_mae"]["tree"] < panel["tune_mae"]["constant"]
    assert 0.9 < panel["retained_reduction"] <= 1.1
    table = cc.leaf_table(panel["tree"], ["a", "b", "c", "d"])
    assert len(table) == panel["leaves"] and sum(r["n"] for r in table) == 100
    assert any("a" in r["conditions"] for r in table)


def test_f_scores_from_counts():
    scores = cc.f_scores(tp=6, fp=2, fn=4, n=20)
    assert scores["precision"] == 0.75 and scores["recall"] == 0.6
    assert scores["f1"] == pytest.approx(2 * 6 / (2 * 6 + 2 + 4))
    assert scores["f2"] == pytest.approx(5 * 6 / (5 * 6 + 4 * 4 + 2))
    assert scores["coverage"] == 0.4
    empty = cc.f_scores(tp=0, fp=0, fn=3, n=10)
    assert empty["precision"] is None and empty["f2"] == 0.0


def test_threshold_respects_precision_and_lift():
    target = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)
    predicted = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)
    result = cc.tune_threshold(target, predicted)
    chosen = predicted > result["threshold"]
    assert result["status"] == "measured"
    precision = (target & chosen).sum() / chosen.sum()
    assert precision >= 0.5 and precision / target.mean() >= 1.5
    assert result["f2"] == pytest.approx(cc.f_scores(*[int((target & chosen).sum()), int((~target & chosen).sum()),
                                                       int((target & ~chosen).sum()), 10])["f2"])
    assert result["f2"] == pytest.approx(1.0)
    flat = cc.tune_threshold(np.array([1, 0, 0, 0, 0, 0], bool), np.array([0, 1, 2, 3, 4, 5], float))
    assert flat["status"] == "none" and flat["threshold"] is None


def test_retained_reduction_formula_and_undefined():
    assert cc.retained_reduction(10.0, 4.0, 2.0) == pytest.approx(0.75)
    assert np.isnan(cc.retained_reduction(10.0, 4.0, 10.0))
    assert np.isnan(cc.retained_reduction(10.0, 4.0, 11.0))


# ---------------------------------------------------------------- CAV negative pools

def original_class_weights(labels, multiplicity, mass):
    labels, multiplicity = np.asarray(labels), np.asarray(multiplicity, float)
    weights = np.zeros(len(labels), float)
    for label in (0, 1):
        at = labels == label
        total = multiplicity[at].sum()
        if total:
            weights[at] = multiplicity[at] * (mass / 2) / total
    return weights


def original_draw_multiplicity(labels, seed, method):
    labels = np.asarray(labels, int)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == label) for label in (0, 1)]
    draws = [rng.choice(pool, len(pool), replace=True) for pool in pools]
    if method == "legacy":
        count = min(map(len, pools))
        draws = [rng.choice(draw, count, replace=False) if len(draw) > count else draw for draw in draws]
    return np.bincount(np.concatenate(draws), minlength=len(labels))


def original_fit_direction(X, labels, multiplicity, mass, patients):
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    weights = original_class_weights(labels, multiplicity, mass)
    support = [len(np.unique(patients[(labels == k) & (weights > 0)])) for k in (0, 1)]
    if min(support) < 2:
        return np.full(X.shape[1], np.nan)
    model = LogisticRegression(C=.1, penalty="l2", solver="liblinear", class_weight=None, max_iter=1000, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, labels, sample_weight=weights)
    norm = np.linalg.norm(model.coef_[0])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught) or not np.isfinite(norm) or norm <= 0:
        return np.full(X.shape[1], np.nan)
    return model.coef_[0] / norm


def cav_fixture(n=80, seed=3):
    rng = np.random.default_rng(seed)
    rule = rng.random(n) < 0.3
    activation = np.where(rng.random(n) < 0.5, 0.0, rng.random(n) + 0.1)
    activation[rule] = rng.random(rule.sum()) + 0.1
    activation[0], rule[0] = 0.0, False
    X = rng.standard_normal((n, 5)) + rule[:, None] * np.array([1.5, 0, 0, 0, 0])
    patients = np.array([f"p{i // 2}" for i in range(n)])
    return X, rule, activation, patients


def test_pool_nesting():
    X, rule, activation, patients = cav_fixture()
    pools = cc.pool_rows(rule, activation)
    assert (pools["eligible"] <= pools["literal"]).all() and (pools["eligible"] & rule).sum() == rule.sum()
    assert pools["literal"].all() and pools["eligible"].sum() < len(rule)
    assert not (pools["eligible"] & ~rule & (activation > activation.min())).any()
    mult = cc.pool_multiplicities(rule, activation, 1001)
    assert ((mult["legacy"] > 0) <= (mult["eligible"] > 0)).all()
    assert ((mult["eligible"] > 0) & ~rule).sum() <= pools["eligible"][~rule].sum()
    assert (mult["literal"][~rule] > 0).sum() > (mult["eligible"][~rule] > 0).sum() or pools["literal"][~rule].sum() > 0
    assert (mult["literal"][~pools["literal"]] == 0).all() and (mult["eligible"][~pools["eligible"]] == 0).all()


def test_positive_multiplicities_shared_across_pools():
    X, rule, activation, patients = cav_fixture()
    for seed in (1001, 1002, 1050):
        mult = cc.pool_multiplicities(rule, activation, seed)
        np.testing.assert_array_equal(mult["literal"][rule], mult["eligible"][rule])
        negatives = ~rule
        pool = cc.pool_rows(rule, activation)["literal"]
        rng = np.random.default_rng([seed, 1])
        expected = np.bincount(rng.choice(np.flatnonzero(pool & negatives), (pool & negatives).sum(), replace=True),
                               minlength=len(rule))
        np.testing.assert_array_equal(mult["literal"][negatives], expected[negatives])


def test_historical_recipe_unchanged():
    X, rule, activation, patients = cav_fixture()
    labels = rule.astype(int)
    elig = cc.pool_rows(rule, activation)["eligible"]
    for method in ("legacy", "all_negatives"):
        for seed in (1001, 1017):
            np.testing.assert_array_equal(cc.draw_multiplicity(labels[elig], seed, method),
                                          original_draw_multiplicity(labels[elig], seed, method))
            mult = cc.pool_multiplicities(rule, activation, seed)["legacy" if method == "legacy" else "eligible"]
            np.testing.assert_array_equal(mult[elig], original_draw_multiplicity(labels[elig], seed, method))
            m = cc.draw_multiplicity(labels[elig], seed, method)
            mass = 2 * min(len(np.unique(patients[elig][labels[elig] == k])) for k in (0, 1))
            np.testing.assert_array_equal(cc.class_weights(labels[elig], m, mass), original_class_weights(labels[elig], m, mass))
            got = cc.fit_direction(X[elig], labels[elig], m, mass, patients[elig])
            want = original_fit_direction(X[elig], labels[elig], m, mass, patients[elig])
            assert got.tobytes() == want.tobytes()


def test_equal_class_mass_and_support_minimum():
    X, rule, activation, patients = cav_fixture()
    for pool in ("legacy", "eligible", "literal"):
        fit = cc.fit_pool(X, rule, activation, patients, 1001, pool)
        assert fit["status"] == "measured"
        mass = fit["class_mass"]
        assert mass[0] == pytest.approx(mass[1])
        rows = cc.pool_rows(rule, activation)["literal" if pool == "literal" else "eligible"]
        assert sum(mass) == pytest.approx(2 * min(len(np.unique(patients[rows & (rule == k)])) for k in (False, True)))
    one = np.array(["a"] * 80)
    thin = cc.fit_classifier(X, rule.astype(int), np.ones(80), 4.0, one)
    assert thin["status"] == "unsupported" and thin["support"] == [1, 1]
    assert np.isnan(cc.fit_direction(X, rule.astype(int), np.ones(80), 4.0, one)).all()


def test_seed_join_keeps_failed_fits():
    seeds = [1001, 1002, 1003, 1004]
    new = np.array([[1.0, 0], [np.nan, np.nan], [0, 1.0], [1.0, 0]])
    saved_seeds = [1004, 1003, 1002, 1001]
    saved = np.array([[1.0, 0], [0, 1.0], [1.0, 0], [np.nan, np.nan]])
    joined = cc.join_saved(seeds, new, saved_seeds, saved)
    assert joined["seed"].tolist() == seeds
    np.testing.assert_allclose(joined["cosine"], [np.nan, np.nan, 1.0, 1.0], equal_nan=True)
    assert joined["failed_new"].tolist() == [False, True, False, False]
    assert joined["failed_saved"].tolist() == [True, False, False, False]


def test_scoring_rows_identical_across_pools():
    X, rule, activation, patients = cav_fixture()
    populations = cc.scoring_populations(rule, activation, patients, float(activation.min()))
    assert populations["all"]["rows"].all() or len(populations["all"]["rows"]) == len(rule)
    restricted = populations["restricted"]["rows"]
    assert set(restricted) <= set(populations["all"]["rows"])
    assert not (~rule[restricted] & (activation[restricted] > activation.min())).any()
    assert populations["all"]["positive_patients"] == len(np.unique(patients[rule]))
    results = []
    for pool in ("legacy", "eligible", "literal"):
        fit = cc.fit_pool(X, rule, activation, patients, 1001, pool)
        results.append(cc.score_populations(populations, rule, X @ fit["coef"] + fit["intercept"]))
    for result in results[1:]:
        for name in ("all", "restricted"):
            assert result[name]["positive_patients"] == results[0][name]["positive_patients"]
            assert result[name]["negative_patients"] == results[0][name]["negative_patients"]
    assert all(0 <= r[name]["auroc"] <= 1 for r in results for name in ("all", "restricted"))


def test_declared_classifier_replays_scores():
    X, rule, activation, patients = cav_fixture()
    from sklearn.linear_model import LogisticRegression
    for pool in ("eligible", "literal"):
        fit = cc.declared_fit(X, rule, activation, patients, pool)
        rows = cc.pool_rows(rule, activation)["literal" if pool == "literal" else "eligible"]
        mass = 2 * min(len(np.unique(patients[rows & (rule == k)])) for k in (False, True))
        weights = original_class_weights(rule[rows].astype(int), np.ones(rows.sum()), mass)
        model = LogisticRegression(C=.1, penalty="l2", solver="liblinear", class_weight=None, max_iter=1000, random_state=42)
        model.fit(X[rows], rule[rows].astype(int), sample_weight=weights)
        np.testing.assert_allclose(X @ fit["coef"] + fit["intercept"], model.decision_function(X), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(fit["direction"], fit["coef"] / np.linalg.norm(fit["coef"]))
        assert fit["status"] == "measured" and fit["support"][0] >= 2


def test_readouts_on_known_geometry():
    gradients = np.tile([0.0, 2.0], (5, 1))
    v = np.array([[0.0, 1.0], [1.0, 0.0], [np.nan, np.nan]])
    angle = cc.evaluation_angles(gradients, v)
    np.testing.assert_allclose(angle[:2], [90.0, 0.0], atol=1e-9)
    assert np.isnan(angle[2])
    np.testing.assert_allclose(cc.mean_sensitivity(gradients, v), [2.0, 0.0, np.nan], equal_nan=True)
    cosines = cc.fit_cosines(v)
    np.testing.assert_allclose(cosines, [0.0])
    assert cc.auroc(np.array([0, 1, 0, 1]), np.array([.1, .9, .2, .8])) == 1.0
    assert cc.auroc(np.array([1, 1]), np.array([.1, .2])) is None


# ---------------------------------------------------------------- source comparison and patient bootstrap

def test_duplicate_merge_keeps_all_labels():
    candidates = [{"id": "s1", "source": "sae", "rule_patients": {"a", "b"}},
                  {"id": "g1", "source": "group", "rule_patients": {"b", "a"}},
                  {"id": "p1", "source": "pca", "rule_patients": {"a", "c"}},
                  {"id": "s2", "source": "sae", "rule_patients": {"a", "b"}},
                  {"id": "r1", "source": "random1", "rule_patients": None}]
    merged = cc.merge_duplicates(candidates)
    assert len(merged) == 2
    first = next(m for m in merged if m["patients"] == {"a", "b"})
    assert first["ids"] == ["s1", "g1", "s2"] and first["sources"] == ["sae", "group", "sae"]
    assert all("r1" not in m["ids"] for m in merged)


def test_best_jaccard():
    candidates = [{"id": "s1", "source": "sae", "rule_patients": {"a", "b", "c", "d"}},
                  {"id": "s2", "source": "sae", "rule_patients": {"a", "b", "c", "d"}},
                  {"id": "g1", "source": "group", "rule_patients": {"a", "b"}},
                  {"id": "p1", "source": "pca", "rule_patients": {"a", "b", "c"}},
                  {"id": "r1", "source": "random1", "rule_patients": None}]
    best = cc.best_jaccard(candidates)
    assert best["s1"] == pytest.approx(0.75) and best["s2"] == pytest.approx(0.75)
    assert best["g1"] == pytest.approx(2 / 3)
    assert best["p1"] == pytest.approx(0.75)
    assert np.isnan(best["r1"])
    only = cc.best_jaccard(candidates[:2])
    assert np.isnan(only["s1"])


def test_source_yield_summary():
    results = [{"source": "sae", "rule_patients": {"a"}, "f1": 0.2}, {"source": "sae", "rule_patients": {"a"}, "f1": 0.4},
               {"source": "sae", "rule_patients": {"a", "b"}, "f1": 0.6}, {"source": "sae", "rule_patients": None, "f1": np.nan},
               {"source": "pca", "rule_patients": None, "f1": np.nan}]
    rows = {r["source"]: r for r in cc.source_yield(results, "f1")}
    assert rows["sae"]["candidates"] == 4 and rows["sae"]["with_rule"] == 3 and rows["sae"]["distinct_rule_sets"] == 2
    assert rows["sae"]["median"] == pytest.approx(0.4)
    assert rows["pca"]["with_rule"] == 0 and np.isnan(rows["pca"]["median"])


def test_patient_bootstrap_paired():
    rng = np.random.default_rng(0)
    patients = np.repeat([f"p{i}" for i in range(40)], 2)
    labels = rng.random(80) < 0.4
    good, noise = labels + 0.5 * rng.standard_normal(80), rng.standard_normal(80)
    seeds = range(3001, 3041)
    weights = cc.patient_weights(patients, seeds)
    assert weights.shape == (40, 80)
    np.testing.assert_array_equal(weights, cc.patient_weights(patients, seeds))
    np.testing.assert_array_equal(weights[:, 0::2], weights[:, 1::2])
    assert (weights.sum(1) == 80).all()
    assert len(cc.patient_weights(patients)) == 500
    result = cc.bootstrap_auroc(labels, {"a": good, "b": good.copy(), "c": noise}, weights)
    np.testing.assert_array_equal(result["difference"][("a", "b")], np.zeros(40))
    np.testing.assert_allclose(result["difference"][("a", "c")], result["draws"]["a"] - result["draws"]["c"])
    from sklearn.metrics import roc_auc_score
    np.testing.assert_allclose(result["draws"]["c"][7], roc_auc_score(labels, noise, sample_weight=weights[7]))
    low, high = result["interval"]["a"]
    assert low <= result["point"]["a"] <= high or low <= high
    assert result["interval"][("a", "b")] == (0.0, 0.0)
    assert result["point"]["a"] > result["point"]["c"]


# ---------------------------------------------------------------- loading anchors

def test_replay_gap_and_nan_pattern():
    saved = np.array([[1.0, 0.0], [np.nan, np.nan], [0.0, 1.0]], np.float32)
    fits = saved.astype(np.float64) + np.array([[1e-8, 0], [np.nan, np.nan], [0, -1e-8]])
    result = cc.replay_gap(fits, saved)
    assert result["nan_pattern_equal"] and result["passed"] and result["max_abs_gap"] == pytest.approx(1e-8, rel=1e-3)
    moved = fits.copy()
    moved[0, 0] += 1e-3
    bad = cc.replay_gap(moved, saved)
    assert not bad["passed"] and bad["nan_pattern_equal"]
    extra_nan = fits.copy()
    extra_nan[2] = np.nan
    assert not cc.replay_gap(extra_nan, saved)["nan_pattern_equal"]
    assert not cc.replay_gap(extra_nan, saved)["passed"]


def test_tree_anchor_counts(tmp_path):
    path = tmp_path / "selected-trees.csv"
    path.write_text('concept,model,aliases\nc1,tree_5,"last_3"\nc2,tree_3,"last_3"\nc3,tree_8,"last_3"\n')
    result = cc.tree_anchor({"c1": "tree_5", "c2": "tree_8", "c9": "tree_3"}, path)
    assert result == {"compared": 2, "equal": 1, "different": 1, "missing_in_selected": 1}


def test_freeze_selection_aurocs_keyed_by_seed():
    fits = [{"seed": 1001, "status": "measured"}, {"seed": 1002, "status": "failed"}, {"seed": 1003, "status": "measured"}]
    discrimination = [{"status": "measured", "auroc": 0.8}, {"status": "unsupported", "auroc": None},
                      {"status": "undefined", "auroc": None}]
    unit = {"joint": {("legacy", 0.1): fits}, "scores": {("legacy", 0.1): {"discrimination": discrimination}}}
    result = cc.freeze_selection_aurocs(unit, "legacy", 0.1)
    assert result == {1001: 0.8, 1002: None, 1003: None}


# ---------------------------------------------------------------- notebook helpers

def test_family_members_keeps_full_families_only():
    families = [{"id": "f1", "kind": "full", "members": [[42, 3], [135, 5]]},
                {"id": "f2", "kind": "partial", "members": [[42, 4]]}]
    assert cc.family_members(families) == {"f1": [(42, 3), (135, 5)]}


def test_group_encoder_scales_on_fit_rows_and_drops_dead_families():
    rng = np.random.default_rng(0)
    fit_codes = {42: rng.random((20, 4)), 135: rng.random((20, 4))}
    fit_codes[42][:, 3] = 0.0
    fit_codes[135][:, 3] = 0.0
    encoders = {s: (lambda raw, s=s: raw[s]) for s in (42, 135)}
    members = {"live": [(42, 0), (135, 1)], "half": [(42, 2), (135, 3)], "dead": [(42, 3), (135, 3)]}
    encode, info, dropped = cc.group_encoder(encoders, fit_codes, members)
    assert dropped == ["dead"] and [i["id"] for i in info] == ["live", "half"]
    assert info[1]["left_out"] == 1 and info[0]["left_out"] == 0
    new = {42: rng.random((7, 4)), 135: rng.random((7, 4))}
    scores = encode(new)
    expected = (new[42][:, 0] / fit_codes[42][:, 0].std() + new[135][:, 1] / fit_codes[135][:, 1].std()) / 2
    np.testing.assert_allclose(scores[:, 0], expected)
    np.testing.assert_allclose(scores[:, 1], new[42][:, 2] / fit_codes[42][:, 2].std())


def test_top_columns_rank_by_positive_count_and_columns_encoder():
    matrix = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 1, 1]], float)
    assert cc.top_columns(matrix, 2).tolist() == [2, 0]
    encode = cc.columns_encoder({"a": lambda z: z[:, :2], "b": lambda z: z[:, 2:]}, [("b", 1), ("a", 0)])
    z = np.arange(8.0).reshape(2, 4)
    np.testing.assert_array_equal(encode(z), z[:, [3, 0]])


def test_replay_concepts_pass_on_own_fits_and_flag_a_changed_pool():
    X, rule, activation, patients = cav_fixture()
    seeds = range(1001, 1006)
    candidate = {"rule": rule, "activation": activation, "patients": patients}
    fitted = cc.fit_candidates(X, [candidate], seeds, 1)[0]
    saved = {pool: np.stack([f["direction"] if f["status"] == "measured" else np.full(5, np.nan) for f in fitted[pool][:-1]])
             for pool in ("legacy", "eligible")}
    result = cc.replay_concepts(X, [candidate, candidate], [saved, saved], seeds, 1)
    assert all(r["legacy"]["passed"] and r["eligible"]["passed"] for r in result)
    changed = {"legacy": saved["legacy"], "eligible": saved["legacy"] + 0.1}
    assert not cc.replay_concepts(X, [candidate], [changed], seeds, 1)[0]["eligible"]["passed"]


def test_anchors_pass_needs_every_exact_check():
    anchor = {"rule_identities_equal": True, "family_counts_equal": True, "candidates_equal": True,
              "concepts_missing_from_index": 0, "A_or_R_row_mismatches": 0, "agreement_max_abs_gap": 0.0}
    replays = [{"legacy": {"passed": True}, "eligible": {"passed": True}}]
    assert cc.anchors_pass(anchor, replays)
    assert not cc.anchors_pass({**anchor, "A_or_R_row_mismatches": 2}, replays)
    assert not cc.anchors_pass(anchor, replays + [{"legacy": {"passed": False}, "eligible": {"passed": True}}])


def description_fixture(seed=0, n=300):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 6))
    y = np.maximum(X[:, 0] * 2 + 0.2 * rng.standard_normal(n), 0)
    fit = np.arange(n) % 2 == 0
    return X, y, fit, ~fit


def test_describe_candidate_row_fields_and_unsupported():
    X, y, fit, tune = description_fixture()
    in_rule = X[:, 1] > 0
    row, panel = cc.describe_candidate(X[fit], y[fit], X[tune], y[tune], X[:100], y[:100], 1.0, in_rule[:100])
    assert row["status"] == "measured" and row["leaves"] in (3, 5, 8)
    assert row["rule_tp"] + row["rule_fn"] == int((y[:100] > 1.0).sum())
    for key in ("tree_mae", "constant_mae", "boosting_mae", "tree_mae_high", "retained_reduction",
                "tree_f2_cutoff", "tree_f2_tuned", "rule_f1", "rule_f2", "threshold_status"):
        assert key in row
    assert panel["tree"].get_n_leaves() == row["leaves"]
    empty, none = cc.describe_candidate(X[:0], y[:0], X[tune], y[tune], X[:10], y[:10], 1.0, in_rule[:10])
    assert empty["status"] == "unsupported" and none is None


def test_fidelity_intervals_are_paired_and_count_undefined_draws():
    rng = np.random.default_rng(2)
    n = 120
    truth = rng.random(n) * 3
    tree = truth + 0.3 * rng.standard_normal(n)
    constant = np.full(n, np.median(truth))
    boost = truth + 0.1 * rng.standard_normal(n)
    in_rule = truth + rng.standard_normal(n) > 1.5
    weights = cc.patient_weights(np.arange(n), range(3001, 3101))
    result = cc.fidelity_intervals(weights, truth, tree, constant, boost, truth > 1.5, in_rule, 1.5)
    assert set(result) >= {"rule_f1", "tree_f1", "f1_difference", "retained_reduction"}
    for value in result.values():
        assert value["low"] <= value["high"] and 0 <= value["undefined_fraction"] <= 1
    again = cc.fidelity_intervals(weights, truth, tree, constant, boost, truth > 1.5, in_rule, 1.5)
    assert again["f1_difference"] == result["f1_difference"]


def sets_fixture():
    X, rule, activation, patients = cav_fixture()
    rng = np.random.default_rng(9)
    n = 50
    Xs = rng.standard_normal((n, 5))
    rs = rng.random(n) < 0.3
    acts = np.where(rng.random(n) < 0.5, 0.0, rng.random(n) + 0.1)
    acts[rs] += 0.1
    ps = np.array([f"q{i // 2}" for i in range(n)])
    gradients = rng.standard_normal((n, 5)) + np.array([1.0, 0, 0, 0, 0])
    one = {"X": Xs, "rule": rs, "activation": acts, "patients": ps, "gradients": gradients}
    return (X, rule, activation, patients), {"evaluation": one, "selection": {**one, "gradients": -gradients}}


def test_run_pools_tables_and_shared_populations():
    fit, sets = sets_fixture()
    result = cc.run_pools(*fit, sets, range(1001, 1006), 1)
    fits = result["fits"]
    assert len(fits) == 3 * 5 and {f["pool"] for f in fits} == {"legacy", "eligible", "literal"}
    first = fits[0]
    for key in ("seed", "status", "auroc_all_evaluation", "auroc_restricted_evaluation", "auroc_all_selection",
                "angle_evaluation", "angle_selection", "sensitivity_evaluation"):
        assert key in first
    assert result["populations"]["evaluation"]["all"]["positive_patients"] == len(np.unique(sets["evaluation"]["patients"][sets["evaluation"]["rule"]]))
    assert set(result["declared"]) == {"legacy", "eligible", "literal"}
    assert result["directions"]["literal"].shape == (5, 5)
    ok = [f["angle_evaluation"] for f in fits if f["pool"] == "legacy" and f["status"] == "measured"]
    flipped = [f["angle_selection"] for f in fits if f["pool"] == "legacy" and f["status"] == "measured"]
    np.testing.assert_allclose(ok, -np.array(flipped), atol=1e-9)
    assert len(result["cosines"]["eligible"]) <= 10


def test_bases_helpers_match_the_encoders(alternative_bases):
    rng = np.random.default_rng(4)
    fit = rng.standard_normal((30, 6)) * np.array([5, 3, 1, 1, 1, 1])
    mean, axes = cc.pca_axes(fit, 4)
    np.testing.assert_allclose(mean, fit.mean(0))
    z = rng.standard_normal((5, 6))
    scores = (z - mean) @ axes.T
    expected = alternative_bases.pca_encoder(fit, 4)(z)
    np.testing.assert_allclose(expected[:, 0], np.maximum(scores[:, 0], 0), atol=1e-5)
    directions = cc.random_directions(6, 10, [1, 2, 3])
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)
    np.testing.assert_array_equal(directions, cc.random_directions(6, 10, [1, 2, 3]))


def test_attach_sweep_pairs_concepts_with_frozen_entries():
    concepts = [{"concept": "a"}, {"concept": "b"}]
    sweep = {"frozen": [{"medoid": [42, 3], "inner_split": {"fit_rows": 5}}, {"medoid": [0, 1], "inner_split": {"fit_rows": 6}}]}
    cc.attach_sweep(concepts, sweep)
    assert concepts[1]["medoid"] == [0, 1] and concepts[0]["inner_split"]["fit_rows"] == 5
    with pytest.raises(ValueError):
        cc.attach_sweep(concepts[:1], sweep)


def test_pool_candidate_rule_equals_source_rule():
    rule = np.array([True, False, True])
    sae = {"f1": {"concept": "last_3/2011/f1", "build": "last_3", "reference": 2011, "family_id": "f1", "rule_dict": {"r": 1}, "cutoff": 0.5, "rule_identity": "a",
                  "in_R": rule, "inner_split": {"seed": 11}, "medoid": [42, 3]}}
    group = {"f1": {"concept": "group0/last_3/2011/f1", "build": "last_3", "reference": 2011, "family_id": "f1", "rule_dict": {"r": 2}, "cutoff": 0.7,
                    "rule_identity": "b", "in_R": ~rule, "inner_split": {"seed": 11}, "medoid": [0, 0]}}
    meta = [{"source": "group", "id": "f1", "column": 0, "positive_count": 9},
            {"source": "sae", "id": "f1", "column": 0, "positive_count": 8},
            {"source": "pca", "id": "p+", "column": 1, "positive_count": 7}]
    concepts, candidates = cc.pool_concepts(meta, {"sae": sae, "group": group, "pca": {}})
    assert [c["family_id"] for c in candidates] == ["group:f1", "sae:f1", "pca:p+"]
    assert [c["medoid"] for c in candidates] == [[0, 0], [0, 1], [0, 2]] and candidates[0]["positive_count"] == 9
    assert [c["family_id"] for c in concepts] == ["group:f1", "sae:f1"]
    for pooled, source in zip(concepts, (group["f1"], sae["f1"])):
        for key in ("rule_dict", "cutoff", "rule_identity", "inner_split"):
            assert pooled[key] == source[key]
        np.testing.assert_array_equal(pooled["in_R"], source["in_R"])
    assert concepts[1]["medoid"] == [0, 1] and concepts[1]["basis"] == "pool"
    assert concepts[0]["concept"] != group["f1"]["concept"]


# ---------------------------------------------------------------- batched readouts and bootstrap (torch)

def test_gpu_weighted_auroc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(5)
    n = 60
    labels = rng.random(n) < 0.4
    scores = np.round(rng.standard_normal((4, n)), 1)
    weights = rng.integers(0, 4, (7, n)).astype(float)
    weights[3] = 0.0
    weights[4, ~labels] = 0.0
    got = cc.weighted_auroc(labels, scores, weights)
    assert got.shape == (4, 7)
    for f in range(4):
        for d in range(7):
            if len(np.unique(labels[weights[d] > 0])) == 2:
                assert abs(got[f, d] - roc_auc_score(labels, scores[f], sample_weight=weights[d])) < 1e-12
            else:
                assert np.isnan(got[f, d])


def test_batch_size_fits_half_free_memory(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1000, 2000))
    cc.BATCH_LOG.clear()
    assert cc.batch_size(100, 50) == 5
    assert cc.batch_size(100, 3) == 3
    assert cc.batch_size(10 ** 6, 50) == 1
    assert [b["size"] for b in cc.BATCH_LOG] == [5, 3, 1]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cc.batch_size(100, 50) == 50


def test_batched_readouts_match_numpy():
    from sklearn.metrics import roc_auc_score
    import tcav_geometry as geometry
    fit, sets = sets_fixture()
    X, rule, activation, patients = fit
    seeds = range(1001, 1006)
    fitted = cc.fit_candidates(X, [{"rule": rule, "activation": activation, "patients": patients}], seeds, 1)[0]
    result = cc.pool_results(fitted, list(seeds), X, rule, activation, patients, sets)
    s = sets["evaluation"]
    for row, fit_ in zip(result["fits"], [f for pool in cc.POOLS for f in fitted[pool][:-1]]):
        assert row["status"] == fit_["status"]
        if fit_["status"] != "measured":
            continue
        scores = s["X"] @ fit_["coef"] + fit_["intercept"]
        pop = result["populations"]["evaluation"]
        for name in ("all", "restricted"):
            rows_ = pop[name]["rows"]
            expected = roc_auc_score(s["rule"][rows_], scores[rows_])
            assert abs(row[f"auroc_{name}_evaluation"] - expected) < 1e-12
        angle = geometry.boundary_angle(geometry.coherence(s["gradients"])["mean_direction"], fit_["direction"][None])[0]
        assert abs(row["angle_evaluation"] - angle) < 1e-10
        sensitivity = (s["gradients"][s["rule"]][np.isfinite(s["gradients"][s["rule"]]).all(1)] @ fit_["direction"]).mean()
        assert abs(row["sensitivity_evaluation"] - sensitivity) < 1e-12
    directions = result["directions"]["legacy"]
    valid = directions[np.isfinite(directions).all(1)]
    np.testing.assert_allclose(result["cosines"]["legacy"], (valid @ valid.T)[np.triu_indices(len(valid), 1)], atol=1e-12)


def per_concept_fixture():
    import pandas as pd
    rows = []
    for source, concept, auroc in (("sae", "c1", 0.9), ("sae", "c2", 0.7), ("random1", "r1", 0.5), ("random2", "r2", 0.6),
                                   ("pool", "p1", 0.99)):
        for pool in ("legacy", "literal"):
            rows.append({"panel": "primary", "source": source, "concept": concept, "pool": pool, "measured_share": 1.0,
                         "auroc_all": auroc, "auroc_restricted": auroc + 0.01, "angle_median": -10.0, "angle_p05": -12.0, "angle_p95": -8.0})
    return pd.DataFrame(rows)


def test_source_pool_summary_keeps_each_random_draw():
    import pandas as pd
    cosines = pd.DataFrame([{"concept": c, "pool": p, "cosine_median": 0.9} for c in ("c1", "c2", "r1", "r2", "p1") for p in ("legacy", "literal")])
    out = cc.source_pool_summary(per_concept_fixture(), cosines).set_index(["source", "pool"])
    assert set(out.index.get_level_values("source")) == {"sae", "random1", "random2", "random (all draws)"}
    assert out.loc[("sae", "legacy"), "concepts"] == 2 and out.loc[("random1", "legacy"), "concepts"] == 1
    assert out.loc[("random (all draws)", "legacy"), "concepts"] == 2
    assert out.loc[("sae", "legacy"), "median_auroc_all"] == pytest.approx(0.8)
    assert out.loc[("random1", "literal"), "median_auroc_restricted"] == pytest.approx(0.51)
    assert out.loc[("random (all draws)", "literal"), "median_auroc_restricted"] == pytest.approx(0.56)
    assert out.loc[("sae", "literal"), "median_fit_cosine"] == pytest.approx(0.9)


def test_cosine_table_and_write_source_tables(tmp_path):
    import pandas as pd
    unit = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [np.nan, np.nan]])
    classifiers = {c: {f"{pool}|coefs": unit * 3 for pool in cc.POOLS} for c in ("c1", "c2", "r1", "r2", "p1")}
    for item in classifiers.values():
        item.update({f"{pool}|statuses": ["measured"] * 3 + ["failed"] for pool in cc.POOLS})
    table = cc.cosine_table(classifiers).set_index(["concept", "pool"])
    assert table.loc[("c1", "legacy"), "pairs"] == 3
    assert table.loc[("c1", "legacy"), "cosine_median"] == pytest.approx(0.0)
    assert table.loc[("c1", "legacy"), "cosine_min"] == pytest.approx(0.0)
    per = per_concept_fixture()
    per.to_csv(tmp_path / "development_by_concept.csv", index=False)
    per.to_csv(tmp_path / "evaluation_by_concept.csv", index=False)
    cc.write_source_tables(tmp_path, classifiers)
    for name in ("fit_cosines_by_concept", "pool_summary_by_source_development", "pool_summary_by_source_evaluation"):
        assert (tmp_path / f"{name}.csv").exists()
    assert "median_fit_cosine" in pd.read_csv(tmp_path / "pool_summary_by_source_evaluation.csv").columns
    saved = {f"{c.replace('/', '~')}|{pool}|{part}": v for c, item in classifiers.items() for pool in cc.POOLS
             for part, v in (("coef", item[f"{pool}|coefs"].astype(np.float32)), ("status", np.array(item[f"{pool}|statuses"])))}
    np.savez(tmp_path / "cav.npz", **saved)
    again = cc.classifiers_from_npz(tmp_path / "cav.npz")
    assert again["c1"]["legacy|statuses"].tolist() == classifiers["c1"]["legacy|statuses"]
